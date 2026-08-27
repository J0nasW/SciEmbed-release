"""Generate paraphrased BFR queries to address lexical-leakage confound.

The original BFR queries are body sentences sampled verbatim from each candidate's
body. BM25 over title+abstract+body therefore reaches 95.2 R@1 by exact-string
match (results.tex:293). The same-field-pool variant (Table 12) addresses
cross-field separability but does NOT address this lexical-leakage confound:
the within-recipe monotonicity (61.7 -> 62.2 -> 67.1 R@10) could equally reflect
"more long-context lexical surface to match" as "long-context comprehension."

This script paraphrases each query so the n-gram set differs substantially from
the source body sentence while preserving the semantic content. The retrieval
pipeline is otherwise identical (same 9,749 candidates, same gold mapping).

Output: a queries.parquet file with the same columns as the original
(query_id, query_text, gold_corpus_id, ...) but with `query_text` replaced by
the paraphrase.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


SYSTEM_PROMPT = (
    "You are a careful scientific writer. Given one sentence drawn from the body "
    "of a research paper, write ONE paraphrase that preserves the exact meaning "
    "but uses different wording, sentence structure, and phrasing. Constraints:\n"
    "  - keep the same factual content (numbers, units, named entities unchanged)\n"
    "  - change at least 60% of the content words to synonyms or rephrasings\n"
    "  - do NOT add new information or change emphasis\n"
    "  - return ONLY the paraphrase; no preamble, no quotes, no list markers\n"
    "  - one sentence, similar length to the source"
)


def build_prompt(tokenizer, sentence: str, enable_thinking: bool = False) -> str:
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Source sentence:\n{sentence}\n\nParaphrase:"},
    ]
    # Qwen3 supports an `enable_thinking` flag in its chat template; ignored on
    # other tokenizers. We set False for one-shot paraphrasing (no chain-of-thought).
    try:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
        )
    except TypeError:
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--queries",
        type=Path,
        required=True,
        help="Path to original queries.parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write paraphrased queries.parquet",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Instruction-tuned LLM for paraphrasing",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument(
        "--audit",
        type=Path,
        default=None,
        help="Optional JSONL path to dump (id, original, paraphrase, jaccard_3gram) tuples",
    )
    args = parser.parse_args()

    log.info("Loading queries from %s", args.queries)
    table = pq.read_table(args.queries)
    df = table.to_pandas()
    log.info("Loaded %d queries", len(df))

    log.info("Loading paraphrase model: %s", args.model)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    sentences = df["query_text"].tolist()
    paraphrases: list[str] = []
    audit_rows: list[dict] = []

    t0 = time.time()
    for start in range(0, len(sentences), args.batch_size):
        batch = sentences[start : start + args.batch_size]
        prompts = [build_prompt(tok, s) for s in batch]
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to("cuda")

        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                repetition_penalty=1.0,
                pad_token_id=tok.pad_token_id,
            )

        # Strip the prompt tokens to keep only the generated continuation.
        gen = out[:, enc["input_ids"].shape[1] :]
        decoded = tok.batch_decode(gen, skip_special_tokens=True)

        for src, txt in zip(batch, decoded):
            ph = txt.strip().splitlines()[0].strip().strip('"').strip("'")
            paraphrases.append(ph)
            if args.audit is not None:
                audit_rows.append({"original": src, "paraphrase": ph})

        if (start // args.batch_size) % 10 == 0:
            elapsed = time.time() - t0
            done = start + len(batch)
            log.info(
                "Paraphrased %d/%d (%.1fs elapsed, %.2fs/sentence)",
                done,
                len(sentences),
                elapsed,
                elapsed / max(done, 1),
            )

    elapsed = time.time() - t0
    log.info("Done. %d paraphrases in %.1fs (%.2fs/sentence)", len(paraphrases), elapsed, elapsed / len(paraphrases))

    df["query_text_original"] = df["query_text"]
    df["query_text"] = paraphrases

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(df), args.output)
    log.info("Wrote %s", args.output)

    if args.audit is not None:
        args.audit.parent.mkdir(parents=True, exist_ok=True)
        with args.audit.open("w") as f:
            for r in audit_rows:
                f.write(json.dumps(r) + "\n")
        log.info("Audit written to %s", args.audit)


if __name__ == "__main__":
    main()
