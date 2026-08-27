"""Signal E: Section-aware embeddings from full-text papers.

Extracts section boundaries (Introduction, Methods, Results, Discussion)
from full-text papers and generates section-type-prefixed training pairs.

Positive pairs: same section type from papers sharing methodology
(identified via cite_method intent from the citation graph).
Negative pairs: cross-section pairs (Methods(A) vs Introduction(B)).
"""

from __future__ import annotations

import logging
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from omegaconf import DictConfig
from tqdm import tqdm

from sciembed.data.datalake import DatalakeConnection
from sciembed.data.text_cleaning import clean_text

logger = logging.getLogger(__name__)

# Section detection patterns

# Regex patterns for common section headings across different formats
# Handles: markdown (#), LaTeX (\section{}), plain text headings
SECTION_PATTERNS = {
    "introduction": re.compile(
        r"(?:^|\n)(?:#{1,3}\s*|\\section\{|\\subsection\{)?"
        r"\s*(?:\d+[\.\)]\s*)?"
        r"(?:introduction|background\s+and\s+motivation|overview)"
        r"(?:\}|\s*\n)",
        re.IGNORECASE | re.MULTILINE,
    ),
    "methods": re.compile(
        r"(?:^|\n)(?:#{1,3}\s*|\\section\{|\\subsection\{)?"
        r"\s*(?:\d+[\.\)]\s*)?"
        r"(?:method(?:s|ology)?|materials?\s+and\s+methods|experimental\s+(?:setup|design|methods)"
        r"|approach|model(?:\s+architecture)?|implementation|system\s+description)"
        r"(?:\}|\s*\n)",
        re.IGNORECASE | re.MULTILINE,
    ),
    "results": re.compile(
        r"(?:^|\n)(?:#{1,3}\s*|\\section\{|\\subsection\{)?"
        r"\s*(?:\d+[\.\)]\s*)?"
        r"(?:results?|experiments?(?:\s+and\s+results)?|evaluation|findings|empirical\s+(?:results|analysis))"
        r"(?:\}|\s*\n)",
        re.IGNORECASE | re.MULTILINE,
    ),
    "discussion": re.compile(
        r"(?:^|\n)(?:#{1,3}\s*|\\section\{|\\subsection\{)?"
        r"\s*(?:\d+[\.\)]\s*)?"
        r"(?:discussion|conclusion(?:s)?|discussion\s+and\s+conclusion(?:s)?"
        r"|summary|limitations?\s+and\s+future\s+work|future\s+work)"
        r"(?:\}|\s*\n)",
        re.IGNORECASE | re.MULTILINE,
    ),
}

# Prefixes used in training
SECTION_PREFIX_MAP = {
    "introduction": "section_intro",
    "methods": "section_methods",
    "results": "section_results",
    "discussion": "section_discussion",
}

# Minimum chars for a section to be usable
MIN_SECTION_CHARS = 200
MAX_SECTION_CHARS = 16_000  # ~4K tokens at ~4 chars/token


@dataclass
class ExtractedSection:
    """A detected section from a paper."""

    section_type: str
    text: str
    start_pos: int
    end_pos: int


# SQL queries

FULLTEXT_STREAM_QUERY = """
SELECT doi, source, text
FROM fulltext.papers
WHERE has_full_text = true
  AND text_length BETWEEN {min_len} AND {max_len}
  AND source = '{source}'
"""


def extract_sections(text: str) -> list[ExtractedSection]:
    """Extract section boundaries from a full-text paper.

    Uses regex-based heuristics to identify Introduction, Methods,
    Results, and Discussion/Conclusion sections.

    Args:
        text: Full text of the paper.

    Returns:
        List of ExtractedSection objects, sorted by position.
    """
    if not text or len(text) < 500:
        return []

    # Find all section heading positions
    headings: list[tuple[str, int]] = []
    for section_type, pattern in SECTION_PATTERNS.items():
        for match in pattern.finditer(text):
            headings.append((section_type, match.start()))

    if not headings:
        return []

    headings.sort(key=lambda x: x[1])

    # Deduplicate: keep first occurrence of each section type
    seen = set()
    unique_headings = []
    for section_type, pos in headings:
        if section_type not in seen:
            seen.add(section_type)
            unique_headings.append((section_type, pos))

    # Extract section text between consecutive headings
    sections = []
    for i, (section_type, start) in enumerate(unique_headings):
        # End at next heading or end of text
        if i + 1 < len(unique_headings):
            end = unique_headings[i + 1][1]
        else:
            end = len(text)

        section_text = text[start:end].strip()

        # Remove the heading line itself
        first_newline = section_text.find("\n")
        if first_newline > 0:
            section_text = section_text[first_newline:].strip()

        # Filter by length
        if len(section_text) < MIN_SECTION_CHARS:
            continue
        if len(section_text) > MAX_SECTION_CHARS:
            section_text = section_text[:MAX_SECTION_CHARS]

        sections.append(
            ExtractedSection(
                section_type=section_type,
                text=section_text,
                start_pos=start,
                end_pos=end,
            )
        )

    return sections


def build_section_pairs(cfg: DictConfig) -> dict[str, Any]:
    """Generate section-aware training pairs from full-text papers.

    Two types of pairs:
    1. Same-section positive: Methods(A) ↔ Methods(B) for methodology-linked papers
    2. Cross-section negative: Methods(A) should NOT match Introduction(C)

    Args:
        cfg: SectionExtractorConfig (as DictConfig).

    Returns:
        Statistics dict.
    """
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dl = DatalakeConnection(
        db_path=cfg.datalake.db_path,
        read_only=cfg.datalake.read_only,
        threads=cfg.datalake.threads,
        memory_limit=cfg.datalake.memory_limit,
    )

    target = cfg.num_pairs
    page_size = cfg.page_size
    shard_size = cfg.shard_size
    sources = list(cfg.sources)

    # Phase 1: Extract sections from full-text papers, index by DOI
    logger.info("Phase 1: Extracting sections from full-text papers...")
    # Store: doi → {section_type: text}
    paper_sections: dict[str, dict[str, str]] = {}
    extraction_stats: dict[str, int] = {"total_papers": 0, "papers_with_sections": 0}
    section_type_counts: dict[str, int] = {st: 0 for st in SECTION_PREFIX_MAP}

    for source in sources:
        max_papers = cfg.max_papers_per_source
        source_extracted = 0

        query = FULLTEXT_STREAM_QUERY.format(
            min_len=cfg.min_text_length,
            max_len=cfg.max_text_length,
            source=source,
        )

        logger.info("  %s: streaming full-text papers...", source)
        pbar = tqdm(desc=f"  {source} sections", unit="papers")

        for batch in dl.stream_query(query, batch_size=50_000):
            if source_extracted >= max_papers:
                break

            dois = batch.column("doi").to_pylist()
            sources_col = batch.column("source").to_pylist()
            texts = batch.column("text").to_pylist()

            for doi, text_source, text in zip(dois, sources_col, texts):
                if source_extracted >= max_papers:
                    break

                if not doi or not text:
                    continue

                extraction_stats["total_papers"] += 1

                cleaned = clean_text(text, source=text_source)
                if not cleaned:
                    continue

                sections = extract_sections(cleaned)
                if not sections:
                    continue

                extraction_stats["papers_with_sections"] += 1
                section_dict = {}
                for sec in sections:
                    section_dict[sec.section_type] = sec.text
                    section_type_counts[sec.section_type] += 1

                paper_sections[doi] = section_dict
                source_extracted += 1

            pbar.update(len(dois))

        pbar.close()
        logger.info("  %s: extracted sections from %d papers", source, source_extracted)

    logger.info(
        "Section extraction: %d/%d papers have sections (%s)",
        extraction_stats["papers_with_sections"],
        extraction_stats["total_papers"],
        ", ".join(f"{k}={v}" for k, v in section_type_counts.items()),
    )

    # Phase 2: Generate section-aware pairs
    logger.info("Phase 2: Generating section-aware pairs...")
    pair_count = 0
    shard_idx = 0
    shard_data: dict[str, list] = {
        "anchor": [],
        "positive": [],
        "anchor_section": [],
        "positive_section": [],
        "pair_type": [],  # "same_section" or "cross_section_negative"
    }

    def _flush_shard() -> None:
        nonlocal shard_idx, shard_data
        if not shard_data["anchor"]:
            return
        table = pa.table(shard_data)
        shard_path = output_dir / f"section_pairs_{shard_idx:05d}.parquet"
        pq.write_table(table, shard_path, compression="zstd")
        shard_idx += 1
        shard_data = {k: [] for k in shard_data}

    # Build DOI list for random pairing
    doi_list = list(paper_sections.keys())
    if len(doi_list) < 2:
        logger.warning("Not enough papers with sections to generate pairs")
        dl.close()
        return {"total_pairs": 0, "num_shards": 0}

    # Generate pairs
    positive_ratio = cfg.get("positive_ratio", 0.6)  # 60% same-section, 40% cross-section
    pbar = tqdm(total=target, desc="Section pairs", unit="pairs")

    attempts = 0
    max_attempts = target * 5  # avoid infinite loop

    while pair_count < target and attempts < max_attempts:
        attempts += 1

        # Pick two random papers
        doi_a, doi_b = random.sample(doi_list, 2)
        sections_a = paper_sections[doi_a]
        sections_b = paper_sections[doi_b]

        if random.random() < positive_ratio:
            # Same-section positive pair
            common_sections = set(sections_a.keys()) & set(sections_b.keys())
            if not common_sections:
                continue

            section_type = random.choice(list(common_sections))
            prefix = SECTION_PREFIX_MAP[section_type]

            shard_data["anchor"].append(f"{prefix}: {sections_a[section_type]}")
            shard_data["positive"].append(f"{prefix}: {sections_b[section_type]}")
            shard_data["anchor_section"].append(section_type)
            shard_data["positive_section"].append(section_type)
            shard_data["pair_type"].append("same_section")
        else:
            # Cross-section negative pair
            sections_a_types = list(sections_a.keys())
            sections_b_types = list(sections_b.keys())
            if not sections_a_types or not sections_b_types:
                continue

            sec_a_type = random.choice(sections_a_types)
            # Pick a different section type from paper B
            diff_types = [t for t in sections_b_types if t != sec_a_type]
            if not diff_types:
                continue
            sec_b_type = random.choice(diff_types)

            prefix_a = SECTION_PREFIX_MAP[sec_a_type]
            prefix_b = SECTION_PREFIX_MAP[sec_b_type]

            shard_data["anchor"].append(f"{prefix_a}: {sections_a[sec_a_type]}")
            shard_data["positive"].append(f"{prefix_b}: {sections_b[sec_b_type]}")
            shard_data["anchor_section"].append(sec_a_type)
            shard_data["positive_section"].append(sec_b_type)
            shard_data["pair_type"].append("cross_section_negative")

        pair_count += 1
        pbar.update(1)

        if len(shard_data["anchor"]) >= shard_size:
            _flush_shard()

    _flush_shard()
    pbar.close()

    dl.close()

    stats = {
        "total_pairs": pair_count,
        "num_shards": shard_idx,
        "output_dir": str(output_dir),
        "extraction_stats": extraction_stats,
        "section_type_counts": section_type_counts,
    }
    logger.info("Generated %d section pairs in %d shards", pair_count, shard_idx)
    return stats
