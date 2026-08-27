"""Tests for novel training signal modules (Signals B, C, E, and data mixer)."""

import pytest

from sciembed.data.citation_contexts import _filter_context
from sciembed.data.intent_triplets import _parse_intents, INTENT_PREFIX_MAP
from sciembed.data.section_extractor import extract_sections, SECTION_PREFIX_MAP


# Signal B: Citation context filtering


class TestFilterContext:
    def test_valid_context(self):
        text = "Smith et al. developed a novel approach to protein folding using attention."
        result = _filter_context(text, min_chars=20, max_chars=500)
        assert result == text

    def test_too_short(self):
        assert _filter_context("Short", min_chars=20, max_chars=500) is None

    def test_too_long(self):
        text = "x " * 300  # 600 chars
        assert _filter_context(text, min_chars=20, max_chars=500) is None

    def test_empty(self):
        assert _filter_context("", min_chars=20, max_chars=500) is None
        assert _filter_context(None, min_chars=20, max_chars=500) is None

    def test_too_few_words(self):
        assert _filter_context("ab", min_chars=1, max_chars=500) is None

    def test_low_alpha_ratio(self):
        # Mostly numbers/symbols
        text = "123 456 789 012 345 678 901"
        assert _filter_context(text, min_chars=5, max_chars=500) is None

    def test_strips_whitespace(self):
        text = "  Smith et al. proposed a method for classification.  "
        result = _filter_context(text, min_chars=10, max_chars=500)
        assert result == text.strip()


# Signal C: Intent parsing


class TestParseIntents:
    def test_list_input(self):
        assert _parse_intents(["background", "methodology"]) == ["background", "methodology"]

    def test_string_input(self):
        assert _parse_intents("background, methodology") == ["background", "methodology"]

    def test_none_input(self):
        assert _parse_intents(None) == []

    def test_empty_list(self):
        assert _parse_intents([]) == []

    def test_strips_and_lowercases(self):
        assert _parse_intents(["  Background ", "RESULT"]) == ["background", "result"]

    def test_filters_empty_strings(self):
        assert _parse_intents(["background", "", "  ", "result"]) == ["background", "result"]


class TestIntentPrefixMap:
    def test_all_intents_mapped(self):
        for intent in ["background", "methodology", "result"]:
            assert intent in INTENT_PREFIX_MAP

    def test_prefix_format(self):
        assert INTENT_PREFIX_MAP["background"] == "cite_background"
        assert INTENT_PREFIX_MAP["methodology"] == "cite_method"
        assert INTENT_PREFIX_MAP["result"] == "cite_result"


# Signal E: Section extraction


class TestExtractSections:
    def test_markdown_sections(self):
        text = """# Introduction

This paper presents a novel approach to scientific embedding.
We build on prior work in contrastive learning and citation analysis.
This is a sufficiently long introduction section that passes the minimum character threshold for extraction.
Additional context about the research problem and motivation is provided here.

# Methods

We use a ModernBERT-base encoder with 8K context length.
Training follows a two-stage pipeline: domain-adaptive MLM pretraining
followed by multi-signal contrastive fine-tuning. The model architecture
incorporates RoPE positional embeddings and Flash Attention 2 for efficiency.

# Results

Our model achieves state-of-the-art performance on SciRepEval, with
X.X average across all task types. The largest improvements come from
the search and proximity tasks, where citation context training provides
the strongest signal. We evaluate on both SciRepEval and MTEB benchmarks.

# Discussion

The results demonstrate that citation context sentences provide a richer
training signal than binary citation edges alone. Intent conditioning
further improves retrieval quality for methodology-specific queries.
We discuss limitations and directions for future research.
"""
        sections = extract_sections(text)
        assert len(sections) >= 2  # at least some sections detected
        section_types = {s.section_type for s in sections}
        # Should find at least introduction and methods
        assert "introduction" in section_types or "methods" in section_types

    def test_numbered_sections(self):
        text = """1. Introduction

This paper addresses the problem of scientific document representation.
Prior approaches are limited to title and abstract inputs. We propose a
full-text approach leveraging modern long-context encoders with support for
up to 8192 tokens per input sequence.

2. Methods

We train a ModernBERT-base model on 13.2M full-text papers using a
multi-stage training pipeline that combines domain-adaptive pretraining
with multi-signal contrastive fine-tuning on citation graph data with
intent labels and section-aware representations.

3. Results

Experimental evaluation shows significant improvements on SciRepEval.
Our ablation study demonstrates the contribution of each novel training
signal, with citation context and intent conditioning providing the
largest gains across all evaluation metrics.
"""
        sections = extract_sections(text)
        assert len(sections) >= 1

    def test_empty_text(self):
        assert extract_sections("") == []
        assert extract_sections("short") == []

    def test_no_sections_found(self):
        text = "This is just a block of text without any section headings. " * 20
        sections = extract_sections(text)
        assert sections == []

    def test_section_prefix_map(self):
        assert "introduction" in SECTION_PREFIX_MAP
        assert "methods" in SECTION_PREFIX_MAP
        assert "results" in SECTION_PREFIX_MAP
        assert "discussion" in SECTION_PREFIX_MAP
        assert SECTION_PREFIX_MAP["methods"] == "section_methods"


# Data mixer: column normalization


class TestDataMixerNormalize:
    def test_normalize_triplet_columns(self):
        import pyarrow as pa
        from sciembed.data.data_mixer import _normalize_columns

        table = pa.table({
            "anchor": ["a1", "a2"],
            "positive": ["p1", "p2"],
            "negative": ["n1", "n2"],
        })
        result = _normalize_columns(table, "citation_triplets")
        assert "signal_type" in result.column_names
        assert result.column("signal_type").to_pylist() == ["citation_triplets"] * 2

    def test_normalize_query_document_columns(self):
        import pyarrow as pa
        from sciembed.data.data_mixer import _normalize_columns

        table = pa.table({
            "query": ["q1", "q2"],
            "document": ["d1", "d2"],
        })
        result = _normalize_columns(table, "citation_contexts")
        assert result.column("anchor").to_pylist() == ["q1", "q2"]
        assert result.column("positive").to_pylist() == ["d1", "d2"]
        assert result.column("negative").to_pylist() == [None, None]

    def test_normalize_section_columns(self):
        import pyarrow as pa
        from sciembed.data.data_mixer import _normalize_columns

        table = pa.table({
            "anchor": ["a1"],
            "positive": ["p1"],
            "pair_type": ["same_section"],
        })
        result = _normalize_columns(table, "section_pairs")
        assert len(result) == 1
        assert result.column("negative").to_pylist() == [None]

    def test_sample_rows(self):
        import pyarrow as pa
        from sciembed.data.data_mixer import _sample_rows

        table = pa.table({"x": list(range(100))})
        sampled = _sample_rows(table, 10)
        assert len(sampled) == 10

    def test_sample_rows_larger_than_table(self):
        import pyarrow as pa
        from sciembed.data.data_mixer import _sample_rows

        table = pa.table({"x": [1, 2, 3]})
        sampled = _sample_rows(table, 100)
        assert len(sampled) == 3
