"""Tests for citation triplet generation."""

from __future__ import annotations

import pytest

from sciembed.data.citation_triplets import (
    format_anchor,
    format_document,
    sample_hard_negative,
)


class TestFormatting:
    def test_format_anchor(self) -> None:
        result = format_anchor("My Paper Title", "This is the abstract.")
        assert result == "search_query: My Paper Title. This is the abstract."

    def test_format_document(self) -> None:
        result = format_document("Doc Title", "Doc abstract.")
        assert result == "search_document: Doc Title. Doc abstract."

    def test_format_empty_abstract(self) -> None:
        result = format_anchor("Title", "")
        assert result == "search_query: Title. "


class TestSampleHardNegative:
    @pytest.fixture
    def papers(self) -> dict[int, dict]:
        return {
            1: {"title": "Paper 1", "abstract_preview": "Abs 1", "field_of_study": "Physics", "citationcount": 5},
            2: {"title": "Paper 2", "abstract_preview": "Abs 2", "field_of_study": "Physics", "citationcount": 3},
            3: {"title": "Paper 3", "abstract_preview": "Abs 3", "field_of_study": "Physics", "citationcount": 8},
            4: {"title": "Paper 4", "abstract_preview": "Abs 4", "field_of_study": "Biology", "citationcount": 2},
            5: {"title": "Paper 5", "abstract_preview": "Abs 5", "field_of_study": "Biology", "citationcount": 1},
        }

    @pytest.fixture
    def field_index(self) -> dict[str, list[int]]:
        return {
            "Physics": [1, 2, 3],
            "Biology": [4, 5],
        }

    def test_returns_valid_negative(
        self, papers: dict, field_index: dict
    ) -> None:
        neg_id = sample_hard_negative(
            anchor_id=1,
            positive_id=2,
            field_of_study="Physics",
            papers=papers,
            field_index=field_index,
            cited_set={2},
            negative_mix={"same_topic": 1.0, "two_hop": 0.0, "random": 0.0},
        )
        assert neg_id is not None
        assert neg_id != 1  # not anchor
        assert neg_id != 2  # not positive
        assert neg_id not in {2}  # not in cited set

    def test_same_field_negative(
        self, papers: dict, field_index: dict
    ) -> None:
        # Force same-field sampling
        neg_id = sample_hard_negative(
            anchor_id=1,
            positive_id=2,
            field_of_study="Physics",
            papers=papers,
            field_index=field_index,
            cited_set={2},
            negative_mix={"same_topic": 1.0, "two_hop": 0.0, "random": 0.0},
        )
        # Should be paper 3 (only remaining in Physics)
        assert neg_id == 3

    def test_random_negative_fallback(
        self, papers: dict, field_index: dict
    ) -> None:
        # Force random sampling
        neg_id = sample_hard_negative(
            anchor_id=1,
            positive_id=2,
            field_of_study=None,  # no field
            papers=papers,
            field_index=field_index,
            cited_set={2},
            negative_mix={"same_topic": 0.0, "two_hop": 0.0, "random": 1.0},
        )
        assert neg_id is not None
        assert neg_id != 1
        assert neg_id != 2
