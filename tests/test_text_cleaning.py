"""Tests for scientific text cleaning."""

from __future__ import annotations

import pytest

from sciembed.data.text_cleaning import (
    clean_latex,
    clean_text,
    clean_xml,
    normalize_whitespace,
    remove_boilerplate,
    remove_references,
    remove_urls_emails,
)


class TestRemoveReferences:
    def test_removes_references_section(self) -> None:
        text = "Some content.\n\nReferences\n[1] Author A. Title.\n[2] Author B. Title."
        result = remove_references(text)
        assert "Some content." in result
        assert "[1] Author A" not in result

    def test_removes_bibliography(self) -> None:
        text = "Content here.\n\nBibliography\nEntry 1\nEntry 2"
        result = remove_references(text)
        assert "Content here." in result
        assert "Entry 1" not in result

    def test_preserves_inline_references(self) -> None:
        text = "As shown in the references above, this is true."
        result = remove_references(text)
        assert "references above" in result


class TestCleanLatex:
    def test_removes_cite_commands(self) -> None:
        text = r"As shown by \cite{smith2024} and \citep{jones2023}, this holds."
        result = clean_latex(text)
        assert r"\cite" not in result
        assert "As shown by" in result

    def test_removes_environments(self) -> None:
        text = r"\begin{figure} caption \end{figure} Real text here."
        result = clean_latex(text)
        assert r"\begin{figure}" not in result
        assert "Real text here." in result

    def test_preserves_math(self) -> None:
        text = r"The equation $x^2 + y^2 = z^2$ is well known."
        result = clean_latex(text)
        assert "$x^2 + y^2 = z^2$" in result

    def test_unwraps_formatting(self) -> None:
        text = r"This is \textbf{important} and \emph{emphasized}."
        result = clean_latex(text)
        assert "important" in result
        assert "emphasized" in result
        assert r"\textbf" not in result

    def test_unwraps_sections(self) -> None:
        text = r"\section{Introduction} Some text. \subsection{Background}"
        result = clean_latex(text)
        assert "Introduction" in result
        assert "Background" in result
        assert r"\section" not in result


class TestCleanXML:
    def test_removes_tags(self) -> None:
        text = "<p>This is a <italic>paragraph</italic> with <bold>tags</bold>.</p>"
        result = clean_xml(text)
        assert result == "This is a paragraph with tags."

    def test_removes_jats_tags(self) -> None:
        text = '<xref ref-type="bibr" rid="ref-1">1</xref> showed that'
        result = clean_xml(text)
        assert result == "1 showed that"


class TestRemoveBoilerplate:
    def test_removes_pmc_courtesy(self) -> None:
        text = "Content here.\nArticles from BMJ are provided here courtesy of BMJ Publishing."
        result = remove_boilerplate(text)
        assert "Content here." in result
        assert "courtesy of" not in result

    def test_removes_copyright(self) -> None:
        text = "Content.\n© 2024 The Authors. All rights reserved.\nMore content."
        result = remove_boilerplate(text)
        assert "More content." in result
        assert "©" not in result


class TestRemoveURLsEmails:
    def test_removes_urls(self) -> None:
        text = "Available at https://example.com/paper and http://test.org/data."
        result = remove_urls_emails(text)
        assert "https://" not in result
        assert "http://" not in result
        assert "Available at" in result

    def test_removes_emails(self) -> None:
        text = "Contact: author@university.edu for details."
        result = remove_urls_emails(text)
        assert "@" not in result


class TestNormalizeWhitespace:
    def test_collapses_spaces(self) -> None:
        text = "Multiple   spaces    here"
        result = normalize_whitespace(text)
        assert result == "Multiple spaces here"

    def test_collapses_newlines(self) -> None:
        text = "First paragraph.\n\n\n\n\nSecond paragraph."
        result = normalize_whitespace(text)
        assert result == "First paragraph.\n\nSecond paragraph."

    def test_nfkc_normalization(self) -> None:
        # ﬁ (U+FB01) → fi
        text = "ﬁber optic"
        result = normalize_whitespace(text)
        assert result == "fiber optic"

    def test_strips_edges(self) -> None:
        text = "  content  "
        result = normalize_whitespace(text)
        assert result == "content"


class TestCleanTextPipeline:
    def test_pmc_text(self) -> None:
        text = (
            "<p>This is a <bold>PMC</bold> paper.</p>\n"
            "Articles from BMJ are provided here courtesy of BMJ Publishing.\n"
            "Contact: test@pmc.gov\n\n"
            "References\n[1] Smith A. Title."
        )
        result = clean_text(text, source="pmc")
        assert "<p>" not in result
        assert "<bold>" not in result
        assert "courtesy" not in result
        assert "[1] Smith" not in result
        assert "PMC" in result

    def test_arxiv_text(self) -> None:
        text = (
            r"\section{Introduction}"
            "\n"
            r"As shown by \cite{smith2024}, the equation $E=mc^2$ holds."
            "\n\n"
            "References\n"
            r"\bibitem{smith2024} Smith, 2024."
        )
        result = clean_text(text, source="arxiv")
        assert "Introduction" in result
        assert "$E=mc^2$" in result
        assert r"\cite" not in result
        assert r"\bibitem" not in result

    def test_empty_text(self) -> None:
        assert clean_text("") == ""
        assert clean_text("", source="pmc") == ""

    def test_none_source(self) -> None:
        text = "Simple text with https://url.com and author@email.com."
        result = clean_text(text, source=None)
        assert "https://" not in result
        assert "@" not in result
        assert "Simple text" in result
