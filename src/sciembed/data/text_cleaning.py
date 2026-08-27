"""Source-aware scientific text normalization and cleaning."""

from __future__ import annotations

import re
import unicodedata


# Reference section: standalone heading to end of document
_RE_REFERENCES = re.compile(
    r"\n\s*(?:References|REFERENCES|Bibliography|BIBLIOGRAPHY)\s*\n.*",
    re.DOTALL,
)

# LaTeX environments (remove \begin{...}...\end{...} for non-math envs)
_RE_LATEX_ENV = re.compile(
    r"\\(?:begin|end)\{(?:document|figure|table|tabular|itemize|enumerate|"
    r"description|thebibliography|abstract|acknowledgments?)\}",
)

# LaTeX citation commands: \cite{...}, \citep{...}, \citet{...}, \citeauthor{...}
_RE_LATEX_CITE = re.compile(r"\\cite[a-z]*\{[^}]*\}")

# LaTeX ref commands: \ref{...}, \eqref{...}, \label{...}
_RE_LATEX_REF = re.compile(r"\\(?:eq)?ref\{[^}]*\}|\\label\{[^}]*\}")

# LaTeX formatting commands (keep content): \textbf{content} → content
_RE_LATEX_FORMAT = re.compile(r"\\(?:textbf|textit|textrm|texttt|emph|underline)\{([^}]*)\}")

# LaTeX section commands: \section{...} → content
_RE_LATEX_SECTION = re.compile(r"\\(?:sub)*section\*?\{([^}]*)\}")

# Stray LaTeX commands (\command without braces)
_RE_LATEX_CMD = re.compile(r"\\(?:noindent|newpage|clearpage|bigskip|medskip|smallskip|\\)")

# XML/HTML tags (PMC JATS artifacts)
_RE_XML_TAGS = re.compile(r"<[^>]+>")

# URLs
_RE_URL = re.compile(r"https?://\S+")

# Email addresses
_RE_EMAIL = re.compile(r"\S+@\S+\.\S+")

# Multiple whitespace (spaces, tabs) collapsed to single space
_RE_MULTI_SPACE = re.compile(r"[^\S\n]+")

# Multiple newlines collapsed to double newline
_RE_MULTI_NEWLINE = re.compile(r"\n{3,}")

# PMC boilerplate patterns
_RE_PMC_BOILERPLATE = re.compile(
    r"(?:This article has been cited by other articles in PMC|"
    r"Articles from .* are provided here courtesy of|"
    r"External link\. Please review our privacy policy|"
    r"Go to:)",
    re.IGNORECASE,
)

# Publisher watermarks / copyright lines
_RE_COPYRIGHT = re.compile(
    r"(?:©|Copyright|All rights reserved|Licensed under|Published by).*?(?:\n|$)",
    re.IGNORECASE,
)


def remove_references(text: str) -> str:
    """Remove the references/bibliography section from the end of a paper."""
    return _RE_REFERENCES.sub("", text)


def clean_latex(text: str) -> str:
    """Clean LaTeX artifacts while preserving meaningful math."""
    text = _RE_LATEX_CITE.sub("", text)
    text = _RE_LATEX_REF.sub("", text)
    text = _RE_LATEX_ENV.sub("", text)
    text = _RE_LATEX_FORMAT.sub(r"\1", text)
    text = _RE_LATEX_SECTION.sub(r"\n\1\n", text)
    text = _RE_LATEX_CMD.sub("", text)
    return text


def clean_xml(text: str) -> str:
    """Remove XML/HTML tags from PMC JATS extraction."""
    return _RE_XML_TAGS.sub("", text)


def remove_boilerplate(text: str) -> str:
    """Remove publisher watermarks, PMC footers, and copyright lines."""
    text = _RE_PMC_BOILERPLATE.sub("", text)
    text = _RE_COPYRIGHT.sub("", text)
    return text


def remove_urls_emails(text: str) -> str:
    """Remove URLs and email addresses."""
    text = _RE_URL.sub("", text)
    text = _RE_EMAIL.sub("", text)
    return text


def normalize_whitespace(text: str) -> str:
    """Normalize unicode (NFKC), collapse whitespace."""
    text = unicodedata.normalize("NFKC", text)
    text = _RE_MULTI_SPACE.sub(" ", text)
    text = _RE_MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()


def clean_text(text: str, source: str | None = None) -> str:
    """Full cleaning pipeline for scientific text.

    Args:
        text: Raw text from the datalake.
        source: Source identifier (pmc, s2orc, arxiv, pes2o) for source-specific cleaning.

    Returns:
        Cleaned text ready for tokenization.
    """
    if not text:
        return ""

    text = remove_references(text)

    # Source-specific cleaning
    if source in ("arxiv", "s2orc"):
        text = clean_latex(text)

    # XML cleaning — once for all sources (PMC JATS + stray tags in S2ORC)
    text = clean_xml(text)
    text = remove_boilerplate(text)
    text = remove_urls_emails(text)
    text = normalize_whitespace(text)

    return text
