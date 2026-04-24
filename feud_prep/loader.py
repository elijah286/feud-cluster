from __future__ import annotations

import io
from collections import Counter

import pandas as pd

from feud_prep.models import AnswerCount

# Spreadsheet layout: columns A–E are metadata (id, times, email, name); first survey prompt is column F (index 5).
METADATA_COLUMN_COUNT = 5


def split_cell_on_commas(text: str) -> list[str]:
    """Split one cell on commas; each non-empty trimmed segment is its own answer."""
    return [p.strip() for p in text.split(",") if p.strip()]


def load_survey_csv(path_or_buffer: str | io.BytesIO | io.StringIO, encoding: str = "utf-8") -> pd.DataFrame:
    if isinstance(path_or_buffer, str):
        return pd.read_csv(path_or_buffer, encoding=encoding, dtype=str, keep_default_na=False)
    if isinstance(path_or_buffer, io.StringIO):
        return pd.read_csv(path_or_buffer, dtype=str, keep_default_na=False)
    return pd.read_csv(path_or_buffer, encoding=encoding, dtype=str, keep_default_na=False)


def prompt_column_headers(columns: list[str]) -> list[str]:
    """Return header names for survey prompt columns (from column F onward)."""
    if len(columns) <= METADATA_COLUMN_COUNT:
        return []
    return list(columns[METADATA_COLUMN_COUNT:])


def extract_raw_answers(
    df: pd.DataFrame,
    column: str,
    ignore_exact: set[str] | None = None,
) -> list[str]:
    """Pull free-text answers from one column.

    Respondents may put several answers in one cell separated by commas (e.g. ``a, b, c``).
    Each comma-separated segment is treated as a separate answer for counting and clustering.
    """
    ignore_exact = ignore_exact or set()
    ignore_norm = {s.strip().lower() for s in ignore_exact if s.strip()}
    if column not in df.columns:
        raise KeyError(f"Column not found: {column!r}")
    series = df[column].astype(str)
    out: list[str] = []
    for v in series:
        s = v.strip()
        if not s or s.lower() in ("nan", "none", "null"):
            continue
        phrases = split_cell_on_commas(s)
        if not phrases:
            continue
        for phrase in phrases:
            if phrase.lower() in ignore_norm:
                continue
            out.append(phrase)
    return out


def aggregate_answer_counts(raw: list[str]) -> list[AnswerCount]:
    counts = Counter(raw)
    # Stable sort: frequency desc, then text asc
    items = sorted(counts.items(), key=lambda x: (-x[1], x[0].lower()))
    return [AnswerCount(text=t, count=c) for t, c in items]
