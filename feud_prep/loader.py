from __future__ import annotations

import io
from collections import Counter

import pandas as pd

from feud_prep.models import AnswerCount


def load_survey_csv(path_or_buffer: str | io.BytesIO | io.StringIO, encoding: str = "utf-8") -> pd.DataFrame:
    if isinstance(path_or_buffer, str):
        return pd.read_csv(path_or_buffer, encoding=encoding, dtype=str, keep_default_na=False)
    if isinstance(path_or_buffer, io.StringIO):
        return pd.read_csv(path_or_buffer, dtype=str, keep_default_na=False)
    return pd.read_csv(path_or_buffer, encoding=encoding, dtype=str, keep_default_na=False)


def filter_column_names(
    columns: list[str],
    skip_first: int = 0,
    exclude_substrings: list[str] | None = None,
) -> list[str]:
    exclude_substrings = [s.strip().lower() for s in (exclude_substrings or []) if s.strip()]
    names = list(columns)
    if skip_first > 0:
        names = names[skip_first:]
    out: list[str] = []
    for c in names:
        low = c.lower()
        if any(ex in low for ex in exclude_substrings):
            continue
        out.append(c)
    return out


def extract_raw_answers(
    df: pd.DataFrame,
    column: str,
    ignore_exact: set[str] | None = None,
) -> list[str]:
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
        if s.lower() in ignore_norm:
            continue
        out.append(s)
    return out


def aggregate_answer_counts(raw: list[str]) -> list[AnswerCount]:
    counts = Counter(raw)
    # Stable sort: frequency desc, then text asc
    items = sorted(counts.items(), key=lambda x: (-x[1], x[0].lower()))
    return [AnswerCount(text=t, count=c) for t, c in items]
