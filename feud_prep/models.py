from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class AnswerCount:
    text: str
    count: int

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "count": self.count}


@dataclass
class Cluster:
    label: str
    members: list[AnswerCount]

    @property
    def count(self) -> int:
        return sum(m.count for m in self.members)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "count": self.count,
            "members": [{"text": m.text, "count": m.count} for m in self.members],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Cluster:
        label = str(d.get("label", "")).strip() or "Unnamed"
        raw_members = d.get("members") or []
        members: list[AnswerCount] = []
        for item in raw_members:
            if isinstance(item, str):
                members.append(AnswerCount(text=item.strip(), count=1))
            elif isinstance(item, dict):
                t = str(item.get("text", "")).strip()
                if not t:
                    continue
                c = int(item.get("count", 1))
                members.append(AnswerCount(text=t, count=max(1, c)))
        return cls(label=label, members=members)


@dataclass
class RunArtifact:
    source_file: str
    column: str
    created_at: str
    top_k: int
    clusters: list[Cluster]
    n_cells_non_empty: int
    n_unique_texts: int
    skip_first_columns: int = 0
    ignore_exact_answers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_file": self.source_file,
            "column": self.column,
            "created_at": self.created_at,
            "top_k": self.top_k,
            "clusters": [c.to_dict() for c in self.clusters],
            "n_cells_non_empty": self.n_cells_non_empty,
            "n_unique_texts": self.n_unique_texts,
            "skip_first_columns": self.skip_first_columns,
            "ignore_exact_answers": self.ignore_exact_answers,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RunArtifact:
        clusters = [Cluster.from_dict(x) for x in (d.get("clusters") or [])]
        return cls(
            source_file=str(d.get("source_file", "")),
            column=str(d.get("column", "")),
            created_at=str(d.get("created_at", "")),
            top_k=int(d.get("top_k", 10)),
            clusters=clusters,
            n_cells_non_empty=int(d.get("n_cells_non_empty", 0)),
            n_unique_texts=int(d.get("n_unique_texts", 0)),
            skip_first_columns=int(d.get("skip_first_columns", 0)),
            ignore_exact_answers=list(d.get("ignore_exact_answers") or []),
        )


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
