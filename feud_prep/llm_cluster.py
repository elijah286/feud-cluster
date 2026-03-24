from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI

from feud_prep.models import AnswerCount, Cluster

DEFAULT_MODEL = "gpt-4o-mini"
CHUNK_SIZE = 45


def _client() -> OpenAI:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to .env or the environment.")
    return OpenAI(api_key=key)


def _model() -> str:
    return os.environ.get("OPENAI_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL


def _chat_json(system: str, user: str) -> dict[str, Any]:
    client = _client()
    resp = client.chat.completions.create(
        model=_model(),
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )
    content = resp.choices[0].message.content or "{}"
    return json.loads(content)


def _build_allowed_map(answers: list[AnswerCount]) -> dict[str, AnswerCount]:
    """Map normalized key -> AnswerCount (first wins for duplicate text after strip)."""
    m: dict[str, AnswerCount] = {}
    for a in answers:
        key = a.text.strip()
        if key and key not in m:
            m[key] = AnswerCount(text=key, count=a.count)
    return m


def _parse_clusters_payload(
    data: dict[str, Any],
    allowed: dict[str, AnswerCount],
) -> list[Cluster]:
    raw_clusters = data.get("clusters")
    if not isinstance(raw_clusters, list):
        return []
    out: list[Cluster] = []
    for item in raw_clusters:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "")).strip() or "Unnamed"
        members_raw = item.get("members")
        if not isinstance(members_raw, list):
            members_raw = []
        members: list[AnswerCount] = []
        for m in members_raw:
            text: str
            if isinstance(m, str):
                text = m.strip()
            elif isinstance(m, dict):
                text = str(m.get("text", "")).strip()
            else:
                continue
            if not text:
                continue
            if text not in allowed:
                continue
            members.append(AnswerCount(text=text, count=allowed[text].count))
        if not members:
            continue
        out.append(Cluster(label=label, members=members))
    return out


def _dedupe_clusters_by_member(clusters: list[Cluster]) -> list[Cluster]:
    """If same text appears in multiple clusters, keep first occurrence."""
    seen: set[str] = set()
    fixed: list[Cluster] = []
    for c in clusters:
        new_members: list[AnswerCount] = []
        for m in c.members:
            if m.text in seen:
                continue
            seen.add(m.text)
            new_members.append(m)
        if new_members:
            fixed.append(Cluster(label=c.label, members=new_members))
    return fixed


def _sort_clusters(clusters: list[Cluster], top_k: int) -> list[Cluster]:
    clusters = sorted(clusters, key=lambda x: (-x.count, x.label.lower()))
    return clusters[:top_k]


SYSTEM_BASE = """You help run a Family Feud style game from free-text survey answers about LabVIEW.
You merge answers that mean the same thing (synonyms, typos, different phrasing).
Output MUST be valid JSON only, no markdown.
Use short, display-friendly labels (a few words).
Each "members" entry must use the EXACT text string from the input (character-for-character match after the input's own trimming)."""


def cluster_single_batch(
    answers: list[AnswerCount],
    top_k: int,
    question_context: str,
) -> list[Cluster]:
    allowed = _build_allowed_map(answers)
    payload = {
        "answers": [{"text": a.text, "count": a.count} for a in answers],
        "top_k": top_k,
        "question_context": question_context,
    }
    user = f"""Task: Group these answers into up to {top_k} clusters for the game board.

Question context (may be empty): {question_context!r}

Input JSON:
{json.dumps(payload, ensure_ascii=False)}

Return JSON shape:
{{"clusters":[{{"label":"...","members":[{{"text":"<exact from input>","count":<int>}}]}}]}}

Rules:
- Order clusters by total people (sum of member counts), descending.
- Every input answer text must appear in exactly one cluster's members (use exact "text" values from input).
- Merge duplicates of meaning; member "count" must match the input count for that text.
- Return at most {top_k} clusters; if fewer distinct ideas exist, return fewer."""
    data = _chat_json(SYSTEM_BASE, user)
    clusters = _parse_clusters_payload(data, allowed)
    clusters = _dedupe_clusters_by_member(clusters)
    # Recover missing texts into catch-all
    assigned = {m.text for c in clusters for m in c.members}
    missing = [a for a in answers if a.text not in assigned]
    if missing:
        clusters.append(Cluster(label="Other", members=missing))
        clusters = _dedupe_clusters_by_member(clusters)
    return _sort_clusters(clusters, top_k)


def cluster_chunked_then_merge(
    answers: list[AnswerCount],
    top_k: int,
    question_context: str,
) -> list[Cluster]:
    allowed = _build_allowed_map(answers)
    chunks: list[list[AnswerCount]] = []
    buf: list[AnswerCount] = []
    for a in answers:
        buf.append(a)
        if len(buf) >= CHUNK_SIZE:
            chunks.append(buf)
            buf = []
    if buf:
        chunks.append(buf)

    partial: list[dict[str, Any]] = []
    for i, ch in enumerate(chunks):
        sub_k = max(top_k + 3, 12)
        cs = cluster_single_batch(ch, sub_k, question_context)
        partial.append(
            {
                "chunk_index": i,
                "clusters": [c.to_dict() for c in cs],
            }
        )

    merge_user = f"""You previously clustered subsets of the same survey column. Now merge into one final list for the game.

Question context: {question_context!r}

Target: at most {top_k} clusters, ordered by total people descending.

Partial results JSON:
{json.dumps({"partial": partial}, ensure_ascii=False)}

Return JSON:
{{"clusters":[{{"label":"...","members":[{{"text":"<must match an input text exactly>","count":<int>}}]}}]}}

Rules:
- Every text in the partial results must appear exactly once in your output (same exact spelling as in partial).
- Merge clusters that represent the same real-world answer.
- Member "count" must match the original count for that text (from partial)."""
    data = _chat_json(SYSTEM_BASE, merge_user)
    clusters = _parse_clusters_payload(data, allowed)
    clusters = _dedupe_clusters_by_member(clusters)
    assigned = {m.text for c in clusters for m in c.members}
    missing = [allowed[t] for t in allowed if t not in assigned]
    if missing:
        clusters.append(Cluster(label="Other", members=missing))
        clusters = _dedupe_clusters_by_member(clusters)
    return _sort_clusters(clusters, top_k)


def cluster_answers(
    answers: list[AnswerCount],
    top_k: int = 10,
    question_context: str = "",
) -> list[Cluster]:
    if not answers:
        return []
    if len(answers) <= CHUNK_SIZE:
        return cluster_single_batch(answers, top_k, question_context)
    return cluster_chunked_then_merge(answers, top_k, question_context)
