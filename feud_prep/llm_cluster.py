from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI

from feud_prep.models import AnswerCount, Cluster

DEFAULT_MODEL = "gpt-4o-mini"
CHUNK_SIZE = 45


def _client() -> OpenAI:
    key = (os.environ.get("OPENAI_API_KEY") or "").strip()
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
All answers come from ONE survey column: each spreadsheet row is one respondent; blank cells are omitted;
comma-separated lists in a cell were already split into separate answer lines with frequencies combined.
Output MUST be valid JSON only, no markdown.
Use short, display-friendly labels (a few words).
Each "members" entry must use the EXACT text string from the input (character-for-character match after the input's own trimming).

CRITICAL RULES:
- NEVER create catch-all categories like "Other", "Miscellaneous", "General", or "Various".
- If an answer does not clearly fit with any other answer, it gets its OWN cluster with a specific, descriptive label.
- A cluster of 1 answer is perfectly fine. Label it with a clear, specific name based on what that answer is.
- Only merge answers that genuinely mean the same thing. Do NOT force unrelated answers together."""


def cluster_single_batch(
    answers: list[AnswerCount],
    top_k: int,
    survey_prompt: str,
    prior_labels: list[str] | None = None,
) -> list[Cluster]:
    allowed = _build_allowed_map(answers)
    prompt_line = survey_prompt.strip() if survey_prompt.strip() else "(header missing — infer from answers)"
    payload = {
        "survey_prompt": prompt_line,
        "answers": [{"text": a.text, "count": a.count} for a in answers],
        "top_k": top_k,
    }

    seed_section = ""
    if prior_labels:
        seed_section = f"""\n\nPrior cluster labels from a previous review session (use these as a starting point — 
assign answers to these existing groups where they fit, and create new specific groups for answers that don't match):
{json.dumps(prior_labels, ensure_ascii=False)}\n"""

    user = f"""Task: Group these answers into up to {top_k} clusters for the game board.

Survey prompt (exact column header from row 1 of the spreadsheet): {prompt_line!r}

The counts below aggregate every respondent in that column: multiple answers from one person and duplicate phrasing across rows are already reflected in "count".
{seed_section}
Input JSON:
{json.dumps(payload, ensure_ascii=False)}

Return JSON shape:
{{"clusters":[{{"label":"...","members":[{{"text":"<exact from input>","count":<int>}}]}}]}}

Rules:
- Order clusters by total people (sum of member counts), descending.
- Every input answer text must appear in exactly one cluster's members (use exact "text" values from input).
- Merge duplicates of meaning; member "count" must match the input count for that text.
- Return at most {top_k} clusters; if fewer distinct ideas exist, return fewer.
- NEVER create clusters named "Other", "Miscellaneous", "General", or any catch-all. If an answer is unique, give it its own cluster with a specific name."""
    data = _chat_json(SYSTEM_BASE, user)
    clusters = _parse_clusters_payload(data, allowed)
    clusters = _dedupe_clusters_by_member(clusters)
    # Recover any answers the LLM missed — each gets its own specific cluster
    assigned = {m.text for c in clusters for m in c.members}
    missing = [a for a in answers if a.text not in assigned]
    for a in missing:
        clusters.append(Cluster(label=a.text.title(), members=[a]))
    clusters = _dedupe_clusters_by_member(clusters)
    return _sort_clusters(clusters, top_k)


def cluster_chunked_then_merge(
    answers: list[AnswerCount],
    top_k: int,
    survey_prompt: str,
    prior_labels: list[str] | None = None,
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
        cs = cluster_single_batch(ch, sub_k, survey_prompt, prior_labels=prior_labels)
        partial.append(
            {
                "chunk_index": i,
                "clusters": [c.to_dict() for c in cs],
            }
        )

    seed_section = ""
    if prior_labels:
        seed_section = f"""\nPrior cluster labels to prefer: {json.dumps(prior_labels, ensure_ascii=False)}\n"""

    _p = survey_prompt.strip() or "(unknown)"
    merge_user = f"""You previously clustered subsets of the same survey column. Now merge into one final list for the game.

Survey prompt (column header): {_p!r}

Target: at most {top_k} clusters, ordered by total people descending.
{seed_section}
Partial results JSON:
{json.dumps({"partial": partial}, ensure_ascii=False)}

Return JSON:
{{"clusters":[{{"label":"...","members":[{{"text":"<must match an input text exactly>","count":<int>}}]}}]}}

Rules:
- Every text in the partial results must appear exactly once in your output (same exact spelling as in partial).
- Merge clusters that represent the same real-world answer.
- Member "count" must match the original count for that text (from partial).
- NEVER create clusters named "Other", "Miscellaneous", "General", or any catch-all. Unique answers get their own specifically-named cluster."""
    data = _chat_json(SYSTEM_BASE, merge_user)
    clusters = _parse_clusters_payload(data, allowed)
    clusters = _dedupe_clusters_by_member(clusters)
    assigned = {m.text for c in clusters for m in c.members}
    missing = [allowed[t] for t in allowed if t not in assigned]
    for a in missing:
        clusters.append(Cluster(label=a.text.title(), members=[a]))
    clusters = _dedupe_clusters_by_member(clusters)
    return _sort_clusters(clusters, top_k)


def cluster_answers(
    answers: list[AnswerCount],
    top_k: int = 10,
    survey_prompt: str = "",
    prior_labels: list[str] | None = None,
) -> list[Cluster]:
    if not answers:
        return []
    if len(answers) <= CHUNK_SIZE:
        return cluster_single_batch(answers, top_k, survey_prompt, prior_labels=prior_labels)
    return cluster_chunked_then_merge(answers, top_k, survey_prompt, prior_labels=prior_labels)
