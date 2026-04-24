from __future__ import annotations

import copy
import io
import json
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from feud_prep.loader import (
    METADATA_COLUMN_COUNT,
    aggregate_answer_counts,
    extract_raw_answers,
    load_survey_csv,
    prompt_column_headers,
)
from feud_prep.llm_cluster import cluster_answers
from feud_prep.models import AnswerCount, Cluster, RunArtifact, utc_now_iso

PROJECT_ROOT = Path(__file__).resolve().parent
RUNS_DIR = PROJECT_ROOT / "runs"
ENV_PATH = PROJECT_ROOT / ".env"
UNDO_STACK_MAX = 50
BUNDLE_FORMAT_VERSION = 2

st.set_page_config(page_title="Feud survey clustering", layout="wide")
load_dotenv(ENV_PATH)


def apply_session_llm_settings() -> None:
    """Apply API key / model from Streamlit session state to os.environ for this run."""
    key = st.session_state.get("_session_api_key")
    if key:
        os.environ["OPENAI_API_KEY"] = key
    model = st.session_state.get("_session_openai_model")
    if model:
        os.environ["OPENAI_MODEL"] = model.strip()


def save_openai_key_to_dotenv(api_key: str) -> None:
    """Create or update OPENAI_API_KEY in .env (quoted for special characters)."""
    key = api_key.strip()
    if not key:
        raise ValueError("API key is empty.")
    lines: list[str] = []
    if ENV_PATH.exists():
        lines = ENV_PATH.read_text(encoding="utf-8").splitlines()
    out: list[str] = []
    found = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("OPENAI_API_KEY="):
            out.append(f'OPENAI_API_KEY="{key}"')
            found = True
        else:
            out.append(line)
    if not found:
        out.append(f'OPENAI_API_KEY="{key}"')
    try:
        ENV_PATH.write_text("\n".join(out).rstrip() + "\n", encoding="utf-8")
    except OSError as e:
        raise OSError(f"Could not write {ENV_PATH}: {e}") from e


def openai_key_is_configured() -> bool:
    """True if OPENAI_API_KEY is set in the process environment (e.g. from .env or session apply)."""
    return bool((os.environ.get("OPENAI_API_KEY") or "").strip())


def dotenv_file_has_openai_key() -> bool:
    """True if project .env defines a non-empty OPENAI_API_KEY (does not load the value into memory beyond parsing the line)."""
    if not ENV_PATH.exists():
        return False
    try:
        text = ENV_PATH.read_text(encoding="utf-8")
    except OSError:
        return False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("OPENAI_API_KEY="):
            raw = stripped.split("=", 1)[-1].strip().strip('"').strip("'")
            return bool(raw)
    return False


def save_openai_model_to_dotenv(model: str) -> None:
    m = model.strip()
    if not m:
        raise ValueError("Model name is empty.")
    lines: list[str] = []
    if ENV_PATH.exists():
        lines = ENV_PATH.read_text(encoding="utf-8").splitlines()
    out: list[str] = []
    found = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("OPENAI_MODEL="):
            out.append(f'OPENAI_MODEL="{m}"')
            found = True
        else:
            out.append(line)
    if not found:
        out.append(f'OPENAI_MODEL="{m}"')
    ENV_PATH.write_text("\n".join(out).rstrip() + "\n", encoding="utf-8")


def push_clusters_undo() -> None:
    cs = st.session_state.get("clusters_state")
    if cs is None:
        return
    stack: list[list[dict]] = st.session_state.setdefault("_undo_stack", [])
    stack.append(copy.deepcopy(cs))
    if len(stack) > UNDO_STACK_MAX:
        del stack[0 : len(stack) - UNDO_STACK_MAX]


def pop_clusters_undo() -> bool:
    stack: list[list[dict]] = st.session_state.get("_undo_stack") or []
    if not stack:
        return False
    st.session_state["clusters_state"] = stack.pop()
    return True


def artifact_meta_from_artifact_dict(ad: dict) -> dict:
    return {
        "source_file": ad.get("source_file", ""),
        "column": ad.get("column", ""),
        "top_k": ad.get("top_k", 10),
        "skip_first": ad.get("skip_first_columns", METADATA_COLUMN_COUNT),
        "ignore_exact": list(ad.get("ignore_exact_answers") or []),
    }


def bundle_from_single_run_file(data: dict) -> dict:
    """Wrap a legacy single-prompt run JSON in bundle shape."""
    art = RunArtifact.from_dict(data)
    col = art.column
    return {
        "format_version": BUNDLE_FORMAT_VERSION,
        "source_file": art.source_file,
        "created_at": art.created_at,
        "top_k": art.top_k,
        "skip_first_columns": art.skip_first_columns,
        "ignore_exact_answers": list(art.ignore_exact_answers),
        "prompt_column_order": [col],
        "prompts": {col: art.to_dict()},
    }


def normalize_run_file_payload(data: dict) -> dict:
    """Return a v2 bundle dict from either multi-prompt or legacy file contents."""
    if isinstance(data.get("prompts"), dict) and data["prompts"]:
        first = next(iter(data["prompts"].values()))
        if not isinstance(first, dict):
            return bundle_from_single_run_file(data)
        return {
            "format_version": int(data.get("format_version") or BUNDLE_FORMAT_VERSION),
            "source_file": str(data.get("source_file") or first.get("source_file", "")),
            "created_at": str(data.get("created_at") or first.get("created_at", "")),
            "top_k": int(data.get("top_k", first.get("top_k", 10))),
            "skip_first_columns": int(
                data.get("skip_first_columns", first.get("skip_first_columns", METADATA_COLUMN_COUNT))
            ),
            "ignore_exact_answers": list(
                data.get("ignore_exact_answers") or first.get("ignore_exact_answers") or []
            ),
            "prompt_column_order": list(data.get("prompt_column_order") or []),
            "prompts": dict(data["prompts"]),
        }
    return bundle_from_single_run_file(data)


def build_by_column_from_bundle_payload(payload: dict) -> dict[str, dict]:
    by_column: dict[str, dict] = {}
    for col_name, art_dict in payload["prompts"].items():
        art = RunArtifact.from_dict(art_dict)
        by_column[col_name] = {
            "artifact": art.to_dict(),
            "clusters_state": clusters_to_state(art.clusters),
        }
    return by_column


def workspace_from_bundle_row(row: dict) -> None:
    st.session_state["clusters_state"] = copy.deepcopy(row["clusters_state"])
    st.session_state["artifact"] = copy.deepcopy(row["artifact"])
    st.session_state["artifact_meta"] = artifact_meta_from_artifact_dict(st.session_state["artifact"])


def flush_workspace_to_bundle_prompt(bundle: dict, column: str | None) -> None:
    if not bundle or not column or column not in bundle.get("by_column", {}):
        return
    bundle["by_column"][column]["clusters_state"] = copy.deepcopy(st.session_state.get("clusters_state"))


def persist_undo_for_prompt(column: str | None) -> None:
    if not column:
        return
    st.session_state.setdefault("_undo_stacks", {})[column] = copy.deepcopy(st.session_state.get("_undo_stack") or [])


def load_undo_for_prompt(column: str) -> None:
    stacks = st.session_state.get("_undo_stacks") or {}
    st.session_state["_undo_stack"] = copy.deepcopy(stacks.get(column, []))


def sync_review_prompt_selection(bundle: dict, selected: str) -> None:
    prev = st.session_state.get("_last_review_sel")
    if prev == selected:
        return
    if prev is not None:
        flush_workspace_to_bundle_prompt(bundle, prev)
        persist_undo_for_prompt(prev)
    workspace_from_bundle_row(bundle["by_column"][selected])
    load_undo_for_prompt(selected)
    st.session_state["_last_review_sel"] = selected
    # New widgets per prompt switch so st.data_editor / dataframes don't reuse stale session state.
    st.session_state["_review_ui_epoch"] = int(st.session_state.get("_review_ui_epoch", 0)) + 1


def ordered_prompt_options(bundle: dict) -> list[str]:
    order = bundle.get("prompt_column_order") or []
    keys = list(bundle["by_column"].keys())
    out: list[str] = []
    seen: set[str] = set()
    for c in order:
        if c in bundle["by_column"] and c not in seen:
            out.append(c)
            seen.add(c)
    for c in keys:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def ensure_legacy_bundle_from_workspace() -> dict | None:
    """If session has a single-artifact workspace but no bundle (old flow), wrap it."""
    if st.session_state.get("all_prompts_bundle"):
        return st.session_state["all_prompts_bundle"]
    artifact_dict = st.session_state.get("artifact")
    state_list = st.session_state.get("clusters_state")
    if not artifact_dict or not state_list:
        return None
    col = str(artifact_dict.get("column", "Prompt"))
    ad = copy.deepcopy(artifact_dict)
    ad["clusters"] = [c.to_dict() for c in state_to_clusters(state_list)]
    payload = {
        "format_version": BUNDLE_FORMAT_VERSION,
        "source_file": str(artifact_dict.get("source_file", "")),
        "created_at": str(artifact_dict.get("created_at", utc_now_iso())),
        "top_k": int(artifact_dict.get("top_k", 10)),
        "skip_first_columns": int(artifact_dict.get("skip_first_columns", METADATA_COLUMN_COUNT)),
        "ignore_exact_answers": list(artifact_dict.get("ignore_exact_answers") or []),
        "prompt_column_order": [col],
        "prompts": {col: ad},
    }
    by_column = build_by_column_from_bundle_payload(payload)
    st.session_state["all_prompts_bundle"] = {
        **{k: v for k, v in payload.items() if k != "prompts"},
        "by_column": by_column,
        "prompt_column_order": payload["prompt_column_order"],
    }
    st.session_state["_last_review_sel"] = col
    st.session_state["review_prompt_sel"] = col
    return st.session_state["all_prompts_bundle"]


def flush_bundle_workspace_if_dirty() -> None:
    bundle = st.session_state.get("all_prompts_bundle")
    col = st.session_state.get("_last_review_sel")
    if bundle and col:
        flush_workspace_to_bundle_prompt(bundle, col)


def serialize_bundle_to_file_dict(bundle: dict) -> dict:
    flush_bundle_workspace_if_dirty()
    prompts_out: dict[str, dict] = {}
    for col, row in bundle["by_column"].items():
        ad = copy.deepcopy(row["artifact"])
        ad["clusters"] = [c.to_dict() for c in state_to_clusters(row["clusters_state"])]
        prompts_out[col] = ad
    return {
        "format_version": BUNDLE_FORMAT_VERSION,
        "source_file": bundle["source_file"],
        "created_at": bundle.get("created_at") or utc_now_iso(),
        "top_k": int(bundle["top_k"]),
        "skip_first_columns": int(bundle["skip_first_columns"]),
        "ignore_exact_answers": list(bundle.get("ignore_exact_answers") or []),
        "prompt_column_order": list(bundle.get("prompt_column_order") or ordered_prompt_options(bundle)),
        "prompts": prompts_out,
    }


def feud_board_question_payload(question: str, state_list: list[dict]) -> dict[str, Any]:
    """One survey question in Feud export shape: { question, answers: [{ answer, score }] }."""
    order = cluster_board_order(state_list)
    answers: list[dict[str, Any]] = []
    for i in order:
        b = state_list[i]
        label = str(b.get("label") or "").strip() or "Unnamed"
        answers.append({"answer": label, "score": int(count_cluster(b))})
    return {"question": question, "answers": answers}


def feud_board_export_json_list_for_prompt(question: str, state_list: list[dict]) -> list[dict[str, Any]]:
    """Top-level JSON array with a single { question, answers } object (matches Feud board export)."""
    return [feud_board_question_payload(question, state_list)]


def feud_board_export_json_list_all_prompts(bundle: dict) -> list[dict[str, Any]]:
    """Array of { question, answers } for every prompt in column order."""
    out: list[dict[str, Any]] = []
    for col in ordered_prompt_options(bundle):
        cs = bundle["by_column"][col]["clusters_state"]
        out.append(feud_board_question_payload(col, cs))
    return out


def clear_work_session(keep_llm_session: bool = True) -> None:
    keys_remove = [
        "_df",
        "_source_name",
        "artifact",
        "artifact_meta",
        "clusters_state",
        "_undo_stack",
        "_undo_stacks",
        "all_prompts_bundle",
        "_last_review_sel",
        "review_prompt_sel",
        "_review_ui_epoch",
        "_opened_saved_run",
        "_confirm_restart",
        "_restart_confirm_check",
    ]
    for k in keys_remove:
        st.session_state.pop(k, None)
    if not keep_llm_session:
        st.session_state.pop("_session_api_key", None)
        st.session_state.pop("_session_openai_model", None)


def slugify(name: str) -> str:
    s = re.sub(r"[^\w]+", "_", name.lower()).strip("_")
    return (s[:80] if s else "column")


def clusters_to_state(clusters: list[Cluster]) -> list[dict]:
    return [
        {
            "label": c.label,
            "members": [{"text": m.text, "count": m.count} for m in c.members],
        }
        for c in clusters
    ]


def state_to_clusters(state: list[dict]) -> list[Cluster]:
    out: list[Cluster] = []
    for block in state:
        label = str(block.get("label", "")).strip() or "Unnamed"
        members_raw = block.get("members") or []
        members: list[AnswerCount] = []
        for m in members_raw:
            if isinstance(m, dict):
                t = str(m.get("text", "")).strip()
                if not t:
                    continue
                c = int(m.get("count", 1))
                members.append(AnswerCount(text=t, count=max(1, c)))
        if members:
            out.append(Cluster(label=label, members=members))
    return out


def count_cluster(block: dict) -> int:
    return sum(int(m.get("count", 1)) for m in (block.get("members") or []))


def move_member(state: list[dict], text: str, dest_idx: int) -> None:
    moved: dict | None = None
    for block in state:
        new_m = []
        for m in block.get("members") or []:
            if isinstance(m, dict) and m.get("text") == text:
                moved = dict(m)
            else:
                new_m.append(m)
        block["members"] = new_m
    if moved is None:
        return
    state[dest_idx]["members"] = list(state[dest_idx].get("members") or [])
    state[dest_idx]["members"].append(moved)


def member_preview(block: dict, max_items: int = 8, max_chars: int = 280) -> str:
    members_raw = [
        m for m in (block.get("members") or []) if isinstance(m, dict) and str(m.get("text", "")).strip()
    ]
    if not members_raw:
        return ""
    members_sorted = sorted(
        members_raw,
        key=lambda m: (-int(m.get("count", 1)), str(m.get("text", "")).lower()),
    )
    parts: list[str] = []
    for m in members_sorted[:max_items]:
        t = str(m.get("text", "")).strip()
        if len(t) > 48:
            t = t[:45] + "…"
        parts.append(t)
    s = ", ".join(parts)
    if len(s) > max_chars:
        s = s[: max_chars - 1] + "…"
    return s


def cluster_board_order(state_list: list[dict]) -> list[int]:
    return sorted(range(len(state_list)), key=lambda i: -count_cluster(state_list[i]))


def main() -> None:
    apply_session_llm_settings()

    if flash := st.session_state.pop("_flash_info", None):
        st.info(flash)

    st.title("LabVIEW Family Feud — survey clustering")
    st.caption(
        "Load a wide CSV, cluster every survey prompt with OpenAI, refine groupings on **Review**, download JSON on **Export**."
    )

    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    with st.sidebar:
        st.header("Menu")
        with st.expander("LLM connection", expanded=False):
            key_configured = openai_key_is_configured()
            key_placeholder = (
                "•••••••• — key is saved; paste here only to replace it"
                if key_configured
                else "Paste your OpenAI API key"
            )
            api_field = st.text_input(
                "OpenAI API key",
                type="password",
                key="menu_openai_key_field",
                placeholder=key_placeholder,
                help="The field stays blank after save (the real key is not shown). "
                "A key from .env is loaded when the app starts. "
                "Apply = this tab only; Save = write to .env next to app.py.",
            )
            row_key = st.columns(2)
            with row_key[0]:
                if st.button("Apply key", key="menu_apply_key"):
                    if api_field.strip():
                        st.session_state["_session_api_key"] = api_field.strip()
                        apply_session_llm_settings()
                        st.success("Key applied for this session.")
                    elif st.session_state.get("_session_api_key"):
                        st.info("Session key already set; enter text above only if replacing it.")
                    else:
                        st.warning("Paste a key first, or save one to .env and restart the app.")
            with row_key[1]:
                save_key_src = api_field.strip() or (st.session_state.get("_session_api_key") or "")
                if st.button("Save key to .env", key="menu_save_key_env"):
                    if not save_key_src:
                        st.warning("No key to save. Apply or paste one first.")
                    else:
                        try:
                            save_openai_key_to_dotenv(save_key_src)
                            st.session_state["_session_api_key"] = save_key_src
                            load_dotenv(ENV_PATH, override=True)
                            apply_session_llm_settings()
                            st.success(f"Saved to {ENV_PATH.name} (path: {ENV_PATH}).")
                        except ValueError as e:
                            st.warning(str(e))
                        except OSError as e:
                            st.error(str(e))
            if st.session_state.get("_session_api_key"):
                if dotenv_file_has_openai_key():
                    st.caption("API key: in use for this tab and saved in `.env` on disk.")
                else:
                    st.caption("API key: in use for this tab only — click **Save key to .env** to store it on disk.")
            elif openai_key_is_configured():
                st.caption(
                    f"API key: loaded from `{ENV_PATH.name}` or your environment (box stays empty so the secret is not shown)."
                )
            else:
                st.caption("No API key yet — paste one above, or add OPENAI_API_KEY to `.env`.")

            if st.button("Clear session API key", key="menu_clear_key"):
                st.session_state.pop("_session_api_key", None)
                if ENV_PATH.exists():
                    load_dotenv(ENV_PATH, override=True)
                else:
                    os.environ.pop("OPENAI_API_KEY", None)
                st.rerun()

            default_model = (
                st.session_state.get("_session_openai_model")
                or os.environ.get("OPENAI_MODEL")
                or "gpt-4o-mini"
            )
            model_field = st.text_input("Model name", value=default_model, key="menu_openai_model_field")
            row_model = st.columns(2)
            with row_model[0]:
                if st.button("Apply model", key="menu_apply_model"):
                    st.session_state["_session_openai_model"] = model_field.strip()
                    apply_session_llm_settings()
                    st.success("Model override applied.")
            with row_model[1]:
                if st.button("Save model to .env", key="menu_save_model_env"):
                    try:
                        save_openai_model_to_dotenv(model_field)
                        st.success(f"Updated OPENAI_MODEL in {ENV_PATH.name}.")
                    except ValueError as e:
                        st.warning(str(e))
            if st.button("Clear session model override", key="menu_clear_model"):
                st.session_state.pop("_session_openai_model", None)
                st.session_state.pop("menu_openai_model_field", None)
                if ENV_PATH.exists():
                    load_dotenv(ENV_PATH, override=True)
                else:
                    os.environ.pop("OPENAI_MODEL", None)
                st.rerun()

        with st.expander("Save / undo / restart", expanded=False):
            stack = st.session_state.get("_undo_stack") or []
            st.caption(f"Undo history depth: **{len(stack)}** (move, sort, delete empty).")
            u1, u2 = st.columns(2)
            with u1:
                undo_disabled = len(stack) == 0 or (st.session_state.get("clusters_state") is None)
                if st.button("Undo", disabled=undo_disabled, key="menu_undo"):
                    if pop_clusters_undo():
                        st.rerun()
            with u2:
                if st.button("Save run to disk", key="menu_save_run"):
                    bundle = st.session_state.get("all_prompts_bundle") or ensure_legacy_bundle_from_workspace()
                    if not bundle or not bundle.get("by_column"):
                        st.warning("Nothing to save yet.")
                    else:
                        payload = serialize_bundle_to_file_dict(bundle)
                        stem = slugify(Path(bundle["source_file"]).stem if bundle.get("source_file") else "run")
                        out_path = RUNS_DIR / f"manual_{utc_now_iso().replace(':', '')}_{stem}_all_prompts.json"
                        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
                        st.success(f"Wrote {out_path.name} ({len(bundle['by_column'])} prompt(s)).")

            rev_disabled = st.session_state.get("artifact") is None or st.session_state.get("clusters_state") is None
            if st.button(
                "Revert review to last clustered snapshot",
                disabled=rev_disabled,
                key="menu_revert_llm",
                help="Restores groups from the last LLM run for the prompt selected on Review, discarding edits there.",
            ):
                art = RunArtifact.from_dict(st.session_state["artifact"])
                st.session_state["clusters_state"] = clusters_to_state(art.clusters)
                cur = st.session_state.get("_last_review_sel")
                if cur and st.session_state.get("all_prompts_bundle"):
                    st.session_state["all_prompts_bundle"]["by_column"][cur]["clusters_state"] = clusters_to_state(
                        art.clusters
                    )
                    st.session_state["all_prompts_bundle"]["by_column"][cur]["artifact"] = art.to_dict()
                st.session_state["_undo_stack"] = []
                st.rerun()

            st.checkbox(
                "I understand this clears the loaded CSV and all in-tab progress",
                key="_restart_confirm_check",
            )
            if st.button("Restart session", key="menu_restart"):
                if not st.session_state.get("_restart_confirm_check"):
                    st.warning("Check the box above to confirm.")
                else:
                    clear_work_session(keep_llm_session=True)
                    st.rerun()

        st.divider()
        st.header("Answers")
        st.caption(
            f"Columns A–E are treated as metadata; every column from F onward is one survey prompt. "
            f"Clustering runs on all of those prompts at once."
        )
        ignore_raw = st.text_area(
            "Ignore answers equal to (comma or newline, case-insensitive)",
            value="n/a,na,none",
            height=80,
        )
        st.caption(
            "If a cell lists several items separated by commas, each item is a separate answer for counting and clustering."
        )
        st.header("Clustering")
        top_k = st.slider("Max clusters (board slots)", min_value=4, max_value=16, value=10)

    ignore_exact: set[str] = set()
    for part in re.split(r"[\n,]+", ignore_raw):
        p = part.strip()
        if p:
            ignore_exact.add(p)

    tab_load, tab_review, tab_export = st.tabs(["1. Load & cluster", "2. Review", "3. Export"])

    ensure_legacy_bundle_from_workspace()

    with tab_load:
        uploaded = st.file_uploader("Survey CSV", type=["csv"])
        existing_runs = sorted(RUNS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        pick_existing = st.selectbox(
            "Or open a saved run",
            options=["—"] + [p.name for p in existing_runs],
        )

        df: pd.DataFrame | None = st.session_state.get("_df")
        source_name = st.session_state.get("_source_name", "")

        if uploaded is not None:
            raw = uploaded.getvalue()
            try:
                df = load_survey_csv(io.BytesIO(raw))  # type: ignore[name-defined]
            except Exception as e:
                st.error(f"Could not read CSV: {e}")
                df = None
            if df is not None:
                source_name = uploaded.name
                st.session_state["_df"] = df
                st.session_state["_source_name"] = source_name
                st.session_state.pop("_opened_saved_run", None)
                st.success(f"Loaded {len(df)} rows, {len(df.columns)} columns from {source_name!r}.")

        if pick_existing == "—":
            st.session_state.pop("_opened_saved_run", None)
        elif pick_existing != st.session_state.get("_opened_saved_run"):
            path = RUNS_DIR / pick_existing
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                payload = normalize_run_file_payload(data)
                by_column = build_by_column_from_bundle_payload(payload)
                order = payload.get("prompt_column_order") or []
                if not order:
                    order = list(payload["prompts"].keys())
                st.session_state["all_prompts_bundle"] = {
                    "source_file": payload["source_file"],
                    "created_at": payload["created_at"],
                    "top_k": payload["top_k"],
                    "skip_first_columns": payload["skip_first_columns"],
                    "ignore_exact_answers": payload["ignore_exact_answers"],
                    "prompt_column_order": order,
                    "by_column": by_column,
                }
                opts = ordered_prompt_options(st.session_state["all_prompts_bundle"])
                first = opts[0]
                workspace_from_bundle_row(by_column[first])
                st.session_state["_last_review_sel"] = first
                st.session_state["review_prompt_sel"] = first
                st.session_state["_undo_stack"] = []
                st.session_state.pop("_undo_stacks", None)
                st.session_state["_opened_saved_run"] = pick_existing
                st.session_state["_review_ui_epoch"] = int(st.session_state.get("_review_ui_epoch", 0)) + 1
                st.session_state["_flash_info"] = (
                    f"Opened saved run: {pick_existing} ({len(by_column)} prompt(s)). "
                    "Use **Review** and **Export** for this run."
                )
            except Exception as e:
                st.error(f"Could not load run: {e}")

        if df is None and uploaded is None and pick_existing == "—":
            st.session_state.pop("_df", None)
            st.session_state.pop("_source_name", None)

        df = st.session_state.get("_df")
        if df is not None:
            all_cols = list(df.columns)
            selectable = prompt_column_headers(all_cols)
            if not selectable:
                st.warning(
                    f"Expected metadata in columns A–E and prompts from column F onward. "
                    f"This file only has {len(all_cols)} column(s)."
                )
            else:
                st.write(f"**{len(selectable)}** survey prompt(s) will be clustered (headers shown to the LLM).")
                with st.expander("Prompt column headers", expanded=False):
                    for h in selectable:
                        st.caption(f"· {h}")
                if st.button("Run LLM clustering (all prompts)", type="primary"):
                    by_column: dict[str, dict] = {}
                    errors: list[str] = []
                    prog = st.progress(0)
                    n = len(selectable)
                    for i, column in enumerate(selectable):
                        prog.progress((i + 1) / max(n, 1))
                        try:
                            raw_answers = extract_raw_answers(df, column, ignore_exact=ignore_exact)
                            if not raw_answers:
                                continue
                            agg = aggregate_answer_counts(raw_answers)
                            clusters = cluster_answers(agg, top_k=int(top_k), survey_prompt=column)
                            art = RunArtifact(
                                source_file=source_name,
                                column=column,
                                created_at=utc_now_iso(),
                                top_k=int(top_k),
                                clusters=clusters,
                                n_cells_non_empty=len(raw_answers),
                                n_unique_texts=len(agg),
                                skip_first_columns=METADATA_COLUMN_COUNT,
                                ignore_exact_answers=sorted(ignore_exact),
                            )
                            by_column[column] = {
                                "artifact": art.to_dict(),
                                "clusters_state": clusters_to_state(clusters),
                            }
                        except Exception as e:
                            errors.append(f"{column!r}: {e}")
                    prog.empty()
                    if not by_column:
                        st.error(
                            "No answers found in any prompt column (all empty after filters), or every prompt failed. "
                            + (" Errors: " + "; ".join(errors) if errors else "")
                        )
                    else:
                        created = utc_now_iso()
                        st.session_state["all_prompts_bundle"] = {
                            "source_file": source_name,
                            "created_at": created,
                            "top_k": int(top_k),
                            "skip_first_columns": METADATA_COLUMN_COUNT,
                            "ignore_exact_answers": sorted(ignore_exact),
                            "prompt_column_order": list(selectable),
                            "by_column": by_column,
                        }
                        first = next(c for c in selectable if c in by_column)
                        workspace_from_bundle_row(by_column[first])
                        st.session_state["_last_review_sel"] = first
                        st.session_state["review_prompt_sel"] = first
                        st.session_state["_undo_stack"] = []
                        st.session_state.pop("_undo_stacks", None)
                        st.session_state.pop("_opened_saved_run", None)
                        st.session_state["_review_ui_epoch"] = int(st.session_state.get("_review_ui_epoch", 0)) + 1
                        file_payload = serialize_bundle_to_file_dict(st.session_state["all_prompts_bundle"])
                        stem = slugify(source_name) if source_name else "run"
                        out_path = RUNS_DIR / f"{created.replace(':', '')}_{stem}_all_prompts.json"
                        out_path.write_text(json.dumps(file_payload, indent=2, ensure_ascii=False), encoding="utf-8")
                        no_answer_cols = [
                            c
                            for c in selectable
                            if c not in by_column and not any(e.startswith(f"{c!r}:") for e in errors)
                        ]
                        msg = (
                            f"Clustered **{len(by_column)}** prompt(s) with answers; saved **{out_path.name}**. "
                            "On **Review**, pick a prompt from the dropdown, then use Review or Export."
                        )
                        if no_answer_cols:
                            msg += f" Skipped **{len(no_answer_cols)}** prompt(s) with no answers (after ignore rules)."
                        if errors:
                            msg += f" **{len(errors)}** prompt(s) failed; see below."
                        st.success(msg)
                        if errors:
                            st.warning("Failures:\n" + "\n".join(errors))

    with tab_review:
        bundle = st.session_state.get("all_prompts_bundle")
        if bundle and bundle.get("by_column"):
            options = ordered_prompt_options(bundle)
            if st.session_state.get("review_prompt_sel") not in options and options:
                st.session_state["review_prompt_sel"] = options[0]
            prompt_choice = st.selectbox(
                "Prompt to review",
                options=options,
                key="review_prompt_sel",
                help="Tables below show clusters for this prompt only. Edits are kept per prompt when you switch.",
            )
            sync_review_prompt_selection(bundle, str(prompt_choice))

        state_list = st.session_state.get("clusters_state")
        meta = st.session_state.get("artifact_meta") or {}
        artifact_dict = st.session_state.get("artifact")

        if not state_list or not bundle:
            st.info("Run clustering in the first tab or open a saved run.")
        else:
            selected = str(st.session_state.get("review_prompt_sel") or "")
            prompt_sk = slugify(selected)[:72] or "prompt"
            epoch = int(st.session_state.get("_review_ui_epoch", 0))
            wk = f"{prompt_sk}_{epoch}"

            st.subheader("Review groupings")
            st.caption(
                f"Viewing: **{selected}** — clusters, previews, and answer lists below are for this prompt only. "
                "Board rank is by total responses. Use **Export** when you’re done."
            )
            if meta:
                st.caption(
                    f"Source: {meta.get('source_file', '')!r} · "
                    f"Unique raw answers: **{artifact_dict.get('n_unique_texts') if artifact_dict else '?'}**"
                )

            order_idx = cluster_board_order(state_list)
            cluster_rows = []
            for rank, i in enumerate(order_idx, start=1):
                b = state_list[i]
                cluster_rows.append(
                    {
                        "_idx": i,
                        "Label": str(b.get("label") or ""),
                        "Rank": rank,
                        "Responses": count_cluster(b),
                        "Preview": member_preview(b),
                    }
                )
            cluster_df = pd.DataFrame(cluster_rows)
            if len(cluster_df) > 0:
                cluster_df = cluster_df[["Label", "Rank", "Responses", "Preview", "_idx"]]
            cluster_fb_key = f"cluster_sel_{wk}"

            st.markdown("**Clusters** — select a row to inspect answers and rename.")
            st.dataframe(
                cluster_df,
                column_order=["Label", "Rank", "Responses", "Preview"],
                column_config={
                    "_idx": None,
                    "Label": st.column_config.TextColumn("Cluster label", width="medium"),
                    "Rank": st.column_config.NumberColumn("Rank", width="small"),
                    "Responses": st.column_config.NumberColumn("Responses", width="small"),
                    "Preview": st.column_config.TextColumn("Preview", width="large"),
                },
                hide_index=True,
                use_container_width=True,
            )
            cluster_fb_options = list(range(len(cluster_df))) if len(cluster_df) else []
            cluster_pick = st.selectbox(
                "Select cluster",
                options=[None] + cluster_fb_options,
                format_func=lambda r: "-- Select a cluster --"
                if r is None
                else f"{str(cluster_df.iloc[r]['Label'])[:56]} -- {int(cluster_df.iloc[r]['Responses'])} responses",
                key=cluster_fb_key,
            )

            detail_cluster_idx: int | None = None
            if cluster_pick is not None and len(cluster_df) > 0:
                detail_row_pos = max(0, min(int(cluster_pick), len(cluster_df) - 1))
                detail_cluster_idx = int(cluster_df.iloc[detail_row_pos]["_idx"])

            if detail_cluster_idx is None:
                st.caption("Select a cluster row in the table above to see unique answers and edit the label.")
            else:
                st.divider()
                block = state_list[detail_cluster_idx]
                lbl_short = (block.get("label") or "").strip() or f"Cluster {detail_cluster_idx + 1}"
                st.markdown("**Selected cluster**")
                st.caption(
                    f"Bucket index **{detail_cluster_idx + 1}** — {lbl_short} — **{count_cluster(block)}** weighted responses"
                )
                lbl_col, del_col = st.columns([4, 1])
                with lbl_col:
                    new_lbl = st.text_input(
                        "Cluster label",
                        value=str(block.get("label") or ""),
                        key=f"cluster_label_edit_{wk}_{detail_cluster_idx}",
                        help="Renames this cluster; the table above updates on the next interaction.",
                    )
                    state_list[detail_cluster_idx]["label"] = new_lbl
                with del_col:
                    st.markdown("<div style='padding-top:1.6rem'></div>", unsafe_allow_html=True)
                    if st.button("Delete this cluster", key=f"del_cluster_{wk}_{detail_cluster_idx}"):
                        push_clusters_undo()
                        del state_list[detail_cluster_idx]
                        st.rerun()

                st.markdown("**Unique answers in this cluster**")
                members_raw = [
                    m
                    for m in (block.get("members") or [])
                    if isinstance(m, dict) and str(m.get("text", "")).strip()
                ]
                members_sorted = sorted(
                    members_raw,
                    key=lambda m: (-int(m.get("count", 1)), str(m.get("text", "")).lower()),
                )
                ans_rows = [
                    {"Answer": str(m.get("text", "")), "Count": int(m.get("count", 1))}
                    for m in members_sorted
                ]
                ans_df = pd.DataFrame(ans_rows)
                ans_fb_key = f"answer_sel_{wk}"
                n_bk = len(state_list)
                can_move = n_bk >= 2 and len(ans_rows) > 0
                st.dataframe(ans_df, hide_index=True, use_container_width=True)
                ans_options = list(range(len(ans_df))) if len(ans_df) else []
                ans_pick = st.selectbox(
                    "Select answer to move",
                    options=[None] + ans_options,
                    format_func=lambda r: "-- Select an answer --"
                    if r is None
                    else f"{str(ans_df.iloc[r]['Answer'])[:72]} -- x{ans_df.iloc[r]['Count']}",
                    key=ans_fb_key,
                )

                picked_text: str | None = None
                if ans_pick is not None and len(ans_df) > 0:
                    answer_row_pos = max(0, min(int(ans_pick), len(ans_df) - 1))
                    picked_text = str(ans_df.iloc[answer_row_pos]["Answer"])

                if picked_text and can_move:
                    dest_candidates = [j for j in range(n_bk) if j != detail_cluster_idx]
                    st.caption("Choose a destination cluster, then click **Move answer**.")
                    c1, c2 = st.columns([2, 1])
                    with c1:
                        dest_pick = st.selectbox(
                            "Re-assign answer to cluster",
                            options=dest_candidates,
                            format_func=lambda j: f"{j + 1}. {(state_list[j].get('label') or '')[:48] or 'Unnamed'} "
                            f"({count_cluster(state_list[j])})",
                            key=f"move_dest_{wk}",
                            disabled=not dest_candidates,
                        )
                    with c2:
                        if st.button("Move answer", key=f"move_btn_{wk}", disabled=not dest_candidates):
                            push_clusters_undo()
                            move_member(state_list, picked_text, dest_pick)
                            st.rerun()
                elif len(ans_rows) == 0:
                    st.caption("No answers in this cluster.")
                elif can_move and not picked_text:
                    st.caption("Select an answer, then choose a destination cluster and click **Move answer**.")

            st.divider()
            b1, b2, b3 = st.columns(3)
            with b1:
                if st.button("Create new cluster", key=f"create_cluster_{wk}"):
                    push_clusters_undo()
                    state_list.append({"label": "New Cluster", "members": []})
                    st.rerun()
            with b2:
                if st.button("Delete empty buckets", key=f"del_empty_{wk}"):
                    push_clusters_undo()
                    state_list[:] = [b for b in state_list if count_cluster(b) > 0]
                    st.rerun()
            with b3:
                if st.button("Sort buckets by total responses (high → low)", key=f"sort_cnt_{wk}"):
                    push_clusters_undo()
                    state_list.sort(key=lambda b: -count_cluster(b))
                    st.rerun()

    with tab_export:
        state_list = st.session_state.get("clusters_state")
        meta = st.session_state.get("artifact_meta") or {}
        artifact_dict = st.session_state.get("artifact")
        bundle = st.session_state.get("all_prompts_bundle")

        if not state_list or not bundle:
            st.info("Run clustering in the first tab or open a saved run.")
        else:
            selected = str(st.session_state.get("review_prompt_sel") or "")
            prompt_sk = slugify(selected)[:72] or "prompt"

            st.subheader("Export (Feud board JSON)")
            st.caption(
                "JSON is a list of objects shaped like "
                '`[{ "question": "…", "answers": [{ "answer": "…", "score": N }, …] }, …]`. '
                "Scores are total responses per cluster (board order: high → low)."
            )
            if meta:
                st.caption(
                    f"**Prompt:** {meta.get('column', '')!r} · "
                    f"Buckets: **{len(state_list)}** · Total weighted answers: **{sum(count_cluster(b) for b in state_list)}**"
                )

            export_list = feud_board_export_json_list_for_prompt(selected, state_list)
            json_out = json.dumps(export_list, indent=2, ensure_ascii=False)
            safe_file = f"feud_export_{prompt_sk[:40]}.json"
            st.download_button(
                "Download JSON for this prompt",
                data=json_out.encode("utf-8"),
                file_name=safe_file,
                mime="application/json",
                key=f"dl_json_{prompt_sk}",
            )

            flush_bundle_workspace_if_dirty()
            bundle_for_all = st.session_state.get("all_prompts_bundle")
            if bundle_for_all and len(bundle_for_all.get("by_column") or {}) > 1:
                all_list = feud_board_export_json_list_all_prompts(bundle_for_all)
                json_all = json.dumps(all_list, indent=2, ensure_ascii=False)
                st.download_button(
                    "Download JSON for all prompts",
                    data=json_all.encode("utf-8"),
                    file_name=f"feud_export_all_{slugify(bundle_for_all.get('source_file') or 'run')[:30]}.json",
                    mime="application/json",
                    key=f"dl_json_all_{prompt_sk}",
                )

            with st.expander("Preview JSON (this prompt)"):
                st.code(json_out, language="json")


if __name__ == "__main__":
    main()
