from __future__ import annotations

import copy
import io
import json
import os
import re
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from feud_prep.loader import (
    aggregate_answer_counts,
    extract_raw_answers,
    filter_column_names,
    load_survey_csv,
)
from feud_prep.llm_cluster import cluster_answers
from feud_prep.models import AnswerCount, Cluster, RunArtifact, utc_now_iso

PROJECT_ROOT = Path(__file__).resolve().parent
RUNS_DIR = PROJECT_ROOT / "runs"
ENV_PATH = PROJECT_ROOT / ".env"
UNDO_STACK_MAX = 50

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
    ENV_PATH.write_text("\n".join(out).rstrip() + "\n", encoding="utf-8")


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


def build_export_dict(state_list: list[dict]) -> dict:
    meta = st.session_state.get("artifact_meta") or {}
    artifact_dict = st.session_state.get("artifact")
    clusters = state_to_clusters(state_list)
    return {
        "source_file": (artifact_dict or {}).get("source_file", meta.get("source_file", "")),
        "column": (artifact_dict or {}).get("column", meta.get("column", "")),
        "created_at": (artifact_dict or {}).get("created_at", utc_now_iso()),
        "top_k": (artifact_dict or {}).get("top_k", meta.get("top_k", 10)),
        "clusters": [c.to_dict() for c in clusters],
        "n_cells_non_empty": (artifact_dict or {}).get("n_cells_non_empty", 0),
        "n_unique_texts": (artifact_dict or {}).get("n_unique_texts", 0),
        "skip_first_columns": (artifact_dict or {}).get("skip_first_columns", meta.get("skip_first", 0)),
        "exclude_name_substrings": (artifact_dict or {}).get(
            "exclude_name_substrings", meta.get("exclude_substrings", [])
        ),
        "ignore_exact_answers": (artifact_dict or {}).get(
            "ignore_exact_answers", meta.get("ignore_exact", [])
        ),
    }


def clear_work_session(keep_llm_session: bool = True) -> None:
    keys_remove = [
        "_df",
        "_source_name",
        "artifact",
        "artifact_meta",
        "clusters_state",
        "_undo_stack",
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


def main() -> None:
    apply_session_llm_settings()

    st.title("LabVIEW Family Feud — survey clustering")
    st.caption("Load a wide CSV, pick one open-ended column, cluster with OpenAI, then review and export JSON.")

    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    with st.sidebar:
        st.header("Menu")
        with st.expander("LLM connection", expanded=False):
            api_field = st.text_input(
                "OpenAI API key",
                type="password",
                key="menu_openai_key_field",
                help="Applied for this browser tab. Use “Save to .env” to persist on disk.",
            )
            row_key = st.columns(2)
            with row_key[0]:
                if st.button("Apply key", key="menu_apply_key"):
                    if api_field.strip():
                        st.session_state["_session_api_key"] = api_field.strip()
                        apply_session_llm_settings()
                        st.success("Key applied for this session.")
                    else:
                        st.warning("Paste a key first.")
            with row_key[1]:
                save_key_src = api_field.strip() or (st.session_state.get("_session_api_key") or "")
                if st.button("Save key to .env", key="menu_save_key_env"):
                    if not save_key_src:
                        st.warning("No key to save. Apply or paste one first.")
                    else:
                        try:
                            save_openai_key_to_dotenv(save_key_src)
                            st.success(f"Updated {ENV_PATH.name}.")
                        except ValueError as e:
                            st.warning(str(e))
            if st.session_state.get("_session_api_key"):
                st.caption("Session API key: set (not shown).")
            elif os.environ.get("OPENAI_API_KEY"):
                st.caption("Using OPENAI_API_KEY from environment or .env.")
            else:
                st.caption("No API key yet — set one above or in .env.")

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
                    cs = st.session_state.get("clusters_state")
                    if not cs:
                        st.warning("Nothing to save yet.")
                    else:
                        ad = st.session_state.get("artifact") or {}
                        col_slug = slugify(str(ad.get("column", "manual")))
                        out_path = RUNS_DIR / f"manual_{utc_now_iso().replace(':', '')}_{col_slug}.json"
                        out_path.write_text(
                            json.dumps(build_export_dict(cs), indent=2, ensure_ascii=False),
                            encoding="utf-8",
                        )
                        st.success(f"Wrote {out_path.name}")

            rev_disabled = st.session_state.get("artifact") is None or st.session_state.get("clusters_state") is None
            if st.button(
                "Revert review to last clustered snapshot",
                disabled=rev_disabled,
                key="menu_revert_llm",
                help="Restores groups from the last LLM run (or loaded run file), discarding edits in the Review tab.",
            ):
                art = RunArtifact.from_dict(st.session_state["artifact"])
                st.session_state["clusters_state"] = clusters_to_state(art.clusters)
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
        st.header("Column visibility")
        skip_first = st.number_input("Skip first N columns (metadata)", min_value=0, value=0, step=1)
        exclude_raw = st.text_area(
            "Exclude columns whose name contains (comma or newline separated)",
            value="timestamp,email",
            height=80,
        )
        ignore_raw = st.text_area(
            "Ignore answers equal to (comma or newline, case-insensitive)",
            value="n/a,na,none",
            height=80,
        )
        st.header("Clustering")
        top_k = st.slider("Max clusters (board slots)", min_value=4, max_value=16, value=10)
        question_hint = st.text_input("Question / context for the LLM (optional)", value="")

    exclude_substrings = []
    for part in re.split(r"[\n,]+", exclude_raw):
        p = part.strip()
        if p:
            exclude_substrings.append(p)

    ignore_exact: set[str] = set()
    for part in re.split(r"[\n,]+", ignore_raw):
        p = part.strip()
        if p:
            ignore_exact.add(p)

    tab_load, tab_review = st.tabs(["1. Load & cluster", "2. Review & export"])

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
                st.success(f"Loaded {len(df)} rows, {len(df.columns)} columns from {source_name!r}.")

        if pick_existing != "—":
            path = RUNS_DIR / pick_existing
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                art = RunArtifact.from_dict(data)
                st.session_state["artifact"] = art.to_dict()
                st.session_state["clusters_state"] = clusters_to_state(art.clusters)
                st.session_state["_undo_stack"] = []
                st.session_state["artifact_meta"] = {
                    "source_file": art.source_file,
                    "column": art.column,
                    "top_k": art.top_k,
                    "question_hint": "",
                    "skip_first": art.skip_first_columns,
                    "exclude_substrings": art.exclude_name_substrings,
                    "ignore_exact": art.ignore_exact_answers,
                }
                st.info(f"Opened saved run: {pick_existing}. Switch to Review tab.")
            except Exception as e:
                st.error(f"Could not load run: {e}")

        if df is None and uploaded is None and pick_existing == "—":
            st.session_state.pop("_df", None)
            st.session_state.pop("_source_name", None)

        df = st.session_state.get("_df")
        if df is not None:
            all_cols = list(df.columns)
            selectable = filter_column_names(all_cols, skip_first=int(skip_first), exclude_substrings=exclude_substrings)
            if not selectable:
                st.warning("No columns left after filters. Lower skip count or remove exclude rules.")
            else:
                column = st.selectbox("Open-ended column for this Feud round", options=selectable)
                if st.button("Run LLM clustering", type="primary"):
                    with st.spinner("Calling OpenAI…"):
                        try:
                            raw_answers = extract_raw_answers(df, column, ignore_exact=ignore_exact)
                            agg = aggregate_answer_counts(raw_answers)
                            clusters = cluster_answers(agg, top_k=int(top_k), question_context=question_hint)
                            art = RunArtifact(
                                source_file=source_name,
                                column=column,
                                created_at=utc_now_iso(),
                                top_k=int(top_k),
                                clusters=clusters,
                                n_cells_non_empty=len(raw_answers),
                                n_unique_texts=len(agg),
                                skip_first_columns=int(skip_first),
                                exclude_name_substrings=exclude_substrings,
                                ignore_exact_answers=sorted(ignore_exact),
                            )
                            st.session_state["artifact_meta"] = {
                                "source_file": source_name,
                                "column": column,
                                "top_k": int(top_k),
                                "question_hint": question_hint,
                                "skip_first": int(skip_first),
                                "exclude_substrings": exclude_substrings,
                                "ignore_exact": sorted(ignore_exact),
                            }
                            st.session_state["artifact"] = art.to_dict()
                            st.session_state["clusters_state"] = clusters_to_state(clusters)
                            st.session_state["_undo_stack"] = []
                            slug = slugify(column)
                            out_path = RUNS_DIR / f"{utc_now_iso().replace(':', '')}_{slug}.json"
                            out_path.write_text(json.dumps(art.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
                            st.success(f"Saved run to {out_path.relative_to(PROJECT_ROOT)}")
                        except Exception as e:
                            st.error(str(e))

    with tab_review:
        state_list = st.session_state.get("clusters_state")
        meta = st.session_state.get("artifact_meta") or {}
        artifact_dict = st.session_state.get("artifact")

        if not state_list:
            st.info("Run clustering in the first tab or open a saved run.")
            return

        st.subheader("Clusters (edit labels, move answers, export)")
        if meta:
            st.caption(
                f"Source: {meta.get('source_file', '')!r} · Column: {meta.get('column', '')!r} · "
                f"Unique texts clustered: {artifact_dict.get('n_unique_texts') if artifact_dict else '?'}"
            )

        # Rebuild flat member list for movers
        for i, block in enumerate(state_list):
            st.divider()
            label_key = f"label_{i}"
            new_label = st.text_input(
                f"Cluster {i + 1} label",
                value=block.get("label", ""),
                key=label_key,
            )
            block["label"] = new_label
            total = count_cluster(block)
            st.write(f"**{total}** responses in this cluster")
            members = block.get("members") or []
            with st.expander(f"Members ({len(members)})", expanded=False):
                for m in members:
                    if isinstance(m, dict):
                        st.write(f"- {m.get('text', '')!r} × {m.get('count', 1)}")

        st.divider()
        st.markdown("**Move an answer**")
        flat_options: list[str] = []
        for block in state_list:
            for m in block.get("members") or []:
                if isinstance(m, dict) and m.get("text"):
                    flat_options.append(str(m["text"]))
        if flat_options:
            which = st.selectbox("Answer text", options=flat_options, key="move_pick")
            dest = st.selectbox(
                "Move to cluster #",
                options=list(range(1, len(state_list) + 1)),
                format_func=lambda x: f"{x}: {state_list[x - 1].get('label', '')}",
            )
            if st.button("Move"):
                push_clusters_undo()
                move_member(state_list, which, dest - 1)
                st.rerun()

        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("Delete empty clusters"):
                push_clusters_undo()
                state_list[:] = [b for b in state_list if count_cluster(b) > 0]
                st.rerun()
        with c2:
            if st.button("Sort by count (desc)"):
                push_clusters_undo()
                state_list.sort(key=lambda b: -count_cluster(b))
                st.rerun()

        export_dict = build_export_dict(state_list)
        json_out = json.dumps(export_dict, indent=2, ensure_ascii=False)
        with c3:
            st.download_button(
                "Download JSON",
                data=json_out.encode("utf-8"),
                file_name="feud_round_export.json",
                mime="application/json",
            )
        with st.expander("Preview JSON"):
            st.code(json_out, language="json")


if __name__ == "__main__":
    main()
