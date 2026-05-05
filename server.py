"""Flask backend for LabVIEW Family Feud survey clustering tool."""
from __future__ import annotations

import copy
import io
import json
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory

from feud_prep.loader import (
    METADATA_COLUMN_COUNT,
    aggregate_answer_counts,
    extract_raw_answers,
    load_survey_csv,
    prompt_column_headers,
)
from feud_prep.llm_cluster import cluster_answers
from feud_prep.models import AnswerCount, Cluster, RunArtifact, utc_now_iso
import db as run_db

PROJECT_ROOT = Path(__file__).resolve().parent
RUNS_DIR = PROJECT_ROOT / "runs"
STATIC_DIR = PROJECT_ROOT / "static"
ENV_PATH = PROJECT_ROOT / ".env"
BUNDLE_FORMAT_VERSION = 2

load_dotenv(ENV_PATH)

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB upload limit


@app.errorhandler(Exception)
def handle_exception(e):
    """Return JSON for any unhandled exception so the frontend always gets parseable errors."""
    import traceback
    traceback.print_exc()
    return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def slugify(name: str) -> str:
    s = re.sub(r"[^\w]+", "_", name.lower()).strip("_")
    return (s[:80] if s else "column")


def clusters_to_state(clusters: list[Cluster]) -> list[dict]:
    return [
        {"label": c.label, "members": [{"text": m.text, "count": m.count} for m in c.members]}
        for c in clusters
    ]


def state_to_clusters(state: list[dict]) -> list[Cluster]:
    out: list[Cluster] = []
    for block in state:
        label = str(block.get("label", "")).strip() or "Unnamed"
        members = []
        for m in block.get("members") or []:
            if isinstance(m, dict):
                t = str(m.get("text", "")).strip()
                if t:
                    members.append(AnswerCount(text=t, count=max(1, int(m.get("count", 1)))))
        if members:
            out.append(Cluster(label=label, members=members))
    return out


def normalize_run_file(data: dict) -> dict:
    """Normalise any saved run JSON to v2 bundle shape."""
    if isinstance(data.get("prompts"), dict) and data["prompts"]:
        first = next(iter(data["prompts"].values()))
        if not isinstance(first, dict):
            # Legacy single-prompt
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
    # Single-prompt legacy
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


def bundle_to_api_response(payload: dict) -> dict:
    """Convert a normalised bundle to the API response shape with by_column."""
    prompts_data: dict[str, dict] = {}
    for col, art_dict in payload["prompts"].items():
        art = RunArtifact.from_dict(art_dict)
        prompts_data[col] = {
            "artifact": art.to_dict(),
            "clusters": clusters_to_state(art.clusters),
        }
    order = payload.get("prompt_column_order") or list(payload["prompts"].keys())
    return {
        "source_file": payload["source_file"],
        "created_at": payload["created_at"],
        "top_k": payload["top_k"],
        "skip_first_columns": payload["skip_first_columns"],
        "ignore_exact_answers": payload.get("ignore_exact_answers", []),
        "prompt_column_order": order,
        "prompts": prompts_data,
    }


def feud_board_export(prompts_data: dict, prompt_order: list[str]) -> list[dict]:
    """Build Feud export JSON: [{ question, answers: [{ answer, score }] }]."""
    out = []
    for col in prompt_order:
        if col not in prompts_data:
            continue
        clusters = prompts_data[col]["clusters"]
        sorted_clusters = sorted(clusters, key=lambda c: -sum(m["count"] for m in c.get("members", [])))
        answers = []
        for c in sorted_clusters:
            total = sum(m["count"] for m in c.get("members", []))
            answers.append({"answer": c["label"], "score": total})
        out.append({"question": col, "answers": answers})
    return out


# ---------------------------------------------------------------------------
# Routes – Static
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


# ---------------------------------------------------------------------------
# Routes – API
# ---------------------------------------------------------------------------

@app.get("/api/runs")
def list_runs():
    """List saved runs from the database."""
    return jsonify(run_db.list_runs())


@app.get("/api/runs/latest")
def latest_run():
    """Load the most recently saved run."""
    rows = run_db.list_runs()
    if not rows:
        return jsonify({"error": "No runs found"}), 404
    latest_filename = rows[0]["filename"]
    data = run_db.load_run(latest_filename)
    if data is None:
        return jsonify({"error": "Run not found"}), 404
    payload = normalize_run_file(data)
    resp = bundle_to_api_response(payload)
    resp["saved_as"] = latest_filename
    return jsonify(resp)


@app.get("/api/runs/latest/version")
def latest_run_version():
    """Return just the version (updated_at) of the latest run — lightweight poll endpoint."""
    info = run_db.get_latest_version()
    if not info:
        return jsonify({"error": "No runs"}), 404
    return jsonify(info)


@app.get("/api/runs/<filename>")
def load_run(filename: str):
    """Load a saved run by filename."""
    safe = Path(filename).name  # sanitise
    data = run_db.load_run(safe)
    if data is None:
        return jsonify({"error": "Run not found"}), 404
    payload = normalize_run_file(data)
    return jsonify(bundle_to_api_response(payload))


@app.post("/api/upload-and-cluster")
def upload_and_cluster():
    """Upload CSV and run LLM clustering on all prompts."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    top_k = int(request.form.get("top_k", 10))
    ignore_raw = request.form.get("ignore_exact", "n/a,na,none")
    seed_json = request.form.get("seed_clusters", "")

    ignore_exact: set[str] = set()
    for part in re.split(r"[\n,]+", ignore_raw):
        p = part.strip()
        if p:
            ignore_exact.add(p)

    # Parse seed clusters if provided (prior labels to guide re-clustering)
    seed_clusters: dict[str, list[str]] = {}
    if seed_json:
        try:
            seed_clusters = json.loads(seed_json)
        except json.JSONDecodeError:
            pass

    raw = file.read()
    try:
        df = load_survey_csv(io.BytesIO(raw))
    except Exception as e:
        return jsonify({"error": f"Could not read CSV: {e}"}), 400

    all_cols = list(df.columns)
    selectable = prompt_column_headers(all_cols)
    if not selectable:
        return jsonify({"error": f"No prompt columns found (only {len(all_cols)} columns)"}), 400

    source_name = file.filename
    by_column: dict[str, dict] = {}
    errors: list[str] = []

    for column in selectable:
        try:
            raw_answers = extract_raw_answers(df, column, ignore_exact=ignore_exact)
            if not raw_answers:
                continue
            agg = aggregate_answer_counts(raw_answers)

            # If we have seed labels for this prompt, pass them to the clusterer
            prior_labels = seed_clusters.get(column, [])
            clusters = cluster_answers(
                agg, top_k=top_k, survey_prompt=column, prior_labels=prior_labels,
            )
            art = RunArtifact(
                source_file=source_name,
                column=column,
                created_at=utc_now_iso(),
                top_k=top_k,
                clusters=clusters,
                n_cells_non_empty=len(raw_answers),
                n_unique_texts=len(agg),
                skip_first_columns=METADATA_COLUMN_COUNT,
                ignore_exact_answers=sorted(ignore_exact),
            )
            by_column[column] = {
                "artifact": art.to_dict(),
                "clusters": clusters_to_state(clusters),
            }
        except Exception as e:
            import traceback
            traceback.print_exc()
            errors.append(f"{column}: {e}")

    if not by_column:
        return jsonify({"error": "No answers found", "details": errors}), 400

    created = utc_now_iso()
    bundle = {
        "format_version": BUNDLE_FORMAT_VERSION,
        "source_file": source_name,
        "created_at": created,
        "top_k": top_k,
        "skip_first_columns": METADATA_COLUMN_COUNT,
        "ignore_exact_answers": sorted(ignore_exact),
        "prompt_column_order": [c for c in selectable if c in by_column],
        "prompts": {col: row["artifact"] for col, row in by_column.items()},
    }

    # Auto-save to database
    stem = slugify(source_name) if source_name else "run"
    save_filename = f"{created.replace(':', '')}_{stem}_all_prompts.json"
    run_db.upsert_run(save_filename, bundle)

    response = bundle_to_api_response(bundle)
    response["saved_as"] = save_filename
    if errors:
        response["errors"] = errors
    return jsonify(response)


@app.post("/api/save")
def save_run():
    """Save the current working state to disk."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data"}), 400

    prompts_data = data.get("prompts", {})
    prompt_order = data.get("prompt_column_order", list(prompts_data.keys()))
    source_file = data.get("source_file", "run")

    # Rebuild the bundle for saving
    prompts_out: dict[str, dict] = {}
    for col, pdata in prompts_data.items():
        art_dict = copy.deepcopy(pdata.get("artifact", {}))
        art_dict["clusters"] = pdata.get("clusters", [])
        prompts_out[col] = art_dict

    bundle = {
        "format_version": BUNDLE_FORMAT_VERSION,
        "source_file": source_file,
        "created_at": data.get("created_at", utc_now_iso()),
        "top_k": data.get("top_k", 10),
        "skip_first_columns": data.get("skip_first_columns", METADATA_COLUMN_COUNT),
        "ignore_exact_answers": data.get("ignore_exact_answers", []),
        "prompt_column_order": prompt_order,
        "prompts": prompts_out,
    }

    stem = slugify(Path(source_file).stem if source_file else "run")
    # Update in-place if the client tells us which file it's editing
    save_filename = data.get("saved_as") or f"manual_{utc_now_iso().replace(':', '')}_{stem}_all_prompts.json"
    run_db.upsert_run(save_filename, bundle)

    return jsonify({"saved_as": save_filename})


@app.post("/api/export")
def export_feud():
    """Generate Feud board JSON from current state."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data"}), 400
    prompts_data = data.get("prompts", {})
    prompt_order = data.get("prompt_column_order", list(prompts_data.keys()))

    # Filter out excluded clusters
    for col, pdata in prompts_data.items():
        clusters = pdata.get("clusters", [])
        pdata["clusters"] = [c for c in clusters if not c.get("excluded")]

    export = feud_board_export(prompts_data, prompt_order)
    return jsonify(export)


def _migrate_json_runs() -> None:
    """One-time migration: import any local JSON run files into the database (batched)."""
    if not RUNS_DIR.is_dir():
        return
    items: list[tuple[str, dict]] = []
    for p in sorted(RUNS_DIR.glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            bundle = normalize_run_file(data)
            items.append((p.name, bundle))
        except Exception as exc:
            print(f"[migrate] skipping {p.name}: {exc}")
    if items:
        n = run_db.bulk_upsert_runs(items)
        print(f"[migrate] {n} runs imported to database.")


with app.app_context():
    try:
        run_db.init_db()
        _migrate_json_runs()
    except Exception as exc:
        print(f"[startup] DB init warning (will retry on first request): {exc}")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    app.run(debug=True, host="0.0.0.0", port=port, use_reloader=False)
