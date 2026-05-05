"""Postgres persistence layer for run artifacts (Supabase-hosted)."""
from __future__ import annotations

import json
import os
from contextlib import contextmanager
from typing import Any, Generator

import psycopg2
import psycopg2.extras


def _dsn() -> str:
    url = os.environ.get("DATABASE_URL", "")
    if not url:
        raise RuntimeError("DATABASE_URL environment variable is not set")
    return url


@contextmanager
def _conn() -> Generator:
    conn = psycopg2.connect(_dsn(), connect_timeout=10)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


_db_ready = False


def _ensure_db() -> None:
    """Lazy init: create table on first use if not done at startup."""
    global _db_ready
    if not _db_ready:
        init_db()


def init_db() -> None:
    """Create the runs table if it doesn't exist."""
    global _db_ready
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    id          SERIAL PRIMARY KEY,
                    filename    TEXT UNIQUE NOT NULL,
                    source_file TEXT NOT NULL DEFAULT '',
                    created_at  TEXT NOT NULL DEFAULT '',
                    n_prompts   INTEGER NOT NULL DEFAULT 0,
                    data        JSONB NOT NULL,
                    inserted_at TIMESTAMPTZ NOT NULL DEFAULT now()
                );
            """)
    _db_ready = True


def upsert_run(filename: str, data: dict) -> None:
    """Insert or update a run by filename."""
    _ensure_db()
    source_file = data.get("source_file", "")
    created_at = data.get("created_at", "")
    n_prompts = len(data.get("prompts", {}))
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO runs (filename, source_file, created_at, n_prompts, data)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (filename) DO UPDATE
                    SET source_file = EXCLUDED.source_file,
                        created_at  = EXCLUDED.created_at,
                        n_prompts   = EXCLUDED.n_prompts,
                        data        = EXCLUDED.data;
                """,
                (filename, source_file, created_at, n_prompts, json.dumps(data)),
            )


def list_runs() -> list[dict[str, Any]]:
    """Return summary rows for every run, newest first."""
    _ensure_db()
    with _conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT filename, source_file, created_at, n_prompts "
                "FROM runs ORDER BY inserted_at DESC"
            )
            return [dict(r) for r in cur.fetchall()]


def load_run(filename: str) -> dict | None:
    """Load a single run's full JSON data by filename. Returns None if not found."""
    _ensure_db()
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT data FROM runs WHERE filename = %s", (filename,))
            row = cur.fetchone()
            return row[0] if row else None


def delete_run(filename: str) -> bool:
    """Delete a run by filename. Returns True if a row was deleted."""
    _ensure_db()
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM runs WHERE filename = %s", (filename,))
            return cur.rowcount > 0


def bulk_upsert_runs(items: list[tuple[str, dict]]) -> int:
    """Insert/update many runs in a single connection. Returns count of rows upserted."""
    _ensure_db()
    if not items:
        return 0
    with _conn() as conn:
        with conn.cursor() as cur:
            for filename, data in items:
                source_file = data.get("source_file", "")
                created_at = data.get("created_at", "")
                n_prompts = len(data.get("prompts", {}))
                cur.execute(
                    """
                    INSERT INTO runs (filename, source_file, created_at, n_prompts, data)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (filename) DO UPDATE
                        SET source_file = EXCLUDED.source_file,
                            created_at  = EXCLUDED.created_at,
                            n_prompts   = EXCLUDED.n_prompts,
                            data        = EXCLUDED.data;
                    """,
                    (filename, source_file, created_at, n_prompts, json.dumps(data)),
                )
    return len(items)
