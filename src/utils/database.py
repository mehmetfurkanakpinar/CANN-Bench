"""
Utility: SQLite database interface for logging benchmark results.

Why SQLite over a flat CSV?
  - Queryable: results can be filtered, grouped, and joined without loading everything into memory.
  - Typed: enforces schema on writes, preventing silent data corruption.
  - Concurrent-safe: multiple benchmark runs can write without file conflicts.
  - Lightweight: zero infrastructure — a single .db file ships with the repo.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Any


def get_connection(db_path: str | Path) -> sqlite3.Connection:
    """Open (or create) a SQLite database."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    _create_tables(conn)
    return conn


def _create_tables(conn: sqlite3.Connection) -> None:
    """Create benchmark results table if it doesn't exist."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS benchmark_runs (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            run_timestamp TEXT    NOT NULL,
            model_name    TEXT    NOT NULL,
            atlas_name    TEXT,
            constrained   INTEGER NOT NULL,   -- 1 = anatomy-constrained, 0 = baseline
            seed          INTEGER NOT NULL,
            accuracy      REAL,
            f1            REAL,
            roc_auc       REAL,
            dataset_name  TEXT,
            notes         TEXT
        )
        """
    )
    conn.commit()


def log_result(
    conn: sqlite3.Connection,
    model_name: str,
    constrained: bool,
    seed: int,
    metrics: dict[str, float],
    atlas_name: str | None = None,
    dataset_name: str | None = None,
    notes: str | None = None,
) -> int:
    """Insert a single benchmark result row.

    Args:
        conn: Active SQLite connection.
        model_name: Name/identifier of the model used.
        constrained: Whether anatomical constraints were applied.
        seed: Random seed for this run.
        metrics: Dict with keys matching column names (accuracy, f1, roc_auc).
        atlas_name: Filename of the brain atlas CSV used.
        dataset_name: Name of the evaluation dataset.
        notes: Optional free-text notes.

    Returns:
        Row ID of the inserted record.
    """
    cursor = conn.execute(
        """
        INSERT INTO benchmark_runs
            (run_timestamp, model_name, atlas_name, constrained, seed,
             accuracy, f1, roc_auc, dataset_name, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.utcnow().isoformat(),
            model_name,
            atlas_name,
            int(constrained),
            seed,
            metrics.get("accuracy"),
            metrics.get("f1"),
            metrics.get("roc_auc"),
            dataset_name,
            notes,
        ),
    )
    conn.commit()
    return cursor.lastrowid


def fetch_results(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Retrieve all benchmark results as a list of dicts."""
    conn.row_factory = sqlite3.Row
    cursor = conn.execute("SELECT * FROM benchmark_runs ORDER BY run_timestamp DESC")
    return [dict(row) for row in cursor.fetchall()]
