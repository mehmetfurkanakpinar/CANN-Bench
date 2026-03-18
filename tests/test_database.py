"""Unit tests for src/utils/database.py"""

import pytest
import sqlite3
import tempfile
from pathlib import Path

from src.utils.database import get_connection, log_result, fetch_results


@pytest.fixture
def db(tmp_path):
    """Provide a fresh in-memory-style SQLite connection for each test."""
    db_path = tmp_path / "test_benchmark.db"
    conn = get_connection(db_path)
    yield conn
    conn.close()


class TestGetConnection:
    def test_creates_file(self, tmp_path):
        db_path = tmp_path / "subdir" / "bench.db"
        conn = get_connection(db_path)
        assert db_path.exists()
        conn.close()

    def test_table_exists(self, db):
        cursor = db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='benchmark_runs'"
        )
        assert cursor.fetchone() is not None


class TestLogResult:
    def test_inserts_row(self, db):
        row_id = log_result(
            db,
            model_name="distilbert-base-uncased",
            constrained=True,
            seed=42,
            metrics={"accuracy": 0.87, "f1": 0.85, "roc_auc": 0.91},
            atlas_name="dopamine_gradient.csv",
            dataset_name="test_dataset",
        )
        assert isinstance(row_id, int)
        assert row_id >= 1

    def test_correct_values_stored(self, db):
        log_result(
            db,
            model_name="test-model",
            constrained=False,
            seed=0,
            metrics={"accuracy": 0.75},
        )
        results = fetch_results(db)
        assert len(results) == 1
        assert results[0]["model_name"] == "test-model"
        assert results[0]["constrained"] == 0
        assert results[0]["accuracy"] == pytest.approx(0.75)

    def test_multiple_rows(self, db):
        for i in range(5):
            log_result(db, model_name="m", constrained=True, seed=i, metrics={})
        assert len(fetch_results(db)) == 5


class TestFetchResults:
    def test_empty_db_returns_empty_list(self, db):
        assert fetch_results(db) == []

    def test_returns_list_of_dicts(self, db):
        log_result(db, model_name="m", constrained=True, seed=1, metrics={"accuracy": 0.9})
        results = fetch_results(db)
        assert isinstance(results, list)
        assert isinstance(results[0], dict)
