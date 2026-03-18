"""Unit tests for src/ingestion/loader.py and src/ingestion/hf_loader.py"""

import pytest
import pandas as pd
import tempfile
from pathlib import Path

from src.ingestion.loader import load_csv
from src.ingestion.hf_loader import load_mock_dataset, EMOTION_LABELS


# CSV Loader

class TestLoadCSV:
    def _write_csv(self, tmp_path, data: dict, filename="data.csv") -> Path:
        df = pd.DataFrame(data)
        path = tmp_path / filename
        df.to_csv(path, index=False)
        return path

    def test_loads_valid_csv(self, tmp_path):
        path = self._write_csv(tmp_path, {"text": ["hello", "world"], "label": [0, 1]})
        df = load_csv(path)
        assert len(df) == 2
        assert "text" in df.columns

    def test_standardises_column_names(self, tmp_path):
        path = self._write_csv(tmp_path, {"body": ["hi"], "sentiment": [1]})
        df = load_csv(path, text_col="body", label_col="sentiment")
        assert "text" in df.columns
        assert "label" in df.columns

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            load_csv("/nonexistent/file.csv")

    def test_empty_file_raises(self, tmp_path):
        path = self._write_csv(tmp_path, {"text": [], "label": []})
        with pytest.raises(ValueError, match="empty"):
            load_csv(path)

    def test_missing_text_column_raises(self, tmp_path):
        path = self._write_csv(tmp_path, {"body": ["hi"], "label": [1]})
        with pytest.raises(ValueError, match="text"):
            load_csv(path, text_col="text")

    def test_max_rows_respected(self, tmp_path):
        path = self._write_csv(tmp_path, {
            "text": [f"sample {i}" for i in range(100)],
            "label": list(range(100))
        })
        df = load_csv(path, max_rows=10)
        assert len(df) == 10

    def test_drops_null_text_rows(self, tmp_path):
        path = self._write_csv(tmp_path, {
            "text": ["hello", None, "world"],
            "label": [0, 1, 2]
        })
        df = load_csv(path)
        assert len(df) == 2

    def test_no_label_column(self, tmp_path):
        path = self._write_csv(tmp_path, {"text": ["hello", "world"]})
        df = load_csv(path, label_col=None)
        assert "text" in df.columns
        assert "label" not in df.columns


# Mock HuggingFace Loader

class TestLoadMockDataset:
    def test_default_size(self):
        df = load_mock_dataset(n=200)
        assert len(df) == 200

    def test_balanced_classes(self):
        df = load_mock_dataset(n=200)
        counts = df["label"].value_counts()
        assert all(counts == 50)

    def test_schema(self):
        df = load_mock_dataset(n=40)
        assert set(df.columns) == {"text", "label", "emotion"}

    def test_emotion_label_consistency(self):
        df = load_mock_dataset(n=40)
        for _, row in df.iterrows():
            assert EMOTION_LABELS[row["label"]] == row["emotion"]

    def test_reproducible_with_same_seed(self):
        df1 = load_mock_dataset(n=40)
        df2 = load_mock_dataset(n=40)
        pd.testing.assert_frame_equal(df1, df2)

    def test_no_empty_text(self):
        df = load_mock_dataset(n=40)
        assert df["text"].str.strip().ne("").all()

    def test_valid_label_range(self):
        df = load_mock_dataset(n=40)
        assert df["label"].between(0, 3).all()
