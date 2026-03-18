"""
Unit tests for the clinical NLP benchmark components.

All tests are offline — no GPU, no internet, no model loading required.
A 200-row synthetic CSV is created in a tmp directory for I/O tests.

The 7 tests cover:
  T1  load_kaggle_suicide returns exactly two DataFrames
  T2  both DataFrames have standardised columns [text, label]
  T3  all label values are binary (0 or 1, no NaN)
  T4  the test split is approximately 20% of total rows (±2 rows)
  T5  class balance is preserved after stratified split (within 5%)
  T6  no empty or whitespace-only text strings remain in either split
  T7  FileNotFoundError raised for a non-existent CSV path
"""

from __future__ import annotations

import csv
import pytest
import tempfile
import os
from pathlib import Path

from src.ingestion.kaggle_loader import load_kaggle_suicide, LABEL_MAP



# Fixtures

@pytest.fixture(scope="module")
def synthetic_csv(tmp_path_factory) -> Path:
    """Create a 200-row balanced CSV matching the Kaggle Suicide Detection format."""
    tmp_dir = tmp_path_factory.mktemp("clinical_data")
    csv_path = tmp_dir / "Suicide_Detection.csv"

    rows = []
    for i in range(100):
        rows.append({
            "": str(i),
            "text": f"This is a suicide-related post number {i}. " * 3,
            "class": "suicide",
        })
    for i in range(100, 200):
        rows.append({
            "": str(i),
            "text": f"This is a non-suicide-related post number {i}. " * 3,
            "class": "non-suicide",
        })

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["", "text", "class"])
        writer.writeheader()
        writer.writerows(rows)

    return csv_path


@pytest.fixture(scope="module")
def loaded_splits(synthetic_csv):
    """Run load_kaggle_suicide once and share result across tests."""
    return load_kaggle_suicide(synthetic_csv, test_size=0.2, max_rows=None, seed=42)


# T1 — Returns exactly two DataFrames

class TestReturnType:
    def test_returns_tuple_of_two_dataframes(self, loaded_splits):
        """T1: load_kaggle_suicide must return (train_df, test_df)."""
        import pandas as pd
        train_df, test_df = loaded_splits
        assert isinstance(train_df, pd.DataFrame), "train split must be a DataFrame"
        assert isinstance(test_df, pd.DataFrame), "test split must be a DataFrame"



# T2 — Standardised columns [text, label]

class TestColumns:
    def test_columns_are_text_and_label(self, loaded_splits):
        """T2: both splits must have exactly columns [text, label]."""
        train_df, test_df = loaded_splits
        assert list(train_df.columns) == ["text", "label"], (
            f"train columns wrong: {list(train_df.columns)}"
        )
        assert list(test_df.columns) == ["text", "label"], (
            f"test columns wrong: {list(test_df.columns)}"
        )



# T3 — All labels are binary integers (0 or 1), no NaN

class TestLabels:
    def test_labels_are_binary_and_no_nan(self, loaded_splits):
        """T3: label column must contain only 0 or 1, with no missing values."""
        train_df, test_df = loaded_splits
        for split_name, df in [("train", train_df), ("test", test_df)]:
            assert df["label"].isna().sum() == 0, (
                f"{split_name}: found NaN in label column"
            )
            invalid = set(df["label"].unique()) - {0, 1}
            assert not invalid, (
                f"{split_name}: unexpected label values {invalid}"
            )


# T4 — Test split is ~20% of total rows (±2 rows tolerance)

class TestSplitRatio:
    def test_test_size_approximately_twenty_percent(self, loaded_splits):
        """T4: test_df must be ~20% of total rows within a tolerance of ±2 rows."""
        train_df, test_df = loaded_splits
        total = len(train_df) + len(test_df)
        expected_test = total * 0.20
        actual_test = len(test_df)
        assert abs(actual_test - expected_test) <= 2, (
            f"Expected ~{expected_test:.0f} test rows, got {actual_test}"
        )



# T5 — Class balance preserved after stratified split (within 5 percentage points)

class TestClassBalance:
    def test_class_balance_preserved(self, loaded_splits):
        """T5: suicide-class proportion in both splits must be within 5 pp of 50%."""
        train_df, test_df = loaded_splits
        for split_name, df in [("train", train_df), ("test", test_df)]:
            positive_rate = df["label"].mean()  # 1 = suicide
            assert abs(positive_rate - 0.5) <= 0.05, (
                f"{split_name}: class imbalance out of tolerance — "
                f"positive rate = {positive_rate:.3f} (expected ~0.5)"
            )



# T6 — No empty or whitespace-only text strings in either split

class TestEmptyTexts:
    def test_no_empty_texts(self, loaded_splits):
        """T6: no row should have an empty or whitespace-only text field."""
        train_df, test_df = loaded_splits
        for split_name, df in [("train", train_df), ("test", test_df)]:
            empty_mask = df["text"].str.strip() == ""
            assert empty_mask.sum() == 0, (
                f"{split_name}: {empty_mask.sum()} empty text(s) found"
            )



# T7 — FileNotFoundError raised for a non-existent path

class TestGuardClauses:
    def test_file_not_found_raises(self, tmp_path):
        """T7: must raise FileNotFoundError when the CSV path does not exist."""
        missing_path = tmp_path / "does_not_exist.csv"
        with pytest.raises(FileNotFoundError, match="not found"):
            load_kaggle_suicide(missing_path)
