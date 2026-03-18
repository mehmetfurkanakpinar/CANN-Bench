"""
Data Ingestion Module: load raw datasets into the CANN-Bench pipeline.

Supports:
  - Local CSV files
  - HuggingFace datasets (e.g. for NLP benchmarks)

All loaded data is returned as a pandas DataFrame with standardised
columns [text, label] for downstream preprocessing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


def load_csv(
    path: str | Path,
    text_col: str = "text",
    label_col: Optional[str] = "label",
    max_rows: Optional[int] = None,
) -> pd.DataFrame:
    """Load a local CSV dataset.

    Args:
        path: Path to CSV file.
        text_col: Column name containing the input text.
        label_col: Column name containing labels (optional).
        max_rows: If set, load only the first N rows (useful for dev/testing).

    Returns:
        DataFrame with standardised columns [text, label].

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the text column is not present, or the file is empty.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    df = pd.read_csv(path, nrows=max_rows)

    if df.empty:
        raise ValueError(f"Dataset at {path} is empty.")

    if text_col not in df.columns:
        raise ValueError(
            f"Text column '{text_col}' not found. Available columns: {list(df.columns)}"
        )

    # Standardise column names
    rename = {text_col: "text"}
    if label_col and label_col in df.columns:
        rename[label_col] = "label"

    df = df.rename(columns=rename)

    keep = ["text"] + (["label"] if "label" in df.columns else [])
    return df[keep].dropna(subset=["text"]).reset_index(drop=True)


def load_huggingface(
    dataset_name: str,
    split: str = "train",
    text_col: str = "text",
    label_col: Optional[str] = "label",
    max_rows: Optional[int] = None,
) -> pd.DataFrame:
    """Load a dataset from the HuggingFace Hub.

    Args:
        dataset_name: HuggingFace dataset identifier (e.g. 'imdb', 'tweet_eval').
        split: Dataset split to load ('train', 'test', 'validation').
        text_col: Column in the HF dataset containing text.
        label_col: Column in the HF dataset containing labels.
        max_rows: If set, truncate to first N rows.

    Returns:
        Standardised DataFrame with columns [text, label].
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install the 'datasets' package: pip install datasets")

    dataset = load_dataset(dataset_name, split=split)

    if max_rows:
        dataset = dataset.select(range(min(max_rows, len(dataset))))

    df = dataset.to_pandas()

    rename = {text_col: "text"}
    if label_col and label_col in df.columns:
        rename[label_col] = "label"

    df = df.rename(columns=rename)
    keep = ["text"] + (["label"] if "label" in df.columns else [])
    return df[keep].dropna(subset=["text"]).reset_index(drop=True)
