"""
Clinical Data Ingestion Module: load the Kaggle Suicide Detection dataset.

Dataset source: Suicide_Detection.csv (Reddit posts, ~1M rows)
Columns expected:
  - 'text'  : raw Reddit post content
  - 'class' : 'suicide' | 'non-suicide'

Labels are converted to integers:
  suicide     -> 1
  non-suicide -> 0

The data is split into train / test (stratified) and returned as two
DataFrames each with standardised columns [text, label].
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

# Canonical label mapping — matches config.yml label_names ordering
LABEL_MAP: dict[str, int] = {
    "non-suicide": 0,
    "suicide": 1,
}

_REQUIRED_COLUMNS = {"text", "class"}


def load_kaggle_suicide(
    path: str | Path,
    test_size: float = 0.2,
    max_rows: Optional[int] = None,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and split the Kaggle Suicide Detection CSV dataset.

    Args:
        path: Path to Suicide_Detection.csv.
        test_size: Fraction of data to hold out for testing (default 0.2).
        max_rows: If set, load only the first N rows (useful for smoke tests).
        seed: Random seed for reproducible train/test split.

    Returns:
        Tuple of (train_df, test_df), each with columns [text, label].
        label is an integer: 0=non-suicide, 1=suicide.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is empty, required columns are missing,
                    label values are not the expected strings, or all
                    texts are empty after cleaning.
    """
    path = Path(path)

    # Guard 1: file exists
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    df = pd.read_csv(path, nrows=max_rows)

    # Guard 2: not empty
    if df.empty:
        raise ValueError(f"Dataset at {path} is empty.")

    # Guard 3: required columns present
    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Required column(s) {sorted(missing)} not found. "
            f"Available columns: {list(df.columns)}"
        )

    # Guard 4: only the two expected label strings exist
    observed_labels = set(df["class"].dropna().unique())
    unexpected = observed_labels - set(LABEL_MAP.keys())
    if unexpected:
        raise ValueError(
            f"Unexpected label value(s) found: {unexpected}. "
            f"Expected only {set(LABEL_MAP.keys())}."
        )

    # Guard 5: no empty texts
    df = df[["text", "class"]].copy()
    df = df.dropna(subset=["text"])
    df["text"] = df["text"].astype(str).str.strip()
    empty_mask = df["text"] == ""
    if empty_mask.all():
        raise ValueError("All text entries are empty after stripping whitespace.")
    df = df[~empty_mask].reset_index(drop=True)

    # Encode labels to integers
    df["label"] = df["class"].map(LABEL_MAP).astype(int)
    df = df[["text", "label"]]

    # Stratified train / test split (preserves class balance)
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df["label"],
    )

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return train_df, test_df
