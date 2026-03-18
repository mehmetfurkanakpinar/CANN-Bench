"""
Preprocessing Engine: text cleaning and complexity quantification.

Shannon entropy is used as a proxy for linguistic complexity:

  H(X) = -Σ p(x) log₂ p(x)

where p(x) is the probability of token x in the text.
High entropy → complex, unpredictable text.
Low entropy  → simple, repetitive text.

This operationalises the Information Theory requirement in the CANN framework:
different entropy regimes may respond differently to anatomical constraints.
"""

import re
import math
from collections import Counter
from typing import Optional

import pandas as pd


# Text cleaning

def clean_text(text: str) -> str:
    """Basic text normalisation.

    - Lowercases
    - Removes URLs
    - Removes non-alphanumeric characters (keeps spaces)
    - Collapses multiple whitespace

    Args:
        text: Raw input string.

    Returns:
        Cleaned string.
    """
    if not isinstance(text, str):
        raise TypeError(f"Expected str, got {type(text).__name__}")

    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)            # strip URLs
    text = re.sub(r"[^a-z0-9\s]", " ", text)              # keep alphanumeric
    text = re.sub(r"\s+", " ", text).strip()               # collapse whitespace
    return text



# Shannon Entropy

def shannon_entropy(text: str) -> float:
    """Compute Shannon entropy over the token distribution of a text.

    Args:
        text: Input string (should already be cleaned).

    Returns:
        Entropy value in bits. Returns 0.0 for empty or single-token texts.
    """
    tokens = text.split()
    if len(tokens) <= 1:
        return 0.0

    counts = Counter(tokens)
    total = len(tokens)
    entropy = -sum((c / total) * math.log2(c / total) for c in counts.values())
    return round(entropy, 4)


# Pipeline

def preprocess_dataframe(
    df: pd.DataFrame,
    text_col: str = "text",
    label_col: Optional[str] = "label",
) -> pd.DataFrame:
    """Apply cleaning and entropy computation to a text DataFrame.

    Args:
        df: Input DataFrame with at least a text column.
        text_col: Name of the column containing raw text.
        label_col: Name of the label column (kept as-is if present).

    Returns:
        New DataFrame with columns [text_clean, entropy, label (if present)].

    Raises:
        ValueError: If the text column is not found in the DataFrame.
        ValueError: If the DataFrame is empty.
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found. Available: {list(df.columns)}")

    result = df.copy()
    result["text_clean"] = result[text_col].apply(clean_text)
    result["entropy"] = result["text_clean"].apply(shannon_entropy)

    keep_cols = ["text_clean", "entropy"]
    if label_col and label_col in result.columns:
        keep_cols.append(label_col)

    return result[keep_cols]
