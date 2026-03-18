"""
HuggingFace Dataset Loader for CANN-Bench.

Primary dataset: tweet_eval/emotion
  - 4 classes: anger (0), joy (1), optimism (2), sadness (3)
  - ~3,257 train / 374 validation / 1,421 test samples
  - Publicly available
  - Emotional valence variation makes it well-suited for testing
    whether anatomical constraints (dopamine gradient) differentially
    affect processing of high vs low-arousal language.

Why this dataset?
  The dopamine gradient (Froudist-Walsh et al., 2021) predicts that
  anterior regions with high dopaminergic tone preferentially process
  complex, emotionally salient information. tweet_eval/emotion provides
  a naturalistic test for this hypothesis at the NLP level.
"""

from __future__ import annotations
from typing import Optional
import pandas as pd


EMOTION_LABELS = {0: "anger", 1: "joy", 2: "optimism", 3: "sadness"}


def load_tweet_eval(
    split: str = "train",
    max_rows: Optional[int] = None,
) -> pd.DataFrame:
    """Load the tweet_eval/emotion dataset from HuggingFace Hub.

    Args:
        split: One of 'train', 'validation', or 'test'.
        max_rows: If set, truncate to first N rows.

    Returns:
        DataFrame with columns [text, label, emotion].

    Raises:
        ImportError: If the 'datasets' package is not installed.
        ValueError: If an invalid split is requested.
    """
    valid_splits = {"train", "validation", "test"}
    if split not in valid_splits:
        raise ValueError(f"Invalid split '{split}'. Choose from: {valid_splits}")

    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "Install HuggingFace datasets"
        )

    print(f"[ingestion] Loading tweet_eval/emotion ({split} split)...")
    dataset = load_dataset("tweet_eval", "emotion", split=split, trust_remote_code=True)

    if max_rows:
        dataset = dataset.select(range(min(max_rows, len(dataset))))

    df = dataset.to_pandas()
    df = df.rename(columns={"text": "text", "label": "label"})
    df["emotion"] = df["label"].map(EMOTION_LABELS)

    print(f"[ingestion] Loaded {len(df)} samples. Label distribution:")
    print(df["emotion"].value_counts().to_string())

    return df[["text", "label", "emotion"]].reset_index(drop=True)


def load_mock_dataset(n: int = 200) -> pd.DataFrame:
    """Generate a small mock dataset for offline testing and CI environments.

    Produces synthetic tweet-like text with realistic entropy variation
    across the four emotion categories.

    Args:
        n: Total number of samples to generate.

    Returns:
        DataFrame with columns [text, label, emotion] — same schema as load_tweet_eval.
    """
    import random, math
    random.seed(42)

    templates = {
        0: [  # anger — short, repetitive, low entropy
            "i hate this so much",
            "this is absolutely awful and terrible",
            "why why why does this keep happening",
            "so angry right now cannot believe this",
            "furious about what just happened today",
        ],
        1: [  # joy — varied vocabulary, higher entropy
            "what an incredible wonderful amazing day this has been",
            "feeling so happy grateful and blessed today",
            "just got the best news ever so excited",
            "love everything about this beautiful sunny morning",
            "celebrated with friends family food and laughter tonight",
        ],
        2: [  # optimism — moderate complexity
            "things will get better i believe in the future",
            "looking forward to new opportunities and challenges ahead",
            "staying positive despite the difficulties we face together",
            "tomorrow brings fresh chances to grow and improve",
            "hopeful that our efforts will lead to great outcomes",
        ],
        3: [  # sadness — longer, more complex sentences
            "feeling so alone and lost in this overwhelming world",
            "miss the way things used to be before everything changed",
            "struggling to find meaning in the monotony of daily life",
            "exhausted by the weight of emotions i cannot express",
            "grief comes in waves unexpected and impossible to control",
        ],
    }

    rows = []
    per_class = n // 4
    for label, texts in templates.items():
        for i in range(per_class):
            text = random.choice(texts)
            rows.append({
                "text": text,
                "label": label,
                "emotion": EMOTION_LABELS[label],
            })

    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"[ingestion] Generated {len(df)} mock samples (offline mode).")
    return df
