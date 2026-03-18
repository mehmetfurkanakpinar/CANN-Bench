"""
CANN-Bench Data Pipeline
==================================
Orchestrates the full ingestion → preprocessing → storage workflow.

Usage:
    python -m src.pipeline                          # full tweet_eval dataset
    python -m src.pipeline --split validation       # validation split
    python -m src.pipeline --max-rows 500           # quick dev run
    python -m src.pipeline --offline                # use mock data (no internet needed)

Output:
    - Processed CSV saved to data/processed/
    - Entropy summary printed to stdout
    - Run metadata logged to results/benchmark.db
"""

from __future__ import annotations
import argparse
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.ingestion.hf_loader import load_tweet_eval, load_mock_dataset
from src.preprocessing.engine import preprocess_dataframe
from src.utils.config import load_config
from src.utils.database import get_connection


# Helpers

def _save_processed(df: pd.DataFrame, output_dir: str, split: str) -> Path:
    """Save processed DataFrame to CSV."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = out / f"tweet_eval_{split}_{timestamp}.csv"
    df.to_csv(path, index=False)
    print(f"[pipeline] Saved processed data → {path}")
    return path


def _log_pipeline_run(conn: sqlite3.Connection, metadata: dict) -> None:
    """Log pipeline run stats to SQLite."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS pipeline_runs (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            run_timestamp   TEXT NOT NULL,
            dataset         TEXT NOT NULL,
            split           TEXT NOT NULL,
            n_samples       INTEGER NOT NULL,
            mean_entropy    REAL,
            std_entropy     REAL,
            offline_mode    INTEGER NOT NULL,
            output_path     TEXT
        )
        """
    )
    conn.execute(
        """
        INSERT INTO pipeline_runs
            (run_timestamp, dataset, split, n_samples,
             mean_entropy, std_entropy, offline_mode, output_path)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.utcnow().isoformat(),
            metadata["dataset"],
            metadata["split"],
            metadata["n_samples"],
            metadata["mean_entropy"],
            metadata["std_entropy"],
            int(metadata["offline"]),
            metadata["output_path"],
        ),
    )
    conn.commit()
    print("[pipeline] Run metadata logged to database.")


def _print_entropy_report(df: pd.DataFrame) -> None:
    """Print a summary of entropy across emotion categories."""
    print("\n" + "=" * 55)
    print("  Shannon Entropy Report — tweet_eval/emotion")
    print("=" * 55)
    print(f"  {'Emotion':<12} {'N':>6} {'Mean H':>8} {'Std H':>8}")
    print("  " + "-" * 38)

    for emotion, group in df.groupby("emotion"):
        print(
            f"  {emotion:<12} {len(group):>6} "
            f"{group['entropy'].mean():>8.3f} "
            f"{group['entropy'].std():>8.3f}"
        )

    print("  " + "-" * 38)
    print(
        f"  {'TOTAL':<12} {len(df):>6} "
        f"{df['entropy'].mean():>8.3f} "
        f"{df['entropy'].std():>8.3f}"
    )
    print("=" * 55)
    print()
    print("  Interpretation:")
    print("  Higher entropy → more lexically complex text.")
    print("  Anatomical constraints (Week 3) will be tested")
    print("  against whether complexity interacts with the")
    print("  dopamine gradient to affect model performance.")
    print("=" * 55 + "\n")


# Main

def run_pipeline(
    split: str = "train",
    max_rows: int | None = None,
    offline: bool = False,
) -> pd.DataFrame:
    """Run the full ingestion → preprocessing → storage pipeline.

    Args:
        split: Dataset split ('train', 'validation', 'test').
        max_rows: Truncate dataset to first N rows (None = all).
        offline: Use mock dataset instead of HuggingFace Hub.

    Returns:
        Processed DataFrame with columns [text_clean, entropy, label, emotion].
    """
    config = load_config()

    # 1. Ingestion
    if offline:
        raw_df = load_mock_dataset(n=max_rows or 200)
    else:
        raw_df = load_tweet_eval(split=split, max_rows=max_rows)

    # 2. Preprocessing
    print(f"\n[pipeline] Preprocessing {len(raw_df)} samples...")
    processed = preprocess_dataframe(raw_df, text_col="text", label_col="label")

    # Re-attach emotion labels for reporting
    processed["emotion"] = raw_df["emotion"].values

    # 3. Report
    _print_entropy_report(processed)

    # 4. Save
    output_path = _save_processed(
        processed,
        output_dir=config["paths"]["data_processed"],
        split="offline" if offline else split,
    )

    # 5. Log to database
    conn = get_connection(config["paths"]["database"])
    _log_pipeline_run(conn, {
        "dataset": "mock" if offline else "tweet_eval/emotion",
        "split": "offline" if offline else split,
        "n_samples": len(processed),
        "mean_entropy": processed["entropy"].mean(),
        "std_entropy": processed["entropy"].std(),
        "offline": offline,
        "output_path": str(output_path),
    })
    conn.close()

    print(f"[pipeline] ✓ Pipeline complete. {len(processed)} samples processed.\n")
    return processed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CANN-Bench data pipeline")
    parser.add_argument("--split", default="train",
                        choices=["train", "validation", "test"])
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Limit number of rows loaded (useful for dev)")
    parser.add_argument("--offline", action="store_true",
                        help="Use mock dataset instead of HuggingFace Hub")
    args = parser.parse_args()

    run_pipeline(
        split=args.split,
        max_rows=args.max_rows,
        offline=args.offline,
    )
