"""
Benchmark Evaluator

Runs two conditions head-to-head:
  1. Baseline:    standard DistilBERT (no anatomical constraints)
  2. Constrained: DistilBERT + dopamine gradient modulation

Metrics computed per condition:
  - Accuracy
  - Macro F1
  - Per-class F1 (anger, joy, optimism, sadness)
  - Performance stratified by Shannon entropy quartile
    (key hypothesis: anatomy constraint helps most on high-entropy text)

All results are written to SQLite via src.utils.database.
"""

from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd



# Tokenisation & dataset prep


def build_dataset(
    df: pd.DataFrame,
    tokenizer,
    max_length: int = 128,
    text_col: str = "text_clean",
    label_col: str = "label",
):
    """Convert a preprocessed DataFrame into a HuggingFace Dataset."""
    try:
        from datasets import Dataset
        import torch
    except ImportError:
        raise ImportError("Install 'datasets' and 'torch' packages.")

    hf_dataset = Dataset.from_pandas(df[[text_col, label_col, "entropy"]].copy())

    def tokenize(batch):
        return tokenizer(
            batch[text_col],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    hf_dataset = hf_dataset.map(tokenize, batched=True)
    hf_dataset = hf_dataset.rename_column(label_col, "labels")
    hf_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels", "entropy"],
    )
    return hf_dataset



# Inference


def run_inference(model, dataset, batch_size: int = 32, device: str = "cpu") -> dict:
    """Run model inference and return predictions + true labels.

    Args:
        model: Loaded HuggingFace model (output of AnatomyConstrainedModel.load()).
        dataset: HuggingFace Dataset with tokenized inputs.
        batch_size: Inference batch size.
        device: 'cpu' or 'cuda'.

    Returns:
        Dict with keys: preds (np.ndarray), labels (np.ndarray), entropies (np.ndarray).
    """
    import torch
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=batch_size)
    model.eval()

    all_preds, all_labels, all_entropies = [], [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]
            entropies = batch["entropy"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = outputs.logits.argmax(dim=-1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            all_entropies.extend(entropies.numpy())

    return {
        "preds": np.array(all_preds),
        "labels": np.array(all_labels),
        "entropies": np.array(all_entropies),
    }



# Metrics


EMOTION_LABELS = {0: "anger", 1: "joy", 2: "optimism", 3: "sadness"}


def compute_metrics(results: dict, label_names: dict = EMOTION_LABELS) -> dict:
    """Compute accuracy, macro F1, per-class F1, and entropy-stratified accuracy.

    Args:
        results: Output dict from run_inference().
        label_names: Mapping from int label → string name.

    Returns:
        Flat metrics dictionary suitable for database logging.
    """
    from sklearn.metrics import accuracy_score, f1_score, classification_report

    preds = results["preds"]
    labels = results["labels"]
    entropies = results["entropies"]

    accuracy = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
    f1_per_class = f1_score(labels, preds, average=None, zero_division=0)

    metrics = {
        "accuracy": round(float(accuracy), 4),
        "f1": round(float(f1_macro), 4),
    }

    # Per-class F1
    for label_id, name in label_names.items():
        if label_id < len(f1_per_class):
            metrics[f"f1_{name}"] = round(float(f1_per_class[label_id]), 4)

    # Entropy-stratified accuracy (quartiles)
    quartiles = np.percentile(entropies, [25, 50, 75])
    q_labels = ["low", "mid_low", "mid_high", "high"]
    bins = [-np.inf] + list(quartiles) + [np.inf]

    for i, q_label in enumerate(q_labels):
        mask = (entropies >= bins[i]) & (entropies < bins[i + 1])
        if mask.sum() > 0:
            q_acc = accuracy_score(labels[mask], preds[mask])
            metrics[f"accuracy_entropy_{q_label}"] = round(float(q_acc), 4)

    return metrics



# Printer


def print_comparison(baseline_metrics: dict, constrained_metrics: dict) -> None:
    """Print a side-by-side comparison of baseline vs constrained results."""
    print("\n" + "=" * 62)
    print("  CANN-Bench Results: Baseline vs Anatomy-Constrained")
    print("=" * 62)
    print(f"  {'Metric':<30} {'Baseline':>12} {'Constrained':>12}")
    print("  " + "-" * 56)

    core = ["accuracy", "f1", "f1_anger", "f1_joy", "f1_optimism", "f1_sadness"]
    for key in core:
        b = baseline_metrics.get(key, "—")
        c = constrained_metrics.get(key, "—")
        delta = ""
        if isinstance(b, float) and isinstance(c, float):
            diff = c - b
            delta = f"({'+'if diff>=0 else ''}{diff:.3f})"
        print(f"  {key:<30} {str(b):>12} {str(c):>10} {delta}")

    print("  " + "-" * 56)
    print("  Entropy-stratified accuracy:")
    for q in ["low", "mid_low", "mid_high", "high"]:
        key = f"accuracy_entropy_{q}"
        b = baseline_metrics.get(key, "—")
        c = constrained_metrics.get(key, "—")
        print(f"  {key:<30} {str(b):>12} {str(c):>12}")

    print("=" * 62 + "\n")
