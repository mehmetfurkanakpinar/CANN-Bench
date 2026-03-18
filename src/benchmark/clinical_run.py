"""
CANN-Bench Clinical NLP Benchmark
===================================
2×2 factorial comparison on the Kaggle Suicide Detection dataset.

Models evaluated
----------------
  A — Fine-tuned,   no constraint : checkpoint-48726 as-is
  B — Fine-tuned  + CANN constraint: checkpoint-48726 + dopamine gradient
  C — Vanilla,      no constraint : distilbert-base-cased, random head
  D — Vanilla     + CANN constraint: distilbert-base-cased + dopamine gradient

Design
------
  Fine-tuning (yes/no) - Constraint (yes/no)

  This separates two effects cleanly:
    - B vs A : does the constraint help a model that already has task knowledge?
    - D vs C : does the constraint help a model with no task knowledge?
    - A vs C : how much does fine-tuning matter on its own?
    - B vs D : does fine-tuning interact with the constraint?

Key clinical metric: recall for the *suicide* class (label = 1).
  In safety-critical screening, false negatives (missed cases) are more
  harmful than false positives, so recall takes priority over precision.

Usage
-----
    python -m src.benchmark.clinical_run                         # full run
    python -m src.benchmark.clinical_run --max-rows 500          # smoke test
    python -m src.benchmark.clinical_run --max-rows 500 --seed 0
    python -m src.benchmark.clinical_run --constraint-strength 0.5
"""

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

from src.ingestion.kaggle_loader import load_kaggle_suicide
from src.preprocessing.engine import preprocess_dataframe
from src.atlas.loader import load_atlas
from src.utils.config import load_config
from src.utils.database import get_connection, log_result

# Label map matching kaggle_loader.py
CLINICAL_LABELS = {0: "non-suicide", 1: "suicide"}


# Model loading


def load_clinical_model(
    model_path: str,
    num_labels: int = 2,
    constraint_strength: float = 0.0,
    atlas=None,
    device: str = "cpu",
):
    """Load a DistilBERT model for binary clinical classification.

    Args:
        model_path: HuggingFace model name or local checkpoint directory.
        num_labels: Number of output classes.
        constraint_strength: If > 0 and atlas is provided, apply dopamine
                             gradient modulation to attention weights.
        atlas: Brain atlas DataFrame (from src.atlas.loader).  Required when
               constraint_strength > 0.
        device: 'cpu' or 'cuda'.

    Returns:
        Loaded (and optionally constrained) HuggingFace model in eval mode.
    """
    import torch
    from transformers import AutoModelForSequenceClassification
    from src.atlas.loader import compute_constraint_vector

    print(f"  [model] Loading from: {model_path}")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    )
    model.to(device)
    model.eval()

    if constraint_strength > 0 and atlas is not None:
        attention_keywords = ["q_lin", "k_lin", "v_lin", "attention"]
        modulated = 0
        with torch.no_grad():
            for name, param in model.named_parameters():
                if not any(kw in name for kw in attention_keywords):
                    continue
                if "weight" not in name:
                    continue
                flat = param.data.cpu().numpy().flatten()
                constraint = compute_constraint_vector(atlas, len(flat))
                gain = 1.0 + constraint_strength * (constraint - 0.5) * 2
                param.data = torch.tensor(
                    (flat * gain).reshape(param.data.shape),
                    dtype=param.dtype,
                    device=device,
                )
                modulated += 1
        print(f"  [model] Applied atlas constraint to {modulated} layers "
              f"(strength={constraint_strength})")

    return model


# Dataset preparation

def build_clinical_dataset(df: pd.DataFrame, tokenizer, max_length: int = 128):
    """Tokenise a preprocessed clinical DataFrame into a HuggingFace Dataset.

    Args:
        df: DataFrame with columns [text_clean, entropy, label].
        tokenizer: Loaded HuggingFace tokenizer.
        max_length: Maximum token length.

    Returns:
        HuggingFace Dataset ready for inference.
    """
    from datasets import Dataset

    hf_ds = Dataset.from_pandas(df[["text_clean", "entropy", "label"]].copy())

    def tokenize(batch):
        return tokenizer(
            batch["text_clean"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    hf_ds = hf_ds.map(tokenize, batched=True)
    hf_ds = hf_ds.rename_column("label", "labels")
    hf_ds.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels", "entropy"],
    )
    return hf_ds


# Inference

def run_clinical_inference(model, dataset, batch_size: int = 32, device: str = "cpu") -> dict:
    """Run inference and return raw predictions + probabilities.

    Args:
        model: Loaded HuggingFace model.
        dataset: Tokenised HuggingFace Dataset.
        batch_size: Inference batch size.
        device: 'cpu' or 'cuda'.

    Returns:
        Dict with preds, labels, probs (positive class), entropies.
    """
    import torch
    from torch.utils.data import DataLoader
    import torch.nn.functional as F

    loader = DataLoader(dataset, batch_size=batch_size)
    model.eval()

    all_preds, all_labels, all_probs, all_entropies = [], [], [], []
    total_batches = len(loader)

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx % 100 == 0:
                print(f"  [inference] batch {batch_idx}/{total_batches} "
                      f"({100*batch_idx/total_batches:.0f}%)", flush=True)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()  # P(suicide)
            preds = logits.argmax(dim=-1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(batch["labels"].numpy())
            all_probs.extend(probs)
            all_entropies.extend(batch["entropy"].numpy())

    return {
        "preds": np.array(all_preds),
        "labels": np.array(all_labels),
        "probs": np.array(all_probs),
        "entropies": np.array(all_entropies),
    }


# Clinical metrics

def compute_clinical_metrics(results: dict) -> dict:
    """Compute accuracy, F1, recall/precision for suicide class, and ROC-AUC.

    Args:
        results: Output of run_clinical_inference().

    Returns:
        Flat metrics dictionary for display and database logging.
    """
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        recall_score,
        precision_score,
        roc_auc_score,
    )

    preds = results["preds"]
    labels = results["labels"]
    probs = results["probs"]
    entropies = results["entropies"]

    accuracy = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
    f1_per_class = f1_score(labels, preds, average=None, zero_division=0)

    # Clinical focus: recall and precision for suicide class (label=1)
    recall_suicide = recall_score(labels, preds, pos_label=1, zero_division=0)
    precision_suicide = precision_score(labels, preds, pos_label=1, zero_division=0)

    try:
        roc_auc = roc_auc_score(labels, probs)
    except ValueError:
        roc_auc = float("nan")

    metrics = {
        "accuracy": round(float(accuracy), 4),
        "f1": round(float(f1_macro), 4),
        "f1_non_suicide": round(float(f1_per_class[0]), 4) if len(f1_per_class) > 0 else 0.0,
        "f1_suicide": round(float(f1_per_class[1]), 4) if len(f1_per_class) > 1 else 0.0,
        "recall_suicide": round(float(recall_suicide), 4),
        "precision_suicide": round(float(precision_suicide), 4),
        "roc_auc": round(float(roc_auc), 4),
    }

    # Entropy-stratified accuracy (quartile bins)
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

def print_clinical_comparison(
    metrics_a: dict,
    metrics_b: dict,
    metrics_c: dict,
    metrics_d: dict,
) -> None:
    """Print a 2x2 factorial comparison table."""
    def delta(x, y):
        if isinstance(x, float) and isinstance(y, float):
            d = y - x
            return f"{'+'if d>=0 else ''}{d:.4f}"
        return ""

    print("\n" + "=" * 84)
    print("  CANN-Bench Clinical Results — Suicide Detection  (2×2 factorial)")
    print("  A: Fine-tuned            B: Fine-tuned+constraint")
    print("  C: Vanilla               D: Vanilla+constraint")
    print("=" * 84)
    print(f"  {'Metric':<28} {'A':>9} {'B':>9}  {'A→B':>8}  {'C':>9} {'D':>9}  {'C→D':>8}")
    print("  " + "-" * 78)

    core = [
        "accuracy",
        "f1",
        "f1_suicide",
        "f1_non_suicide",
        "recall_suicide",
        "precision_suicide",
        "roc_auc",
    ]
    for key in core:
        a = metrics_a.get(key, "—")
        b = metrics_b.get(key, "—")
        c = metrics_c.get(key, "—")
        d = metrics_d.get(key, "—")
        print(f"  {key:<28} {str(a):>9} {str(b):>9}  {delta(a,b):>8}  "
              f"{str(c):>9} {str(d):>9}  {delta(c,d):>8}")

    print("  " + "-" * 78)
    print("  Entropy-stratified accuracy:")
    for q in ["low", "mid_low", "mid_high", "high"]:
        key = f"accuracy_entropy_{q}"
        a = metrics_a.get(key, "—")
        b = metrics_b.get(key, "—")
        c = metrics_c.get(key, "—")
        d = metrics_d.get(key, "—")
        print(f"  {key:<28} {str(a):>9} {str(b):>9}  {delta(a,b):>8}  "
              f"{str(c):>9} {str(d):>9}  {delta(c,d):>8}")
    print("=" * 84 + "\n")



# Main entry point

def run_clinical_benchmark(
    max_rows: int | None = None,
    seed: int = 42,
    batch_size: int = 32,
    constraint_strength: float | None = None,
) -> dict:
    """Run the 2×2 factorial clinical benchmark.

    Conditions:
      A — Fine-tuned,   no constraint
      B — Fine-tuned  + constraint
      C — Vanilla,      no constraint
      D — Vanilla     + constraint

    Args:
        max_rows: Rows to sample from the dataset (None = full dataset).
        seed: Random seed for reproducibility.
        batch_size: Inference batch size.
        constraint_strength: Override atlas constraint_strength from config.

    Returns:
        Dict with keys 'model_a', 'model_b', 'model_c', 'model_d'.
    """
    import torch
    torch.manual_seed(seed)

    config = load_config()
    clin_cfg = config["clinical_benchmark"]
    strength = constraint_strength if constraint_strength is not None \
        else config["atlas"]["constraint_strength"]

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"\n[clinical] Device: {device}")

    
    # 1. Data
    
    print("\n[clinical] Step 1/5 — Loading Suicide Detection dataset...")
    _, test_df = load_kaggle_suicide(
        clin_cfg["dataset"],
        test_size=clin_cfg["test_size"],
        max_rows=max_rows,
        seed=seed,
    )
    print(f"[clinical] Test set: {len(test_df):,} rows  "
          f"(suicide={test_df['label'].sum():,}, "
          f"non-suicide={(test_df['label']==0).sum():,})")


    # 2. Preprocessing (text clean + Shannon entropy)
    
    print("[clinical] Step 2/5 — Preprocessing (clean + entropy)...")
    processed_df = preprocess_dataframe(test_df, text_col="text", label_col="label")
    print(f"[clinical] Entropy range: "
          f"[{processed_df['entropy'].min():.2f}, {processed_df['entropy'].max():.2f}]")

    
    # 3. Tokeniser + atlas (shared by all conditions)
    
    print("\n[clinical] Step 3/5 — Loading tokenizer and brain atlas...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(clin_cfg["checkpoint"])
    dataset = build_clinical_dataset(
        processed_df,
        tokenizer,
        max_length=config["model"]["max_length"],
    )
    atlas_path = f"{config['paths']['atlases']}{config['atlas']['default']}"
    atlas = load_atlas(atlas_path)
    print(f"[clinical] Atlas: {config['atlas']['default']} ({len(atlas)} regions)")

    
    # 4. Run all four conditions
    
    checkpoint_path = clin_cfg["checkpoint"]
    base_model = "distilbert-base-cased"
    num_labels = clin_cfg["num_labels"]
    all_metrics = {}

    conditions = [
        ("model_a", checkpoint_path, 0.0,     "A — Fine-tuned,   no constraint"),
        ("model_b", checkpoint_path, strength, f"B — Fine-tuned  + constraint (s={strength})"),
        ("model_c", base_model,      0.0,     "C — Vanilla,       no constraint"),
        ("model_d", base_model,      strength, f"D — Vanilla      + constraint (s={strength})"),
    ]

    for label, model_path, s, tag in conditions:
        print(f"\n[clinical] Step 4/5 — Running {tag}...")

        model = load_clinical_model(
            model_path=model_path,
            num_labels=num_labels,
            constraint_strength=s,
            atlas=atlas if s > 0 else None,
            device=device,
        )
        results = run_clinical_inference(model, dataset, batch_size=batch_size, device=device)
        metrics = compute_clinical_metrics(results)
        all_metrics[label] = metrics

        print(f"  accuracy={metrics['accuracy']:.4f}  "
              f"recall_suicide={metrics['recall_suicide']:.4f}  "
              f"roc_auc={metrics['roc_auc']:.4f}")

        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    
    # 5. Results + logging
    
    print_clinical_comparison(
        all_metrics["model_a"],
        all_metrics["model_b"],
        all_metrics["model_c"],
        all_metrics["model_d"],
    )

    print("[clinical] Step 5/5 — Logging results to database...")
    conn = get_connection(config["paths"]["database"])
    log_rows = [
        ("model_a", "checkpoint-48726", False, "finetuned_no_constraint"),
        ("model_b", "checkpoint-48726", True,  f"finetuned_constrained_s{strength}"),
        ("model_c", base_model,         False, "vanilla_no_constraint"),
        ("model_d", base_model,         True,  f"vanilla_constrained_s{strength}"),
    ]
    for model_key, model_name, constrained, notes_tag in log_rows:
        m = all_metrics[model_key]
        log_result(
            conn,
            model_name=model_name,
            constrained=constrained,
            seed=seed,
            metrics={"accuracy": m["accuracy"], "f1": m["f1"], "roc_auc": m["roc_auc"]},
            atlas_name=config["atlas"]["default"] if constrained else None,
            dataset_name="Suicide_Detection",
            notes=f"{notes_tag}, recall_suicide={m['recall_suicide']}, "
                  f"max_rows={max_rows}",
        )
    conn.close()
    print("[clinical] Results logged. ✓")

    return all_metrics



# CLI


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CANN-Bench Clinical NLP Benchmark — fine-tuned vs fine-tuned + constraint"
    )
    parser.add_argument(
        "--max-rows", type=int, default=None,
        help="Limit dataset rows (e.g. 500 for a smoke test).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--constraint-strength", type=float, default=None,
        help="Override atlas constraint_strength from config.yml (default: use config value).",
    )
    args = parser.parse_args()

    run_clinical_benchmark(
        max_rows=args.max_rows,
        seed=args.seed,
        batch_size=args.batch_size,
        constraint_strength=args.constraint_strength,
    )
