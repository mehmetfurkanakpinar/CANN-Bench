
from __future__ import annotations
import argparse

from src.pipeline import run_pipeline
from src.atlas.loader import load_atlas
from src.utils.config import load_config
from src.utils.database import get_connection, log_result
from src.benchmark.model import AnatomyConstrainedModel, load_tokenizer
from src.benchmark.evaluator import build_dataset, run_inference, compute_metrics, print_comparison


def run_benchmark(
    offline: bool = False,
    max_rows: int | None = None,
    constraint_strength: float = 1.0,
    split: str = "validation",
    seed: int = 42,
) -> dict:
    
    import torch
    torch.manual_seed(seed)

    config = load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[benchmark] Device: {device}")

    
    # 1. Data
    
    print("\n[benchmark] Step 1/5 — Loading and preprocessing data...")
    processed_df = run_pipeline(
        split=split,
        max_rows=max_rows,
        offline=offline,
    )

   
    # 2. Atlas
    
    print("[benchmark] Step 2/5 — Loading brain atlas...")
    atlas_path = f"{config['paths']['atlases']}{config['atlas']['default']}"
    atlas = load_atlas(atlas_path)
    print(f"[benchmark] Atlas: {config['atlas']['default']} ({len(atlas)} regions)")

   
    # 3. Tokeniser (shared between both conditions)
    
    model_name = config["model"]["name"]
    print(f"\n[benchmark] Step 3/5 — Loading tokenizer ({model_name})...")
    tokenizer = load_tokenizer(model_name)
    dataset = build_dataset(
        processed_df,
        tokenizer,
        max_length=config["model"]["max_length"],
    )

    
    # 4. Baseline run
    
    print("\n[benchmark] Step 4/5 — Running BASELINE (no anatomical constraint)...")
    baseline_wrapper = AnatomyConstrainedModel(
        model_name=model_name,
        atlas=atlas,
        constraint_strength=0.0,   # <-- no constraint
        num_labels=4,
        device=device,
    ).load()

    baseline_results = run_inference(baseline_wrapper.model, dataset, device=device)
    baseline_metrics = compute_metrics(baseline_results)
    print(f"[benchmark] Baseline accuracy: {baseline_metrics['accuracy']:.4f}")

    
    # 5. Constrained run
    
    print(f"\n[benchmark] Step 5/5 — Running CONSTRAINED (strength={constraint_strength})...")
    constrained_wrapper = AnatomyConstrainedModel(
        model_name=model_name,
        atlas=atlas,
        constraint_strength=constraint_strength,
        num_labels=4,
        device=device,
    ).load()

    constrained_results = run_inference(constrained_wrapper.model, dataset, device=device)
    constrained_metrics = compute_metrics(constrained_results)
    print(f"[benchmark] Constrained accuracy: {constrained_metrics['accuracy']:.4f}")

    
    # Results
    
    print_comparison(baseline_metrics, constrained_metrics)

    # Log to database
    conn = get_connection(config["paths"]["database"])
    log_result(
        conn,
        model_name=model_name,
        constrained=False,
        seed=seed,
        metrics={k: v for k, v in baseline_metrics.items()
                 if k in ["accuracy", "f1"]},
        atlas_name=None,
        dataset_name="tweet_eval/emotion" if not offline else "mock",
        notes=f"split={split}",
    )
    log_result(
        conn,
        model_name=model_name,
        constrained=True,
        seed=seed,
        metrics={k: v for k, v in constrained_metrics.items()
                 if k in ["accuracy", "f1"]},
        atlas_name=config["atlas"]["default"],
        dataset_name="tweet_eval/emotion" if not offline else "mock",
        notes=f"split={split}, strength={constraint_strength}",
    )
    conn.close()
    print("[benchmark] Results logged to database. ✓")

    return {"baseline": baseline_metrics, "constrained": constrained_metrics}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CANN-Bench: anatomy-constrained AI benchmark")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--split", default="validation",
                        choices=["train", "validation", "test"])
    parser.add_argument("--constraint-strength", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_benchmark(
        offline=args.offline,
        max_rows=args.max_rows,
        constraint_strength=args.constraint_strength,
        split=args.split,
        seed=args.seed,
    )
