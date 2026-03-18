"""
Unit tests for benchmark components.
All tests use mock data and mock models — no GPU or internet required.
"""

import pytest
import numpy as np
import pandas as pd

from src.benchmark.evaluator import compute_metrics, print_comparison, EMOTION_LABELS
from src.benchmark.model import AnatomyConstrainedModel


# Helpers

def _make_results(n=100, correct_frac=0.75, seed=42):
    """Generate synthetic inference results for testing."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 4, size=n)
    preds = labels.copy()
    # Flip a fraction of predictions to simulate errors
    n_wrong = int(n * (1 - correct_frac))
    wrong_idx = rng.choice(n, size=n_wrong, replace=False)
    preds[wrong_idx] = (labels[wrong_idx] + 1) % 4
    entropies = rng.uniform(1.5, 4.0, size=n)
    return {"preds": preds, "labels": labels, "entropies": entropies}



# Compute_metrics

class TestComputeMetrics:
    def test_returns_dict(self):
        results = _make_results()
        metrics = compute_metrics(results)
        assert isinstance(metrics, dict)

    def test_accuracy_in_range(self):
        results = _make_results(correct_frac=0.75)
        metrics = compute_metrics(results)
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_perfect_accuracy(self):
        labels = np.array([0, 1, 2, 3] * 10)
        results = {"preds": labels.copy(), "labels": labels, "entropies": np.ones(40)}
        metrics = compute_metrics(results)
        assert metrics["accuracy"] == pytest.approx(1.0)

    def test_f1_macro_present(self):
        metrics = compute_metrics(_make_results())
        assert "f1" in metrics

    def test_per_class_f1_present(self):
        metrics = compute_metrics(_make_results())
        for emotion in EMOTION_LABELS.values():
            assert f"f1_{emotion}" in metrics

    def test_entropy_quartile_keys_present(self):
        metrics = compute_metrics(_make_results())
        for q in ["low", "mid_low", "mid_high", "high"]:
            assert f"accuracy_entropy_{q}" in metrics

    def test_entropy_quartile_values_in_range(self):
        metrics = compute_metrics(_make_results())
        for q in ["low", "mid_low", "mid_high", "high"]:
            key = f"accuracy_entropy_{q}"
            assert 0.0 <= metrics[key] <= 1.0

    def test_constrained_vs_baseline_structure_identical(self):
        """Both conditions should produce the same set of metric keys."""
        baseline = compute_metrics(_make_results(correct_frac=0.70, seed=1))
        constrained = compute_metrics(_make_results(correct_frac=0.78, seed=2))
        assert set(baseline.keys()) == set(constrained.keys())



# AnatomyConstrainedModel (no-load tests)

class TestAnatomyConstrainedModelInit:
    def _make_atlas(self):
        return pd.DataFrame({
            "region_name": ["PFC", "HPC", "V1"],
            "density": [0.9, 0.5, 0.2],
        })

    def test_initialises_without_error(self):
        model = AnatomyConstrainedModel(
            model_name="distilbert-base-uncased",
            atlas=self._make_atlas(),
            constraint_strength=1.0,
        )
        assert model.model_name == "distilbert-base-uncased"

    def test_constraint_strength_stored(self):
        model = AnatomyConstrainedModel(
            model_name="distilbert-base-uncased",
            atlas=self._make_atlas(),
            constraint_strength=0.5,
        )
        assert model.constraint_strength == 0.5

    def test_model_not_loaded_before_load_call(self):
        model = AnatomyConstrainedModel(
            model_name="distilbert-base-uncased",
            atlas=self._make_atlas(),
        )
        with pytest.raises(RuntimeError, match="load"):
            _ = model.model



# print_comparison (smoke test — just to ensure it doesn't crash)

class TestPrintComparison:
    def test_runs_without_error(self, capsys):
        baseline = compute_metrics(_make_results(correct_frac=0.70))
        constrained = compute_metrics(_make_results(correct_frac=0.78))
        print_comparison(baseline, constrained)
        captured = capsys.readouterr()
        assert "Baseline" in captured.out
        assert "Constrained" in captured.out
        assert "accuracy" in captured.out
