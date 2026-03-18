"""Unit tests for src/atlas/loader.py"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import os

from src.atlas.loader import load_atlas, compute_constraint_vector, apply_constraint


def _write_atlas(tmp_path, data: dict) -> Path:
    """Helper: write a temporary atlas CSV."""
    df = pd.DataFrame(data)
    path = tmp_path / "test_atlas.csv"
    df.to_csv(path, index=False)
    return path


class TestLoadAtlas:
    def test_valid_atlas_loads(self, tmp_path):
        path = _write_atlas(tmp_path, {
            "region_name": ["PFC", "ACC", "HPC"],
            "density": [0.9, 0.5, 0.1],
        })
        df = load_atlas(path)
        assert len(df) == 3
        assert "density" in df.columns

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            load_atlas("/nonexistent/path/atlas.csv")

    def test_missing_column_raises(self, tmp_path):
        path = _write_atlas(tmp_path, {"region_name": ["PFC"], "dopamine": [0.5]})
        with pytest.raises(ValueError, match="density"):
            load_atlas(path)

    def test_out_of_range_density_raises(self, tmp_path):
        path = _write_atlas(tmp_path, {
            "region_name": ["PFC"],
            "density": [1.5],  # invalid
        })
        with pytest.raises(ValueError, match="density"):
            load_atlas(path)

    def test_empty_csv_raises(self, tmp_path):
        path = tmp_path / "empty.csv"
        pd.DataFrame(columns=["region_name", "density"]).to_csv(path, index=False)
        with pytest.raises(ValueError, match="empty"):
            load_atlas(path)


class TestConstraintVector:
    def _make_atlas(self, densities):
        return pd.DataFrame({
            "region_name": [f"R{i}" for i in range(len(densities))],
            "density": densities,
        })

    def test_same_length_returns_exact(self):
        atlas = self._make_atlas([0.1, 0.5, 0.9])
        vec = compute_constraint_vector(atlas, 3)
        np.testing.assert_array_almost_equal(vec, [0.1, 0.5, 0.9])

    def test_output_length_matches_request(self):
        atlas = self._make_atlas([0.2, 0.8])
        vec = compute_constraint_vector(atlas, 10)
        assert len(vec) == 10

    def test_values_in_range(self):
        atlas = self._make_atlas([0.0, 0.5, 1.0])
        vec = compute_constraint_vector(atlas, 100)
        assert vec.min() >= 0.0 and vec.max() <= 1.0


class TestApplyConstraint:
    def _make_atlas(self):
        return pd.DataFrame({
            "region_name": ["PFC", "HPC"],
            "density": [1.0, 0.0],
        })

    def test_output_shape_preserved(self):
        weights = np.ones((4, 4))
        result = apply_constraint(weights, self._make_atlas())
        assert result.shape == (4, 4)

    def test_zero_strength_unchanged(self):
        weights = np.ones((6,))
        result = apply_constraint(weights, self._make_atlas(), strength=0.0)
        np.testing.assert_array_almost_equal(result, weights)
