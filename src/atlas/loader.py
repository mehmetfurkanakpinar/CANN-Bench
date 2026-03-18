"""
Brain Atlas Module: loader and constraint logic.

A brain atlas in CANN-Bench is a CSV with (at minimum) two columns:
  - region_name : str   — name of the cortical/subcortical region
  - density     : float — neurotransmitter density, normalised to [0, 1]

The constraint logic modulates model learning rates by these density values,
operationalising the dopamine gradient hypothesis:

  Froudist-Walsh et al. (2021). A dopamine gradient controls access to
  distributed working memory in the large-scale monkey cortex. Neuron.

High density → faster learning (higher effective LR).
Low density  → slower learning (lower effective LR).
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path


def load_atlas(path: str | Path) -> pd.DataFrame:
    """Load a brain atlas CSV file.

    Args:
        path: Path to CSV file with columns [region_name, density].

    Returns:
        DataFrame with validated atlas data.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If required columns are missing or density values are out of range.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Atlas file not found: {path}")

    df = pd.read_csv(path)

    # Validate schema
    required_cols = {"region_name", "density"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Atlas CSV missing required columns: {missing}")

    # Validate density range
    if not df["density"].between(0.0, 1.0).all():
        raise ValueError("All density values must be in [0, 1].")

    if df.empty:
        raise ValueError("Atlas CSV is empty.")

    return df.reset_index(drop=True)


def compute_constraint_vector(atlas: pd.DataFrame, n_weights: int) -> np.ndarray:
    """Map atlas density values to a weight-space constraint vector.

    Tiles or interpolates the atlas densities to match the number of model
    weight dimensions, producing a multiplicative scaling vector.

    Args:
        atlas: Validated atlas DataFrame (output of load_atlas).
        n_weights: Number of model weight dimensions to constrain.

    Returns:
        1-D numpy array of shape (n_weights,) with values in [0, 1].
    """
    densities = atlas["density"].values.astype(np.float32)

    if len(densities) == n_weights:
        return densities

    # Interpolate densities to match weight dimensionality
    indices_from = np.linspace(0, len(densities) - 1, n_weights)
    constraint = np.interp(indices_from, np.arange(len(densities)), densities)
    return constraint.astype(np.float32)


def apply_constraint(weights: np.ndarray, atlas: pd.DataFrame, strength: float = 1.0) -> np.ndarray:
    """Apply anatomical constraint to a weight matrix.

    Scales weights by the constraint vector derived from the atlas.
    strength=0 returns weights unchanged; strength=1 applies full modulation.

    Args:
        weights: Model weight array of arbitrary shape.
        atlas: Validated atlas DataFrame.
        strength: Interpolation factor between unconstrained (0) and fully constrained (1).

    Returns:
        Modulated weight array with the same shape as input.
    """
    original_shape = weights.shape
    flat = weights.flatten()

    constraint = compute_constraint_vector(atlas, len(flat))
    modulation = 1.0 + strength * (constraint - 0.5) * 2  # maps [0,1] → [-1, 1] gain

    modulated = flat * modulation
    return modulated.reshape(original_shape)
