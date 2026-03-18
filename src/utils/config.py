"""
Utility: Configuration loader.
Reads config.yml and exposes settings as a plain dict.
"""

from __future__ import annotations

import yaml
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict:
    """Load YAML configuration file.

    Args:
        path: Path to config file. Defaults to <project_root>/config.yml.

    Returns:
        Configuration dictionary.

    Raises:
        FileNotFoundError: If config file does not exist.
    """
    config_path = Path(path) if path else _ROOT / "config.yml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config
