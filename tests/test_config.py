"""Unit tests for src/utils/config.py"""

import pytest
import yaml
from pathlib import Path

from src.utils.config import load_config


class TestLoadConfig:
    def test_loads_default_config(self):
        config = load_config()
        assert isinstance(config, dict)

    def test_required_top_level_keys(self):
        config = load_config()
        for key in ["project", "paths", "model", "atlas", "benchmark"]:
            assert key in config, f"Missing top-level key: '{key}'"

    def test_paths_section_has_database(self):
        config = load_config()
        assert "database" in config["paths"]

    def test_model_section_has_name(self):
        config = load_config()
        assert "name" in config["model"]

    def test_custom_config_path(self, tmp_path):
        custom = {"project": {"name": "test"}, "paths": {"database": "test.db"},
                   "model": {"name": "m"}, "atlas": {"default": "a.csv"},
                   "benchmark": {"metrics": []}}
        path = tmp_path / "custom.yml"
        path.write_text(yaml.dump(custom))
        config = load_config(path)
        assert config["project"]["name"] == "test"

    def test_nonexistent_config_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nonexistent.yml")
