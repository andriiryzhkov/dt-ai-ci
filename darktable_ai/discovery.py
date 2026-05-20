"""Discover available models from models/*/model.yaml."""

from __future__ import annotations

from pathlib import Path

from darktable_ai.config import ModelConfig, load_model_config


def discover_models(root_dir: Path) -> list[ModelConfig]:
    """Find all model.yaml files and load them."""
    models_dir = root_dir / "models"
    configs = []
    for model_dir in sorted(models_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        yaml_path = model_dir / "model.yaml"
        if yaml_path.is_file():
            config = load_model_config(model_dir, root_dir)
            configs.append(config)
    return configs


def find_project_root() -> Path:
    """Walk up from cwd to find the project root (contains models/ and samples/)."""
    cwd = Path.cwd()
    for p in [cwd, *cwd.parents]:
        if (p / "models").is_dir() and (p / "pyproject.toml").is_file():
            return p
    raise FileNotFoundError(
        "Could not find project root (no models/ directory with pyproject.toml found)"
    )
