"""Configuration loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    if cfg.get("mode") != "paper":
        raise ValueError(
            "Only paper trading is supported. Set mode: paper in config.yaml. "
            "Real-money trading is intentionally not implemented."
        )
    return cfg


def resolve_path(relative: str) -> Path:
    """Resolve a config-relative path against the project root."""
    p = Path(relative)
    return p if p.is_absolute() else PROJECT_ROOT / p
