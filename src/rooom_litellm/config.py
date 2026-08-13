from __future__ import annotations

import os
from pathlib import Path

import yaml

from .models import ModelSpec, RouterConfig, RouterWeights


def _env(value):
    if isinstance(value, str) and value.startswith("env:"):
        return os.getenv(value[4:], "")
    return value


def load_config(path: str | Path) -> RouterConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    models = []
    for item in raw.get("models", []):
        item = dict(item)
        item["regions"] = tuple(item.get("regions", ["global"]))
        item["data_classes"] = tuple(item.get("data_classes", ["public", "internal"]))
        item["capabilities"] = tuple(item.get("capabilities", ["chat"]))
        item["metadata"] = {k: _env(v) for k, v in item.get("metadata", {}).items()}
        models.append(ModelSpec(**item))
    weights = RouterWeights(**raw.get("weights", {}))
    kwargs = {k: v for k, v in raw.items() if k not in {"models", "weights"}}
    return RouterConfig(models=models, weights=weights, **kwargs)
