"""YAML 配置加载器。"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml

from core.paths import CONFIG_DIR


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    return data


@lru_cache(maxsize=None)
def load_app_config() -> Dict[str, Any]:
    return _load_yaml(CONFIG_DIR / "app_config.yaml")


@lru_cache(maxsize=None)
def load_llm_config() -> Dict[str, Any]:
    return _load_yaml(CONFIG_DIR / "llm_config.yaml")


@lru_cache(maxsize=None)
def load_param_ranges() -> Dict[str, Any]:
    return _load_yaml(CONFIG_DIR / "param_ranges.yaml")


@lru_cache(maxsize=None)
def load_material_db() -> Dict[str, Any]:
    return _load_yaml(CONFIG_DIR / "material_db.yaml")
