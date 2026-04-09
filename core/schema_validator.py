"""JSON Schema 校验工具。"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict

import json
from jsonschema import Draft202012Validator

from core.paths import SCHEMA_DIR


class SchemaValidationError(ValueError):
    """Schema 校验失败。"""


@lru_cache(maxsize=None)
def load_schema(schema_name: str) -> Dict[str, Any]:
    schema_path = SCHEMA_DIR / schema_name
    with schema_path.open("r", encoding="utf-8") as file:
        return json.load(file)


@lru_cache(maxsize=None)
def get_validator(schema_name: str) -> Draft202012Validator:
    return Draft202012Validator(load_schema(schema_name))


def validate_or_raise(schema_name: str, payload: Dict[str, Any]) -> None:
    validator = get_validator(schema_name)
    errors = sorted(validator.iter_errors(payload), key=lambda err: list(err.path))
    if errors:
        first = errors[0]
        path = ".".join(str(part) for part in first.path) or "<root>"
        raise SchemaValidationError(f"{schema_name} 校验失败: {path} -> {first.message}")
