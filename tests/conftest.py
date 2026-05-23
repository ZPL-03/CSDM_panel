"""测试环境公共配置。"""

from __future__ import annotations

import os

import pytest

from core.paths import TASKS_DIR


os.environ.setdefault("CSDM_panel_DISABLE_LLM_AUTO", "1")


@pytest.fixture(autouse=True)
def clean_task_registry_after_test():
    existing_tasks = {path.name for path in TASKS_DIR.glob("TASK_*.json")}
    yield
    for path in TASKS_DIR.glob("TASK_*.json"):
        if path.name not in existing_tasks:
            path.unlink(missing_ok=True)
