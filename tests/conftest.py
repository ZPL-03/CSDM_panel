"""测试环境公共配置。"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pytest

from core import paths


os.environ.setdefault("CSDM_panel_DISABLE_LLM_AUTO", "1")


@pytest.fixture(scope="session")
def test_models(tmp_path_factory):
    target = tmp_path_factory.mktemp("panel-models") / "models"
    shutil.copytree(paths.MODELS_DIR, target)
    return target


@pytest.fixture(autouse=True)
def isolated_runtime(monkeypatch, tmp_path, test_models):
    """隔离运行数据及已导入的路径别名，正式知识包保持只读。"""
    targets = {
        "DATA_DIR": tmp_path / "data",
        "IO_DIR": tmp_path / "data/io",
        "TASKS_DIR": tmp_path / "data/tasks",
        "RESULTS_DIR": tmp_path / "data/results",
        "CASES_DIR": tmp_path / "data/cases",
        "ABAQUS_RUNS_DIR": tmp_path / "data/abaqus_runs",
        "CHROMA_DIR": tmp_path / "knowledge/chroma_db",
        "CASE_LIBRARY_DIR": tmp_path / "knowledge/case_library",
        "MODELS_DIR": test_models,
    }
    original = {name: getattr(paths, name) for name in targets}
    for target in targets.values():
        target.mkdir(parents=True, exist_ok=True)
    for module in tuple(sys.modules.values()):
        file = getattr(module, "__file__", None)
        if not file or not Path(file).resolve().is_relative_to(paths.ROOT_DIR):
            continue
        for name, target in targets.items():
            if getattr(module, name, None) == original[name]:
                monkeypatch.setattr(module, name, target)
