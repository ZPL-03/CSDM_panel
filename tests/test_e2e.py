from __future__ import annotations

import shutil
from pathlib import Path

from abaqus.job_utils import is_abaqus_available
from agents.orchestrator import OrchestratorAgent
from core.paths import ABAQUS_RUNS_DIR, CASES_DIR, CASE_LIBRARY_DIR, IO_DIR, RESULTS_DIR, TASKS_DIR


def _snapshot_names(folder: Path, pattern: str = "*") -> set[str]:
    return {path.name for path in folder.glob(pattern)}


def _restore_report_files(snapshot: dict[Path, bytes | None]) -> None:
    for path, content in snapshot.items():
        if content is None:
            path.unlink(missing_ok=True)
        else:
            path.write_bytes(content)


def test_orchestrator_end_to_end_with_real_abaqus() -> None:
    assert is_abaqus_available("abaqus"), "当前环境未找到 abaqus 命令，端到端真实有限元测试不能运行"

    existing_cases = _snapshot_names(CASES_DIR, "CASE_*.json")
    existing_library = _snapshot_names(CASE_LIBRARY_DIR, "CASE_*.json")
    existing_io = _snapshot_names(IO_DIR)
    existing_runs = _snapshot_names(ABAQUS_RUNS_DIR, "C*")
    existing_tasks = _snapshot_names(TASKS_DIR, "TASK_*.json")
    report_paths = [RESULTS_DIR / "latest_report.md", RESULTS_DIR / "latest_report.pdf"]
    report_snapshot = {path: path.read_bytes() if path.exists() else None for path in report_paths}

    orchestrator = None
    try:
        orchestrator = OrchestratorAgent()
        result = orchestrator.run(
            "请为机翼下蒙皮壁板生成1个T型筋方案，压缩载荷120kN/m，"
            "长度360mm，宽度300mm，BLF目标0.1，初筛保留1个"
        )

        assert result["task"]["task_id"].startswith("TASK_")
        assert len(result["candidates"]) == 1
        assert all("task_id" not in candidate for candidate in result["candidates"])
        assert len(result["top_candidates"]) == 1
        assert len(result["results"]) == 1
        assert result["results"][0]["status"] == "success"
        assert result["results"][0]["error_type"] is None
        assert result["results"][0]["BLF_global"] is not None
    finally:
        new_case_ids = {
            path.stem
            for folder in [CASES_DIR, CASE_LIBRARY_DIR]
            for path in folder.glob("CASE_*.json")
            if path.name not in (existing_cases if folder == CASES_DIR else existing_library)
        }
        for path in CASES_DIR.glob("CASE_*.json"):
            if path.name not in existing_cases:
                path.unlink(missing_ok=True)
        for path in CASE_LIBRARY_DIR.glob("CASE_*.json"):
            if path.name not in existing_library:
                path.unlink(missing_ok=True)
        for path in IO_DIR.glob("*"):
            if path.name not in existing_io:
                path.unlink(missing_ok=True)
        for path in ABAQUS_RUNS_DIR.glob("C*"):
            if path.name not in existing_runs and path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
        for path in TASKS_DIR.glob("TASK_*.json"):
            if path.name not in existing_tasks:
                path.unlink(missing_ok=True)
        if new_case_ids and orchestrator is not None:
            try:
                orchestrator.knowledge_agent.case_memory.engine.collection.delete(ids=sorted(new_case_ids))
            except Exception:
                pass
        _restore_report_files(report_snapshot)
