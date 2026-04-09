import os
import shutil

from agents.orchestrator import OrchestratorAgent
from core.paths import ABAQUS_RUNS_DIR, CASES_DIR, CASE_LIBRARY_DIR, IO_DIR


def test_orchestrator_end_to_end() -> None:
    os.environ["CSDM_USE_MOCK_ABAQUS"] = "1"
    existing_cases = {path.name for path in CASES_DIR.glob("CASE_*.json")}
    existing_library = {path.name for path in CASE_LIBRARY_DIR.glob("CASE_*.json")}
    existing_io = {path.name for path in IO_DIR.glob("*")}
    existing_runs = {path.name for path in ABAQUS_RUNS_DIR.glob("C*")}

    try:
        orchestrator = OrchestratorAgent()
        result = orchestrator.run("请为机翼下翼面壁板设计一个 T 形筋方案，压缩载荷 850 kN/m")
        assert result["task"]["task_id"].startswith("TASK_")
        assert len(result["candidates"]) > 0
        assert len(result["top_candidates"]) > 0
        assert len(result["results"]) > 0
    finally:
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
