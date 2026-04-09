from pathlib import Path

from agents.orchestrator import OrchestratorAgent
from core.paths import TASKS_DIR


def test_parse_instruction_persists_task_record() -> None:
    TASKS_DIR.mkdir(parents=True, exist_ok=True)
    existing = {path.name for path in TASKS_DIR.glob("TASK_*.json")}
    agent = OrchestratorAgent()

    try:
        task = agent.parse_instruction("请设计一个T形加筋方案，压缩荷载为1200kN/m")
        task_path = TASKS_DIR / f"{task['task_id']}.json"
        assert task_path.exists()
    finally:
        for path in TASKS_DIR.glob("TASK_*.json"):
            if path.name not in existing:
                path.unlink(missing_ok=True)
