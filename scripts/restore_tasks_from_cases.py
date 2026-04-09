"""从 data/cases 中恢复 data/tasks 任务记录。"""

from __future__ import annotations

from collections import OrderedDict

from core.io_utils import read_json, write_json
from core.paths import CASES_DIR, TASKS_DIR
from core.schema_validator import validate_or_raise
from core.task_contract import normalize_task_payload


def _normalize_task(task: dict) -> dict:
    return normalize_task_payload(dict(task))


def restore_tasks_from_cases() -> int:
    task_map: OrderedDict[str, dict] = OrderedDict()

    for case_path in sorted(CASES_DIR.glob("CASE_*.json")):
        payload = read_json(case_path)
        task = payload.get("task") or {}
        task_id = str(task.get("task_id", "")).strip()
        if not task_id or task_id in task_map:
            continue
        task = _normalize_task(task)
        validate_or_raise("task.schema.json", task)
        task_map[task_id] = task

    TASKS_DIR.mkdir(parents=True, exist_ok=True)
    for stale_path in TASKS_DIR.glob("TASK_*.json"):
        stale_path.unlink(missing_ok=True)

    for task_id, task in task_map.items():
        write_json(TASKS_DIR / f"{task_id}.json", task)

    return len(task_map)


if __name__ == "__main__":
    restored = restore_tasks_from_cases()
    print(f"restored_tasks={restored}")
