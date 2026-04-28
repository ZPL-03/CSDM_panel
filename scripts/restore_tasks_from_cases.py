"""根据历史案例导出任务快照的兼容脚本。"""

from __future__ import annotations

import sys
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.id_utils import format_task_id, next_task_index, task_file_name
from core.io_utils import read_json, write_json
from core.paths import CASES_DIR, TASKS_DIR
from core.schema_validator import validate_or_raise
from core.task_contract import build_task_request_record, normalize_task_payload


def _restore_task_record(payload: dict, task_id: str) -> dict | None:
    task = normalize_task_payload(dict(payload.get("task") or {}))
    if not task:
        return None
    record = build_task_request_record(
        task,
        task_id=task_id,
        source=str(payload.get("source") or "restored_from_case"),
        created_at=str(payload.get("created_at") or ""),
    )
    validate_or_raise("task.schema.json", record["task"])
    return record


def _allocate_task_ids(case_count: int) -> list[str]:
    start_index = next_task_index()
    return [format_task_id(start_index + offset) for offset in range(case_count)]


def _load_case_payloads() -> list[dict]:
    return [read_json(case_path) for case_path in sorted(CASES_DIR.glob("CASE_*.json"))]


def _validated_task_records(case_payloads: list[dict]) -> list[dict]:
    task_ids = _allocate_task_ids(len(case_payloads))
    records: list[dict] = []
    for payload, task_id in zip(case_payloads, task_ids):
        record = _restore_task_record(payload, task_id)
        if record is not None:
            records.append(record)
    return records


def _rewrite_task_directory(task_records: list[dict]) -> None:
    TASKS_DIR.mkdir(parents=True, exist_ok=True)
    for stale_path in TASKS_DIR.glob("*.json"):
        stale_path.unlink(missing_ok=True)
    for task_record in task_records:
        task_id = task_record["task_id"]
        write_json(TASKS_DIR / task_file_name(task_id), task_record)




def restore_tasks_from_cases() -> int:
    case_payloads = _load_case_payloads()
    task_records = _validated_task_records(case_payloads)
    ordered_task_records = OrderedDict((record["task_id"], record) for record in task_records)
    _rewrite_task_directory(list(ordered_task_records.values()))
    return len(ordered_task_records)


if __name__ == "__main__":
    restored = restore_tasks_from_cases()
    print(f"restored_tasks={restored}")
