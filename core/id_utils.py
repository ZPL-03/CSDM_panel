"""统一管理任务、候选方案与案例编号。"""

from __future__ import annotations

import re
from core.io_utils import read_json
from core.paths import ABAQUS_DIR, ABAQUS_RUNS_DIR, CASES_DIR, IO_DIR, TASKS_DIR


TASK_PATTERN = re.compile(r"^TASK_(\d+)$")
CANDIDATE_PATTERN = re.compile(r"^C(\d+)$")
TEMP_CANDIDATE_PATTERN = re.compile(r"^TMP_(\d+)$")
CASE_PATTERN = re.compile(r"^CASE_(\d+)$")


def _parse_index(value: str, pattern: re.Pattern[str]) -> int | None:
    match = pattern.match(value)
    if not match:
        return None
    return int(match.group(1))


def format_task_id(index: int) -> str:
    return f"TASK_{index}"


def format_candidate_id(index: int) -> str:
    return f"C{index}"


def format_temp_candidate_id(index: int) -> str:
    return f"TMP_{index}"


def format_case_id(index: int) -> str:
    return f"CASE_{index}"


def candidate_index(candidate_id: str) -> int | None:
    return _parse_index(str(candidate_id), CANDIDATE_PATTERN)


def temp_candidate_index(candidate_id: str) -> int | None:
    return _parse_index(str(candidate_id), TEMP_CANDIDATE_PATTERN)


def case_id_for_candidate(candidate_id: str) -> str | None:
    index = candidate_index(candidate_id)
    if index is None:
        return None
    return format_case_id(index)


def _max_task_index() -> int:
    indices = []
    for path in TASKS_DIR.glob("TASK_*.json"):
        task_id = path.stem
        parsed = _parse_index(task_id, TASK_PATTERN)
        if parsed is not None:
            indices.append(parsed)
    return max(indices, default=0)


def _max_candidate_index() -> int:
    indices = []
    for path in IO_DIR.glob("input_*.json"):
        parsed = _parse_index(path.stem.replace("input_", ""), CANDIDATE_PATTERN)
        if parsed is not None:
            indices.append(parsed)
    for path in IO_DIR.glob("result_*.json"):
        parsed = _parse_index(path.stem.replace("result_", ""), CANDIDATE_PATTERN)
        if parsed is not None:
            indices.append(parsed)
    for folder in (ABAQUS_DIR,):
        for path in folder.glob("C*.odb"):
            parsed = _parse_index(path.stem, CANDIDATE_PATTERN)
            if parsed is not None:
                indices.append(parsed)
        for path in folder.glob("C*.inp"):
            parsed = _parse_index(path.stem, CANDIDATE_PATTERN)
            if parsed is not None:
                indices.append(parsed)
    for path in ABAQUS_RUNS_DIR.glob("C*"):
        if not path.is_dir():
            continue
        parsed = _parse_index(path.name, CANDIDATE_PATTERN)
        if parsed is not None:
            indices.append(parsed)
    for path in CASES_DIR.glob("CASE_*.json"):
        try:
            payload = read_json(path)
        except Exception:
            continue
        for value in [
            payload.get("design", {}).get("candidate_id"),
            payload.get("abaqus_results", {}).get("candidate_id"),
        ]:
            parsed = _parse_index(str(value), CANDIDATE_PATTERN)
            if parsed is not None:
                indices.append(parsed)
    return max(indices, default=0)


def _max_case_index() -> int:
    indices = []
    for path in CASES_DIR.glob("CASE_*.json"):
        parsed = _parse_index(path.stem, CASE_PATTERN)
        if parsed is not None:
            indices.append(parsed)
            continue
        try:
            case_id = str(read_json(path).get("case_id", ""))
        except Exception:
            continue
        parsed = _parse_index(case_id, CASE_PATTERN)
        if parsed is not None:
            indices.append(parsed)
    return max(indices, default=0)


def next_task_id() -> str:
    return format_task_id(_max_task_index() + 1)


def next_candidate_index() -> int:
    return _max_candidate_index() + 1


def next_case_id(candidate_id: str | None = None) -> str:
    preferred = case_id_for_candidate(candidate_id) if candidate_id else None
    if preferred is not None:
        return preferred
    return format_case_id(_max_case_index() + 1)
