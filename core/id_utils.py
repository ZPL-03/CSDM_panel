"""统一管理任务实例、候选方案与案例编号。"""

from __future__ import annotations

import hashlib
import json
import re
import uuid

from core.io_utils import read_json
from core.paths import ABAQUS_DIR, ABAQUS_RUNS_DIR, CASES_DIR, IO_DIR, TASKS_DIR


TASK_PATTERN = re.compile(r"^TASK_(\d+)$")
CANDIDATE_PATTERN = re.compile(r"^C(\d+)$")
TEMP_CANDIDATE_PATTERN = re.compile(r"^TMP_(\d+)$")
CASE_PATTERN = re.compile(r"^CASE_(\d+)$")
REQUEST_PATTERN = re.compile(r"^REQ_[A-F0-9]{12}$")
TASK_FINGERPRINT_PATTERN = re.compile(r"^TFP_[A-F0-9]{16}$")


def _parse_index(value: str, pattern: re.Pattern[str]) -> int | None:
    match = pattern.match(value)
    if not match:
        return None
    return int(match.group(1))


def format_task_id(index: int) -> str:
    return f"TASK_{index}"


def format_candidate_id(index: int) -> str:
    return f"C{index}"


def new_request_id() -> str:
    return f"REQ_{uuid.uuid4().hex[:12].upper()}"


def format_task_fingerprint(digest: str) -> str:
    return f"TFP_{digest[:16].upper()}"


def task_index(task_id: str) -> int | None:
    return _parse_index(str(task_id), TASK_PATTERN)


def next_task_index() -> int:
    indices = []
    for path in TASKS_DIR.glob("TASK_*.json"):
        parsed = _parse_index(path.stem, TASK_PATTERN)
        if parsed is not None:
            indices.append(parsed)
    return max(indices, default=0) + 1


def next_task_id() -> str:
    return format_task_id(next_task_index())


def task_file_name(task_id: str) -> str:
    return f"{str(task_id).strip()}.json"


def task_display_label(task_id: str | None) -> str:
    return str(task_id or "").strip() or "-"


def task_file_name_for_payload(payload: dict) -> str:
    return task_file_name(str(payload.get("task_id") or "").strip())


def task_identity_payload(task_id: str) -> dict:
    return {"task_id": str(task_id).strip()}


def task_fingerprint_seed(task: dict) -> str:
    normalized = dict(task)
    normalized.pop("task_id", None)
    return json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def task_fingerprint(task: dict) -> str:
    normalized = task_fingerprint_seed(task)
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()
    return format_task_fingerprint(digest)


def request_identity_payload(request_id: str | None, task_fp: str | None) -> dict:
    payload = {}
    if str(request_id or "").strip():
        payload["request_id"] = str(request_id).strip()
    if str(task_fp or "").strip():
        payload["task_fingerprint"] = str(task_fp).strip()
    return payload


def is_task_id(value: str | None) -> bool:
    return bool(TASK_PATTERN.match(str(value or "")))


def is_request_id(value: str | None) -> bool:
    return bool(REQUEST_PATTERN.match(str(value or "")))


def is_task_fingerprint(value: str | None) -> bool:
    return bool(TASK_FINGERPRINT_PATTERN.match(str(value or "")))


def request_file_stem(request_id: str) -> str:
    return str(request_id).strip()


def request_file_name(request_id: str) -> str:
    return f"{request_file_stem(request_id)}.json"


def request_display_label(request_id: str | None) -> str:
    text = str(request_id or "").strip()
    return text[:12] if text else "-"




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


def next_candidate_index() -> int:
    return _max_candidate_index() + 1


def next_case_id(candidate_id: str | None = None) -> str:
    preferred = case_id_for_candidate(candidate_id) if candidate_id else None
    if preferred is not None:
        return preferred
    return format_case_id(_max_case_index() + 1)
