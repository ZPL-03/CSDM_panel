"""把历史任务、案例与 JSON 工件迁移到当前结构化契约。"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.io_utils import read_json, write_json
from core.paths import ABAQUS_RUNS_DIR, CASES_DIR, CASE_LIBRARY_DIR, CHROMA_DIR, IO_DIR, RESULTS_DIR
from core.rag_engine import RAGEngine
from core.schema_validator import validate_or_raise
from core.surrogate_model import SurrogateModelManager
from core.task_contract import (
    build_task_request_record,
    describe_boundary_conditions,
    describe_load_conditions,
    normalize_boundary_conditions,
    normalize_load_conditions,
    normalize_task_payload,
)


def normalize_result_document(result: Dict, design: Dict | None = None) -> Dict:
    normalized = dict(result)
    load_conditions = {}
    boundary_conditions = {}
    if isinstance(design, dict):
        load_conditions = normalize_load_conditions(design.get("load_conditions", {}))
        boundary_conditions = normalize_boundary_conditions(design.get("boundary_conditions", {}))
    normalized["load_summary"] = normalized.get("load_summary") or describe_load_conditions(load_conditions)
    normalized["boundary_summary"] = normalized.get("boundary_summary") or describe_boundary_conditions(boundary_conditions)
    if not normalized.get("diagnosis_summary"):
        if normalized.get("status") == "success":
            verdict = normalized.get("verdict", "未判定")
            normalized["diagnosis_summary"] = f"历史结果已迁移，当前结论为“{verdict}”。"
        else:
            normalized["diagnosis_summary"] = "历史结果已迁移，请结合错误日志继续复核。"
    return normalized


def _task_payload(task: Dict | None) -> Dict:
    return normalize_task_payload(dict(task or {}))


def _normalize_candidate_source(source: object) -> str:
    raw = str(source or "").strip().upper()
    if raw in {"LLM", "DOE", "CASE_TRANSFER"}:
        return raw
    return "DOE"


def normalize_candidate_document(candidate: Dict, task: Dict | None = None) -> Dict:
    normalized = dict(candidate)
    task_payload = _task_payload(task)
    normalized["source"] = _normalize_candidate_source(normalized.get("source"))
    normalized.pop("task_id", None)
    normalized.pop("request_id", None)
    normalized.pop("task_fingerprint", None)
    normalized["load_conditions"] = normalize_load_conditions(
        normalized.get("load_conditions", task_payload.get("load_conditions", {}))
    )
    normalized["boundary_conditions"] = normalize_boundary_conditions(
        normalized.get("boundary_conditions", task_payload.get("boundary_conditions", {}))
    )
    normalized["design_targets"] = dict(normalized.get("design_targets") or task_payload.get("design_targets", {}))
    normalized["material_system"] = dict(normalized.get("material_system") or task_payload.get("material_system", {}))
    normalized.setdefault("origin_summary", "")
    normalized.setdefault("screening_summary", None)
    normalized.setdefault("selection_reason", None)
    normalized.setdefault("display_name", normalized.get("candidate_id"))
    normalized.setdefault("persistent_candidate_id", None)
    return normalized


def normalize_case_record(record: Dict) -> Dict:
    normalized = dict(record)
    task_payload = _task_payload(normalized.get("task", {}))
    design = normalize_candidate_document(
        dict(normalized.get("design", {})),
        task_payload,
    )
    abaqus_results = normalize_result_document(dict(normalized.get("abaqus_results", {})), design)

    normalized.pop("task_id", None)
    normalized.pop("request_id", None)
    normalized.pop("task_fingerprint", None)
    normalized["candidate_id"] = str(normalized.get("candidate_id") or design.get("candidate_id") or "") or None
    normalized["task"] = task_payload
    normalized["design"] = design
    normalized["abaqus_results"] = abaqus_results
    normalized["verdict"] = normalized.get("verdict") or abaqus_results.get("verdict") or "未知"
    normalized["fem_agent_retry_count"] = int(normalized.get("fem_agent_retry_count", abaqus_results.get("retry_count", 0)) or 0)
    return normalized


def _migrate_json_files(paths: Iterable[Path], normalizer) -> Dict[str, int]:
    summary = {"updated": 0, "skipped": 0, "failed": 0}
    for path in paths:
        try:
            payload = read_json(path)
            normalized = normalizer(payload)
            write_json(path, normalized)
            summary["updated"] += 1
        except Exception:
            summary["failed"] += 1
    return summary


def migrate_cases() -> Dict[str, int]:
    def _normalize(payload: Dict) -> Dict:
        normalized = normalize_case_record(payload)
        validate_or_raise("task.schema.json", normalized["task"])
        validate_or_raise("candidate.schema.json", normalized["design"])
        validate_or_raise("abaqus_result.schema.json", normalized["abaqus_results"])
        validate_or_raise("case_record.schema.json", normalized)
        return normalized

    return _migrate_json_files(sorted(CASES_DIR.glob("CASE_*.json")), _normalize)


def migrate_case_library() -> Dict[str, int]:
    def _normalize(payload: Dict) -> Dict:
        normalized = normalize_case_record(payload)
        validate_or_raise("task.schema.json", normalized["task"])
        validate_or_raise("candidate.schema.json", normalized["design"])
        validate_or_raise("abaqus_result.schema.json", normalized["abaqus_results"])
        validate_or_raise("case_record.schema.json", normalized)
        return normalized

    return _migrate_json_files(sorted(CASE_LIBRARY_DIR.glob("CASE_*.json")), _normalize)


def migrate_io_payloads() -> Dict[str, int]:
    summary = {"updated": 0, "skipped": 0, "failed": 0}
    for path in sorted(IO_DIR.glob("*.json")):
        try:
            payload = read_json(path)
            if path.name.startswith("input_"):
                normalized = normalize_candidate_document(payload)
                validate_or_raise("candidate.schema.json", normalized)
            elif path.name.startswith("result_"):
                normalized = normalize_result_document(payload)
                validate_or_raise("abaqus_result.schema.json", normalized)
            else:
                summary["skipped"] += 1
                continue
            write_json(path, normalized)
            summary["updated"] += 1
        except Exception:
            summary["failed"] += 1
    return summary


def migrate_run_inputs() -> Dict[str, int]:
    summary = {"updated": 0, "skipped": 0, "failed": 0}
    for path in sorted(ABAQUS_RUNS_DIR.glob("C*/candidate_input.json")):
        try:
            normalized = normalize_candidate_document(read_json(path))
            validate_or_raise("candidate.schema.json", normalized)
            write_json(path, normalized)
            summary["updated"] += 1
        except Exception:
            summary["failed"] += 1
    return summary


def rebuild_rag_index() -> Dict[str, int]:
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR, ignore_errors=True)
    engine = RAGEngine()
    records: List[Dict] = []
    for path in sorted(CASE_LIBRARY_DIR.glob("CASE_*.json")):
        payload = normalize_case_record(read_json(path))
        if payload.get("abaqus_results", {}).get("status") == "success" and payload.get("verdict") == "通过":
            records.append(payload)
    engine.upsert_records(records, id_key="case_id")
    return {"indexed_records": len(records)}


def maybe_retrain_surrogate() -> Dict | None:
    manager = SurrogateModelManager()
    records = manager.load_training_records()
    if len(records) < 10:
        return {"training_size": len(records), "skipped": True}
    return manager.train_from_records(records)


def run_migration(rebuild_rag: bool, retrain_surrogate: bool) -> Dict:
    summary = {
        "cases": migrate_cases(),
        "case_library": migrate_case_library(),
        "io": migrate_io_payloads(),
        "abaqus_runs": migrate_run_inputs(),
    }
    if rebuild_rag:
        summary["rag"] = rebuild_rag_index()
    if retrain_surrogate:
        summary["surrogate"] = maybe_retrain_surrogate()
    write_json(RESULTS_DIR / "contract_migration_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-rag", action="store_true")
    parser.add_argument("--retrain-surrogate", action="store_true")
    args = parser.parse_args()

    summary = run_migration(
        rebuild_rag=not args.skip_rag,
        retrain_surrogate=args.retrain_surrogate,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
