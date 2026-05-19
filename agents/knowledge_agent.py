"""知识回流智能体。"""

from __future__ import annotations

from datetime import datetime
from typing import Dict

from agents.base import BaseAgent
from core.case_memory import CaseMemoryIndex
from core.config_loader import load_app_config
from core.id_utils import next_case_id
from core.io_utils import write_json
from core.paths import CASE_LIBRARY_DIR, CASES_DIR
from core.schema_validator import validate_or_raise
from core.surrogate_model import SurrogateModelManager
from core.task_contract import (
    normalize_boundary_conditions,
    normalize_load_conditions,
    normalize_task_payload,
    task_payload_from_request,
)


class KnowledgeAgent(BaseAgent):
    agent_name = "KNOWLEDGE_AGENT"

    def __init__(self, progress_callback=None) -> None:
        super().__init__(progress_callback=progress_callback)
        self.case_memory = CaseMemoryIndex()
        self.model_manager = SurrogateModelManager()
        self.config = load_app_config()
        self.min_case_records_for_retrain = int(self.config["pipeline"]["min_case_records_for_retrain"])

    def _sanitize_task(self, task: Dict) -> Dict:
        normalized = task_payload_from_request(task)
        return {
            "application": normalized.get("application"),
            "load_conditions": dict(normalized.get("load_conditions", {})),
            "boundary_conditions": dict(normalized.get("boundary_conditions", {})),
            "geometry_envelope": dict(normalized.get("geometry_envelope", {})),
            "material_system": dict(normalized.get("material_system", {})),
            "layup_constraints": dict(normalized.get("layup_constraints", {})),
            "candidate_generation_preferences": dict(normalized.get("candidate_generation_preferences", {})),
            "screening_preferences": dict(normalized.get("screening_preferences", {})),
            "stiffener_type": normalized.get("stiffener_type", "T"),
            "design_targets": dict(normalized.get("design_targets", {})),
        }

    def _sanitize_design(self, design: Dict) -> Dict:
        return {
            "candidate_id": design.get("candidate_id"),
            "source": design.get("source"),
            "stiffener_type": design.get("stiffener_type", "T"),
            "geometry": dict(design.get("geometry", {})),
            "layup": dict(design.get("layup", {})),
            "material_system": dict(design.get("material_system", {})),
            "load_conditions": dict(normalize_load_conditions(design.get("load_conditions", {}))),
            "boundary_conditions": dict(normalize_boundary_conditions(design.get("boundary_conditions", {}))),
            "design_targets": dict(design.get("design_targets", {})),
            "rule_check": dict(design.get("rule_check", {})),
            "surrogate_BLF": design.get("surrogate_BLF"),
            "rationale": design.get("rationale", ""),
        }

    def _sanitize_abaqus_results(self, abaqus_results: Dict) -> Dict:
        keys = [
            "candidate_id",
            "status",
            "retry_count",
            "BLF_global",
            "BLF_local",
            "failure_mode",
            "max_displacement_mm",
            "weight_kg_per_m2",
            "verdict",
            "abaqus_odb",
            "abaqus_inp",
            "visualization_json",
            "artifact_dir",
            "error_type",
            "error_log",
            "mode_eigenvalues",
            "load_summary",
            "boundary_summary",
            "diagnosis_summary",
        ]
        return {key: abaqus_results.get(key) for key in keys}

    def _build_record(self, task: Dict, design: Dict, abaqus_results: Dict) -> Dict:
        clean_task = self._sanitize_task(task)
        clean_design = self._sanitize_design(design)
        clean_results = self._sanitize_abaqus_results(abaqus_results)
        case_id = next_case_id(clean_design.get("candidate_id"))
        verdict = clean_results.get("verdict") or ("失败" if clean_results.get("status") != "success" else "未知")
        record = {
            "case_id": case_id,
            "task_id": task.get("task_id"),
            "created_at": datetime.utcnow().isoformat(),
            "source": "abaqus_auto",
            "task": clean_task,
            "design": clean_design,
            "abaqus_results": clean_results,
            "verdict": verdict,
            "surrogate_BLF_error_pct": None
            if clean_design.get("surrogate_BLF") is None or clean_results.get("BLF_global") is None
            else round(
                abs(clean_design["surrogate_BLF"] - clean_results["BLF_global"])
                / max(clean_results["BLF_global"], 1e-6)
                * 100.0,
                3,
            ),
            "fem_agent_retry_count": int(clean_results.get("retry_count", 0) or 0),
        }
        validate_or_raise("case_record.schema.json", record)
        return record

    def _should_store_record(self, abaqus_results: Dict) -> bool:
        return abaqus_results.get("status") == "success" and abaqus_results.get("verdict") == "通过"

    def _store_record(self, record: Dict) -> None:
        write_json(CASES_DIR / f"{record['case_id']}.json", record)
        if self._should_store_record(record.get("abaqus_results", {})):
            write_json(CASE_LIBRARY_DIR / f"{record['case_id']}.json", record)
            self.case_memory.upsert_cases([record], scope="formal")
        else:
            self.case_memory.upsert_cases([record], scope="archive")

    def _maybe_retrain_surrogate(self) -> Dict | None:
        records = self.model_manager.load_training_records()
        record_count = len(records)
        if record_count < self.min_case_records_for_retrain:
            return None
        if record_count % self.min_case_records_for_retrain != 0:
            return None

        summary = self.model_manager.train_from_records(records)
        self.emit(
            "代理模型已重训："
            f"{summary['selected_model']} | "
            f"RF MAPE={summary['rf']['mape']:.4f} | "
            f"MLP MAPE={summary['mlp']['mape']:.4f}"
        )
        return summary

    def run(self, input_data: Dict) -> Dict:
        task = input_data["task"]
        design = input_data["design"]
        abaqus_results = input_data["abaqus_results"]

        record = self._build_record(task, design, abaqus_results)
        self._store_record(record)
        if self._should_store_record(record["abaqus_results"]):
            self.emit(f"案例 {record['case_id']} 已进入正式案例库")
        else:
            self.emit(f"案例 {record['case_id']} 已归档到评估档案，未进入正式案例库")

        retrain_summary = self._maybe_retrain_surrogate()
        return {
            "status": "stored" if self._should_store_record(record["abaqus_results"]) else "archived_only",
            "case_id": record["case_id"],
            "retrained": retrain_summary is not None,
            "surrogate_summary": retrain_summary,
        }
