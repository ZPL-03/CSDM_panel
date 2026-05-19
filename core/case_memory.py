"""历史案例记忆索引。

案例 JSON 仍是事实源；向量库只保存可检索摘要和元数据，用于辅助召回与排序。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

from core.config_loader import load_app_config
from core.rag_engine import RAGEngine
from core.stiffener_profile import describe_geometry_text
from core.task_contract import (
    describe_boundary_conditions,
    describe_load_conditions,
    normalize_boundary_conditions,
    normalize_load_conditions,
    task_payload_from_request,
)


DEFAULT_CASE_MEMORY_COLLECTION = "csdm_case_memory"


@dataclass
class CaseMemoryHit:
    case_id: str
    distance: float | None
    metadata: Dict[str, Any]
    document: str


def is_pass_verdict(value: object) -> bool:
    """仅将明确的“通过”结论视为通过，避免把“不通过”误判为通过。"""
    verdict = str(value or "").strip()
    if not verdict or verdict in {"None", "失败"}:
        return False
    return verdict == "通过"


def _safe_number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _material_name(payload: Dict[str, Any]) -> str:
    material = payload.get("material_system", {})
    if not isinstance(material, dict):
        return ""
    return str(material.get("name") or material.get("display_name") or "").strip()


def is_transferable_case(record: Dict[str, Any]) -> bool:
    results = record.get("abaqus_results", {})
    design = record.get("design", {})
    if not isinstance(results, dict) or not isinstance(design, dict) or not design:
        return False
    if results.get("status") != "success":
        return False
    return is_pass_verdict(results.get("verdict") or record.get("verdict"))


def case_record_to_search_text(record: Dict[str, Any]) -> str:
    """生成紧凑、稳定、适合向量检索的案例摘要文本。"""
    task = task_payload_from_request(record.get("task", {}))
    design = record.get("design", {}) if isinstance(record.get("design"), dict) else {}
    results = record.get("abaqus_results", {}) if isinstance(record.get("abaqus_results"), dict) else {}
    geometry = design.get("geometry", {}) if isinstance(design.get("geometry"), dict) else {}
    layup = design.get("layup", {}) if isinstance(design.get("layup"), dict) else {}
    material = _material_name(design) or _material_name(task)

    stype = design.get("stiffener_type") or task.get("stiffener_type", "T")
    geometry_text = describe_geometry_text(stype, geometry, layup)
    layup_text = (
        f"{layup.get('skin_layup', '')}; "
        f"f0={layup.get('skin_f0', '')}, "
        f"f45={layup.get('skin_f45', '')}, "
        f"f90={layup.get('skin_f90', '')}"
    )
    result_text = (
        f"status={results.get('status', '')}; "
        f"verdict={results.get('verdict') or record.get('verdict', '')}; "
        f"BLF_global={results.get('BLF_global', '')}; "
        f"BLF_local={results.get('BLF_local', '')}; "
        f"weight={results.get('weight_kg_per_m2', '')}; "
        f"failure_mode={results.get('failure_mode', '')}"
    )

    return "\n".join(
        [
            f"case_id: {record.get('case_id', '')}",
            f"应用: {task.get('application', '复合材料加筋壁板')}",
            f"工况: {describe_load_conditions(task.get('load_conditions', {}))}",
            f"边界: {describe_boundary_conditions(task.get('boundary_conditions', {}))}",
            f"筋型: {design.get('stiffener_type') or task.get('stiffener_type', 'T')}",
            f"材料: {material}",
            f"目标: BLF >= {task.get('design_targets', {}).get('BLF_min', 1.2)}, "
            f"{task.get('design_targets', {}).get('primary_objective', '最小重量')}",
            f"几何: {geometry_text}",
            f"铺层: {layup_text}",
            f"结果: {result_text}",
            f"设计理由: {design.get('rationale', '')}",
        ]
    )


def case_record_metadata(record: Dict[str, Any], scope: str = "archive") -> Dict[str, Any]:
    task = task_payload_from_request(record.get("task", {}))
    design = record.get("design", {}) if isinstance(record.get("design"), dict) else {}
    load_conditions = normalize_load_conditions(design.get("load_conditions") or task.get("load_conditions", {}))
    boundary_conditions = normalize_boundary_conditions(
        design.get("boundary_conditions") or task.get("boundary_conditions", {})
    )
    results = record.get("abaqus_results", {}) if isinstance(record.get("abaqus_results"), dict) else {}
    return {
        "kind": "case",
        "case_id": str(record.get("case_id") or ""),
        "case_scope": str(scope),
        "transferable": "true" if is_transferable_case(record) else "false",
        "status": str(results.get("status") or ""),
        "verdict": str(results.get("verdict") or record.get("verdict") or ""),
        "load_type": str(load_conditions.get("type") or ""),
        "boundary_type": str(boundary_conditions.get("type") or ""),
        "stiffener_type": str(design.get("stiffener_type") or task.get("stiffener_type") or "T"),
        "material_name": (_material_name(design) or _material_name(task)).lower(),
    }


def task_to_case_query_text(task: Dict[str, Any]) -> str:
    task_payload = task_payload_from_request(task)
    envelope = task_payload.get("geometry_envelope", {})
    material = _material_name(task_payload)
    target = task_payload.get("design_targets", {})
    return "\n".join(
        [
            f"应用: {task_payload.get('application', '复合材料加筋壁板')}",
            f"工况: {describe_load_conditions(task_payload.get('load_conditions', {}))}",
            f"边界: {describe_boundary_conditions(task_payload.get('boundary_conditions', {}))}",
            f"筋型: {task_payload.get('stiffener_type', 'T')}",
            f"材料: {material}",
            f"几何包络: L={envelope.get('panel_length_mm')}, "
            f"W={envelope.get('panel_width_mm')}, h_max={envelope.get('max_stiffener_height_mm')}",
            f"目标: BLF >= {target.get('BLF_min', 1.2)}, {target.get('primary_objective', '最小重量')}",
        ]
    )


class CaseMemoryIndex:
    """案例记忆向量索引，独立于外部知识库/知识图谱。"""

    def __init__(self, collection_name: str | None = None) -> None:
        config = load_app_config()
        case_memory_config = dict(config.get("case_memory", {}))
        self.collection_name = collection_name or str(
            case_memory_config.get("collection_name", DEFAULT_CASE_MEMORY_COLLECTION)
        )
        self.engine = RAGEngine(collection_name=self.collection_name)

    def upsert_cases(self, records: Iterable[Dict[str, Any]], scope: str = "archive") -> None:
        records = [record for record in records if isinstance(record, dict) and record.get("case_id")]
        if not records:
            return
        self.engine.upsert_documents(
            ids=[str(record["case_id"]) for record in records],
            documents=[case_record_to_search_text(record) for record in records],
            metadatas=[case_record_metadata(record, scope=scope) for record in records],
        )

    def query(self, task: Dict[str, Any], top_k: int = 10, where: Dict[str, Any] | None = None) -> List[CaseMemoryHit]:
        query_text = task_to_case_query_text(task)
        query_where = {"kind": "case"}
        if where:
            query_where.update(where)
        matches = self.engine.query_text(query_text, top_k=top_k, where=query_where)
        hits: List[CaseMemoryHit] = []
        for item in matches:
            metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
            case_id = str(metadata.get("case_id") or item.get("id") or "").strip()
            if not case_id:
                continue
            distance = item.get("distance")
            hits.append(
                CaseMemoryHit(
                    case_id=case_id,
                    distance=float(distance) if isinstance(distance, (int, float)) else None,
                    metadata=dict(metadata),
                    document=str(item.get("document") or ""),
                )
            )
        return hits
