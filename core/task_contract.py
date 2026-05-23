"""任务契约与工况/边界条件规范化工具。"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, Iterable


DEFAULT_ALLOWED_ANGLES = [0, 45, -45, 90]
DEFAULT_GEOMETRY_ENVELOPE = {
    "panel_length_mm": [600.0, 800.0],
    "panel_width_mm": [500.0, 700.0],
    "max_stiffener_height_mm": 50.0,
}
DEFAULT_DESIGN_TARGETS = {
    "BLF_min": 1.2,
    "primary_objective": "最小重量",
}
DEFAULT_SCREENING_PREFERENCES = {
    "top_k_candidates": 5,
}
DEFAULT_CANDIDATE_GENERATION_PREFERENCES = {
    "total_candidates": 10,
    "source_allocation_mode": "ratio",
    "source_ratio": {
        "llm": 2.0,
        "case_transfer": 1.0,
        "doe": 1.0,
    },
}
DEFAULT_LAYUP_CONSTRAINTS = {
    "allowed_angles": DEFAULT_ALLOWED_ANGLES,
    "symmetric": True,
    "balanced": True,
    "min_ratio_per_angle": 0.1,
}
DEFAULT_BOUNDARY_TYPE = "SSSS"
DEFAULT_LOAD_TYPE = "axial_compression"

LOAD_CASE_LABELS = {
    "axial_compression": "单轴压缩",
    "in_plane_shear": "面内剪切",
    "compression_shear": "压剪组合",
}

BOUNDARY_CONDITION_LIBRARY = {
    "SSSS": {
        "type": "SSSS",
        "label": "四边简支（SSSS）",
        "description": "四条边均按简支处理，适合基准线性屈曲评估。",
        "simply_supported_edges": ["X0", "X1", "Y0", "Y1"],
        "clamped_edges": [],
    },
    "CCCC": {
        "type": "CCCC",
        "label": "四边固支（CCCC）",
        "description": "四条边均按固支处理，边界刚度最高。",
        "simply_supported_edges": [],
        "clamped_edges": ["X0", "X1", "Y0", "Y1"],
    },
    "SSCC": {
        "type": "SSCC",
        "label": "X 向简支 + Y 向固支（SSCC）",
        "description": "X0/X1 两边简支，Y0/Y1 两边固支。",
        "simply_supported_edges": ["X0", "X1"],
        "clamped_edges": ["Y0", "Y1"],
    },
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def boundary_condition_payload(boundary_type: str) -> Dict[str, Any]:
    boundary_key = str(boundary_type or DEFAULT_BOUNDARY_TYPE).upper()
    if boundary_key not in BOUNDARY_CONDITION_LIBRARY:
        boundary_key = DEFAULT_BOUNDARY_TYPE
    return deepcopy(BOUNDARY_CONDITION_LIBRARY[boundary_key])


def normalize_boundary_conditions(boundary_conditions: Any) -> Dict[str, Any]:
    if isinstance(boundary_conditions, dict):
        boundary_type = str(boundary_conditions.get("type", DEFAULT_BOUNDARY_TYPE)).upper()
        normalized = boundary_condition_payload(boundary_type)
        normalized["label"] = str(boundary_conditions.get("label") or normalized["label"])
        normalized["description"] = str(boundary_conditions.get("description") or normalized["description"])
        return normalized

    text = str(boundary_conditions or "").strip().upper()
    if "SSCC" in text or ("简支" in text and "固支" in text):
        return boundary_condition_payload("SSCC")
    if "CCCC" in text or "固支" in text:
        return boundary_condition_payload("CCCC")
    return boundary_condition_payload("SSSS")


def describe_boundary_conditions(boundary_conditions: Any) -> str:
    normalized = normalize_boundary_conditions(boundary_conditions)
    return str(normalized.get("label", BOUNDARY_CONDITION_LIBRARY[DEFAULT_BOUNDARY_TYPE]["label"]))


def load_condition_payload(load_type: str, nx_kN_per_m: float = 0.0, nxy_kN_per_m: float = 0.0) -> Dict[str, Any]:
    normalized_type = str(load_type or DEFAULT_LOAD_TYPE)
    if normalized_type not in LOAD_CASE_LABELS:
        normalized_type = DEFAULT_LOAD_TYPE
    return {
        "type": normalized_type,
        "label": LOAD_CASE_LABELS[normalized_type],
        "Nx_kN_per_m": round(max(_safe_float(nx_kN_per_m, 0.0), 0.0), 3),
        "Nxy_kN_per_m": round(max(_safe_float(nxy_kN_per_m, 0.0), 0.0), 3),
    }


def normalize_load_conditions(load_conditions: Any) -> Dict[str, Any]:
    if isinstance(load_conditions, dict):
        raw_type = str(load_conditions.get("type", "")).strip()
        nx_value = _safe_float(load_conditions.get("Nx_kN_per_m"), 0.0)
        nxy_value = _safe_float(load_conditions.get("Nxy_kN_per_m"), 0.0)

        type_mapping = {
            "单轴压缩": "axial_compression",
            "轴压": "axial_compression",
            "AXIAL_COMPRESSION": "axial_compression",
            "面内剪切": "in_plane_shear",
            "剪切": "in_plane_shear",
            "IN_PLANE_SHEAR": "in_plane_shear",
            "压剪组合": "compression_shear",
            "压剪": "compression_shear",
            "COMPRESSION_SHEAR": "compression_shear",
        }
        normalized_type = type_mapping.get(raw_type, raw_type)

        if normalized_type not in LOAD_CASE_LABELS:
            if nx_value > 0.0 and nxy_value > 0.0:
                normalized_type = "compression_shear"
            elif nxy_value > 0.0:
                normalized_type = "in_plane_shear"
            else:
                normalized_type = "axial_compression"

        payload = load_condition_payload(normalized_type, nx_value, nxy_value)
        if normalized_type == "axial_compression":
            payload["Nxy_kN_per_m"] = 0.0
        elif normalized_type == "in_plane_shear":
            payload["Nx_kN_per_m"] = 0.0
        return payload

    text = str(load_conditions or "").strip()
    lowered = text.lower()
    if "剪" in text or "shear" in lowered or "nxy" in lowered:
        return load_condition_payload("in_plane_shear")
    return load_condition_payload("axial_compression")


def describe_load_conditions(load_conditions: Any) -> str:
    normalized = normalize_load_conditions(load_conditions)
    load_type = normalized["type"]
    nx_value = normalized.get("Nx_kN_per_m", 0.0)
    nxy_value = normalized.get("Nxy_kN_per_m", 0.0)
    if load_type == "compression_shear":
        return f"{normalized['label']}：Nx={nx_value} kN/m，Nxy={nxy_value} kN/m"
    if load_type == "in_plane_shear":
        return f"{normalized['label']}：Nxy={nxy_value} kN/m"
    return f"{normalized['label']}：Nx={nx_value} kN/m"


def equivalent_in_plane_load(load_conditions: Any) -> float:
    normalized = normalize_load_conditions(load_conditions)
    nx_value = float(normalized.get("Nx_kN_per_m", 0.0))
    nxy_value = float(normalized.get("Nxy_kN_per_m", 0.0))
    load_type = normalized["type"]
    if load_type == "in_plane_shear":
        return round(max(0.85 * nxy_value, 0.0), 3)
    if load_type == "compression_shear":
        return round(max(nx_value + 0.65 * nxy_value, 0.0), 3)
    return round(max(nx_value, 0.0), 3)


def boundary_stiffness_factor(boundary_conditions: Any) -> float:
    normalized = normalize_boundary_conditions(boundary_conditions)
    mapping = {
        "SSSS": 1.00,
        "SSCC": 1.10,
        "CCCC": 1.22,
    }
    return mapping.get(normalized["type"], 1.00)


def load_case_code(load_conditions: Any) -> float:
    normalized = normalize_load_conditions(load_conditions)
    mapping = {
        "axial_compression": 0.0,
        "in_plane_shear": 1.0,
        "compression_shear": 2.0,
    }
    return mapping.get(normalized["type"], 0.0)


def boundary_condition_code(boundary_conditions: Any) -> float:
    normalized = normalize_boundary_conditions(boundary_conditions)
    mapping = {
        "SSSS": 0.0,
        "CCCC": 1.0,
        "SSCC": 2.0,
    }
    return mapping.get(normalized["type"], 0.0)


def normalize_geometry_envelope(envelope: Dict[str, Any] | None) -> Dict[str, Any]:
    data = deepcopy(DEFAULT_GEOMETRY_ENVELOPE)
    if not isinstance(envelope, dict):
        return data

    panel_length = envelope.get("panel_length_mm", data["panel_length_mm"])
    panel_width = envelope.get("panel_width_mm", data["panel_width_mm"])
    if isinstance(panel_length, Iterable) and not isinstance(panel_length, (str, bytes)):
        panel_length_values = list(panel_length)
        if len(panel_length_values) >= 2:
            data["panel_length_mm"] = [_safe_float(panel_length_values[0], 600.0), _safe_float(panel_length_values[1], 800.0)]
    if isinstance(panel_width, Iterable) and not isinstance(panel_width, (str, bytes)):
        panel_width_values = list(panel_width)
        if len(panel_width_values) >= 2:
            data["panel_width_mm"] = [_safe_float(panel_width_values[0], 500.0), _safe_float(panel_width_values[1], 700.0)]
    data["max_stiffener_height_mm"] = _safe_float(
        envelope.get("max_stiffener_height_mm"),
        data["max_stiffener_height_mm"],
    )
    return data


def normalize_screening_preferences(preferences: Dict[str, Any] | None) -> Dict[str, Any]:
    normalized = deepcopy(DEFAULT_SCREENING_PREFERENCES)
    if not isinstance(preferences, dict):
        return normalized
    top_k_value = preferences.get("top_k_candidates", normalized["top_k_candidates"])
    normalized["top_k_candidates"] = int(max(1, min(_safe_float(top_k_value, normalized["top_k_candidates"]), 30.0)))
    return normalized


def normalize_candidate_generation_preferences(preferences: Dict[str, Any] | None) -> Dict[str, Any]:
    normalized = deepcopy(DEFAULT_CANDIDATE_GENERATION_PREFERENCES)
    if not isinstance(preferences, dict):
        return normalized
    total_value = preferences.get("total_candidates", normalized["total_candidates"])
    normalized["total_candidates"] = int(max(1, min(_safe_float(total_value, normalized["total_candidates"]), 60.0)))
    allocation_mode = str(preferences.get("source_allocation_mode") or normalized["source_allocation_mode"]).strip().lower()
    normalized["source_allocation_mode"] = allocation_mode if allocation_mode == "ratio" else "ratio"
    ratio = dict(preferences.get("source_ratio") or {})
    normalized_ratio = {
        key: max(_safe_float(ratio.get(key), value), 0.0)
        for key, value in normalized["source_ratio"].items()
    }
    if sum(normalized_ratio.values()) <= 0.0:
        normalized_ratio = deepcopy(DEFAULT_CANDIDATE_GENERATION_PREFERENCES["source_ratio"])
    normalized["source_ratio"] = normalized_ratio
    from core.stiffener_profile import resolve_stiffener_type
    raw_stiffener_types = preferences.get("stiffener_types")
    if isinstance(raw_stiffener_types, (list, tuple)):
        stiffener_types: list[str] = []
        for raw_type in raw_stiffener_types:
            stype = resolve_stiffener_type(str(raw_type))
            if stype not in stiffener_types:
                stiffener_types.append(stype)
        if stiffener_types:
            normalized["stiffener_types"] = stiffener_types
    return normalized


def normalize_task_payload(
    task: Dict[str, Any],
    *,
    material_system: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    normalized = dict(task)
    normalized.pop("task_id", None)
    if material_system:
        normalized["material_system"] = dict(material_system)

    normalized["application"] = str(normalized.get("application") or "复合材料加筋壁板")
    normalized["load_conditions"] = normalize_load_conditions(normalized.get("load_conditions"))
    normalized["boundary_conditions"] = normalize_boundary_conditions(normalized.get("boundary_conditions"))
    normalized["geometry_envelope"] = normalize_geometry_envelope(normalized.get("geometry_envelope"))
    normalized["candidate_generation_preferences"] = normalize_candidate_generation_preferences(
        normalized.get("candidate_generation_preferences")
    )
    normalized["screening_preferences"] = normalize_screening_preferences(normalized.get("screening_preferences"))
    normalized["layup_constraints"] = {
        **deepcopy(DEFAULT_LAYUP_CONSTRAINTS),
        **dict(normalized.get("layup_constraints", {})),
    }
    normalized["design_targets"] = {
        **deepcopy(DEFAULT_DESIGN_TARGETS),
        **dict(normalized.get("design_targets", {})),
    }
    from core.stiffener_profile import resolve_stiffener_type
    normalized["stiffener_type"] = resolve_stiffener_type(normalized.get("stiffener_type"))
    return normalized


def task_instance_label(task_record: Dict[str, Any] | None) -> str:
    task_id = str((task_record or {}).get("task_id") or "").strip()
    return task_id or "-"


def task_payload_from_request(task_record: Dict[str, Any] | None) -> Dict[str, Any]:
    if not isinstance(task_record, dict):
        return normalize_task_payload({})
    if isinstance(task_record.get("task"), dict):
        return normalize_task_payload(dict(task_record.get("task") or {}))
    return normalize_task_payload(dict(task_record))


def build_task_request_record(
    task: Dict[str, Any],
    *,
    task_id: str,
    source: str,
    created_at: str | None = None,
) -> Dict[str, Any]:
    return {
        "task_id": str(task_id).strip(),
        "created_at": str(created_at or datetime.now(timezone.utc).isoformat()),
        "source": str(source),
        "task": normalize_task_payload(task),
    }


def task_identity_payload(task_record: Dict[str, Any] | None) -> Dict[str, str]:
    task_id = str((task_record or {}).get("task_id") or "").strip()
    return {"task_id": task_id} if task_id else {}


def summarize_task(task: Dict[str, Any]) -> Dict[str, str]:
    normalized_task = task_payload_from_request(task)
    return {
        "application": normalized_task["application"],
        "load_conditions": describe_load_conditions(normalized_task["load_conditions"]),
        "boundary_conditions": describe_boundary_conditions(normalized_task["boundary_conditions"]),
        "candidate_pool": f"候选池目标 {normalized_task['candidate_generation_preferences']['total_candidates']} 个",
        "top_k": f"初筛保留 Top-{normalized_task['screening_preferences']['top_k_candidates']}",
        "objective": (
            f"BLF >= {normalized_task['design_targets']['BLF_min']}，"
            f"{normalized_task['design_targets']['primary_objective']}"
        ),
    }


def requested_candidate_pool_size(task: Dict[str, Any] | None) -> int:
    normalized_task = task_payload_from_request(task or {})
    return int(normalized_task["candidate_generation_preferences"]["total_candidates"])


def requested_screen_top_k(task: Dict[str, Any] | None) -> int:
    normalized_task = task_payload_from_request(task or {})
    return int(normalized_task["screening_preferences"]["top_k_candidates"])


def effective_screen_top_k(task: Dict[str, Any] | None, available_count: int) -> int:
    return max(0, min(requested_screen_top_k(task), max(int(available_count), 0)))
