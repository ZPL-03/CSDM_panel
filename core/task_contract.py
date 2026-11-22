"""任务契约与工况/边界条件规范化工具。"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict
import math
import re


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
        "llm": 3.0,
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


def _finite_number(value: Any, field: str) -> float:
    """数值字段只转换表示形式；无效输入保持为错误，不替换工程条件。"""
    if isinstance(value, bool):
        raise ValueError(f"{field} 必须为有限数值")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} 必须为有限数值") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field} 必须为有限数值")
    return number


def boundary_condition_payload(boundary_type: str) -> Dict[str, Any]:
    boundary_key = str(boundary_type).strip().upper()
    if boundary_key not in BOUNDARY_CONDITION_LIBRARY:
        raise ValueError(f"不支持的边界类型: {boundary_type}")
    return deepcopy(BOUNDARY_CONDITION_LIBRARY[boundary_key])


def normalize_boundary_conditions(boundary_conditions: Any) -> Dict[str, Any]:
    if isinstance(boundary_conditions, dict):
        boundary_type = str(boundary_conditions.get("type", DEFAULT_BOUNDARY_TYPE)).upper()
        normalized = boundary_condition_payload(boundary_type)
        for key in ("simply_supported_edges", "clamped_edges"):
            if key in boundary_conditions and (
                not isinstance(boundary_conditions[key], (list, tuple))
                or sorted(boundary_conditions[key]) != sorted(normalized[key])
            ):
                raise ValueError(f"边界类型与边集合不一致: {key}")
        normalized["label"] = str(boundary_conditions.get("label") or normalized["label"])
        normalized["description"] = str(boundary_conditions.get("description") or normalized["description"])
        return normalized

    text = str(boundary_conditions or "").strip().upper()
    explicit = re.search(r"(?<![A-Z])[SCF]{4}(?![A-Z])|(?:边界(?:条件)?|BOUNDARY)\s*[:：=]?\s*([A-Z]{4})(?![A-Z])", text)
    if explicit:
        return boundary_condition_payload(explicit.group(1) or explicit.group(0))
    if "SSCC" in text or ("简支" in text and "固支" in text):
        return boundary_condition_payload("SSCC")
    if "CCCC" in text or "固支" in text:
        return boundary_condition_payload("CCCC")
    return boundary_condition_payload("SSSS")


def describe_boundary_conditions(boundary_conditions: Any) -> str:
    normalized = normalize_boundary_conditions(boundary_conditions)
    return str(normalized.get("label", BOUNDARY_CONDITION_LIBRARY[DEFAULT_BOUNDARY_TYPE]["label"]))


def load_condition_payload(load_type: str, nx_kN_per_m: float = 0.0, nxy_kN_per_m: float = 0.0) -> Dict[str, Any]:
    """正 Nx 表示压缩，正 Nxy 表示当前模型的剪切方向；分量须与工况一致。"""
    normalized_type = str(load_type).strip()
    if normalized_type not in LOAD_CASE_LABELS:
        raise ValueError(f"不支持的载荷类型: {load_type}")
    nx = _finite_number(nx_kN_per_m, "Nx_kN_per_m")
    nxy = _finite_number(nxy_kN_per_m, "Nxy_kN_per_m")
    if nx < 0 or nxy < 0:
        raise ValueError("Nx 与 Nxy 必须为非负数")
    if (normalized_type == "axial_compression" and nxy != 0) or (
        normalized_type == "in_plane_shear" and nx != 0
    ) or (normalized_type == "compression_shear" and (nx == 0) != (nxy == 0)):
        raise ValueError("载荷分量与工况类型不一致")
    return {
        "type": normalized_type,
        "label": LOAD_CASE_LABELS[normalized_type],
        "Nx_kN_per_m": nx,
        "Nxy_kN_per_m": nxy,
    }


def normalize_load_conditions(load_conditions: Any) -> Dict[str, Any]:
    if isinstance(load_conditions, dict):
        raw_type = str(load_conditions.get("type", "")).strip()
        nx_value = _finite_number(load_conditions.get("Nx_kN_per_m", 0.0), "Nx_kN_per_m")
        nxy_value = _finite_number(load_conditions.get("Nxy_kN_per_m", 0.0), "Nxy_kN_per_m")

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

        if not normalized_type:
            if nx_value > 0.0 and nxy_value > 0.0:
                normalized_type = "compression_shear"
            elif nxy_value > 0.0:
                normalized_type = "in_plane_shear"
            else:
                normalized_type = "axial_compression"

        return load_condition_payload(normalized_type, nx_value, nxy_value)

    text = str(load_conditions or "").strip()
    lowered = text.lower()
    if "剪" in text or "shear" in lowered or "nxy" in lowered:
        return load_condition_payload("in_plane_shear")
    return load_condition_payload("axial_compression")


def describe_load_conditions(load_conditions: Any, *, display: bool = False) -> str:
    """描述工况；显示出口可简化载荷，检索和计算说明保留原值。"""
    normalized = normalize_load_conditions(load_conditions)
    load_type = normalized["type"]
    nx_value = normalized.get("Nx_kN_per_m", 0.0)
    nxy_value = normalized.get("Nxy_kN_per_m", 0.0)
    if display:
        from core.display import format_number
        nx_value, nxy_value = format_number(nx_value), format_number(nxy_value)
    if load_type == "compression_shear":
        return f"{normalized['label']}：Nx={nx_value} kN/m，Nxy={nxy_value} kN/m"
    if load_type == "in_plane_shear":
        return f"{normalized['label']}：Nxy={nxy_value} kN/m"
    return f"{normalized['label']}：Nx={nx_value} kN/m"


def load_case_code(load_conditions: Any) -> float:
    normalized = normalize_load_conditions(load_conditions)
    mapping = {
        "axial_compression": 0.0,
        "in_plane_shear": 1.0,
        "compression_shear": 2.0,
    }
    return mapping[normalized["type"]]


def boundary_condition_code(boundary_conditions: Any) -> float:
    normalized = normalize_boundary_conditions(boundary_conditions)
    mapping = {
        "SSSS": 0.0,
        "CCCC": 1.0,
        "SSCC": 2.0,
    }
    return mapping[normalized["type"]]


def normalize_geometry_envelope(envelope: Dict[str, Any] | None) -> Dict[str, Any]:
    data = deepcopy(DEFAULT_GEOMETRY_ENVELOPE)
    if envelope is None:
        return data
    if not isinstance(envelope, dict):
        raise ValueError("geometry_envelope 必须为对象")
    for key in ("panel_length_mm", "panel_width_mm"):
        pair = envelope.get(key, data[key])
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(f"{key} 必须给出上下界")
        lower, upper = (_finite_number(value, key) for value in pair)
        if not 0 < lower <= upper:
            raise ValueError(f"无效几何包络: {key}")
        data[key] = [lower, upper]
    key = "max_stiffener_height_mm"
    data[key] = _finite_number(envelope.get(key, data[key]), key)
    if data[key] <= 0:
        raise ValueError(f"{key} 必须为正数")
    return data


def _candidate_count(value: Any, field: str, maximum: int) -> int:
    number = _finite_number(value, field)
    if not 1 <= number <= maximum or not number.is_integer():
        raise ValueError(f"{field} 必须为 1 到 {maximum} 的整数")
    return int(number)


def normalize_screening_preferences(preferences: Dict[str, Any] | None) -> Dict[str, Any]:
    normalized = deepcopy(DEFAULT_SCREENING_PREFERENCES)
    if preferences is None:
        return normalized
    if not isinstance(preferences, dict):
        raise ValueError("screening_preferences 必须为对象")
    top_k_value = preferences.get("top_k_candidates", normalized["top_k_candidates"])
    normalized["top_k_candidates"] = _candidate_count(top_k_value, "top_k_candidates", 30)
    return normalized


def normalize_candidate_generation_preferences(preferences: Dict[str, Any] | None) -> Dict[str, Any]:
    normalized = deepcopy(DEFAULT_CANDIDATE_GENERATION_PREFERENCES)
    if preferences is None:
        return normalized
    if not isinstance(preferences, dict):
        raise ValueError("candidate_generation_preferences 必须为对象")
    total_value = preferences.get("total_candidates", normalized["total_candidates"])
    normalized["total_candidates"] = _candidate_count(total_value, "total_candidates", 60)
    allocation_mode = str(preferences.get("source_allocation_mode", normalized["source_allocation_mode"])).strip().lower()
    if allocation_mode != "ratio":
        raise ValueError("候选来源分配方式必须为 ratio")
    normalized["source_allocation_mode"] = allocation_mode
    ratio = preferences.get("source_ratio")
    if ratio is None:
        ratio = normalized["source_ratio"]
    if not isinstance(ratio, dict) or set(ratio) - set(normalized["source_ratio"]):
        raise ValueError("source_ratio 只接受 llm、case_transfer、doe")
    normalized_ratio = {
        key: _finite_number(ratio.get(key, 0.0), key)
        for key in normalized["source_ratio"]
    }
    if min(normalized_ratio.values()) < 0 or sum(normalized_ratio.values()) <= 0.0:
        raise ValueError("候选来源比例必须非负且至少有一个正数")
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


def build_design_constraints(task: Dict, stiffener_type: str | None = None) -> Dict:
    """生成候选与求解共用的任务约束；任务包络优先于全局采样范围。"""
    from core.stiffener_profile import load_param_ranges_for_type, required_geometry_params

    payload = task_payload_from_request(task)
    stype = stiffener_type or payload["stiffener_type"]
    allowed_types = payload["candidate_generation_preferences"].get("stiffener_types", [payload["stiffener_type"]])
    if stype not in allowed_types:
        raise ValueError(f"筋型 {stype} 不在任务允许集合中")
    ranges = deepcopy(load_param_ranges_for_type(stype))
    envelope = payload["geometry_envelope"]
    facts = payload.get("user_input_facts", {}).get("geometry", {})
    fixed = dict(payload.get("fixed_geometry", {}))
    fixed.update({key: value for key, value in facts.items() if key in required_geometry_params(stype)})
    for key in ("panel_length_mm", "panel_width_mm"):
        lower, upper = envelope[key]
        ranges[key] = {"min": float(lower), "max": float(upper)}
    for key, value in fixed.items():
        if key not in required_geometry_params(stype):
            raise ValueError(f"固定几何字段不适用于 {stype}: {key}")
        ranges[key] = {"min": float(value), "max": float(value)}
    height_limit = float(facts.get("max_stiffener_height_mm", envelope["max_stiffener_height_mm"]))
    ranges["stiffener_height_mm"]["max"] = min(ranges["stiffener_height_mm"]["max"], height_limit)
    for key in required_geometry_params(stype):
        lower, upper = ranges[key]["min"], ranges[key]["max"]
        if not all(math.isfinite(v) and v > 0 for v in (lower, upper)) or lower > upper:
            raise ValueError(f"无效任务几何范围: {key}")
    return {"geometry_ranges": ranges, "stiffener_type": stype,
            "layup_constraints": deepcopy(payload["layup_constraints"]),
            "load_conditions": deepcopy(payload["load_conditions"]),
            "boundary_conditions": deepcopy(payload["boundary_conditions"])}


def validate_candidate_constraints(candidate: Dict, constraints: Dict) -> None:
    """检查任务固定项和工况，拒绝跨阶段更换尺寸、筋型或载荷。"""
    if candidate.get("stiffener_type") != constraints["stiffener_type"]:
        raise ValueError("候选筋型与任务约束不一致")
    for key, bounds in constraints["geometry_ranges"].items():
        if key not in candidate.get("geometry", {}):
            raise ValueError(f"候选缺少任务几何: {key}")
        value = float(candidate["geometry"][key])
        if not math.isfinite(value) or not bounds["min"] <= value <= bounds["max"]:
            raise ValueError(f"候选 {key}={value} 不满足任务范围 {bounds}")
    if normalize_load_conditions(candidate.get("load_conditions")) != constraints["load_conditions"]:
        raise ValueError("候选载荷与任务不一致")
    if normalize_boundary_conditions(candidate.get("boundary_conditions"))["type"] != constraints["boundary_conditions"]["type"]:
        raise ValueError("候选边界与任务不一致")
