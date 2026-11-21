"""维护入口的只读案例校验，不补写来源、物理字段或历史求解契约。"""

from __future__ import annotations

import json
import re
from pathlib import Path

from core.io_utils import read_json
from core.laminate import analysis_contract
from core.layup import normalize_layup
from core.schema_validator import get_validator
from core.stiffener_profile import stiffener_footprints
from core.task_contract import BOUNDARY_CONDITION_LIBRARY


def check_schema(name: str, payload: dict, *, schema: dict | None = None) -> None:
    validator = get_validator(name)
    if schema is not None:
        validator = validator.evolve(schema=schema)
    error = next(validator.iter_errors(payload), None)
    if error is not None:
        location = ".".join(map(str, error.path)) or "<root>"
        raise ValueError(f"{name}: {location}: {error.message}")
    json.dumps(payload, allow_nan=False)


def validate_task(task: dict) -> None:
    """任务必须包含明确的结构化工况和边界；无效分量不能被归一化掩盖。"""
    check_schema("task.schema.json", task)
    loads = task["load_conditions"]
    nx, nxy = loads["Nx_kN_per_m"], loads["Nxy_kN_per_m"]
    if nx + nxy <= 0 or (loads["type"] == "axial_compression" and nxy != 0) or (
        loads["type"] == "in_plane_shear" and nx != 0
    ) or (loads["type"] == "compression_shear" and (nx <= 0 or nxy <= 0)):
        raise ValueError("载荷分量与工况类型不一致")
    boundary = task["boundary_conditions"]
    expected = BOUNDARY_CONDITION_LIBRARY[boundary["type"]]
    for key in ("simply_supported_edges", "clamped_edges"):
        if sorted(boundary[key]) != sorted(expected[key]):
            raise ValueError("边界类型与边集合不一致: " + key)
    for key in ("panel_length_mm", "panel_width_mm"):
        lower, upper = task["geometry_envelope"][key]
        if not 0 < lower <= upper:
            raise ValueError("无效几何包络: " + key)


def validate_case_record(record: dict, path: Path) -> None:
    """校验归档设计及身份关联；显示名不属于知识回流保存的物理设计。"""
    try:
        check_schema("case_record.schema.json", record)
        if not re.fullmatch(r"CASE_[1-9][0-9]*", record["case_id"]) or path.stem != record["case_id"]:
            raise ValueError("案例编号与文件名不一致")
        validate_task(record["task"])
        schema = dict(get_validator("candidate.schema.json").schema)
        schema["required"] = [key for key in schema["required"] if key != "display_name"] + [
            "material_system", "load_conditions", "boundary_conditions", "design_targets",
        ]
        design, result = record["design"], record["abaqus_results"]
        check_schema("candidate.schema.json", design, schema=schema)
        check_schema("abaqus_result.schema.json", result)
        candidate_id = design["candidate_id"]
        if not re.fullmatch(r"C[1-9][0-9]*", candidate_id) or result["candidate_id"] != candidate_id:
            raise ValueError("设计与求解结果的候选编号不一致")
        if "candidate_id" in record and record["candidate_id"] != candidate_id:
            raise ValueError("案例顶层候选编号与设计不一致")
        if not isinstance(design["material_system"].get("name"), str) or not design["material_system"]["name"].strip():
            raise ValueError("候选缺少材料名称")
        for key in ("load_conditions", "boundary_conditions", "design_targets"):
            check_schema("task.schema.json", design[key],
                         schema=get_validator("task.schema.json").schema["properties"][key])
        expected_boundary = BOUNDARY_CONDITION_LIBRARY[design["boundary_conditions"]["type"]]
        for key in ("simply_supported_edges", "clamped_edges"):
            if sorted(design["boundary_conditions"][key]) != sorted(expected_boundary[key]):
                raise ValueError("候选边界类型与边集合不一致")
        # 校验已保存的物理输入，不把计算出的契约赋给历史结果。
        analysis_contract(design)
        stiffener_footprints(design["stiffener_type"], design["geometry"])
        normalized = normalize_layup(design["layup"]["skin_layup"])
        for key in ("skin_f0", "skin_f45", "skin_f90"):
            if abs(design["layup"][key] - normalized[key]) > 1e-3:
                raise ValueError("铺层比例与物理层序不一致: " + key)
        for key in ("type", "Nx_kN_per_m", "Nxy_kN_per_m"):
            if design["load_conditions"][key] != record["task"]["load_conditions"][key]:
                raise ValueError("候选与任务载荷不一致: " + key)
        if design["boundary_conditions"]["type"] != record["task"]["boundary_conditions"]["type"]:
            raise ValueError("候选与任务边界不一致")
        if design["design_targets"] != record["task"]["design_targets"]:
            raise ValueError("候选与任务设计目标不一致")
        if result["status"] == "success":
            if not isinstance(result.get("BLF_global"), (int, float)) or result["BLF_global"] <= 0:
                raise ValueError("成功结果缺少有限正 BLF")
            if result.get("verdict") not in ("通过", "不通过") or record["verdict"] != result["verdict"]:
                raise ValueError("成功结果与案例结论不一致")
        elif result.get("BLF_global") is not None or record["verdict"] == "通过":
            raise ValueError("失败结果不能保留承载预测或通过结论")
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"{path}: {exc}") from exc


def read_case_record(path: Path) -> dict:
    record = read_json(path)
    validate_case_record(record, path)
    return record
