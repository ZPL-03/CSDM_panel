# -*- coding: utf-8 -*-
"""ABAQUS ODB 线性屈曲结果提取入口。"""

from __future__ import annotations

import argparse
import math
import re
import traceback
from pathlib import Path
from typing import Dict, List

from core.io_utils import read_json, write_json
from core.task_contract import describe_boundary_conditions, describe_load_conditions


DEFAULT_DENSITY_KG_PER_M3 = 1600.0


def _safe_float(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _candidate_id_from_paths(odb_path: Path, input_json: Path | None) -> str:
    if input_json is not None and input_json.exists():
        try:
            payload = read_json(input_json)
            return str(payload.get("candidate_id", odb_path.stem))
        except Exception:
            return odb_path.stem
    return odb_path.stem


def _load_candidate(input_json: Path | None, candidate_id: str, result_json: Path) -> Dict:
    if input_json is not None and input_json.exists():
        return read_json(input_json)

    inferred_path = result_json.parent / f"input_{candidate_id}.json"
    if inferred_path.exists():
        return read_json(inferred_path)
    return {"candidate_id": candidate_id, "geometry": {}, "design_targets": {}}


def _infer_stiffener_count(geometry: Dict) -> int:
    panel_width = _safe_float(geometry.get("panel_width_mm"), 600.0)
    pitch = max(_safe_float(geometry.get("pitch_mm"), 120.0), 1.0)
    return max(1, int(round(panel_width / pitch)))


def _estimate_weight_kg_per_m2(candidate: Dict) -> float:
    geometry = dict(candidate.get("geometry", {}))
    material = dict(candidate.get("material_system", {}))

    panel_length_mm = _safe_float(geometry.get("panel_length_mm"), 700.0)
    panel_width_mm = _safe_float(geometry.get("panel_width_mm"), 600.0)
    skin_t_mm = _safe_float(geometry.get("skin_thickness_mm"), 2.5)
    stiffener_height_mm = _safe_float(geometry.get("stiffener_height_mm"), 28.0)
    web_t_mm = _safe_float(geometry.get("web_thickness_mm"), 2.0)
    flange_w_mm = _safe_float(geometry.get("flange_width_mm"), 16.0)
    flange_t_mm = _safe_float(geometry.get("flange_thickness_mm"), 2.0)

    panel_area_m2 = max(panel_length_mm * panel_width_mm * 1e-6, 1e-9)
    skin_volume_m3 = panel_length_mm * panel_width_mm * skin_t_mm * 1e-9
    stiffener_volume_m3 = (
        _infer_stiffener_count(geometry)
        * panel_length_mm
        * (web_t_mm * stiffener_height_mm + flange_t_mm * flange_w_mm)
        * 1e-9
    )
    density = _safe_float(material.get("density_kg_per_m3"), DEFAULT_DENSITY_KG_PER_M3)
    total_mass_kg = density * (skin_volume_m3 + stiffener_volume_m3)
    return total_mass_kg / panel_area_m2


def _parse_eigenvalue(description: str) -> float | None:
    match = re.search(r"(?:eigen\s*value|eigenvalue)\s*[:=]?\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", description or "", flags=re.IGNORECASE)
    if match:
        return _safe_float(match.group(1), 0.0)
    match = re.search(r"([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", description or "")
    if match:
        return _safe_float(match.group(1), 0.0)
    return None


def _resolve_effective_eigenvalues(eigenvalues: List[float], positive_tol: float = 1.0e-8) -> tuple[List[float], List[float]]:
    raw_values = [float(value) for value in eigenvalues]
    positive_values = [value for value in raw_values if value > positive_tol]
    return raw_values, positive_values


def _frame_max_displacement(frame) -> float | None:
    if "U" not in frame.fieldOutputs:
        return None
    max_value = None
    for value in frame.fieldOutputs["U"].values:
        magnitude = getattr(value, "magnitude", None)
        if magnitude is None:
            data = getattr(value, "data", ())
            magnitude = math.sqrt(sum(component * component for component in data))
        if max_value is None or magnitude > max_value:
            max_value = magnitude
    return max_value


def _visualization_path(odb_path: Path) -> Path:
    return odb_path.with_name(f"{odb_path.stem}_mode1.json")


def _write_visualization(output_path: Path, points: List[List[float]], faces: List[List[int]], scalars: List[float], title: str) -> None:
    write_json(
        output_path,
        {
            "points": points,
            "faces": faces,
            "scalars": scalars,
            "scalar_name": "ModeMagnitude",
            "title": title,
        },
    )


def _mock_visualization(candidate: Dict, odb_path: Path) -> Path:
    geometry = dict(candidate.get("geometry", {}))
    panel_length = _safe_float(geometry.get("panel_length_mm"), 700.0)
    panel_width = _safe_float(geometry.get("panel_width_mm"), 600.0)
    output_path = _visualization_path(odb_path)
    _write_visualization(
        output_path,
        points=[
            [0.0, 0.0, 0.0],
            [panel_length, 0.0, 0.0],
            [panel_length, panel_width, 6.0],
            [0.0, panel_width, 4.0],
        ],
        faces=[[4, 0, 1, 2, 3]],
        scalars=[0.0, 0.3, 1.0, 0.5],
        title="Mock First Buckling Mode",
    )
    return output_path


def _export_mode_visualization(odb, frame, odb_path: Path) -> Path | None:
    points: List[List[float]] = []
    faces: List[List[int]] = []
    point_index: Dict[tuple[str, int], int] = {}

    for instance_name, instance in odb.rootAssembly.instances.items():
        for node in instance.nodes:
            coordinates = list(getattr(node, "coordinates", (0.0, 0.0, 0.0)))
            while len(coordinates) < 3:
                coordinates.append(0.0)
            point_index[(instance_name, int(node.label))] = len(points)
            points.append([float(coordinates[0]), float(coordinates[1]), float(coordinates[2])])

        for element in instance.elements:
            connectivity = []
            for label in getattr(element, "connectivity", ()):
                idx = point_index.get((instance_name, int(label)))
                if idx is not None:
                    connectivity.append(idx)
            if len(connectivity) >= 3:
                faces.append([len(connectivity)] + connectivity[:4])

    if not points or not faces:
        return None

    scalars = [0.0 for _ in points]
    if "U" in frame.fieldOutputs:
        for value in frame.fieldOutputs["U"].values:
            instance = getattr(value, "instance", None)
            if instance is None:
                continue
            idx = point_index.get((instance.name, int(value.nodeLabel)))
            if idx is None:
                continue
            magnitude = getattr(value, "magnitude", None)
            if magnitude is None:
                data = getattr(value, "data", ())
                magnitude = math.sqrt(sum(component * component for component in data))
            scalars[idx] = float(magnitude)

    output_path = _visualization_path(odb_path)
    _write_visualization(output_path, points, faces, scalars, "First Buckling Mode")
    return output_path


def _write_mock_result(odb_path: Path, result_json: Path, input_json: Path | None) -> None:
    candidate_id = _candidate_id_from_paths(odb_path, input_json)
    candidate = _load_candidate(input_json, candidate_id, result_json)
    weight = round(_estimate_weight_kg_per_m2(candidate), 3)
    visualization_path = _mock_visualization(candidate, odb_path)
    verdict = "通过" if 1.40 >= candidate.get("design_targets", {}).get("BLF_min", 1.2) else "不通过"
    write_json(
        result_json,
        {
            "candidate_id": candidate_id,
            "status": "success",
            "retry_count": 0,
            "BLF_global": 1.40,
            "BLF_local": 1.78,
            "failure_mode": "整体屈曲",
            "max_displacement_mm": 2.5,
            "weight_kg_per_m2": weight,
            "verdict": verdict,
            "abaqus_odb": str(odb_path),
            "abaqus_inp": str(odb_path.with_suffix(".inp")),
            "visualization_json": str(visualization_path),
            "artifact_dir": str(odb_path.parent),
            "error_type": None,
            "error_log": None,
            "mode_eigenvalues": [1.40, 1.78, 2.05],
            "load_summary": describe_load_conditions(candidate.get("load_conditions", {})),
            "boundary_summary": describe_boundary_conditions(candidate.get("boundary_conditions", {})),
            "diagnosis_summary": f"mock 线性屈曲计算完成，当前结论为“{verdict}”。",
        },
    )


def _write_failure_result(candidate_id: str, odb_path: Path, result_json: Path, error_type: str, error_log: object) -> None:
    candidate = _load_candidate(None, candidate_id, result_json)
    write_json(
        result_json,
        {
            "candidate_id": candidate_id,
            "status": "failed",
            "retry_count": 0,
            "BLF_global": None,
            "BLF_local": None,
            "failure_mode": None,
            "max_displacement_mm": None,
            "weight_kg_per_m2": None,
            "verdict": None,
            "abaqus_odb": str(odb_path),
            "abaqus_inp": str(odb_path.with_suffix(".inp")),
            "visualization_json": None,
            "artifact_dir": str(odb_path.parent),
            "error_type": error_type,
            "error_log": error_log,
            "load_summary": describe_load_conditions(candidate.get("load_conditions", {})),
            "boundary_summary": describe_boundary_conditions(candidate.get("boundary_conditions", {})),
            "diagnosis_summary": "线性屈曲结果提取未完成，请结合 ODB 日志继续排查。",
        },
    )


def extract_blf(odb_path: Path, result_json: Path, input_json: Path | None = None, mock: bool = False) -> None:
    if mock:
        _write_mock_result(odb_path, result_json, input_json)
        return

    candidate_id = _candidate_id_from_paths(odb_path, input_json)
    candidate = _load_candidate(input_json, candidate_id, result_json)

    try:  # pragma: no cover
        from odbAccess import openOdb  # type: ignore
    except Exception as exc:  # pragma: no cover
        _write_failure_result(candidate_id, odb_path, result_json, "process_crash", f"ODB 模块导入失败: {exc}")
        return

    try:  # pragma: no cover
        odb = openOdb(path=str(odb_path))
        try:
            if not odb.steps:
                _write_failure_result(candidate_id, odb_path, result_json, "convergence_fail", "ODB 中未找到分析步")
                return

            step = odb.steps["Buckling"] if "Buckling" in odb.steps else next(iter(odb.steps.values()))
            eigenvalues: List[float] = []
            first_mode_displacement = None
            first_mode_frame = step.frames[1] if len(step.frames) > 1 else None
            for frame in step.frames[1:]:
                eigenvalue = _parse_eigenvalue(getattr(frame, "description", ""))
                if eigenvalue is None:
                    eigenvalue = _safe_float(getattr(frame, "frameValue", 0.0), 0.0)
                eigenvalues.append(eigenvalue)
                if first_mode_displacement is None:
                    first_mode_displacement = _frame_max_displacement(frame)

            if not eigenvalues:
                _write_failure_result(candidate_id, odb_path, result_json, "convergence_fail", "屈曲步未输出特征值")
                return

            raw_eigenvalues, positive_eigenvalues = _resolve_effective_eigenvalues(eigenvalues)
            if not positive_eigenvalues:
                _write_failure_result(
                    candidate_id,
                    odb_path,
                    result_json,
                    "blf_negative",
                    {"eigenvalues": [round(value, 6) for value in raw_eigenvalues[:8]]},
                )
                return

            blf_global = positive_eigenvalues[0]
            blf_local = positive_eigenvalues[1] if len(positive_eigenvalues) > 1 else round(blf_global * 1.15, 6)
            failure_mode = "整体屈曲" if blf_global <= blf_local else "局部屈曲"
            verdict = "通过" if blf_global >= candidate.get("design_targets", {}).get("BLF_min", 1.2) else "不通过"
            visualization_path = _export_mode_visualization(odb, first_mode_frame, odb_path) if first_mode_frame is not None else None
            first_positive_mode_index = raw_eigenvalues.index(blf_global) + 1
            negative_modes_skipped = sum(1 for value in raw_eigenvalues if value < 0.0)

            write_json(
                result_json,
                {
                    "candidate_id": candidate_id,
                    "status": "success",
                    "retry_count": 0,
                    "BLF_global": round(blf_global, 6),
                    "BLF_local": round(blf_local, 6),
                    "failure_mode": failure_mode,
                    "max_displacement_mm": round(first_mode_displacement, 6) if first_mode_displacement is not None else None,
                    "weight_kg_per_m2": round(_estimate_weight_kg_per_m2(candidate), 6),
                    "verdict": verdict,
                    "abaqus_odb": str(odb_path),
                    "abaqus_inp": str(odb_path.with_suffix(".inp")),
                    "visualization_json": str(visualization_path) if visualization_path is not None else None,
                    "artifact_dir": str(odb_path.parent),
                    "error_type": None,
                    "error_log": None,
                    "mode_eigenvalues": [round(value, 6) for value in raw_eigenvalues[:8]],
                    "effective_mode_eigenvalues": [round(value, 6) for value in positive_eigenvalues[:5]],
                    "analysis_flags": {
                        "negative_modes_skipped": int(negative_modes_skipped),
                        "first_positive_mode_index": int(first_positive_mode_index),
                    },
                    "load_summary": describe_load_conditions(candidate.get("load_conditions", {})),
                    "boundary_summary": describe_boundary_conditions(candidate.get("boundary_conditions", {})),
                    "diagnosis_summary": (
                        f"线性屈曲结果提取完成，按第 {first_positive_mode_index} 阶正特征值作为有效 BLF，"
                        f"当前结论为“{verdict}”。"
                    ),
                },
            )
        finally:
            odb.close()
    except Exception:  # pragma: no cover
        _write_failure_result(candidate_id, odb_path, result_json, "process_crash", traceback.format_exc())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--odb", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--input")
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()
    extract_blf(
        odb_path=Path(args.odb),
        result_json=Path(args.result),
        input_json=Path(args.input) if args.input else None,
        mock=args.mock,
    )


if __name__ == "__main__":
    main()
