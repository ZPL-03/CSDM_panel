# -*- coding: utf-8 -*-
"""ABAQUS 运行时 ODB 结果提取脚本，兼容旧版 Python。"""

from __future__ import absolute_import

import codecs
import imp
import json
import math
import os
import re
import sys
import traceback


_laminate = imp.load_source("csdm_laminate", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core", "laminate.py"))

try:
    text_type = unicode
    binary_type = str
except NameError:
    text_type = str
    binary_type = bytes


def read_json(path):
    with codecs.open(path, "r", "utf-8") as file_handle:
        return json.load(file_handle)


def normalize_json_value(value):
    if isinstance(value, dict):
        normalized = {}
        for key, item in value.items():
            normalized[normalize_json_value(key)] = normalize_json_value(item)
        return normalized
    if isinstance(value, (list, tuple)):
        return [normalize_json_value(item) for item in value]
    if isinstance(value, text_type):
        return value
    if isinstance(value, binary_type):
        for encoding_name in ("utf-8", "mbcs", "latin-1"):
            try:
                return value.decode(encoding_name)
            except Exception:
                pass
        return value.decode("utf-8", "ignore")
    return value


def write_json(path, data):
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    with codecs.open(path, "w", "utf-8") as file_handle:
        json.dump(normalize_json_value(data), file_handle, ensure_ascii=True, indent=2)


def safe_float(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def candidate_id_from_paths(odb_path, input_json):
    if input_json and os.path.exists(input_json):
        try:
            payload = read_json(input_json)
            return str(payload.get("candidate_id", os.path.splitext(os.path.basename(odb_path))[0]))
        except Exception:
            return os.path.splitext(os.path.basename(odb_path))[0]
    return os.path.splitext(os.path.basename(odb_path))[0]


def load_candidate(input_json, candidate_id, result_json):
    if input_json and os.path.exists(input_json):
        return read_json(input_json)

    inferred_path = os.path.join(os.path.dirname(result_json), "input_%s.json" % candidate_id)
    if os.path.exists(inferred_path):
        return read_json(inferred_path)
    return {"candidate_id": candidate_id, "geometry": {}, "design_targets": {}}


def estimate_weight_kg_per_m2(candidate):
    return _laminate.areal_density(candidate)


def parse_eigenvalue(description):
    match = re.search(r"(?:eigen\s*value|eigenvalue)\s*[:=]?\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", description or "", re.IGNORECASE)
    if match:
        return safe_float(match.group(1), 0.0)
    match = re.search(r"([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", description or "")
    if match:
        return safe_float(match.group(1), 0.0)
    return None


def resolve_effective_eigenvalues(eigenvalues, positive_tol=1.0e-8):
    raw_values = [float(value) for value in eigenvalues]
    if any(math.isnan(value) or math.isinf(value) for value in raw_values):
        raise ValueError("屈曲特征值包含非有限数值")
    positive_values = [value for value in raw_values if value > positive_tol]
    return raw_values, positive_values


def frame_max_displacement(frame):
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


def warning_node_sets(odb):
    """读取求解器写入的警告节点集，保留各实例节点数量。"""
    warnings = {}
    for name, node_set in odb.rootAssembly.nodeSets.items():
        if name.lower().startswith(("warn", "err")):
            count = sum(len(nodes) for nodes in node_set.nodes)
            if count:
                warnings[name] = count
    return warnings


def visualization_path_from_odb(odb_path):
    return os.path.splitext(odb_path)[0] + "_mode1.json"


def write_visualization(path, points, faces, scalars, title):
    write_json(
        path,
        {
            "points": points,
            "faces": faces,
            "scalars": scalars,
            "scalar_name": "ModeMagnitude",
            "title": title,
        },
    )


def export_mode_visualization(odb, frame, odb_path):
    points = []
    faces = []
    point_index = {}

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

    output_path = visualization_path_from_odb(odb_path)
    write_visualization(output_path, points, faces, scalars, "First Buckling Mode")
    return output_path


def write_failure_result(candidate_id, odb_path, result_json, error_type, error_log):
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
            "abaqus_odb": odb_path,
            "abaqus_inp": os.path.splitext(odb_path)[0] + ".inp",
            "visualization_json": None,
            "artifact_dir": os.path.dirname(odb_path),
            "error_type": error_type,
            "error_log": error_log,
        },
    )


def extract_blf(odb_path, result_json, input_json=None):
    candidate_id = candidate_id_from_paths(odb_path, input_json)
    candidate = load_candidate(input_json, candidate_id, result_json)

    try:
        from odbAccess import openOdb
    except Exception as exc:
        write_failure_result(candidate_id, odb_path, result_json, "process_crash", "ODB 模块导入失败: %s" % exc)
        return

    try:
        odb = openOdb(path=str(odb_path))
        try:
            node_warnings = warning_node_sets(odb)
            incomplete_connections = {name: count for name, count in node_warnings.items()
                                      if name.lower().startswith(("warnnodemiss", "warnnodeunconnected", "errnode"))}
            if incomplete_connections:
                write_failure_result(candidate_id, odb_path, result_json, "connection_error",
                                     {"reason": "有限元连接存在未关联或错误节点", "node_sets": incomplete_connections})
                return
            if not odb.steps:
                write_failure_result(candidate_id, odb_path, result_json, "convergence_fail", "ODB 中未找到分析步")
                return

            if "Buckling" in odb.steps:
                step = odb.steps["Buckling"]
            else:
                step = list(odb.steps.values())[-1]
            eigenvalues = []
            frame_count = len(step.frames)
            for frame_index in range(1, frame_count):
                frame = step.frames[frame_index]
                eigenvalue = parse_eigenvalue(getattr(frame, "description", ""))
                if eigenvalue is None:
                    eigenvalue = safe_float(getattr(frame, "frameValue", 0.0), 0.0)
                eigenvalues.append(eigenvalue)

            if not eigenvalues:
                write_failure_result(candidate_id, odb_path, result_json, "convergence_fail", "屈曲步未输出特征值")
                return

            raw_eigenvalues, positive_eigenvalues = resolve_effective_eigenvalues(eigenvalues)
            if not positive_eigenvalues:
                write_failure_result(
                    candidate_id,
                    odb_path,
                    result_json,
                    "blf_negative",
                    {"eigenvalues": [round(value, 6) for value in raw_eigenvalues[:8]]},
                )
                return

            blf_global = positive_eigenvalues[0]
            # 特征值顺序不提供局部/整体模态分类；没有额外正模态时不构造数值。
            second_positive_blf = positive_eigenvalues[1] if len(positive_eigenvalues) > 1 else None
            verdict = "通过" if blf_global >= candidate.get("design_targets", {}).get("BLF_min", 1.2) else "不通过"
            first_positive_mode_index = raw_eigenvalues.index(blf_global) + 1
            first_frame = step.frames[first_positive_mode_index]
            first_mode_displacement = frame_max_displacement(first_frame)
            visualization_path = export_mode_visualization(odb, first_frame, odb_path)
            negative_modes_skipped = sum(1 for value in raw_eigenvalues if value < 0.0)

            write_json(
                result_json,
                {
                    "candidate_id": candidate_id,
                    "status": "success",
                    "retry_count": 0,
                    "BLF_global": blf_global,
                    "BLF_local": None,
                    "second_positive_BLF": second_positive_blf,
                    "failure_mode": None,
                    "max_displacement_mm": round(first_mode_displacement, 6) if first_mode_displacement is not None else None,
                    "weight_kg_per_m2": estimate_weight_kg_per_m2(candidate),
                    "verdict": verdict,
                    "abaqus_odb": odb_path,
                    "abaqus_inp": os.path.splitext(odb_path)[0] + ".inp",
                    "visualization_json": visualization_path,
                    "artifact_dir": os.path.dirname(odb_path),
                    "error_type": None,
                    "error_log": None,
                    "mode_eigenvalues": raw_eigenvalues,
                    "effective_mode_eigenvalues": positive_eigenvalues,
                    "analysis_flags": {
                        "warning_node_sets": node_warnings,
                        "negative_modes_skipped": int(negative_modes_skipped),
                        "first_positive_mode_index": int(first_positive_mode_index),
                        "mode_classification": "not_evaluated",
                    },
                },
            )
        finally:
            odb.close()
    except Exception:
        write_failure_result(candidate_id, odb_path, result_json, "process_crash", traceback.format_exc())


def main(argv=None):
    args = list(argv or sys.argv[1:])
    if len(args) < 2:
        raise SystemExit("Usage: abaqus python runtime_extract_blf.py <odb_path> <result_json> [input_json]")
    odb_path = args[0]
    result_json = args[1]
    input_json = args[2] if len(args) > 2 else None
    extract_blf(odb_path, result_json, input_json=input_json)


if __name__ == "__main__":
    main()
