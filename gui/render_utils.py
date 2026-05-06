"""几何与模态可视化辅助函数。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import pyvista as pv

from core.stiffener_profile import build_stiffener_meshes, stiffener_positions


def build_candidate_scene(candidate: Dict) -> Tuple[list[tuple[pv.DataSet, Dict]], str] | None:
    geometry = dict(candidate.get("geometry", {}))
    if not geometry:
        return None

    stype = str(candidate.get("stiffener_type", "T"))
    panel_length = float(geometry.get("panel_length_mm", 700.0))
    panel_width = float(geometry.get("panel_width_mm", 600.0))
    skin_t = max(float(geometry.get("skin_thickness_mm", 2.5)), 0.5)
    pitch = float(geometry.get("pitch_mm", 120.0))

    meshes: list[tuple[pv.DataSet, Dict]] = []
    skin = pv.Box(bounds=(0.0, panel_length, 0.0, panel_width, 0.0, skin_t))
    meshes.append((skin, {"color": "#9bb7d4", "smooth_shading": True, "opacity": 0.95, "name": "skin"}))

    positions = stiffener_positions(panel_width, pitch)
    stiffener_meshes = build_stiffener_meshes(stype, geometry, positions)
    if stiffener_meshes:
        meshes.extend(stiffener_meshes)

    title = candidate.get("display_name") or candidate.get("candidate_id") or "候选方案"
    return meshes, str(title)


def load_mode_shape_payload(result: Dict) -> Dict | None:
    visualization_json = result.get("visualization_json")
    if not visualization_json:
        return None
    path = Path(str(visualization_json))
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def build_mode_shape_scene(result: Dict) -> Tuple[pv.PolyData, str | None, str] | None:
    payload = load_mode_shape_payload(result)
    if not payload:
        return None

    points = payload.get("points", [])
    faces = payload.get("faces", [])
    scalars = payload.get("scalars", [])
    if not points or not faces:
        return None

    flattened_faces: List[int] = []
    for face in faces:
        flattened_faces.extend(int(value) for value in face)

    mesh = pv.PolyData(points, flattened_faces)
    scalar_name = payload.get("scalar_name", "ModeMagnitude") if scalars and len(scalars) == len(points) else None
    if scalar_name is not None:
        mesh.point_data[scalar_name] = scalars
    title = payload.get("title") or result.get("candidate_id") or "First Buckling Mode"
    return mesh, scalar_name, str(title)
