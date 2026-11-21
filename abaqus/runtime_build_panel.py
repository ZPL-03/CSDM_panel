# -*- coding: utf-8 -*-
"""Abaqus 运行时复合材料加筋壁板建模脚本，兼容 Python 2.7/3。"""

from __future__ import absolute_import

import codecs
import imp
import json
import os
import traceback


# CAE 旧版解释器直接加载纯标准库模块，避免导入宿主 Python 依赖。
_laminate = imp.load_source("csdm_laminate", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core", "laminate.py"))


BOUNDARY_CONDITION_LIBRARY = {
    "SSSS": {
        "type": "SSSS",
        "simply_supported_edges": ["X0", "X1", "Y0", "Y1"],
        "clamped_edges": [],
    },
    "CCCC": {
        "type": "CCCC",
        "simply_supported_edges": [],
        "clamped_edges": ["X0", "X1", "Y0", "Y1"],
    },
    "SSCC": {
        "type": "SSCC",
        "simply_supported_edges": ["X0", "X1"],
        "clamped_edges": ["Y0", "Y1"],
    },
}

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


def _assemble_blade(assembly, model, web_part, skin_instance, spec,
                    index, position_mm, display_index, tol):
    """BLADE 型筋：仅竖腹板，无翼缘。"""
    from abaqusConstants import OFF, ON, SPECIFIED
    web_name = "Web-%02d" % display_index
    assembly.Instance(name=web_name, part=web_part, dependent=ON)
    assembly.rotate(
        instanceList=(web_name,),
        axisPoint=(0.0, 0.0, 0.0),
        axisDirection=(1.0, 0.0, 0.0),
        angle=90.0,
    )
    assembly.translate(instanceList=(web_name,), vector=(0.0, position_mm, 0.0))
    skin_face = skin_instance.faces
    web_instance = assembly.instances[web_name]
    web_edge = web_instance.edges.getByBoundingBox(
        xMin=-tol, xMax=spec["panel_length_mm"] + tol,
        yMin=position_mm - tol, yMax=position_mm + tol,
        zMin=-tol, zMax=tol,
    )
    if len(web_edge) != 1:
        raise ValueError("BLADE 腹板底边选取失败")
    model.Tie(
        "Tie_Web_Skin_%02d" % display_index,
        assembly.Surface(name="SurfSkinW_%02d" % display_index, side1Faces=skin_face),
        assembly.Surface(name="SurfWebB_%02d" % display_index, side1Edges=web_edge),
        positionToleranceMethod=SPECIFIED, positionTolerance=tol,
        adjust=OFF, tieRotations=ON, thickness=OFF,
    )


def _assemble_t(assembly, model, web_part, flange_part, skin_instance, spec,
                index, position_mm, display_index, tol):
    """T 型筋：竖腹板 + 左右半翼缘。"""
    from abaqusConstants import OFF, ON, SPECIFIED
    flange_width = spec["flange_width_mm"]
    half_flange_width = flange_width / 2.0
    panel_length = spec["panel_length_mm"]
    flange_z = (spec["skin_thickness_mm"] + spec["flange_thickness_mm"]) / 2.0

    web_name = "Web-%02d" % display_index
    flange_left_name = "FlangeL-%02d" % display_index
    flange_right_name = "FlangeR-%02d" % display_index

    assembly.Instance(name=web_name, part=web_part, dependent=ON)
    assembly.rotate(
        instanceList=(web_name,),
        axisPoint=(0.0, 0.0, 0.0),
        axisDirection=(1.0, 0.0, 0.0),
        angle=90.0,
    )
    assembly.translate(instanceList=(web_name,), vector=(0.0, position_mm, flange_z))

    assembly.Instance(name=flange_left_name, part=flange_part, dependent=ON)
    assembly.translate(instanceList=(flange_left_name,), vector=(0.0, position_mm - half_flange_width, flange_z))

    assembly.Instance(name=flange_right_name, part=flange_part, dependent=ON)
    assembly.translate(instanceList=(flange_right_name,), vector=(0.0, position_mm, flange_z))

    web_instance = assembly.instances[web_name]
    web_bottom_edge = web_instance.edges.getByBoundingBox(
        xMin=-tol, xMax=panel_length + tol,
        yMin=position_mm - tol, yMax=position_mm + tol,
        zMin=flange_z - tol, zMax=flange_z + tol,
    )
    flange_left_instance = assembly.instances[flange_left_name]
    flange_right_instance = assembly.instances[flange_right_name]
    flange_left_inner = flange_left_instance.edges.getByBoundingBox(
        xMin=-tol, xMax=panel_length + tol,
        yMin=position_mm - tol, yMax=position_mm + tol,
        zMin=flange_z - tol, zMax=flange_z + tol,
    )
    flange_right_inner = flange_right_instance.edges.getByBoundingBox(
        xMin=-tol, xMax=panel_length + tol,
        yMin=position_mm - tol, yMax=position_mm + tol,
        zMin=flange_z - tol, zMax=flange_z + tol,
    )
    flange_inner_edges = flange_left_inner + flange_right_inner
    if len(web_bottom_edge) != 1 or len(flange_inner_edges) != 2:
        raise ValueError("T 型筋连接边选取失败")

    model.Tie(
        "Tie_Skin_FlangeL_%02d" % display_index,
        assembly.Surface(name="SurfSkinL_%02d" % display_index, side1Faces=skin_instance.faces),
        assembly.Surface(name="SurfFlangeFaceL_%02d" % display_index, side1Faces=flange_left_instance.faces),
        positionToleranceMethod=SPECIFIED, positionTolerance=flange_z + tol,
        adjust=OFF, tieRotations=ON, thickness=OFF,
    )
    model.Tie(
        "Tie_Skin_FlangeR_%02d" % display_index,
        assembly.Surface(name="SurfSkinR_%02d" % display_index, side1Faces=skin_instance.faces),
        assembly.Surface(name="SurfFlangeFaceR_%02d" % display_index, side1Faces=flange_right_instance.faces),
        positionToleranceMethod=SPECIFIED, positionTolerance=flange_z + tol,
        adjust=OFF, tieRotations=ON, thickness=OFF,
    )
    model.Tie(
        "Tie_Web_Flange_%02d" % display_index,
        assembly.Surface(name="SurfFlangeInner_%02d" % display_index, side1Edges=flange_inner_edges),
        assembly.Surface(name="SurfWebBase_%02d" % display_index, side1Edges=web_bottom_edge),
        positionToleranceMethod=SPECIFIED, positionTolerance=tol,
        adjust=OFF, tieRotations=ON, thickness=OFF,
    )


def _assemble_l(assembly, model, web_part, flange_part, skin_instance, spec,
                index, position_mm, display_index, tol):
    """L 型角材：竖腹板 + 单侧翼缘（右侧全宽）。"""
    from abaqusConstants import OFF, ON, SPECIFIED
    panel_length = spec["panel_length_mm"]
    flange_z = (spec["skin_thickness_mm"] + spec["flange_thickness_mm"]) / 2.0

    web_name = "Web-%02d" % display_index
    flange_name = "Flange-%02d" % display_index

    assembly.Instance(name=web_name, part=web_part, dependent=ON)
    assembly.rotate(
        instanceList=(web_name,),
        axisPoint=(0.0, 0.0, 0.0),
        axisDirection=(1.0, 0.0, 0.0),
        angle=90.0,
    )
    assembly.translate(instanceList=(web_name,), vector=(0.0, position_mm, flange_z))

    # L 型 flange 全宽，从 position_mm 向右延伸
    assembly.Instance(name=flange_name, part=flange_part, dependent=ON)
    assembly.translate(instanceList=(flange_name,), vector=(0.0, position_mm, flange_z))

    web_instance = assembly.instances[web_name]
    web_bottom_edge = web_instance.edges.getByBoundingBox(
        xMin=-tol, xMax=panel_length + tol,
        yMin=position_mm - tol, yMax=position_mm + tol,
        zMin=flange_z - tol, zMax=flange_z + tol,
    )
    flange_instance = assembly.instances[flange_name]
    flange_inner = flange_instance.edges.getByBoundingBox(
        xMin=-tol, xMax=panel_length + tol,
        yMin=position_mm - tol, yMax=position_mm + tol,
        zMin=flange_z - tol, zMax=flange_z + tol,
    )
    if len(web_bottom_edge) != 1 or len(flange_inner) != 1:
        raise ValueError("L 型筋连接边选取失败")

    model.Tie(
        "Tie_Skin_Flange_%02d" % display_index,
        assembly.Surface(name="SurfSkinF_%02d" % display_index, side1Faces=skin_instance.faces),
        assembly.Surface(name="SurfFlangeF_%02d" % display_index, side1Faces=flange_instance.faces),
        positionToleranceMethod=SPECIFIED, positionTolerance=flange_z + tol,
        adjust=OFF, tieRotations=ON, thickness=OFF,
    )
    model.Tie(
        "Tie_Web_Flange_%02d" % display_index,
        assembly.Surface(name="SurfFlangeIn_%02d" % display_index, side1Edges=flange_inner),
        assembly.Surface(name="SurfWebB_%02d" % display_index, side1Edges=web_bottom_edge),
        positionToleranceMethod=SPECIFIED, positionTolerance=tol,
        adjust=OFF, tieRotations=ON, thickness=OFF,
    )


def _assemble_hat(assembly, model, web_part, flange_part, cap_part, skin_instance,
                  spec, index, position_mm, display_index, tol):
    """HAT 帽型筋：外侧底部连接板 + 左右斜腹板 + 顶部帽顶。"""
    from abaqusConstants import OFF, ON, SPECIFIED
    import math as _math
    flange_width = spec["flange_width_mm"]
    cap_width = spec["cap_width_mm"]
    height = spec["stiffener_height_mm"]
    flange_z = (spec["skin_thickness_mm"] + spec["flange_thickness_mm"]) / 2.0
    panel_length = spec["panel_length_mm"]
    half_flange = flange_width / 2.0
    half_cap = cap_width / 2.0
    foot_width = half_flange - half_cap

    incline_angle = _math.degrees(_math.atan2(height, foot_width))

    # 两侧斜腹板连接底部翼缘中面和帽顶中面。
    web_left_name = "WebL-%02d" % display_index
    assembly.Instance(name=web_left_name, part=web_part, dependent=ON)
    assembly.rotate(
        instanceList=(web_left_name,),
        axisPoint=(0.0, 0.0, 0.0),
        axisDirection=(1.0, 0.0, 0.0),
        angle=incline_angle,
    )
    assembly.translate(instanceList=(web_left_name,),
                       vector=(0.0, position_mm - half_flange, flange_z))

    web_right_name = "WebR-%02d" % display_index
    assembly.Instance(name=web_right_name, part=web_part, dependent=ON)
    assembly.rotate(
        instanceList=(web_right_name,),
        axisPoint=(0.0, 0.0, 0.0),
        axisDirection=(1.0, 0.0, 0.0),
        angle=180.0 - incline_angle,
    )
    assembly.translate(instanceList=(web_right_name,),
                       vector=(0.0, position_mm + half_flange, flange_z))

    # 左外侧底部连接板（从 pos-half_flange-foot_width 到 pos-half_flange）
    flange_left_name = "FlangeL-%02d" % display_index
    assembly.Instance(name=flange_left_name, part=flange_part, dependent=ON)
    assembly.translate(instanceList=(flange_left_name,),
                       vector=(0.0, position_mm - half_flange - foot_width, flange_z))

    # 右外侧底部连接板（从 pos+half_flange 到 pos+half_flange+foot_width）
    flange_right_name = "FlangeR-%02d" % display_index
    assembly.Instance(name=flange_right_name, part=flange_part, dependent=ON)
    assembly.translate(instanceList=(flange_right_name,),
                       vector=(0.0, position_mm + half_flange, flange_z))

    # 顶部 cap（全宽 = cap_width，居中于 position_mm）
    cap_name = "Cap-%02d" % display_index
    assembly.Instance(name=cap_name, part=cap_part, dependent=ON)
    assembly.translate(instanceList=(cap_name,),
                       vector=(0.0, position_mm - half_cap, flange_z + height))

    # 翼缘与蒙皮保持中面偏置，Tie 不移动节点。
    flange_left_instance = assembly.instances[flange_left_name]
    model.Tie(
        "Tie_Skin_FlangeL_%02d" % display_index,
        assembly.Surface(name="SurfSkinFL_%02d" % display_index, side1Faces=skin_instance.faces),
        assembly.Surface(name="SurfFlangeL_%02d" % display_index, side1Faces=flange_left_instance.faces),
        positionToleranceMethod=SPECIFIED, positionTolerance=flange_z + tol,
        adjust=OFF, tieRotations=ON, thickness=OFF,
    )

    # Tie: 右 flange → skin
    flange_right_instance = assembly.instances[flange_right_name]
    model.Tie(
        "Tie_Skin_FlangeR_%02d" % display_index,
        assembly.Surface(name="SurfSkinFR_%02d" % display_index, side1Faces=skin_instance.faces),
        assembly.Surface(name="SurfFlangeR_%02d" % display_index, side1Faces=flange_right_instance.faces),
        positionToleranceMethod=SPECIFIED, positionTolerance=flange_z + tol,
        adjust=OFF, tieRotations=ON, thickness=OFF,
    )

    # Tie: 左 web 底边 → 左外侧连接板内边
    web_left_instance = assembly.instances[web_left_name]
    web_left_bottom = web_left_instance.edges.getByBoundingBox(
        xMin=-tol, xMax=panel_length + tol,
        yMin=position_mm - half_flange - tol, yMax=position_mm - half_flange + tol,
        zMin=flange_z - tol, zMax=flange_z + tol,
    )
    flange_left_web_edge = flange_left_instance.edges.getByBoundingBox(
        xMin=-tol, xMax=panel_length + tol,
        yMin=position_mm - half_flange - tol, yMax=position_mm - half_flange + tol,
        zMin=flange_z - tol, zMax=flange_z + tol,
    )
    if len(web_left_bottom) != 1 or len(flange_left_web_edge) != 1:
        raise ValueError("HAT 左腹板底部连接边选取失败")
    model.Tie(
        "Tie_WebL_FlangeL_%02d" % display_index,
        assembly.Surface(name="SurfFlangeLWebEdge_%02d" % display_index, side1Edges=flange_left_web_edge),
        assembly.Surface(name="SurfWebLBottom_%02d" % display_index, side1Edges=web_left_bottom),
        positionToleranceMethod=SPECIFIED, positionTolerance=tol,
        adjust=OFF, tieRotations=ON, thickness=OFF,
    )

    # Tie: 右 web 底边 → 右外侧连接板内边
    web_right_instance = assembly.instances[web_right_name]
    web_right_bottom = web_right_instance.edges.getByBoundingBox(
        xMin=-tol, xMax=panel_length + tol,
        yMin=position_mm + half_flange - tol, yMax=position_mm + half_flange + tol,
        zMin=flange_z - tol, zMax=flange_z + tol,
    )
    flange_right_web_edge = flange_right_instance.edges.getByBoundingBox(
        xMin=-tol, xMax=panel_length + tol,
        yMin=position_mm + half_flange - tol, yMax=position_mm + half_flange + tol,
        zMin=flange_z - tol, zMax=flange_z + tol,
    )
    if len(web_right_bottom) != 1 or len(flange_right_web_edge) != 1:
        raise ValueError("HAT 右腹板底部连接边选取失败")
    model.Tie(
        "Tie_WebR_FlangeR_%02d" % display_index,
        assembly.Surface(name="SurfFlangeRWebEdge_%02d" % display_index, side1Edges=flange_right_web_edge),
        assembly.Surface(name="SurfWebRBottom_%02d" % display_index, side1Edges=web_right_bottom),
        positionToleranceMethod=SPECIFIED, positionTolerance=tol,
        adjust=OFF, tieRotations=ON, thickness=OFF,
    )

    # Tie: 左 web 顶边 → cap 左边
    cap_instance = assembly.instances[cap_name]
    cap_left_edge = cap_instance.edges.getByBoundingBox(
        xMin=-tol, xMax=panel_length + tol,
        yMin=position_mm - half_cap - tol, yMax=position_mm - half_cap + tol,
        zMin=flange_z + height - tol, zMax=flange_z + height + tol,
    )
    web_left_top = web_left_instance.edges.getByBoundingBox(
        xMin=-tol, xMax=panel_length + tol,
        yMin=position_mm - half_cap - tol, yMax=position_mm - half_cap + tol,
        zMin=flange_z + height - tol, zMax=flange_z + height + tol,
    )
    if len(cap_left_edge) != 1 or len(web_left_top) != 1:
        raise ValueError("HAT 左腹板顶部连接边选取失败")
    model.Tie(
        "Tie_WebL_Cap_%02d" % display_index,
        assembly.Surface(name="SurfCapLEdge_%02d" % display_index, side1Edges=cap_left_edge),
        assembly.Surface(name="SurfWebLTop_%02d" % display_index, side1Edges=web_left_top),
        positionToleranceMethod=SPECIFIED, positionTolerance=tol,
        adjust=OFF, tieRotations=ON, thickness=OFF,
    )

    # Tie: 右 web 顶边 → cap 右边
    cap_right_edge = cap_instance.edges.getByBoundingBox(
        xMin=-tol, xMax=panel_length + tol,
        yMin=position_mm + half_cap - tol, yMax=position_mm + half_cap + tol,
        zMin=flange_z + height - tol, zMax=flange_z + height + tol,
    )
    web_right_top = web_right_instance.edges.getByBoundingBox(
        xMin=-tol, xMax=panel_length + tol,
        yMin=position_mm + half_cap - tol, yMax=position_mm + half_cap + tol,
        zMin=flange_z + height - tol, zMax=flange_z + height + tol,
    )
    if len(cap_right_edge) != 1 or len(web_right_top) != 1:
        raise ValueError("HAT 右腹板顶部连接边选取失败")
    model.Tie(
        "Tie_WebR_Cap_%02d" % display_index,
        assembly.Surface(name="SurfCapREdge_%02d" % display_index, side1Edges=cap_right_edge),
        assembly.Surface(name="SurfWebRTop_%02d" % display_index, side1Edges=web_right_top),
        positionToleranceMethod=SPECIFIED, positionTolerance=tol,
        adjust=OFF, tieRotations=ON, thickness=OFF,
    )


def resolve_analysis_spec(candidate):
    contract = _laminate.analysis_contract(candidate)
    geometry = contract["geometry"]
    load_conditions = contract["load_conditions"]
    boundary_conditions = BOUNDARY_CONDITION_LIBRARY[contract["boundary_type"]]
    analysis = contract["analysis"]
    stiffener_type = contract["stiffener_type"]
    panel_length_mm = geometry["panel_length_mm"]
    panel_width_mm = geometry["panel_width_mm"]
    pitch_mm = geometry["pitch_mm"]
    stiffener_height_mm = geometry["stiffener_height_mm"]
    flange_width_mm = geometry["flange_width_mm"] if stiffener_type != "BLADE" else 0.0
    cap_width_mm = geometry["cap_width_mm"] if stiffener_type == "HAT" else 0.0
    cap_thickness_mm = geometry["cap_thickness_mm"] if stiffener_type == "HAT" else 0.0

    max_span = max(
        max(flange_width_mm, 8.0),
        cap_width_mm if stiffener_type == "HAT" else 0.0,
    )
    mesh_size_mm = max(
        12.0,
        min(
            panel_length_mm / 20.0,
            panel_width_mm / 16.0,
            pitch_mm / 5.0,
            max(stiffener_height_mm, 8.0) / 2.5,
            max(max_span, 8.0) / 1.8,
        ),
    )

    default_buckling_modes = 8
    if load_conditions.get("type") == "in_plane_shear":
        default_buckling_modes = 14
    elif load_conditions.get("type") == "compression_shear":
        default_buckling_modes = 16

    return {
        "analysis_contract": contract,
        "candidate_id": str(candidate.get("candidate_id", "UNKNOWN")),
        "job_name": str(candidate.get("candidate_id", "UNKNOWN")),
        "stiffener_type": stiffener_type,
        "panel_length_mm": panel_length_mm,
        "panel_width_mm": panel_width_mm,
        "skin_thickness_mm": geometry["skin_thickness_mm"],
        "pitch_mm": pitch_mm,
        "stiffener_height_mm": stiffener_height_mm,
        "web_thickness_mm": geometry["web_thickness_mm"],
        "flange_width_mm": flange_width_mm,
        "flange_thickness_mm": geometry["flange_thickness_mm"] if stiffener_type != "BLADE" else 0.0,
        "cap_width_mm": cap_width_mm,
        "cap_thickness_mm": cap_thickness_mm,
        "stiffener_positions_mm": _laminate.stiffener_positions(panel_width_mm, pitch_mm),
        "mesh_size_mm": float(analysis.get("mesh_size_mm", round(mesh_size_mm, 3))),
        "buckling_modes": int(analysis.get("buckling_modes", default_buckling_modes)),
        "preload_nx_kN_per_m": load_conditions["Nx_kN_per_m"],
        "preload_nxy_kN_per_m": load_conditions["Nxy_kN_per_m"],
        "load_case_type": load_conditions["type"],
        "boundary_type": boundary_conditions["type"],
        "simply_supported_edges": boundary_conditions["simply_supported_edges"],
        "clamped_edges": boundary_conditions["clamped_edges"],
        "material": contract["material"],
    }


def create_composite_sections(model, spec, material_name):
    """SectionLayer 顺序与负法向到正法向的物理层序一致，只展开一次对称层序。"""
    from section import SectionLayer
    from abaqusConstants import DEFAULT, GRADIENT, NO_IDEALIZATION, OFF, SIMPSON

    for component, layers in spec["analysis_contract"]["sections"].items():
        model.CompositeShellSection(
            name=str(component.capitalize() + "Section"),
            layup=tuple(SectionLayer(thickness=ply["thickness_mm"], orientAngle=ply["angle_deg"],
                                   material=material_name, numIntPts=3,
                                   plyName=str("Ply-%03d" % (index + 1)))
                        for index, ply in enumerate(layers)),
            symmetric=OFF, preIntegrate=OFF, idealization=NO_IDEALIZATION,
            poissonDefinition=DEFAULT, temperature=GRADIENT, integrationRule=SIMPSON,
        )


def assign_material_orientation(part):
    """部件局部 X 为零度方向；装配绕 X 旋转保留筋条轴向及逐层法向顺序。"""
    from abaqusConstants import AXIS_3, CARTESIAN, ROTATION_NONE, STACK_3, SYSTEM
    import regionToolset

    datum = part.DatumCsysByThreePoints(name="PlyAxes", coordSysType=CARTESIAN,
                                      origin=(0.0, 0.0, 0.0), point1=(1.0, 0.0, 0.0),
                                      point2=(0.0, 1.0, 0.0))
    part.MaterialOrientation(region=regionToolset.Region(faces=part.faces), orientationType=SYSTEM,
                             axis=AXIS_3, localCsys=part.datums[datum.id],
                             additionalRotationType=ROTATION_NONE, angle=0.0, stackDirection=STACK_3)


def write_failure_result(candidate_id, result_json, error_type, error_log):
    run_dir = os.getcwd()
    odb_path = os.path.join(run_dir, "%s.odb" % candidate_id)
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
            "artifact_dir": run_dir,
            "error_type": error_type,
            "error_log": error_log,
        },
    )


def edge_mode(spec, edge_name):
    if edge_name in spec.get("clamped_edges", []):
        return "clamped"
    if edge_name in spec.get("simply_supported_edges", []):
        return "simply_supported"
    return "free"


def edge_nodal_forces(edge_nodes, nx, nxy):
    """按边节点控制长度积分均匀膜内牵引，合并角点处两条边的分量。"""
    forces = {}
    tractions = {"X0": (nx, -nxy), "X1": (-nx, nxy),
                 "Y0": (-nxy, 0.0), "Y1": (nxy, 0.0)}
    for edge_name, traction in tractions.items():
        axis = 1 if edge_name.startswith("X") else 0
        nodes = sorted(edge_nodes[edge_name], key=lambda node: node.coordinates[axis])
        if len(nodes) < 2:
            raise ValueError("载荷边至少需要两个节点: " + edge_name)
        weights = [0.0] * len(nodes)
        for index in range(len(nodes) - 1):
            distance = nodes[index + 1].coordinates[axis] - nodes[index].coordinates[axis]
            if distance <= 0:
                raise ValueError("载荷边节点坐标重复: " + edge_name)
            weights[index] += distance / 2.0
            weights[index + 1] += distance / 2.0
        for node, weight in zip(nodes, weights):
            value = forces.setdefault(node.label, [0.0, 0.0])
            value[0] += weight * traction[0]
            value[1] += weight * traction[1]
    return forces


def apply_panel_boundary_and_load(model, assembly, skin_instance, spec, edge_nodes):
    """支承限定板外运动；两个角点的三个面内约束消除平移和刚体转动。"""
    import regionToolset
    from abaqusConstants import UNIFORM

    for edge_name, nodes in edge_nodes.items():
        region = assembly.Set(name="Edge_" + edge_name, nodes=nodes)
        mode = edge_mode(spec, edge_name)
        if mode not in ("clamped", "simply_supported"):
            raise ValueError("边界类型无效: " + edge_name)
        values = {"u3": 0.0}
        if mode == "clamped":
            values.update(ur1=0.0, ur2=0.0)
        model.DisplacementBC(name="BC_" + edge_name, createStepName="Initial", region=region, **values)
    bottom = set(node.label for node in edge_nodes["Y0"])
    for name, edge, values in (("Origin", "X0", {"u1": 0.0, "u2": 0.0}),
                               ("Axis", "X1", {"u2": 0.0})):
        labels = tuple(node.label for node in edge_nodes[edge] if node.label in bottom)
        if len(labels) != 1:
            raise ValueError("面内锚点必须对应一个角点: " + name)
        region = assembly.Set(name="Anchor_" + name, nodes=skin_instance.nodes.sequenceFromLabels(labels=labels))
        model.DisplacementBC(name="BC_Anchor_" + name, createStepName="Initial", region=region, **values)

    model.BuckleStep(name="Buckling", previous="Initial", numEigen=max(spec["buckling_modes"], 3),
                     vectors=max(spec["buckling_modes"] * 2, 10), maxIterations=200)
    forces = edge_nodal_forces(edge_nodes, spec["preload_nx_kN_per_m"], spec["preload_nxy_kN_per_m"])
    for label, components in sorted(forces.items()):
        if components == [0.0, 0.0]:
            continue
        region = regionToolset.Region(nodes=skin_instance.nodes.sequenceFromLabels(labels=(label,)))
        model.ConcentratedForce(name="EdgeForce_%d" % label, createStepName="Buckling", region=region,
                                cf1=components[0], cf2=components[1], distributionType=UNIFORM)


def detect_error_type(message):
    lowered = (message or "").lower()
    if "too many attempts" in lowered or "mesh" in lowered:
        return "mesh_error"
    if "assembly" in lowered or "boolean merge" in lowered or "geometry" in lowered:
        return "geometry_issue"
    if "eigen" in lowered and "negative" in lowered:
        return "blf_negative"
    if "converg" in lowered:
        return "convergence_fail"
    return "process_crash"


def load_runtime_extract_module(root_dir):
    module_path = os.path.join(root_dir, "abaqus", "runtime_extract_blf.py")
    return imp.load_source("csdm_runtime_extract_blf", module_path)


def build_panel(input_json, result_json):
    candidate = read_json(input_json)
    spec = resolve_analysis_spec(candidate)
    candidate_id = spec["candidate_id"]
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    try:
        from abaqus import Mdb
        from abaqusConstants import (
            ANALYSIS,
            CARTESIAN,
            DEFORMABLE_BODY,
            LAMINA,
            FREE,
            FROM_SECTION,
            MIDDLE_SURFACE,
            ON,
            PERCENTAGE,
            QUAD,
            S4R,
            STANDARD,
            THREE_D,
        )
        from mesh import ElemType
        import interaction  # noqa: F401  初始化 Abaqus 交互模块，供装配 Tie 使用。
        import regionToolset
    except Exception as exc:
        write_failure_result(candidate_id, result_json, "process_crash", "ABAQUS 模块导入失败: %s" % exc)
        return

    try:
        runtime_extract = load_runtime_extract_module(root_dir)

        mdb = Mdb()
        model_name = str("CSDM_panel_%s" % candidate_id)
        model = mdb.Model(name=model_name)
        if "Model-1" in mdb.models:
            del mdb.models["Model-1"]

        material_name = "LAMINA_MATERIAL"
        material = model.Material(name=material_name)
        material.Density(table=((spec["material"]["density_kg_per_m3"] * 1e-12,),))
        material.Elastic(
            type=LAMINA,
            table=(
                (
                    spec["material"]["E1_GPa"] * 1000.0,
                    spec["material"]["E2_GPa"] * 1000.0,
                    spec["material"]["nu12"],
                    spec["material"]["G12_GPa"] * 1000.0,
                    spec["material"]["G13_GPa"] * 1000.0,
                    spec["material"]["G23_GPa"] * 1000.0,
                ),
            ),
        )

        create_composite_sections(model, spec, material_name)

        panel_length = spec["panel_length_mm"]
        panel_width = spec["panel_width_mm"]
        stiffener_height = spec["stiffener_height_mm"]
        flange_width = spec["flange_width_mm"]
        stiffener_type = spec.get("stiffener_type", "T")
        half_flange_width = flange_width / 2.0
        tol = max(panel_length, panel_width) * 1e-7

        skin_sketch = model.ConstrainedSketch(name="SkinSketch", sheetSize=max(panel_length, panel_width) * 2.0)
        skin_sketch.rectangle(point1=(0.0, 0.0), point2=(panel_length, panel_width))
        skin_part = model.Part(name="SkinPanel", dimensionality=THREE_D, type=DEFORMABLE_BODY)
        skin_part.BaseShell(sketch=skin_sketch)
        del skin_sketch

        if stiffener_type != "BLADE":
            partition_sketch = model.ConstrainedSketch(name="SkinPartitionSketch", sheetSize=max(panel_length, panel_width) * 2.0)
            partition_y_positions = []
            for position_mm in spec["stiffener_positions_mm"]:
                if stiffener_type == "HAT":
                    cap_width = spec["cap_width_mm"]
                    half_cap_width = cap_width / 2.0
                    foot_width = half_flange_width - half_cap_width
                    y_values = (
                        position_mm - half_flange_width - foot_width,
                        position_mm - half_flange_width,
                        position_mm - half_cap_width,
                        position_mm + half_cap_width,
                        position_mm + half_flange_width,
                        position_mm + half_flange_width + foot_width,
                    )
                else:
                    y_values = (position_mm - half_flange_width, position_mm, position_mm + half_flange_width)
                for y_value in y_values:
                    if 0.0 <= y_value <= panel_width:
                        partition_y_positions.append(round(y_value, 6))
            for y_value in sorted(set(partition_y_positions)):
                partition_sketch.Line(point1=(0.0, y_value), point2=(panel_length, y_value))
            skin_part.PartitionFaceBySketch(faces=skin_part.faces, sketch=partition_sketch)
            del partition_sketch

        web_sketch = model.ConstrainedSketch(name="WebSketch", sheetSize=max(panel_length, stiffener_height) * 2.0)
        if stiffener_type == "HAT":
            cap_width = spec["cap_width_mm"]
            foot_width = (flange_width - cap_width) / 2.0
            incline_len = (foot_width ** 2 + stiffener_height ** 2) ** 0.5
            web_sketch.rectangle(point1=(0.0, 0.0), point2=(panel_length, incline_len))
        else:
            web_sketch.rectangle(point1=(0.0, 0.0), point2=(panel_length, stiffener_height))
        web_part = model.Part(name="WebPart", dimensionality=THREE_D, type=DEFORMABLE_BODY)
        web_part.BaseShell(sketch=web_sketch)
        del web_sketch

        flange_part = None
        cap_part = None
        if stiffener_type != "BLADE":
            if stiffener_type == "HAT":
                cap_width = spec["cap_width_mm"]
                foot_width = (flange_width - cap_width) / 2.0
                flange_half_width = foot_width
            elif stiffener_type == "L":
                flange_half_width = flange_width  # L 型单侧翼缘全宽
            else:
                flange_half_width = half_flange_width  # T 型半宽
            flange_sketch = model.ConstrainedSketch(name="FlangeHalfSketch", sheetSize=max(panel_length, flange_width) * 2.0)
            flange_sketch.rectangle(point1=(0.0, 0.0), point2=(panel_length, flange_half_width))
            flange_part = model.Part(name="FlangeHalfPart", dimensionality=THREE_D, type=DEFORMABLE_BODY)
            flange_part.BaseShell(sketch=flange_sketch)
            del flange_sketch

        if stiffener_type == "HAT":
            cap_width = spec["cap_width_mm"]
            cap_sketch = model.ConstrainedSketch(name="CapSketch", sheetSize=max(panel_length, cap_width) * 2.0)
            cap_sketch.rectangle(point1=(0.0, 0.0), point2=(panel_length, cap_width))
            cap_part = model.Part(name="CapPart", dimensionality=THREE_D, type=DEFORMABLE_BODY)
            cap_part.BaseShell(sketch=cap_sketch)
            del cap_sketch

        skin_part.SectionAssignment(
            region=regionToolset.Region(faces=skin_part.faces),
            sectionName="SkinSection",
            offset=0.0,
            offsetType=MIDDLE_SURFACE,
            offsetField="",
            thicknessAssignment=FROM_SECTION,
        )
        web_part.SectionAssignment(
            region=regionToolset.Region(faces=web_part.faces),
            sectionName="WebSection",
            offset=0.0,
            offsetType=MIDDLE_SURFACE,
            offsetField="",
            thicknessAssignment=FROM_SECTION,
        )
        if flange_part is not None:
            flange_part.SectionAssignment(
                region=regionToolset.Region(faces=flange_part.faces),
                sectionName="FlangeSection",
                offset=0.0,
                offsetType=MIDDLE_SURFACE,
                offsetField="",
                thicknessAssignment=FROM_SECTION,
            )
        if cap_part is not None:
            cap_part.SectionAssignment(
                region=regionToolset.Region(faces=cap_part.faces),
                sectionName="CapSection",
                offset=0.0,
                offsetType=MIDDLE_SURFACE,
                offsetField="",
                thicknessAssignment=FROM_SECTION,
            )

        elem_type = ElemType(elemCode=S4R, elemLibrary=STANDARD)
        mesh_parts = [skin_part, web_part]
        if flange_part is not None:
            mesh_parts.append(flange_part)
        if cap_part is not None:
            mesh_parts.append(cap_part)
        for part in mesh_parts:
            assign_material_orientation(part)
            part.setElementType(regions=(part.faces,), elemTypes=(elem_type,))
            part.seedPart(size=spec["mesh_size_mm"], deviationFactor=0.1, minSizeFactor=0.1)
            part.setMeshControls(regions=part.faces, elemShape=QUAD, technique=FREE)
            part.generateMesh()

        assembly = model.rootAssembly
        assembly.DatumCsysByDefault(CARTESIAN)
        skin_instance = assembly.Instance(name="Skin-1", part=skin_part, dependent=ON)

        for index, position_mm in enumerate(spec["stiffener_positions_mm"]):
            display_index = index + 1

            if stiffener_type == "BLADE":
                _assemble_blade(assembly, model, web_part, skin_instance, spec,
                                index, position_mm, display_index, tol)
            elif stiffener_type == "HAT":
                _assemble_hat(assembly, model, web_part, flange_part, cap_part,
                              skin_instance, spec, index, position_mm,
                              display_index, tol)
            elif stiffener_type == "L":
                _assemble_l(assembly, model, web_part, flange_part, skin_instance,
                            spec, index, position_mm, display_index, tol)
            else:
                _assemble_t(assembly, model, web_part, flange_part, skin_instance,
                            spec, index, position_mm, display_index, tol)

        # 几何容差只选择边界节点，不随网格尺寸扩大到邻近内部节点。
        edge_tol = max(panel_length, panel_width) * 1e-7
        edge_nodes = {}
        for edge_name, bounds in (
                ("X0", (0.0, 0.0, 0.0, panel_width)),
                ("X1", (panel_length, panel_length, 0.0, panel_width)),
                ("Y0", (0.0, panel_length, 0.0, 0.0)),
                ("Y1", (0.0, panel_length, panel_width, panel_width))):
            edge_nodes[edge_name] = skin_instance.nodes.getByBoundingBox(
                xMin=bounds[0] - edge_tol, xMax=bounds[1] + edge_tol,
                yMin=bounds[2] - edge_tol, yMax=bounds[3] + edge_tol,
                zMin=-edge_tol, zMax=edge_tol)
        apply_panel_boundary_and_load(model, assembly, skin_instance, spec, edge_nodes)
        job = mdb.Job(
            name=spec["job_name"],
            model=model_name,
            type=ANALYSIS,
            memory=85,
            memoryUnits=PERCENTAGE,
            numCpus=4,
            numDomains=4,
        )
        job.submit(consistencyChecking=ON)
        job.waitForCompletion()

        default_odb_path = os.path.join(os.getcwd(), "%s.odb" % spec["job_name"])
        runtime_extract.extract_blf(default_odb_path, result_json, input_json=input_json)
        result = read_json(result_json)
        result["analysis_contract"] = spec["analysis_contract"]
        write_json(result_json, result)
    except Exception:
        write_failure_result(candidate_id, result_json, detect_error_type(traceback.format_exc()), traceback.format_exc())
