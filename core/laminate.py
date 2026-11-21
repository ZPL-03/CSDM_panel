# -*- coding: utf-8 -*-
"""物理铺层、截面与求解来源契约；兼容 Abaqus Python 2.7/3。"""

from __future__ import division, unicode_literals

import hashlib
import json
import math
import os
import re


ANALYSIS_VERSION = "panel-composite-shell-v2"
COMPONENT_THICKNESS = {
    "skin": "skin_thickness_mm", "web": "web_thickness_mm",
    "flange": "flange_thickness_mm", "cap": "cap_thickness_mm",
}


def finite_number(value):
    number = float(value)
    if math.isnan(number) or math.isinf(number):
        raise ValueError("数值必须有限")
    return number


def stiffener_positions(panel_width_mm, pitch_mm):
    """按给定间距居中布筋，筋数采用一致的四舍五入规则。"""
    width, pitch = finite_number(panel_width_mm), finite_number(pitch_mm)
    if width <= 0 or pitch <= 0:
        raise ValueError("壁板宽度和筋距必须为有限正数")
    count = max(1, int(math.floor(width / pitch + 0.5)))
    margin = (width - pitch * (count - 1)) / 2.0
    return [margin + index * pitch for index in range(count)]


def fingerprint(value):
    content = json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def solver_fingerprint():
    """绑定截面、建模和提取代码；统一换行以支持 Windows/WSL 的同源校验。"""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sources = {}
    for relative in ("core/laminate.py", "abaqus/runtime_build_panel.py", "abaqus/runtime_extract_blf.py"):
        with open(os.path.join(root, relative), "rb") as stream:
            sources[relative] = hashlib.sha256(stream.read().replace(b"\r\n", b"\n")).hexdigest()
    return fingerprint(sources)


def parse_angles(value):
    """按壳法向从负厚度面到正厚度面展开物理层序，不推断缺失角度。"""
    text = re.sub(r"^skin_layup\s*[:：]\s*", "", value.strip(), flags=re.IGNORECASE)
    match = re.match(r"^\[([^\[\]]+)\]([sS]?)$", text)
    if not match:
        raise ValueError("铺层必须使用 [角度/角度/...] 或 [...]s 格式")
    base = []
    for token in match.group(1).split("/"):
        token = token.strip().replace("deg", "").replace("°", "")
        values = [float(token[1:]), -float(token[1:])] if token.startswith("±") else [float(token)]
        if any(angle not in (0.0, 45.0, -45.0, 90.0) for angle in values):
            raise ValueError("铺层角度只允许 0、45、-45、90 度")
        base.extend(values)
    return base + list(reversed(base)) if match.group(2) else base


def section_layers(candidate):
    """各部件可指定层序/层厚；未单列的部件继承 skin_layup 并等分部件总厚度。"""
    stype = candidate["stiffener_type"]
    if stype not in ("BLADE", "T", "L", "HAT"):
        raise ValueError("不支持的筋型")
    components = ["skin", "web"] + ([] if stype == "BLADE" else ["flange"])
    if stype == "HAT":
        components.append("cap")
    layup = candidate["layup"]
    sections = {}
    for component in components:
        angles = parse_angles(layup.get(component + "_layup", layup["skin_layup"]))
        total = finite_number(candidate["geometry"][COMPONENT_THICKNESS[component]])
        thicknesses = layup.get(component + "_ply_thicknesses_mm", [total / len(angles)] * len(angles))
        thicknesses = [finite_number(value) for value in thicknesses]
        if total <= 0 or len(thicknesses) != len(angles) or any(value <= 0 for value in thicknesses):
            raise ValueError("部件层厚必须为正且与物理层数一致: " + component)
        if abs(sum(thicknesses) - total) > max(1e-9, total * 1e-9):
            raise ValueError("各层厚度之和必须等于部件几何厚度: " + component)
        sections[component] = [{"angle_deg": angle, "thickness_mm": thickness}
                               for angle, thickness in zip(angles, thicknesses)]
    return sections


def lamina_material(candidate):
    """N/mm 单位制的单层弹性输入；横向剪切缺省采用明确的横观各向同性假设。"""
    source = candidate["material_system"]
    material = {key: finite_number(source[key]) for key in
                ("E1_GPa", "E2_GPa", "G12_GPa", "nu12", "density_kg_per_m3")}
    material["G13_GPa"] = finite_number(source.get("G13_GPa", material["G12_GPa"]))
    material["G23_GPa"] = finite_number(source.get("G23_GPa", material["E2_GPa"] / (2 * (1 + 0.35))))
    if any(value <= 0 for key, value in material.items() if key != "nu12"):
        raise ValueError("材料模量及密度必须为正")
    if not 0 <= material["nu12"] < 0.5:
        raise ValueError("nu12 必须在 [0, 0.5) 内")
    if 1 - material["nu12"] ** 2 * material["E2_GPa"] / material["E1_GPa"] <= 0:
        raise ValueError("单层面内刚度不满足正定条件")
    return material


def analysis_contract(candidate):
    """绑定实际层序、材料、几何、载荷和边界；历史均质结果不具有此版本。"""
    for key in ("panel_length_mm", "panel_width_mm", "pitch_mm", "stiffener_height_mm"):
        if finite_number(candidate["geometry"][key]) <= 0:
            raise ValueError("几何尺寸必须为有限正数: " + key)
    if candidate["stiffener_type"] != "BLADE" and finite_number(candidate["geometry"]["flange_width_mm"]) <= 0:
        raise ValueError("翼缘宽度必须为有限正数")
    if candidate["stiffener_type"] == "HAT":
        if finite_number(candidate["geometry"]["cap_width_mm"]) <= 0:
            raise ValueError("帽顶宽度必须为有限正数")
        if finite_number(candidate["geometry"]["cap_width_mm"]) >= finite_number(candidate["geometry"]["flange_width_mm"]):
            raise ValueError("HAT 底部跨度必须大于帽顶宽度")
    analysis = dict(candidate.get("analysis", {}))
    if "mesh_size_mm" in analysis and finite_number(analysis["mesh_size_mm"]) <= 0:
        raise ValueError("网格尺寸必须为有限正数")
    if "buckling_modes" in analysis:
        modes = finite_number(analysis["buckling_modes"])
        if modes < 1 or modes != int(modes):
            raise ValueError("屈曲模态数量必须为正整数")
    loads = candidate["load_conditions"]
    nx, nxy = [finite_number(loads[key]) for key in ("Nx_kN_per_m", "Nxy_kN_per_m")]
    if nx < 0 or nxy < 0 or nx + nxy <= 0:
        raise ValueError("参考载荷必须为非负数且不全为零")
    if loads["type"] not in ("axial_compression", "in_plane_shear", "compression_shear"):
        raise ValueError("不支持的载荷类型")
    if (loads["type"] == "axial_compression" and nxy != 0) or (loads["type"] == "in_plane_shear" and nx != 0):
        raise ValueError("载荷分量与类型不一致")
    if candidate["boundary_conditions"]["type"] not in ("SSSS", "CCCC", "SSCC"):
        raise ValueError("不支持的边界类型")
    payload = {
        "version": ANALYSIS_VERSION,
        "solver_fingerprint": solver_fingerprint(),
        "stiffener_type": candidate["stiffener_type"],
        "geometry": {key: finite_number(value) for key, value in candidate["geometry"].items()},
        "sections": section_layers(candidate),
        "material": lamina_material(candidate),
        "load_conditions": {"type": loads["type"], "Nx_kN_per_m": nx, "Nxy_kN_per_m": nxy},
        "boundary_type": candidate["boundary_conditions"]["type"],
        "boundary_definition": {"support": "skin_edge_transverse", "in_plane_anchors": "origin_xy_axis_y"},
        "load_definition": "self_equilibrated_skin_edge_tractions",
        "connection_definition": "fixed_midsurface_offset_ties",
        "analysis": analysis,
        "orientation": {"zero_axis": "part_X", "stack_axis": "part_normal",
                        "order": "negative_to_positive_normal", "offset": "MIDDLE_SURFACE"},
    }
    payload["fingerprint"] = fingerprint(payload)
    return payload


def areal_density(candidate):
    """按建模部件的中面面积和厚度计算面密度，单位 kg/m^2。"""
    geometry, stype = candidate["geometry"], candidate["stiffener_type"]
    width, pitch = float(geometry["panel_width_mm"]), float(geometry["pitch_mm"])
    height, web_t = float(geometry["stiffener_height_mm"]), float(geometry["web_thickness_mm"])
    count = len(stiffener_positions(width, pitch))
    area = height * web_t
    if stype in ("T", "L"):
        area += geometry["flange_width_mm"] * geometry["flange_thickness_mm"]
    elif stype == "HAT":
        flange_w, cap_w = geometry["flange_width_mm"], geometry["cap_width_mm"]
        foot = (flange_w - cap_w) / 2
        if foot <= 0:
            raise ValueError("HAT 底部跨度必须大于帽顶宽度")
        area = (2 * math.hypot(height, foot) * web_t + 2 * foot * geometry["flange_thickness_mm"]
                + cap_w * geometry["cap_thickness_mm"])
    elif stype != "BLADE":
        raise ValueError("不支持的筋型")
    return candidate["material_system"]["density_kg_per_m3"] * (geometry["skin_thickness_mm"] + count * area / width) * 1e-3


def laminate_abd(layers, material):
    """经典层合板积分，A/B/D 单位分别为 N/mm、N、N*mm。"""
    e1, e2, g = [material[key] * 1000.0 for key in ("E1_GPa", "E2_GPa", "G12_GPa")]
    nu = material["nu12"]
    denominator = 1 - nu * nu * e2 / e1
    q11, q22, q12, q66 = e1 / denominator, e2 / denominator, nu * e2 / denominator, g
    matrices = [[[0.0] * 3 for _ in range(3)] for _ in range(3)]
    z = -sum(layer["thickness_mm"] for layer in layers) / 2
    for layer in layers:
        theta = math.radians(layer["angle_deg"])
        m, n = math.cos(theta), math.sin(theta)
        qbar11 = q11*m**4 + 2*(q12+2*q66)*m*m*n*n + q22*n**4
        qbar22 = q11*n**4 + 2*(q12+2*q66)*m*m*n*n + q22*m**4
        qbar12 = (q11+q22-4*q66)*m*m*n*n + q12*(m**4+n**4)
        qbar16 = (q11-q12-2*q66)*m**3*n - (q22-q12-2*q66)*m*n**3
        qbar26 = (q11-q12-2*q66)*m*n**3 - (q22-q12-2*q66)*m**3*n
        qbar66 = (q11+q22-2*q12-2*q66)*m*m*n*n + q66*(m**4+n**4)
        qbar = [[qbar11, qbar12, qbar16], [qbar12, qbar22, qbar26], [qbar16, qbar26, qbar66]]
        upper = z + layer["thickness_mm"]
        for order, matrix in enumerate(matrices, 1):
            weight = (upper**order - z**order) / order
            for i in range(3):
                for j in range(3):
                    matrix[i][j] += qbar[i][j] * weight
        z = upper
    return matrices
