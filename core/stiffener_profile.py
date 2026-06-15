"""筋条类型单点真理模块 —— 所有类型相关逻辑的唯一来源。

支持筋型：BLADE（板式筋）、T（T型筋）、HAT（帽型筋）、L（L型角材）。
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Any, Dict, List

from core.config_loader import load_param_ranges as _load_param_ranges_raw

# ═══════════════════════════════════════════════════════════════════
# 常量定义
# ═══════════════════════════════════════════════════════════════════

STIFFENER_TYPES = ["BLADE", "T", "HAT", "L"]

# 所有可能几何参数（超集）
ALL_GEOMETRY_PARAMS = [
    "panel_length_mm",
    "panel_width_mm",
    "skin_thickness_mm",
    "pitch_mm",
    "stiffener_height_mm",
    "web_thickness_mm",
    "flange_width_mm",
    "flange_thickness_mm",
    "cap_width_mm",
    "cap_thickness_mm",
]

# 每种类型的必需几何参数
REQUIRED_GEOMETRY_PARAMS: Dict[str, List[str]] = {
    "BLADE": [
        "panel_length_mm", "panel_width_mm", "skin_thickness_mm",
        "pitch_mm", "stiffener_height_mm", "web_thickness_mm",
    ],
    "T": [
        "panel_length_mm", "panel_width_mm", "skin_thickness_mm",
        "pitch_mm", "stiffener_height_mm", "web_thickness_mm",
        "flange_width_mm", "flange_thickness_mm",
    ],
    "HAT": [
        "panel_length_mm", "panel_width_mm", "skin_thickness_mm",
        "pitch_mm", "stiffener_height_mm", "web_thickness_mm",
        "flange_width_mm", "flange_thickness_mm",
        "cap_width_mm", "cap_thickness_mm",
    ],
    "L": [
        "panel_length_mm", "panel_width_mm", "skin_thickness_mm",
        "pitch_mm", "stiffener_height_mm", "web_thickness_mm",
        "flange_width_mm", "flange_thickness_mm",
    ],
}

# 每种类型的默认几何值
DEFAULT_GEOMETRY: Dict[str, Dict[str, float]] = {
    "BLADE": {
        "panel_length_mm": 700.0, "panel_width_mm": 600.0,
        "skin_thickness_mm": 2.5, "pitch_mm": 120.0,
        "stiffener_height_mm": 28.0, "web_thickness_mm": 2.0,
    },
    "T": {
        "panel_length_mm": 700.0, "panel_width_mm": 600.0,
        "skin_thickness_mm": 2.5, "pitch_mm": 120.0,
        "stiffener_height_mm": 28.0, "web_thickness_mm": 2.0,
        "flange_width_mm": 16.0, "flange_thickness_mm": 2.0,
    },
    "HAT": {
        "panel_length_mm": 700.0, "panel_width_mm": 600.0,
        "skin_thickness_mm": 2.5, "pitch_mm": 120.0,
        "stiffener_height_mm": 28.0, "web_thickness_mm": 2.0,
        "flange_width_mm": 40.0, "flange_thickness_mm": 2.0,
        "cap_width_mm": 20.0, "cap_thickness_mm": 2.0,
    },
    "L": {
        "panel_length_mm": 700.0, "panel_width_mm": 600.0,
        "skin_thickness_mm": 2.5, "pitch_mm": 120.0,
        "stiffener_height_mm": 28.0, "web_thickness_mm": 2.0,
        "flange_width_mm": 16.0, "flange_thickness_mm": 2.0,
    },
}

# 中文显示名
TYPE_DISPLAY_NAMES: Dict[str, str] = {
    "BLADE": "板式筋 (Blade)",
    "T": "T 型筋",
    "HAT": "帽型筋 (Hat)",
    "L": "L 型角材 (Angle)",
}

# 几何参数中文标签
GEOMETRY_LABELS: Dict[str, str] = {
    "panel_length_mm": "壁板长度 (mm)",
    "panel_width_mm": "壁板宽度 (mm)",
    "skin_thickness_mm": "蒙皮厚度 (mm)",
    "pitch_mm": "筋条间距 (mm)",
    "stiffener_height_mm": "筋条高度 (mm)",
    "web_thickness_mm": "腹板厚度 (mm)",
    "flange_width_mm": "翼缘宽度 (mm)",
    "flange_thickness_mm": "翼缘厚度 (mm)",
    "cap_width_mm": "帽顶宽度 (mm)",
    "cap_thickness_mm": "帽顶厚度 (mm)",
}

# 中文别名 → 标准类型名
_TYPE_ALIASES: Dict[str, str] = {
    "板式": "BLADE", "板式筋": "BLADE", "blade": "BLADE",
    "平板筋": "BLADE", "刀型": "BLADE", "刀型筋": "BLADE",
    "t型": "T", "t 型": "T", "t形": "T", "t": "T",
    "t型筋": "T", "t 型筋": "T", "t形筋": "T",
    "帽型": "HAT", "帽形": "HAT", "帽型筋": "HAT",
    "帽形筋": "HAT", "帽加筋": "HAT", "帽筋": "HAT",
    "hat": "HAT", "hat型": "HAT", "槽型": "HAT", "槽型筋": "HAT",
    "l型": "L", "l 型": "L", "l形": "L", "l": "L",
    "l型角材": "L", "角材": "L", "角型": "L", "角型筋": "L",
}


# ═══════════════════════════════════════════════════════════════════
# 类型验证与标准化
# ═══════════════════════════════════════════════════════════════════

def validate_stiffener_type(name: str) -> str:
    """标准化类型名。接受中文别名，返回大写英文标准名。未知值抛出 ValueError。"""
    if not name:
        return "T"
    key = str(name).strip()
    direct = key.upper()
    if direct in STIFFENER_TYPES:
        return direct
    lowered = key.lower()
    if lowered in _TYPE_ALIASES:
        return _TYPE_ALIASES[lowered]
    for alias, stype in _TYPE_ALIASES.items():
        if alias in lowered:
            return stype
    raise ValueError(
        f"未知筋型 '{name}'，支持的筋型：{', '.join(STIFFENER_TYPES)}"
    )


def resolve_stiffener_type(raw: str | None) -> str:
    """宽松解析：未知值回退为 'T' 而非报错。"""
    try:
        return validate_stiffener_type(raw or "T")
    except ValueError:
        return "T"


# ═══════════════════════════════════════════════════════════════════
# 参数管理
# ═══════════════════════════════════════════════════════════════════

def required_geometry_params(stype: str) -> List[str]:
    """返回该类型必需的几何参数列表。"""
    return list(REQUIRED_GEOMETRY_PARAMS.get(validate_stiffener_type(stype), REQUIRED_GEOMETRY_PARAMS["T"]))


def all_geometry_params() -> List[str]:
    """返回所有可能几何参数（超集）。"""
    return list(ALL_GEOMETRY_PARAMS)


def default_geometry(stype: str) -> Dict[str, float]:
    """返回该类型的默认几何值。"""
    return dict(DEFAULT_GEOMETRY.get(validate_stiffener_type(stype), DEFAULT_GEOMETRY["T"]))


def normalize_geometry(stype: str, raw_geom: Dict | None) -> Dict[str, float]:
    """补全默认值并验证必需键存在。"""
    stype = validate_stiffener_type(stype)
    defaults = default_geometry(stype)
    geometry = dict(defaults)
    if isinstance(raw_geom, dict):
        for key in defaults:
            if key in raw_geom and raw_geom[key] is not None:
                try:
                    geometry[key] = float(raw_geom[key])
                except (TypeError, ValueError):
                    pass
    return geometry


# ═══════════════════════════════════════════════════════════════════
# 筋条位置计算（从 render_utils + runtime_build_panel 提取的统一实现）
# ═══════════════════════════════════════════════════════════════════

def stiffener_positions(panel_width_mm: float, pitch_mm: float) -> List[float]:
    """计算筋条在面板宽度方向的位置列表。"""
    if pitch_mm <= 0:
        return [panel_width_mm / 2.0]
    count = max(1, int(round(panel_width_mm / pitch_mm)))
    if count == 1:
        return [panel_width_mm / 2.0]
    occupied = pitch_mm * (count - 1)
    edge_margin = max((panel_width_mm - occupied) / 2.0, 0.0)
    return [edge_margin + index * pitch_mm for index in range(count)]


# ═══════════════════════════════════════════════════════════════════
# 参数范围加载
# ═══════════════════════════════════════════════════════════════════

@lru_cache(maxsize=8)
def load_param_ranges_for_type(stype: str) -> Dict[str, Dict[str, float]]:
    """合并 common + 类型专属参数范围，返回 {param_name: {min, max}}。"""
    stype = validate_stiffener_type(stype)
    all_ranges = _load_param_ranges_raw()
    merged: Dict[str, Dict[str, float]] = {}
    common = all_ranges.get("common", {})
    if isinstance(common, dict):
        for key, value in common.items():
            if isinstance(value, dict) and "min" in value and "max" in value:
                merged[key] = {"min": float(value["min"]), "max": float(value["max"])}
    type_specific = all_ranges.get(stype, {})
    if isinstance(type_specific, dict):
        for key, value in type_specific.items():
            if isinstance(value, dict) and "min" in value and "max" in value:
                merged[key] = {"min": float(value["min"]), "max": float(value["max"])}
    return merged


# ═══════════════════════════════════════════════════════════════════
# ML 特征向量转换
# ═══════════════════════════════════════════════════════════════════

# 标准 8 几何特征顺序（与 FEATURE_ORDER 前 8 一致）
CANONICAL_GEOMETRY_ORDER = [
    "panel_length_mm",
    "panel_width_mm",
    "skin_thickness_mm",
    "pitch_mm",
    "stiffener_height_mm",
    "web_thickness_mm",
    "flange_width_mm",
    "flange_thickness_mm",
]


def geometry_to_feature_vector(geometry: Dict) -> List[float]:
    """将 geometry dict 转为固定 8 浮点列表，缺失参数填 0.0。"""
    return [float(geometry.get(key, 0.0)) for key in CANONICAL_GEOMETRY_ORDER]


# ═══════════════════════════════════════════════════════════════════
# 规则检查参数
# ═══════════════════════════════════════════════════════════════════

def rule_check_param_keys(stype: str) -> List[str]:
    """返回该类型需要范围检查的几何参数键列表。"""
    return list(REQUIRED_GEOMETRY_PARAMS.get(validate_stiffener_type(stype), []))


def solver_safe_window_keys(stype: str) -> Dict[str, tuple]:
    """返回该类型的求解安全区检查（参数名 → (min, max)）。"""
    base = {
        "solver_skin_thickness_ok": (1.2, 3.2),
        "solver_pitch_ok": (90.0, 140.0),
        "solver_height_ok": (18.0, 38.0),
        "solver_height_pitch_band_ok": (0.14, 0.34),
    }
    stype = validate_stiffener_type(stype)
    if stype == "BLADE":
        return base
    if stype == "HAT":
        base["solver_flange_width_ok"] = (24.0, 50.0)
    else:
        base["solver_flange_width_ok"] = (12.0, 22.0)
    return base


def hat_incline_angle_deg(flange_width: float, cap_width: float, stiffener_height: float) -> float:
    """计算 HAT 型腹板倾斜角度（度）。"""
    web_run = (flange_width - cap_width) / 2.0
    if web_run <= 0 or stiffener_height <= 0:
        return 0.0
    return math.degrees(math.atan(stiffener_height / web_run))


# ═══════════════════════════════════════════════════════════════════
# 文本描述生成
# ═══════════════════════════════════════════════════════════════════

def _safe_number(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def describe_geometry_text(stype: str, geometry: Dict, layup: Dict | None = None) -> str:
    """生成人类可读的几何摘要（用于 case_memory 检索文本）。"""
    stype = resolve_stiffener_type(stype)
    parts = [
        f"L={_safe_number(geometry.get('panel_length_mm')):.1f}mm",
        f"W={_safe_number(geometry.get('panel_width_mm')):.1f}mm",
        f"t_skin={_safe_number(geometry.get('skin_thickness_mm')):.2f}mm",
        f"pitch={_safe_number(geometry.get('pitch_mm')):.1f}mm",
        f"h={_safe_number(geometry.get('stiffener_height_mm')):.1f}mm",
        f"t_web={_safe_number(geometry.get('web_thickness_mm')):.2f}mm",
    ]
    if stype in ("T", "L"):
        parts.append(f"b_flange={_safe_number(geometry.get('flange_width_mm')):.1f}mm")
        parts.append(f"t_flange={_safe_number(geometry.get('flange_thickness_mm')):.2f}mm")
    elif stype == "HAT":
        parts.append(f"b_flange={_safe_number(geometry.get('flange_width_mm')):.1f}mm")
        parts.append(f"t_flange={_safe_number(geometry.get('flange_thickness_mm')):.2f}mm")
        parts.append(f"cap_w={_safe_number(geometry.get('cap_width_mm')):.1f}mm")
        parts.append(f"cap_t={_safe_number(geometry.get('cap_thickness_mm')):.2f}mm")
    return ", ".join(parts)


# ═══════════════════════════════════════════════════════════════════
# ABAQUS 部件规格生成
# ═══════════════════════════════════════════════════════════════════

def build_stiffener_part_specs(
    stype: str, geometry: Dict, positions: List[float] | None = None
) -> List[Dict]:
    """生成 ABAQUS 建模所需的部件规格列表。

    每个 spec 描述一个需要创建的 shell 部件实例：
      - part_type: "web" | "flange_half" | "cap_half"
      - width_mm: Y 方向宽度
      - height_mm: Z 方向高度（腹板）或 0（水平件）
      - thickness_mm: 截面厚度
      - section_name: 截面名称
      - rotation_degrees: 绕 X 轴旋转角度（0=水平, 90=垂直）
      - y_offset_mm: 从筋条中心线的 Y 偏移
      - z_offset_mm: Z 偏移（蒙皮表面以上）

    返回列表包含每根筋条所有部件的规格。
    """
    stype = validate_stiffener_type(stype)
    panel_length = float(geometry.get("panel_length_mm", 700.0))
    panel_width = float(geometry.get("panel_width_mm", 600.0))
    skin_t = float(geometry.get("skin_thickness_mm", 2.5))
    pitch = float(geometry.get("pitch_mm", 120.0))
    height = float(geometry.get("stiffener_height_mm", 28.0))
    web_t = float(geometry.get("web_thickness_mm", 2.0))

    if positions is None:
        positions = stiffener_positions(panel_width, pitch)

    specs: List[Dict] = []

    if stype == "BLADE":
        for pos in positions:
            specs.append({
                "part_type": "web",
                "width_mm": web_t,
                "height_mm": height,
                "thickness_mm": web_t,
                "section_name": "WebSection",
                "rotation_degrees": 90.0,
                "y_offset_mm": pos,
                "z_offset_mm": skin_t,
            })

    elif stype == "T":
        flange_w = float(geometry.get("flange_width_mm", 16.0))
        flange_t = float(geometry.get("flange_thickness_mm", 2.0))
        half_w = flange_w / 2.0
        for pos in positions:
            specs.append({
                "part_type": "web",
                "width_mm": web_t,
                "height_mm": height,
                "thickness_mm": web_t,
                "section_name": "WebSection",
                "rotation_degrees": 90.0,
                "y_offset_mm": pos,
                "z_offset_mm": skin_t + flange_t,
            })
            for side, y_off in [("left", pos - half_w), ("right", pos + half_w)]:
                specs.append({
                    "part_type": "flange_half",
                    "width_mm": half_w,
                    "height_mm": 0.0,
                    "thickness_mm": flange_t,
                    "section_name": "FlangeSection",
                    "rotation_degrees": 0.0,
                    "y_offset_mm": y_off,
                    "z_offset_mm": skin_t,
                    "side": side,
                })

    elif stype == "HAT":
        flange_w = float(geometry.get("flange_width_mm", 40.0))
        flange_t = float(geometry.get("flange_thickness_mm", 2.0))
        cap_w = float(geometry.get("cap_width_mm", 20.0))
        cap_t = float(geometry.get("cap_thickness_mm", 2.0))
        half_bottom = flange_w / 2.0
        half_cap = cap_w / 2.0
        foot_w = half_bottom - half_cap
        if foot_w <= 0:
            foot_w = 1.0
        incline_length = math.sqrt(foot_w ** 2 + height ** 2)
        incline_angle = math.degrees(math.atan2(height, foot_w))

        for pos in positions:
            specs.append({
                "part_type": "web_left",
                "width_mm": web_t,
                "height_mm": incline_length,
                "thickness_mm": web_t,
                "section_name": "WebSection",
                "rotation_degrees": 90.0 - incline_angle,
                "y_offset_mm": pos - (half_bottom + half_cap) / 2.0,
                "z_offset_mm": skin_t + flange_t + height / 2.0,
            })
            specs.append({
                "part_type": "web_right",
                "width_mm": web_t,
                "height_mm": incline_length,
                "thickness_mm": web_t,
                "section_name": "WebSection",
                "rotation_degrees": -(90.0 - incline_angle),
                "y_offset_mm": pos + (half_bottom + half_cap) / 2.0,
                "z_offset_mm": skin_t + flange_t + height / 2.0,
            })
            specs.append({
                "part_type": "cap",
                "width_mm": cap_w,
                "height_mm": 0.0,
                "thickness_mm": cap_t,
                "section_name": "CapSection",
                "rotation_degrees": 0.0,
                "y_offset_mm": pos,
                "z_offset_mm": skin_t + flange_t + height,
            })
            for side, y_off in [
                ("left", pos - half_bottom - foot_w),
                ("right", pos + half_bottom),
            ]:
                specs.append({
                    "part_type": "flange_half",
                    "width_mm": foot_w,
                    "height_mm": 0.0,
                    "thickness_mm": flange_t,
                    "section_name": "FlangeSection",
                    "rotation_degrees": 0.0,
                    "y_offset_mm": y_off,
                    "z_offset_mm": skin_t,
                    "side": side,
                })

    elif stype == "L":
        flange_w = float(geometry.get("flange_width_mm", 16.0))
        flange_t = float(geometry.get("flange_thickness_mm", 2.0))
        for pos in positions:
            specs.append({
                "part_type": "web",
                "width_mm": web_t,
                "height_mm": height,
                "thickness_mm": web_t,
                "section_name": "WebSection",
                "rotation_degrees": 90.0,
                "y_offset_mm": pos,
                "z_offset_mm": skin_t + flange_t,
            })
            specs.append({
                "part_type": "flange_half",
                "width_mm": flange_w,
                "height_mm": 0.0,
                "thickness_mm": flange_t,
                "section_name": "FlangeSection",
                "rotation_degrees": 0.0,
                "y_offset_mm": pos + flange_w / 2.0,
                "z_offset_mm": skin_t,
                "side": "right",
            })

    return specs


# ═══════════════════════════════════════════════════════════════════
# 3D 可视化网格生成
# ═══════════════════════════════════════════════════════════════════

def build_stiffener_meshes(
    stype: str, geometry: Dict, positions: List[float]
) -> List[tuple]:
    """返回 (mesh, style_dict) 列表，供 PyVista 渲染。

    每种筋型返回不同颜色和结构的网格以正确表示几何形状。
    """
    import numpy as np

    try:
        import pyvista as pv
    except ImportError:
        return []

    stype = resolve_stiffener_type(stype)
    panel_length = float(geometry.get("panel_length_mm", 700.0))
    skin_t = float(geometry.get("skin_thickness_mm", 2.5))
    height = float(geometry.get("stiffener_height_mm", 28.0))
    web_t = max(float(geometry.get("web_thickness_mm", 2.0)), 0.5)

    meshes: List[tuple] = []

    if stype == "BLADE":
        for idx, pos in enumerate(positions, start=1):
            web = pv.Box(bounds=(
                0.0, panel_length,
                pos - web_t / 2.0, pos + web_t / 2.0,
                skin_t, skin_t + height,
            ))
            meshes.append((web, {"color": "#d88c5a", "smooth_shading": True, "name": f"web_{idx}"}))

    elif stype == "T":
        flange_w = float(geometry.get("flange_width_mm", 16.0))
        flange_t = max(float(geometry.get("flange_thickness_mm", 2.0)), 0.5)
        for idx, pos in enumerate(positions, start=1):
            flange = pv.Box(bounds=(
                0.0, panel_length,
                pos - flange_w / 2.0, pos + flange_w / 2.0,
                skin_t, skin_t + flange_t,
            ))
            web = pv.Box(bounds=(
                0.0, panel_length,
                pos - web_t / 2.0, pos + web_t / 2.0,
                skin_t + flange_t, skin_t + flange_t + height,
            ))
            meshes.append((flange, {"color": "#e3b36a", "smooth_shading": True, "name": f"flange_{idx}"}))
            meshes.append((web, {"color": "#d88c5a", "smooth_shading": True, "name": f"web_{idx}"}))

    elif stype == "HAT":
        flange_w = float(geometry.get("flange_width_mm", 40.0))
        flange_t = max(float(geometry.get("flange_thickness_mm", 2.0)), 0.5)
        cap_w = float(geometry.get("cap_width_mm", 20.0))
        cap_t = max(float(geometry.get("cap_thickness_mm", 2.0)), 0.5)
        half_flange = flange_w / 2.0
        half_cap = cap_w / 2.0
        foot_w = half_flange - half_cap
        if foot_w <= 0:
            foot_w = 1.0
        for idx, pos in enumerate(positions, start=1):
            # 左外侧底部连接板
            left_flange = pv.Box(bounds=(
                0.0, panel_length,
                pos - half_flange - foot_w, pos - half_flange,
                skin_t, skin_t + flange_t,
            ))
            # 右外侧底部连接板
            right_flange = pv.Box(bounds=(
                0.0, panel_length,
                pos + half_flange, pos + half_flange + foot_w,
                skin_t, skin_t + flange_t,
            ))
            # 顶帽
            cap = pv.Box(bounds=(
                0.0, panel_length,
                pos - half_cap, pos + half_cap,
                skin_t + flange_t + height, skin_t + flange_t + height + cap_t,
            ))
            # 左斜腹板（用 PolyData 四边形）
            left_web = _hat_web_mesh(
                panel_length, pos - half_flange, pos - half_cap,
                skin_t + flange_t, skin_t + flange_t + height,
            )
            # 右斜腹板
            right_web = _hat_web_mesh(
                panel_length, pos + half_flange, pos + half_cap,
                skin_t + flange_t, skin_t + flange_t + height,
            )
            meshes.append((left_flange, {"color": "#e3b36a", "smooth_shading": True, "name": f"flange_L_{idx}"}))
            meshes.append((right_flange, {"color": "#e3b36a", "smooth_shading": True, "name": f"flange_R_{idx}"}))
            meshes.append((left_web, {"color": "#d88c5a", "smooth_shading": True, "name": f"web_L_{idx}"}))
            meshes.append((right_web, {"color": "#d88c5a", "smooth_shading": True, "name": f"web_R_{idx}"}))
            meshes.append((cap, {"color": "#c17d4a", "smooth_shading": True, "name": f"cap_{idx}"}))

    elif stype == "L":
        flange_w = float(geometry.get("flange_width_mm", 16.0))
        flange_t = max(float(geometry.get("flange_thickness_mm", 2.0)), 0.5)
        for idx, pos in enumerate(positions, start=1):
            flange = pv.Box(bounds=(
                0.0, panel_length,
                pos, pos + flange_w,
                skin_t, skin_t + flange_t,
            ))
            web = pv.Box(bounds=(
                0.0, panel_length,
                pos - web_t / 2.0, pos + web_t / 2.0,
                skin_t + flange_t, skin_t + flange_t + height,
            ))
            meshes.append((flange, {"color": "#e3b36a", "smooth_shading": True, "name": f"flange_{idx}"}))
            meshes.append((web, {"color": "#d88c5a", "smooth_shading": True, "name": f"web_{idx}"}))

    return meshes


def _hat_web_mesh(
    length: float, y_bottom: float, y_top: float, z_bottom: float, z_top: float
):
    """为 HAT 斜腹板创建四边形 PolyData 网格。"""
    import numpy as np
    import pyvista as pv

    points = np.array([
        [0.0, y_bottom, z_bottom],
        [length, y_bottom, z_bottom],
        [length, y_top, z_top],
        [0.0, y_top, z_top],
    ])
    faces = np.array([[4, 0, 1, 2, 3]])
    return pv.PolyData(points, faces)
