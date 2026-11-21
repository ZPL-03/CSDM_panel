"""候选方案规则检查器。"""

from __future__ import annotations

import math
from typing import Dict, List

from core.config_loader import load_param_ranges
from core.laminate import section_layers
from core.layup import normalize_layup
from core.task_contract import build_design_constraints, validate_candidate_constraints
from core.stiffener_profile import (
    hat_incline_angle_deg,
    load_param_ranges_for_type,
    rule_check_param_keys,
    resolve_stiffener_type,
    stiffener_footprints,
)


class RuleChecker:
    """检查任务几何范围、物理铺层与采样比例约束。"""

    def __init__(self) -> None:
        self.param_ranges = load_param_ranges()

    def _check_range(self, key: str, value: float, errors: List[str], details: Dict[str, bool],
                     type_ranges: Dict) -> None:
        rule = type_ranges.get(key, {})
        if not math.isfinite(value) or value <= 0:
            errors.append(f"{key} 必须为有限正数")
            details[f"{key}_ok"] = False
            return
        if rule.get("min") is not None and value < rule["min"]:
            errors.append(f"{key} 小于最小值 {rule['min']}")
            details[f"{key}_ok"] = False
            return
        if rule.get("max") is not None and value > rule["max"]:
            errors.append(f"{key} 大于最大值 {rule['max']}")
            details[f"{key}_ok"] = False
            return
        details[f"{key}_ok"] = True

    def _layup_text(self, layup_value: object) -> str:
        if isinstance(layup_value, str):
            return layup_value
        if isinstance(layup_value, (list, tuple)):
            items = [str(item).strip() for item in layup_value if str(item).strip()]
            if not items:
                return ""
            suffix = ""
            if items[-1].lower() == "s":
                suffix = "s"
                items = items[:-1]
            return f"[{'/'.join(items)}]{suffix}"
        return str(layup_value or "")

    def run(self, candidate: Dict, stiffener_type: str | None = None,
            task: Dict | None = None) -> Dict:
        geometry = candidate.get("geometry", {})
        layup = candidate.get("layup", {})
        errors: List[str] = []
        suggestions: List[str] = []
        details: Dict[str, bool] = {}

        stype = resolve_stiffener_type(stiffener_type or candidate.get("stiffener_type"))
        constraints = build_design_constraints(task, stype) if task is not None else candidate.get("task_constraints")
        type_ranges = constraints["geometry_ranges"] if constraints else load_param_ranges_for_type(stype)
        if constraints:
            try:
                validate_candidate_constraints(candidate, constraints)
            except ValueError as exc:
                errors.append(str(exc))
        for key in rule_check_param_keys(stype):
            try:
                value = float(geometry[key])
            except (KeyError, TypeError, ValueError):
                errors.append(f"{key} 缺失或不是数值")
                details[f"{key}_ok"] = False
                continue
            self._check_range(key, value, errors, details, type_ranges)

        layup_rules = constraints["layup_constraints"] if constraints else self.param_ranges["layup"]
        try:
            normalized = normalize_layup(self._layup_text(layup.get("skin_layup", "")))
            for key in ("skin_f0", "skin_f45", "skin_f90"):
                if key in layup and not math.isclose(float(layup[key]), normalized[key], abs_tol=0.002):
                    errors.append(f"{key} 与物理层序不一致")
            sections = section_layers({**candidate, "stiffener_type": stype})
            details["symmetric"] = all(layers == list(reversed(layers)) for layers in sections.values())
            details["balanced"] = all(math.isclose(
                sum(p["thickness_mm"] for p in layers if p["angle_deg"] == 45),
                sum(p["thickness_mm"] for p in layers if p["angle_deg"] == -45), abs_tol=1e-9)
                for layers in sections.values())
            allowed = set(layup_rules["allowed_angles"])
            if any(p["angle_deg"] not in allowed for layers in sections.values() for p in layers):
                errors.append("物理铺层使用了任务不允许的角度")
            for property_name in ("symmetric", "balanced"):
                if layup_rules.get(property_name) and not details[property_name]:
                    errors.append(f"物理铺层不满足 {property_name} 约束")
            min_ratio = float(layup_rules["min_ratio_per_angle"])
            details["min_ratio_ok"] = all(normalized[key] >= min_ratio for key in ("skin_f0", "skin_f45", "skin_f90"))
            if not details["min_ratio_ok"]:
                errors.append("铺层角度族比例低于任务最小值")
        except (ValueError, KeyError, TypeError) as exc:
            errors.append(f"物理铺层无效: {exc}")

        try:
            stiffener_footprints(stype, geometry)
            details["footprint_ok"] = True
        except (KeyError, TypeError, ValueError) as exc:
            details["footprint_ok"] = False
            errors.append(f"截面几何无效: {exc}")

        # HAT 的比例与倾角是配置设计约束，截面不重叠和 cap < flange 是几何条件。
        if stype == "HAT" and details["footprint_ok"]:
            design_rules = self.param_ranges.get("HAT_constraints", {})
            flange_w, cap_w, height = (float(geometry[key]) for key in
                                      ("flange_width_mm", "cap_width_mm", "stiffener_height_mm"))
            if "cap_to_flange_width_max" in design_rules:
                maximum = float(design_rules["cap_to_flange_width_max"])
                if not math.isfinite(maximum) or not 0 < maximum <= 1:
                    raise ValueError("HAT_constraints.cap_to_flange_width_max 必须在 (0,1] 内")
                details["hat_cap_to_flange_ok"] = cap_w / flange_w <= maximum
                if not details["hat_cap_to_flange_ok"]:
                    errors.append(f"HAT 帽顶/底部跨度比例超过配置上限 {maximum:g}")
            if "web_incline_min_deg" in design_rules:
                minimum = float(design_rules["web_incline_min_deg"])
                if not math.isfinite(minimum) or not 0 <= minimum < 90:
                    raise ValueError("HAT_constraints.web_incline_min_deg 必须在 [0,90) 内")
                details["hat_web_incline_ok"] = hat_incline_angle_deg(flange_w, cap_w, height) >= minimum
                if not details["hat_web_incline_ok"]:
                    errors.append(f"HAT 腹板倾角小于配置下限 {minimum:g} 度")

        return {
            "is_valid": not errors,
            "errors": errors,
            "suggestions": suggestions,
            "details": details,
        }
