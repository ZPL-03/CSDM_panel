"""候选方案规则检查器。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from core.config_loader import load_param_ranges
from core.stiffener_profile import (
    hat_incline_angle_deg,
    load_param_ranges_for_type,
    rule_check_param_keys,
    resolve_stiffener_type,
)


@dataclass
class RuleCheckResult:
    is_valid: bool
    errors: List[str]
    suggestions: List[str]
    details: Dict[str, bool]


class RuleChecker:
    """执行一期固定范围内的几何与铺层规则校验。"""

    def __init__(self) -> None:
        self.param_ranges = load_param_ranges()

    def _check_range(self, key: str, value: float, errors: List[str], details: Dict[str, bool],
                     type_ranges: Dict) -> None:
        rule = type_ranges.get(key, {})
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

    def run(self, candidate: Dict, strict_solver_window: bool = False, stiffener_type: str = "T") -> Dict:
        geometry = candidate.get("geometry", {})
        layup = candidate.get("layup", {})
        errors: List[str] = []
        suggestions: List[str] = []
        details: Dict[str, bool] = {}

        stype = resolve_stiffener_type(stiffener_type)
        type_ranges = load_param_ranges_for_type(stype)
        for key in rule_check_param_keys(stype):
            self._check_range(key, float(geometry.get(key, 0.0)), errors, details, type_ranges)

        layup_text = self._layup_text(layup.get("skin_layup", ""))
        f0 = float(layup.get("skin_f0", 0.0))
        f45 = float(layup.get("skin_f45", 0.0))
        f90 = float(layup.get("skin_f90", 0.0))
        min_ratio = float(self.param_ranges["layup"]["min_ratio_per_angle"])

        details["symmetric"] = layup_text.endswith("s")
        details["balanced"] = f45 >= min_ratio
        details["min_ratio_ok"] = all(value >= min_ratio for value in [f0, f45, f90] if value > 0)

        if not details["symmetric"]:
            errors.append("铺层未标记为对称层合板")
            suggestions.append("将铺层格式改为 [...]s")
        if not details["balanced"]:
            errors.append("±45° 层比例过低，不满足平衡铺层要求")
            suggestions.append("提高 ±45° 铺层比例")
        if not details["min_ratio_ok"]:
            errors.append("至少一个铺层角度比例低于最小约束")
            suggestions.append("将各角度比例提高到 10% 以上")

        pitch = float(geometry.get("pitch_mm", 0.0))
        height = float(geometry.get("stiffener_height_mm", 0.0))
        skin_t = float(geometry.get("skin_thickness_mm", 1.0))
        details["height_pitch_ratio_ok"] = 0 < height / max(pitch, 1.0) <= 0.6
        details["height_thickness_ratio_ok"] = height / max(skin_t, 0.1) <= 35
        if not details["height_pitch_ratio_ok"]:
            errors.append("筋高/筋距比例超出一期合理范围")
            suggestions.append("减小筋高或增大筋距")
        if not details["height_thickness_ratio_ok"]:
            errors.append("筋高/蒙皮厚度比例过大")
            suggestions.append("提高蒙皮厚度或降低筋高")

        # HAT 专用几何约束
        if stype == "HAT":
            flange_w = float(geometry.get("flange_width_mm", 0.0))
            cap_w = float(geometry.get("cap_width_mm", 0.0))
            cap_flange_ratio = cap_w / max(flange_w, 0.1)
            details["hat_cap_to_flange_ok"] = cap_flange_ratio <= 0.8
            if not details["hat_cap_to_flange_ok"]:
                errors.append("HAT 帽顶宽度/翼缘宽度比例超过 0.8")
                suggestions.append("减小帽顶宽度或增大翼缘宽度")

            incline_deg = hat_incline_angle_deg(flange_w, cap_w, height)
            details["hat_web_incline_ok"] = incline_deg >= 20.0
            if not details["hat_web_incline_ok"]:
                if cap_w >= flange_w:
                    errors.append("HAT 翼缘宽度必须大于帽顶宽度")
                    suggestions.append("增大翼缘宽度或减小帽顶宽度")
                else:
                    errors.append(f"HAT 腹板倾角 {incline_deg:.1f}° 小于最小要求 20°")
                    suggestions.append("增大筋高或减小翼缘与帽顶的宽度差")

        safe_window_checks = {
            "solver_skin_thickness_ok": 1.2 <= skin_t <= 3.2,
            "solver_pitch_ok": 90.0 <= pitch <= 140.0,
            "solver_height_ok": 18.0 <= height <= 38.0,
            "solver_height_pitch_band_ok": 0.14 <= height / max(pitch, 1.0) <= 0.34,
        }
        if stype == "HAT":
            safe_window_checks["solver_flange_width_ok"] = (
                24.0 <= float(geometry.get("flange_width_mm", 0.0)) <= 50.0
            )
        elif stype != "BLADE":
            safe_window_checks["solver_flange_width_ok"] = (
                12.0 <= float(geometry.get("flange_width_mm", 0.0)) <= 22.0
            )
        details.update(safe_window_checks)
        if strict_solver_window:
            if not safe_window_checks["solver_skin_thickness_ok"]:
                errors.append("蒙皮厚度超出当前真实求解安全区")
                suggestions.append("将蒙皮厚度控制在 1.2-3.2 mm")
            if not safe_window_checks["solver_pitch_ok"]:
                errors.append("筋距超出当前真实求解安全区")
                suggestions.append("将筋距控制在 90-140 mm")
            if not safe_window_checks["solver_height_ok"]:
                errors.append("筋高超出当前真实求解安全区")
                suggestions.append("将筋高控制在 18-38 mm")
            if not safe_window_checks.get("solver_flange_width_ok", True):
                errors.append("翼缘宽度超出当前真实求解安全区")
                suggestions.append("将翼缘宽度控制在 12-22 mm")
            if not safe_window_checks["solver_height_pitch_band_ok"]:
                errors.append("筋高/筋距比例超出当前真实求解安全带")
                suggestions.append("将筋高/筋距比例控制在 0.14-0.34")

        return {
            "is_valid": not errors,
            "errors": errors,
            "suggestions": suggestions,
            "details": details,
        }
