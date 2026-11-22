"""需求语义抽取、原文证据核对与任务契约归一化。"""

from __future__ import annotations

import re
import json
import math
from typing import Any, Dict

from core.config_loader import load_app_config, load_material_db
from core.id_utils import next_task_id
from core.schema_validator import validate_or_raise
from core.llm_backend import LLMBackend, auto_llm_enabled
from core.stiffener_profile import TYPE_DISPLAY_NAMES
from core.task_contract import (
    DEFAULT_DESIGN_TARGETS,
    DEFAULT_GEOMETRY_ENVELOPE,
    DEFAULT_LAYUP_CONSTRAINTS,
    build_task_request_record,
    describe_boundary_conditions,
    describe_load_conditions,
    normalize_boundary_conditions,
    normalize_candidate_generation_preferences,
    normalize_geometry_envelope,
    normalize_load_conditions,
    normalize_task_payload,
    task_payload_from_request,
)

NUMBER_PATTERN = r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
BLF_PATTERN = r"(?:BLF|屈曲载荷因子|屈曲因子)\s*(?:至少|不小于|at least)?\s*[>=≥:：]*\s*" + NUMBER_PATTERN

class TaskParser:
    """负责把自然语言设计需求解析成结构化任务。"""

    def __init__(self) -> None:
        self.app_config = load_app_config()
        self.material_db = load_material_db()
        self.default_source_ratio = self._configured_source_ratio()

    def _configured_source_ratio(self) -> Dict[str, float]:
        pipeline = dict(self.app_config.get("pipeline", {}))
        return normalize_candidate_generation_preferences(
            {"source_ratio": pipeline.get("candidate_source_ratio", {})}
        )["source_ratio"]

    def _extract_float(self, pattern: str, text: str) -> float | None:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            return None
        return float(match.group(1))

    def _extract_material(self, text: str) -> tuple[Dict[str, Any], bool]:
        lowered = text.lower()
        for material_key, payload in self.material_db.items():
            display_name = str(payload.get("display_name", ""))
            if material_key.lower() in lowered or display_name.lower() in lowered:
                material = dict(payload)
                material["name"] = material.get("display_name", material_key)
                material["material_key"] = material_key
                material["is_user_specified"] = True
                return material, True
        material = dict(self.material_db["T300_5208"])
        material["name"] = material.get("display_name", "T300/5208")
        material["material_key"] = "T300_5208"
        material["is_user_specified"] = False
        return material, False

    def _stiffener_type_patterns(self) -> list[tuple[str, list[str]]]:
        return [
            (
                "BLADE",
                [
                    r"板\s*式\s*(?:筋|加筋|筋条|壁板|方案)?",
                    r"平\s*板\s*筋",
                    r"刀\s*型\s*(?:筋|加筋|筋条|壁板|方案)?",
                    r"(?<![A-Za-z0-9])blade(?:\s*stiffener)?(?![A-Za-z0-9])",
                ],
            ),
            (
                "HAT",
                [
                    r"帽\s*(?:(?:型|形|式|状)\s*(?:加筋|筋条|筋|壁板|方案)?|(?:加筋|筋条|筋|壁板|方案))",
                    r"槽\s*型\s*(?:筋|加筋|筋条|壁板|方案)?",
                    r"(?<![A-Za-z0-9])hat(?:\s*型|\s*stiffener)?(?![A-Za-z0-9])",
                ],
            ),
            (
                "L",
                [
                    r"角\s*(?:材|型|型筋|形筋)",
                    r"(?<![A-Za-z0-9])L\s*(?:型|形|型筋|形筋|[-\s]*stiffener)?(?![A-Za-z0-9])",
                ],
            ),
            (
                "T",
                [
                    r"(?<![A-Za-z0-9])T\s*(?:型|形|型筋|形筋|[-\s]*stiffener)?(?![A-Za-z0-9])",
                ],
            ),
        ]

    def _extract_stiffener_types(self, text: str) -> list[str]:
        matches: list[tuple[int, int, str]] = []
        for priority, (stype, patterns) in enumerate(self._stiffener_type_patterns()):
            for pattern in patterns:
                for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                    token = match.group(0).strip()
                    if token:
                        matches.append((match.start(), priority, stype))
        matches.sort(key=lambda item: (item[0], item[1]))
        stiffener_types: list[str] = []
        for _, _, stype in matches:
            if stype not in stiffener_types:
                stiffener_types.append(stype)
        return stiffener_types

    def _extract_stiffener_type(self, text: str) -> str:
        """从用户自然语言中识别筋条类型。"""
        stiffener_types = self._extract_stiffener_types(text)
        if stiffener_types:
            return stiffener_types[0]
        mapping = [
            ("板式筋", "BLADE"), ("板式", "BLADE"), ("blade", "BLADE"),
            ("平板筋", "BLADE"), ("刀型筋", "BLADE"), ("刀型", "BLADE"),
            ("帽型筋", "HAT"), ("帽形筋", "HAT"), ("帽型", "HAT"), ("帽形", "HAT"),
            ("帽加筋", "HAT"), ("帽筋", "HAT"), ("hat", "HAT"), ("hat型", "HAT"),
            ("槽型筋", "HAT"), ("槽型", "HAT"),
            ("L型角材", "L"), ("L 型角材", "L"), ("角材", "L"),
            ("角型筋", "L"), ("角型", "L"), ("l型", "L"),
            ("l stiffener", "L"), ("l-stiffener", "L"),
            ("T型筋", "T"), ("T 型筋", "T"), ("T型", "T"),
            ("T 型", "T"), ("T形筋", "T"), ("T形", "T"),
            ("t型", "T"), ("t 型", "T"),
            ("t stiffener", "T"), ("t-stiffener", "T"),
        ]
        lowered = text.lower()
        for keyword, stype in mapping:
            if keyword.lower() in lowered:
                return stype
        if re.search(r"帽\s*(?:型|形|式|状)?\s*(?:加筋|筋条|筋|壁板|方案)", text, flags=re.IGNORECASE):
            return "HAT"
        return "T"

    def _stiffener_type_was_specified(self, text: str) -> bool:
        if self._extract_stiffener_types(text):
            return True
        lowered = text.lower()
        compact = re.sub(r"\s+", "", lowered)
        keywords = [
            "板式", "刀型", "blade", "帽型", "帽形", "帽加筋", "帽筋", "hat", "hat型",
            "槽型", "角材", "角型", "t型", "t形", "l型", "l形",
        ]
        if any(keyword in compact for keyword in keywords):
            return True
        return bool(re.search(r"帽\s*(?:型|形|式|状)?\s*(?:加筋|筋条|筋|壁板|方案)", text, flags=re.IGNORECASE))

    def _extract_application(self, text: str) -> str:
        if "尾翼" in text:
            return "尾翼壁板"
        if "舱段" in text:
            return "舱段壁板"
        if "机翼" in text or "翼面" in text:
            return "机翼下蒙皮壁板"
        return "复合材料加筋壁板"

    def _extract_load_conditions(self, text: str) -> Dict[str, Any]:
        axial_label = r"(?:Nx(?![A-Za-z])|轴压|压缩载荷|压缩荷载|压缩工况|压缩|受压)"
        shear_label = r"(?:Nxy(?![A-Za-z])|面内剪切|剪切载荷|剪切荷载|剪切工况|剪切|受剪|shear)"
        separator = r"\s*(?:为|是)?\s*[:：=]?[\s,，]*"
        nx_value = self._extract_float(axial_label + separator + NUMBER_PATTERN, text)
        nxy_value = self._extract_float(shear_label + separator + NUMBER_PATTERN, text)
        has_axial = re.search(axial_label, text, re.IGNORECASE) is not None or "压剪" in text
        has_shear = re.search(shear_label, text, re.IGNORECASE) is not None or "压剪" in text
        # 缺省工况采用交互默认值；明确给出的零分量参与实际工况判定。
        if nx_value is None:
            nx_value = 850.0 if has_axial or not has_shear else 0.0
        if nxy_value is None:
            nxy_value = 220.0 if has_shear else 0.0
        return normalize_load_conditions(
            {
                "Nx_kN_per_m": nx_value,
                "Nxy_kN_per_m": nxy_value,
            }
        )

    def _extract_boundary_conditions(self, text: str) -> Dict[str, Any]:
        return normalize_boundary_conditions(text)

    def _extract_geometry_envelope(self, text: str) -> Dict[str, Any]:
        geometry = dict(DEFAULT_GEOMETRY_ENVELOPE)
        for key, value in self._extract_geometry_values(text).items():
            geometry[key] = [value, value] if key in ("panel_length_mm", "panel_width_mm") else value
        return normalize_geometry_envelope(geometry)

    def _extract_geometry_values(self, text: str) -> Dict[str, float]:
        values: Dict[str, float] = {}
        separator = r"\s*(?:为|是|不超过|小于等于)?\s*[:：=≤]?\s*"
        length_value = self._extract_float(r"(?:长度|长)" + separator + NUMBER_PATTERN + r"\s*mm", text)
        width_value = self._extract_float(r"(?<!帽顶)(?<!翼缘)(?:宽度|宽)" + separator + NUMBER_PATTERN + r"\s*mm", text)
        height_value = self._extract_float(r"(?:最大筋高|筋高)" + separator + NUMBER_PATTERN + r"\s*mm", text)
        if length_value is not None:
            values["panel_length_mm"] = length_value
        if width_value is not None:
            values["panel_width_mm"] = width_value
        if height_value is not None:
            values["max_stiffener_height_mm"] = height_value
        return values

    def _extract_design_targets(self, text: str) -> Dict[str, Any]:
        blf_value = self._extract_float(BLF_PATTERN, text)
        objective = "最小重量"
        if "最小面密度" in text:
            objective = "最小面密度"
        elif "刚度优先" in text:
            objective = "刚度优先"
        return {
            "BLF_min": DEFAULT_DESIGN_TARGETS["BLF_min"] if blf_value is None else blf_value,
            "primary_objective": objective,
        }

    def _total_candidate_patterns(self) -> list[str]:
        return [
            r"(?:总候选|候选总数|候选池|初始候选|候选方案综述)[\s为是:：=]{0,6}" + NUMBER_PATTERN,
            r"(?:生成|给出|提供|输出)\s*" + NUMBER_PATTERN,
            NUMBER_PATTERN + r"\s*(?:个)?候选",
            r"(?:候选(?:数量)?|样本(?:数量)?)[\s为是:：=]{0,6}" + NUMBER_PATTERN,
            r"(?:generate|create|produce)\s*" + NUMBER_PATTERN,
        ]

    def _top_k_patterns(self) -> list[str]:
        return [
            r"(?:初筛保留|代理模型初筛保留|筛选后保留|初步保留)\s*" + NUMBER_PATTERN,
            r"(?:初筛数量|筛选数量|TopK|Top-K)[\s为是:：=]{0,4}" + NUMBER_PATTERN,
            NUMBER_PATTERN + r"\s*(?:个)?初筛",
            r"(?:初筛|筛选|代理模型初筛)\s*Top[- ]?\s*" + NUMBER_PATTERN,
            r"(?:初筛|筛选|代理模型初筛|筛)\s*" + NUMBER_PATTERN,
            r"(?:screen|select)\s*(?:top[- ]?)?\s*" + NUMBER_PATTERN,
            r"(?:top[- ]?k|top)[\s为是:：=]{0,8}" + NUMBER_PATTERN,
        ]

    def _pattern_was_specified(self, patterns: list[str], text: str) -> bool:
        return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)

    def _match_is_screening_context(self, text: str, match: re.Match[str]) -> bool:
        prefix = text[max(0, match.start() - 12): match.start()]
        return any(token in prefix for token in ["初筛", "筛选", "保留", "Top", "top", "TOP"])

    def _extract_top_k_candidates(self, text: str) -> float:
        for pattern in self._top_k_patterns():
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return float(match.group(1))
        raise ValueError("任务缺少初筛保留数量，请在自然语言需求中明确指定。")

    def _extract_total_candidates(self, text: str) -> float:
        for pattern in self._total_candidate_patterns():
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                if self._match_is_screening_context(text, match):
                    continue
                return float(match.group(1))
        raise ValueError("任务缺少候选池总数，请在自然语言需求中明确指定。")

    def _extract_candidate_generation_preferences(self, text: str) -> Dict[str, Any]:
        preferences = {
            "total_candidates": self._extract_total_candidates(text),
            "source_allocation_mode": "ratio",
            "source_ratio": dict(self.default_source_ratio),
        }
        stiffener_types = self._extract_stiffener_types(text)
        if stiffener_types:
            preferences["stiffener_types"] = stiffener_types
        return preferences

    def _extract_user_load_fact(self, text: str) -> Dict[str, Any] | None:
        compact_text = re.sub(r"\s+", "", text)
        lowered = text.lower()
        has_axial = (
            re.search(r"\bNx\b", text, flags=re.IGNORECASE) is not None
            or re.search(r"Nx(?!y)", compact_text, flags=re.IGNORECASE) is not None
            or any(token in text for token in ["轴压", "压缩载荷", "压缩荷载", "压缩", "受压"])
        )
        has_shear = (
            re.search(r"\bNxy\b", text, flags=re.IGNORECASE) is not None
            or re.search(r"Nxy", compact_text, flags=re.IGNORECASE) is not None
            or any(token in text for token in ["剪切载荷", "剪切荷载", "剪切", "面内剪切", "受剪"])
            or "shear" in lowered
        )
        if not has_axial and not has_shear and "压剪" not in text:
            return None

        load = self._extract_load_conditions(text)
        fact: Dict[str, Any] = {"type": load["type"]}
        nx_value = load.get("Nx_kN_per_m")
        nxy_value = load.get("Nxy_kN_per_m")
        if has_axial and nx_value is not None:
            fact["Nx_kN_per_m"] = nx_value
        if has_shear and nxy_value is not None:
            fact["Nxy_kN_per_m"] = nxy_value
        return fact

    def _user_boundary_conditions(self, text: str) -> Dict[str, Any] | None:
        lowered = text.lower()
        keywords = ["边界", "简支", "固支", "固定", "ssss", "cccc", "sscc", "clamped", "simply", "boundary"]
        if not any(keyword in text or keyword in lowered for keyword in keywords):
            return None
        boundary = self._extract_boundary_conditions(text)
        return {"type": boundary.get("type"), "label": boundary.get("label")}

    def _extract_user_input_facts(self, text: str, hints: Dict[str, Any]) -> Dict[str, Any]:
        facts: Dict[str, Any] = {"explicit_fields": []}

        application = self._extract_application(text)
        if any(keyword in text for keyword in ["尾翼", "舱段", "机翼", "翼面"]):
            facts["application"] = application
            facts["explicit_fields"].append("application")

        load_fact = self._extract_user_load_fact(text)
        if load_fact is not None:
            facts["load_conditions"] = load_fact
            facts["explicit_fields"].append("load_conditions")

        boundary_fact = self._user_boundary_conditions(text)
        if boundary_fact is not None:
            facts["boundary_conditions"] = boundary_fact
            facts["explicit_fields"].append("boundary_conditions")

        geometry_values = self._extract_geometry_values(text)
        if geometry_values:
            facts["geometry"] = geometry_values
            facts["explicit_fields"].append("geometry")

        material, is_user_specified = self._extract_material(text)
        if is_user_specified:
            facts["material_system"] = {
                "name": material.get("name"),
                "material_key": material.get("material_key"),
            }
            facts["explicit_fields"].append("material_system")

        stiffener_type = self._extract_stiffener_type(text)
        if any(keyword.lower() in text.lower() for keyword in TYPE_DISPLAY_NAMES.values()) or self._stiffener_type_was_specified(text):
            facts["stiffener_type"] = stiffener_type
            facts["explicit_fields"].append("stiffener_type")
            stiffener_types = self._extract_stiffener_types(text)
            if len(stiffener_types) > 1:
                facts["stiffener_types"] = stiffener_types
                facts["explicit_fields"].append("stiffener_types")

        candidate_generation: Dict[str, Any] = {}
        if self._pattern_was_specified(self._total_candidate_patterns(), text):
            candidate_generation["total_candidates"] = int(hints["candidate_generation_preferences"]["total_candidates"])
            facts["explicit_fields"].append("candidate_generation_preferences.total_candidates")
        if self._pattern_was_specified(self._top_k_patterns(), text):
            candidate_generation["top_k_candidates"] = int(hints["screening_preferences"]["top_k_candidates"])
            facts["explicit_fields"].append("screening_preferences.top_k_candidates")
        if candidate_generation:
            facts["candidate_generation"] = candidate_generation

        design_targets: Dict[str, Any] = {}
        blf_value = self._extract_float(BLF_PATTERN, text)
        if blf_value is not None:
            design_targets["BLF_min"] = blf_value
            facts["explicit_fields"].append("design_targets.BLF_min")
        if "最小面密度" in text or "刚度优先" in text:
            design_targets["primary_objective"] = hints["design_targets"]["primary_objective"]
            facts["explicit_fields"].append("design_targets.primary_objective")
        if design_targets:
            facts["design_targets"] = design_targets

        facts["explicit_fields"] = sorted(set(facts["explicit_fields"]))
        return facts

    def _rule_hints(self, text: str) -> Dict[str, Any]:
        material, is_user_specified = self._extract_material(text)
        hints = {
            "application": self._extract_application(text),
            "load_conditions": self._extract_load_conditions(text),
            "boundary_conditions": self._extract_boundary_conditions(text),
            "geometry_envelope": self._extract_geometry_envelope(text),
            "candidate_generation_preferences": self._extract_candidate_generation_preferences(text),
            "screening_preferences": {"top_k_candidates": self._extract_top_k_candidates(text)},
            "material_system": {
                "name": material["name"],
                "density_kg_per_m3": material["density_kg_per_m3"],
                "E1_GPa": material["E1_GPa"],
                "E2_GPa": material["E2_GPa"],
                "G12_GPa": material["G12_GPa"],
                "nu12": material["nu12"],
                "material_key": material.get("material_key"),
                "is_user_specified": is_user_specified,
            },
            "layup_constraints": dict(DEFAULT_LAYUP_CONSTRAINTS),
            "stiffener_type": self._extract_stiffener_type(text),
            "design_targets": self._extract_design_targets(text),
        }
        hints = normalize_task_payload(hints)
        hints["user_input_facts"] = self._extract_user_input_facts(text, hints)
        return hints

    def _interpret_requirements(self, text: str) -> Dict[str, Any]:
        """LLM 只抽取带原文引句的需求，程序检查字段、数值、单位与明确约束。"""
        numeric_labels = {"total_candidates": "生成", "top_k_candidates": "初筛保留",
                          "Nx_kN_per_m": "Nx", "Nxy_kN_per_m": "Nxy", "BLF_min": "BLF >=",
                          "panel_length_mm": "长度", "panel_width_mm": "宽度",
                          "max_stiffener_height_mm": "最大筋高"}
        geometry_fields = {"skin_thickness_mm", "pitch_mm", "stiffener_height_mm", "web_thickness_mm",
                           "flange_width_mm", "flange_thickness_mm", "cap_width_mm", "cap_thickness_mm"}
        geometry_labels = {
            "panel_length_mm": r"(?:长度|长)", "panel_width_mm": r"(?<!帽顶)(?<!翼缘)(?:宽度|宽)",
            "max_stiffener_height_mm": r"最大筋高",
            "skin_thickness_mm": r"(?:蒙皮厚度|蒙皮厚)", "pitch_mm": r"(?:筋条间距|筋距)",
            "stiffener_height_mm": r"(?<!最大)筋高", "web_thickness_mm": r"(?:腹板厚度|腹板厚)",
            "flange_width_mm": r"(?:翼缘宽度|翼缘宽)", "flange_thickness_mm": r"(?:翼缘厚度|翼缘厚)",
            "cap_width_mm": r"(?:帽顶宽度|帽顶宽)", "cap_thickness_mm": r"(?:帽顶厚度|帽顶厚)",
        }
        explicit_geometry = {}
        # 显式尺寸不能因单位不受支持或模型省略而退回默认设计包络。
        for field, label in geometry_labels.items():
            pattern = label + r"\s*(?:为|是|不超过|小于等于)?\s*[:：=≤]?\s*" + NUMBER_PATTERN
            for match in re.finditer(pattern, text, re.IGNORECASE):
                if not re.match(r"\s*mm(?![A-Za-z])", text[match.end():], re.IGNORECASE):
                    raise ValueError(f"明确几何 {field} 必须使用 mm，不能省略或转换单位")
                value = float(match.group(1))
                if field in explicit_geometry and explicit_geometry[field] != value:
                    raise ValueError(f"明确几何 {field} 包含冲突数值")
                explicit_geometry[field] = value
        allowed = set(numeric_labels) | geometry_fields | {"stiffener_types", "boundary_type", "material_key", "primary_objective"}
        explicit_types = self._extract_stiffener_types(text)
        answer = LLMBackend().chat(
            "你负责抽取复合材料加筋壁板的用户需求，不生成设计方案。只返回 JSON 对象，唯一字段 facts 为数组。"
            "每项恰含 field、value、quote；quote 必须逐字引用用户原文中包含该值及其含义和单位的连续文字。"
            "只提取明确给出的值，禁止推算、补默认值或转换单位。数值使用原文数字；几何单位必须为 mm，载荷为 kN/m。"
            "stiffener_types 为 BLADE/T/L/HAT 数组，boundary_type 为 SSSS/SSCC/CCCC，material_key 为材料库键。"
            "explicit_stiffener_types 非空时必须返回同序的 stiffener_types 和覆盖全部指定筋型的原文引句，单一筋型也要返回。"
            "primary_objective 为 最小重量/最小面密度/刚度优先。不存在或含糊的信息不要输出。"
            "字段白名单：" + ", ".join(sorted(allowed)) + "；材料库：" + ", ".join(self.material_db),
            json.dumps({"instruction": text, "explicit_stiffener_types": explicit_types}, ensure_ascii=False),
            max_tokens_override=2600, json_mode=True)
        response = json.loads(answer)
        if not isinstance(response, dict) or set(response) != {"facts"} or not isinstance(response["facts"], list):
            raise ValueError("需求抽取必须返回 facts 数组")
        extracted, canonical = {}, []
        for item in response["facts"]:
            if not isinstance(item, dict) or set(item) != {"field", "value", "quote"}:
                raise ValueError("需求事实只允许 field、value、quote")
            field, value, quote = item["field"], item["value"], item["quote"]
            if not isinstance(field, str) or field not in allowed or field in extracted:
                raise ValueError("需求字段未知或重复")
            if not isinstance(quote, str) or not quote.strip() or quote not in text:
                raise ValueError(f"需求 {field} 缺少用户原文证据")
            if field in set(numeric_labels) | geometry_fields:
                if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                        or value not in [float(token) for token in re.findall(NUMBER_PATTERN, quote)]):
                    raise ValueError(f"需求 {field} 数值与原文不一致")
                unit = "mm" if field.endswith("_mm") else "kN/m" if field.endswith("_kN_per_m") else ""
                if unit and not re.search(re.escape(unit), quote, re.IGNORECASE):
                    raise ValueError(f"需求 {field} 必须明确使用 {unit}")
                if field in numeric_labels:
                    canonical.append(f"{numeric_labels[field]} {value:g} {unit}")
            elif field == "stiffener_types":
                if not isinstance(value, list) or not value or value != list(dict.fromkeys(self._extract_stiffener_types(quote))):
                    raise ValueError("需求筋型与引句不一致")
                canonical.append("、".join(value) + "型筋")
            elif field == "boundary_type":
                if value not in {"SSSS", "SSCC", "CCCC"} or value != self._extract_boundary_conditions(quote)["type"]:
                    raise ValueError("需求边界与引句不一致")
                canonical.append(value)
            elif field == "material_key":
                if value not in self.material_db or self._extract_material(quote)[0].get("material_key") != value:
                    raise ValueError("需求材料与引句不一致")
                canonical.append(self.material_db[value]["display_name"])
            elif field == "primary_objective":
                if value not in {"最小重量", "最小面密度", "刚度优先"} or value not in quote:
                    raise ValueError("需求目标与引句不一致")
                canonical.append(value)
            extracted[field] = value
        # 原文的数字表达仍经规则解析；明确字段冲突不能被模型重写。
        explicit = explicit_geometry
        for key, value in explicit.items():
            if extracted.get(key) != value:
                raise ValueError(f"需求抽取遗漏或改变了明确几何 {key}")
        for field, label in (("Nx_kN_per_m", r"(?:Nx(?![A-Za-z])|轴压|压缩载荷|压缩荷载|压缩|受压)"),
                             ("Nxy_kN_per_m", r"(?:Nxy(?![A-Za-z])|面内剪切|剪切载荷|剪切荷载|剪切|受剪|shear)")):
            value = self._extract_float(label + r"\s*(?:为|是)?\s*[:：=]?[\s,，]*" + NUMBER_PATTERN, text)
            if value is not None and extracted.get(field) != value:
                raise ValueError(f"需求抽取遗漏或改变了明确载荷 {field}")
        boundary = self._user_boundary_conditions(text)
        if boundary and extracted.get("boundary_type") != boundary["type"]:
            raise ValueError("需求抽取遗漏或改变了明确边界")
        if explicit_types and extracted.get("stiffener_types") != explicit_types:
            raise ValueError("需求抽取遗漏或改变了明确筋型")
        material, specified = self._extract_material(text)
        if specified and extracted.get("material_key") != material["material_key"]:
            raise ValueError("需求抽取遗漏或改变了明确材料")
        blf = self._extract_float(BLF_PATTERN, text)
        if blf is not None and extracted.get("BLF_min") != blf:
            raise ValueError("需求抽取遗漏或改变了明确 BLF 目标")
        for field, patterns in (("total_candidates", self._total_candidate_patterns()),
                                ("top_k_candidates", self._top_k_patterns())):
            values = [value for pattern in patterns if (value := self._extract_float(pattern, text)) is not None]
            if values and (len(set(values)) != 1 or extracted.get(field) != values[0]):
                raise ValueError(f"需求抽取遗漏或改变了明确数量 {field}")
        task = self._rule_hints("，".join(canonical))
        task["application"] = self._extract_application(text)
        task["fixed_geometry"] = {field: extracted[field] for field in geometry_fields if field in extracted}
        task["requirement_extraction"] = {"method": "llm_grounded", "instruction": text, "facts": response["facts"]}
        return task

    def parse_instruction(self, text: str) -> Dict[str, Any]:
        task = self._interpret_requirements(text) if auto_llm_enabled() else self._rule_hints(text)
        validate_or_raise("task.schema.json", task)
        return build_task_request_record(
            task,
            task_id=next_task_id(),
            source="gui_instruction",
        )

    def describe_parse_result(self, task: Dict[str, Any]) -> str:
        normalized = task_payload_from_request(task)
        return (
            f"已解析任务：{normalized['application']} | "
            f"{describe_load_conditions(normalized['load_conditions'])} | "
            f"{describe_boundary_conditions(normalized['boundary_conditions'])} | "
            f"候选池目标 {normalized['candidate_generation_preferences']['total_candidates']} 个 | "
            f"初筛保留 Top-{normalized['screening_preferences']['top_k_candidates']}"
        )
