"""任务解析器：规则抽取 + 契约归一化。"""

from __future__ import annotations

import re
from typing import Any, Dict

from core.config_loader import load_app_config, load_material_db
from core.id_utils import next_task_id
from core.schema_validator import validate_or_raise
from core.stiffener_profile import TYPE_DISPLAY_NAMES
from core.task_contract import (
    DEFAULT_DESIGN_TARGETS,
    DEFAULT_GEOMETRY_ENVELOPE,
    DEFAULT_LAYUP_CONSTRAINTS,
    build_task_request_record,
    describe_boundary_conditions,
    describe_load_conditions,
    normalize_boundary_conditions,
    normalize_geometry_envelope,
    normalize_load_conditions,
    normalize_task_payload,
    task_payload_from_request,
)

class TaskParser:
    """负责把自然语言设计需求解析成结构化任务。"""

    def __init__(self) -> None:
        self.app_config = load_app_config()
        self.material_db = load_material_db()
        self.default_source_ratio = self._configured_source_ratio()

    def _configured_source_ratio(self) -> Dict[str, float]:
        pipeline = dict(self.app_config.get("pipeline", {}))
        ratio = pipeline.get("candidate_source_ratio")
        if not isinstance(ratio, dict):
            ratio = {"llm": 2.0, "case_transfer": 1.0, "doe": 1.0}
        normalized = {
            "llm": max(self._safe_ratio_value(ratio.get("llm"), 2.0), 0.0),
            "case_transfer": max(self._safe_ratio_value(ratio.get("case_transfer"), 1.0), 0.0),
            "doe": max(self._safe_ratio_value(ratio.get("doe"), 1.0), 0.0),
        }
        if sum(normalized.values()) <= 0.0:
            return {"llm": 2.0, "case_transfer": 1.0, "doe": 1.0}
        return normalized

    def _safe_ratio_value(self, value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _extract_float(self, pattern: str, text: str) -> float | None:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            return None
        try:
            return float(match.group(1))
        except (TypeError, ValueError):
            return None

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

    def _extract_stiffener_type(self, text: str) -> str:
        """从用户自然语言中识别筋条类型。"""
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
        nx_value = None
        nxy_value = None

        axial_patterns = [
            r"(?:\bNx\b|轴压|压缩载荷|压缩荷载|压缩)\s*(?:为|是)?\s*[:：=]?[\s,，]*([0-9]+(?:\.[0-9]+)?)\s*(?:[kK]\s*[nN]\s*/\s*[mM])?",
            r"(?:受压|压缩工况)\s*[:：=]?[\s,，]*([0-9]+(?:\.[0-9]+)?)\s*(?:[kK]\s*[nN]\s*/\s*[mM])?",
        ]
        shear_patterns = [
            r"(?:\bNxy\b|剪切载荷|剪切荷载|剪切|面内剪切)\s*(?:为|是)?\s*[:：=]?[\s,，]*([0-9]+(?:\.[0-9]+)?)\s*(?:[kK]\s*[nN]\s*/\s*[mM])?",
            r"(?:受剪|剪切工况)\s*[:：=]?[\s,，]*([0-9]+(?:\.[0-9]+)?)\s*(?:[kK]\s*[nN]\s*/\s*[mM])?",
        ]

        compact_text = re.sub(r"\s+", "", text)
        compact_axial_patterns = [r"(?:压缩载荷|压缩荷载|压缩|轴压|Nx)([0-9]+(?:\.[0-9]+)?)(?:[kK][nN]/[mM])?"]
        compact_shear_patterns = [r"(?:剪切载荷|剪切荷载|剪切|面内剪切|Nxy)([0-9]+(?:\.[0-9]+)?)(?:[kK][nN]/[mM])?"]

        for pattern in axial_patterns:
            nx_value = self._extract_float(pattern, text)
            if nx_value is not None:
                break
        if nx_value is None:
            for pattern in compact_axial_patterns:
                nx_value = self._extract_float(pattern, compact_text)
                if nx_value is not None:
                    break

        for pattern in shear_patterns:
            nxy_value = self._extract_float(pattern, text)
            if nxy_value is not None:
                break
        if nxy_value is None:
            for pattern in compact_shear_patterns:
                nxy_value = self._extract_float(pattern, compact_text)
                if nxy_value is not None:
                    break

        lowered = text.lower()
        has_explicit_nx = re.search(r"\bNx\b", text, flags=re.IGNORECASE) is not None or re.search(r"Nx(?!y)", compact_text, flags=re.IGNORECASE) is not None
        has_explicit_nxy = re.search(r"\bNxy\b", text, flags=re.IGNORECASE) is not None or re.search(r"Nxy", compact_text, flags=re.IGNORECASE) is not None
        has_axial_text = any(token in text for token in ["轴压", "压缩载荷", "压缩荷载", "压缩", "受压"])
        has_shear_text = any(token in text for token in ["剪切载荷", "剪切荷载", "剪切", "面内剪切", "受剪"])
        has_axial = has_explicit_nx or has_axial_text
        has_shear = has_explicit_nxy or has_shear_text or "shear" in lowered
        explicit_compression_shear = "压剪" in text or ((has_explicit_nx or has_axial_text) and (has_explicit_nxy or has_shear_text or "shear" in lowered))

        if not has_axial:
            nx_value = None
        if not has_shear:
            nxy_value = None

        if explicit_compression_shear:
            return normalize_load_conditions(
                {
                    "type": "compression_shear",
                    "Nx_kN_per_m": nx_value or 850.0,
                    "Nxy_kN_per_m": nxy_value or 220.0,
                }
            )
        if has_shear:
            return normalize_load_conditions(
                {
                    "type": "in_plane_shear",
                    "Nx_kN_per_m": 0.0,
                    "Nxy_kN_per_m": nxy_value or 220.0,
                }
            )
        return normalize_load_conditions(
            {
                "type": "axial_compression",
                "Nx_kN_per_m": nx_value or 850.0,
                "Nxy_kN_per_m": 0.0,
            }
        )

    def _extract_boundary_conditions(self, text: str) -> Dict[str, Any]:
        return normalize_boundary_conditions(text)

    def _extract_geometry_envelope(self, text: str) -> Dict[str, Any]:
        geometry = dict(DEFAULT_GEOMETRY_ENVELOPE)
        length_value = self._extract_float(r"(?:长度|长)\D*([0-9]+(?:\.[0-9]+)?)\s*mm", text)
        width_value = self._extract_float(r"(?:宽度|宽)\D*([0-9]+(?:\.[0-9]+)?)\s*mm", text)
        height_value = self._extract_float(r"(?:筋高|最大筋高)\D*([0-9]+(?:\.[0-9]+)?)\s*mm", text)

        if length_value is not None:
            geometry["panel_length_mm"] = [max(300.0, length_value - 80.0), length_value + 80.0]
        if width_value is not None:
            geometry["panel_width_mm"] = [max(300.0, width_value - 80.0), width_value + 80.0]
        if height_value is not None:
            geometry["max_stiffener_height_mm"] = height_value
        return normalize_geometry_envelope(geometry)

    def _extract_geometry_values(self, text: str) -> Dict[str, float]:
        values: Dict[str, float] = {}
        length_value = self._extract_float(r"(?:长度|长)\D*([0-9]+(?:\.[0-9]+)?)\s*mm", text)
        width_value = self._extract_float(r"(?:宽度|宽)\D*([0-9]+(?:\.[0-9]+)?)\s*mm", text)
        height_value = self._extract_float(r"(?:筋高|最大筋高)\D*([0-9]+(?:\.[0-9]+)?)\s*mm", text)
        if length_value is not None:
            values["panel_length_mm"] = length_value
        if width_value is not None:
            values["panel_width_mm"] = width_value
        if height_value is not None:
            values["max_stiffener_height_mm"] = height_value
        return values

    def _extract_design_targets(self, text: str) -> Dict[str, Any]:
        blf_value = self._extract_float(r"(?:BLF|屈曲载荷因子|屈曲因子)\D*([0-9]+(?:\.[0-9]+)?)", text)
        objective = "最小重量"
        if "最小面密度" in text:
            objective = "最小面密度"
        elif "刚度优先" in text:
            objective = "刚度优先"
        return {
            "BLF_min": blf_value or DEFAULT_DESIGN_TARGETS["BLF_min"],
            "primary_objective": objective,
        }

    def _total_candidate_patterns(self) -> list[str]:
        return [
            r"(?:总候选|候选总数|候选池|初始候选|候选方案综述)\D{0,6}([1-9][0-9]?)\s*(?:个)?(?:候选|样本|方案)?",
            r"(?:生成|给出|提供|输出)\s*([1-9][0-9]?)\s*(?:个)?(?:候选|样本|方案)?",
            r"([1-9][0-9]?)\s*(?:个)?候选",
            r"(?:候选(?:数量)?|样本(?:数量)?)\D{0,6}([1-9][0-9]?)\s*(?:个)?(?:候选|样本|方案)?",
            r"(?:generate|create|produce)\s*([1-9][0-9]?)\s*(?:candidates|designs|samples)?",
        ]

    def _top_k_patterns(self) -> list[str]:
        return [
            r"(?:初筛保留|代理模型初筛保留|筛选后保留|初步保留)\s*([1-9][0-9]?)\s*(?:个)?(?:候选|样本|方案)?",
            r"(?:初筛数量|筛选数量|TopK|Top-K)\D{0,4}([1-9][0-9]?)",
            r"([1-9][0-9]?)\s*(?:个)?初筛",
            r"(?:初筛|筛选|代理模型初筛)\D{0,8}(?:Top[- ]?|TOP[- ]?|top[- ]?)\s*([1-9][0-9]?)",
            r"(?:初筛|筛选|代理模型初筛)\s*([1-9][0-9]?)\s*(?:个)?(?:候选|样本|方案)?",
            r"(?:筛)\s*([1-9][0-9]?)\s*(?:个)?(?:候选|样本|方案)?",
            r"(?:screen|select)\s*(?:top[- ]?)?\s*([1-9][0-9]?)",
            r"(?:top[- ]?k|top)\D{0,8}([1-9][0-9]?)",
        ]

    def _pattern_was_specified(self, patterns: list[str], text: str) -> bool:
        return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)

    def _match_is_screening_context(self, text: str, match: re.Match[str]) -> bool:
        prefix = text[max(0, match.start() - 12): match.start()]
        return any(token in prefix for token in ["初筛", "筛选", "保留", "Top", "top", "TOP"])

    def _extract_top_k_candidates(self, text: str) -> int:
        for pattern in self._top_k_patterns():
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return int(max(1, min(float(match.group(1)), 30.0)))
        raise ValueError("任务缺少初筛保留数量，请在自然语言需求中明确指定。")

    def _extract_total_candidates(self, text: str) -> int:
        for pattern in self._total_candidate_patterns():
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                if self._match_is_screening_context(text, match):
                    continue
                return int(max(1, min(float(match.group(1)), 60.0)))
        raise ValueError("任务缺少候选池总数，请在自然语言需求中明确指定。")

    def _extract_candidate_generation_preferences(self, text: str) -> Dict[str, Any]:
        return {
            "total_candidates": self._extract_total_candidates(text),
            "source_allocation_mode": "ratio",
            "source_ratio": dict(self.default_source_ratio),
        }

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
        blf_value = self._extract_float(r"(?:BLF|屈曲载荷因子|屈曲因子)\D*([0-9]+(?:\.[0-9]+)?)", text)
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
        hints["user_input_facts"] = self._extract_user_input_facts(text, hints)
        return hints

    def _apply_locked_rule_hints(self, hints: Dict[str, Any]) -> Dict[str, Any]:
        """保留规则抽取出的高置信字段和用户显式事实。"""

        normalized = dict(hints)
        normalized.setdefault("candidate_generation_preferences", {})
        normalized.setdefault("screening_preferences", {})

        normalized["candidate_generation_preferences"]["total_candidates"] = int(
            hints["candidate_generation_preferences"]["total_candidates"]
        )
        normalized["candidate_generation_preferences"]["source_allocation_mode"] = "ratio"
        normalized["candidate_generation_preferences"]["source_ratio"] = dict(
            hints.get("candidate_generation_preferences", {}).get("source_ratio", self.default_source_ratio)
        )
        normalized["screening_preferences"]["top_k_candidates"] = int(
            hints["screening_preferences"]["top_k_candidates"]
        )

        user_facts = dict(hints.get("user_input_facts", {"explicit_fields": []}))
        normalized["user_input_facts"] = user_facts
        if "application" in user_facts:
            normalized["application"] = hints.get("application")
        if "load_conditions" in user_facts:
            normalized["load_conditions"] = hints.get("load_conditions")
        if "boundary_conditions" in user_facts:
            normalized["boundary_conditions"] = hints.get("boundary_conditions")
        if "geometry" in user_facts:
            normalized["geometry_envelope"] = hints.get("geometry_envelope")
        if "design_targets" in user_facts:
            normalized["design_targets"] = hints.get("design_targets")

        hint_material = dict(hints.get("material_system", {}))
        if hint_material.get("is_user_specified", False):
            normalized["material_system"] = hint_material

        # 筋型由规则提取锁定，不交给 LLM 判断
        hint_stype = hints.get("stiffener_type")
        if hint_stype:
            normalized["stiffener_type"] = hint_stype

        return normalized

    def parse_instruction(self, text: str) -> Dict[str, Any]:
        hints = self._rule_hints(text)
        merged = self._apply_locked_rule_hints(hints)
        task = normalize_task_payload(merged)
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
