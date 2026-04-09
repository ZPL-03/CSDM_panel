"""任务解析器：规则抽取 + LLM 结构化解析 + 契约归一化。"""

from __future__ import annotations

import json
import re
from typing import Any, Dict

from core.config_loader import load_app_config, load_llm_config, load_material_db
from core.id_utils import next_task_id
from core.llm_backend import LLMBackend
from core.schema_validator import validate_or_raise
from core.task_contract import (
    DEFAULT_CANDIDATE_GENERATION_PREFERENCES,
    DEFAULT_DESIGN_TARGETS,
    DEFAULT_GEOMETRY_ENVELOPE,
    DEFAULT_LAYUP_CONSTRAINTS,
    describe_boundary_conditions,
    describe_load_conditions,
    normalize_boundary_conditions,
    normalize_geometry_envelope,
    normalize_load_conditions,
    normalize_task_payload,
)


def _deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        elif value is not None:
            merged[key] = value
    return merged


class TaskParser:
    """负责把自然语言设计需求解析成结构化任务 JSON。"""

    def __init__(self) -> None:
        self.app_config = load_app_config()
        self.material_db = load_material_db()
        self.default_top_k = int(self.app_config.get("pipeline", {}).get("top_k", 5))
        self.default_total_candidates = int(
            self.app_config.get("pipeline", {}).get("llm_candidates", 4)
            + self.app_config.get("pipeline", {}).get("case_transfer_candidates", 2)
            + self.app_config.get("pipeline", {}).get("doe_candidates", 4)
        )
        self.llm_backend: LLMBackend | None = None
        try:
            self.llm_backend = LLMBackend(load_llm_config())
        except Exception:
            self.llm_backend = None

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
            r"(?:Nx|轴压|压缩载荷|压缩)\D*([0-9]+(?:\.[0-9]+)?)\s*(?:kN/m|kn/m)",
            r"([0-9]+(?:\.[0-9]+)?)\s*(?:kN/m|kn/m)",
        ]
        shear_patterns = [
            r"(?:Nxy|剪切载荷|剪切|面内剪切)\D*([0-9]+(?:\.[0-9]+)?)\s*(?:kN/m|kn/m)",
        ]

        for pattern in axial_patterns:
            nx_value = self._extract_float(pattern, text)
            if nx_value is not None:
                break
        for pattern in shear_patterns:
            nxy_value = self._extract_float(pattern, text)
            if nxy_value is not None:
                break

        lowered = text.lower()
        if "压剪" in text or ("压缩" in text and ("剪切" in text or "shear" in lowered)):
            return normalize_load_conditions(
                {
                    "type": "compression_shear",
                    "Nx_kN_per_m": nx_value or 850.0,
                    "Nxy_kN_per_m": nxy_value or 220.0,
                }
            )
        if "剪切" in text or "nxy" in lowered or "shear" in lowered:
            return normalize_load_conditions(
                {
                    "type": "in_plane_shear",
                    "Nx_kN_per_m": 0.0,
                    "Nxy_kN_per_m": nxy_value or nx_value or 220.0,
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

    def _extract_top_k_candidates(self, text: str) -> int:
        patterns = [
            r"(?:初筛保留|DNN初筛保留|筛选后保留|初步保留)\s*([1-9][0-9]?)\s*(?:个)?(?:候选|样本|方案)",
            r"(?:初筛|筛选|DNN初筛)\D{0,8}(?:Top[- ]?|TOP[- ]?|top[- ]?)\s*([1-9][0-9]?)",
            r"(?:初筛数量|筛选数量|TopK|Top-K)\D{0,4}([1-9][0-9]?)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return int(max(1, min(float(match.group(1)), 30.0)))
        return self.default_top_k

    def _extract_total_candidates(self, text: str) -> int:
        patterns = [
            r"(?:总候选|候选总数|候选池|初始候选|候选方案综述)\D{0,6}([1-9][0-9]?)\s*(?:个)?(?:候选|样本|方案)?",
            r"(?:生成|给出|提供|输出)\s*([1-9][0-9]?)\s*(?:个)?(?:候选|样本|方案)",
            r"(?:候选(?:数量)?|样本(?:数量)?)\D{0,6}([1-9][0-9]?)\s*(?:个)?(?:候选|样本|方案)?",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return int(max(1, min(float(match.group(1)), 60.0)))
        return max(1, DEFAULT_CANDIDATE_GENERATION_PREFERENCES["total_candidates"] if self.default_total_candidates <= 0 else self.default_total_candidates)

    def _rule_hints(self, text: str) -> Dict[str, Any]:
        material, is_user_specified = self._extract_material(text)
        return {
            "application": self._extract_application(text),
            "load_conditions": self._extract_load_conditions(text),
            "boundary_conditions": self._extract_boundary_conditions(text),
            "geometry_envelope": self._extract_geometry_envelope(text),
            "candidate_generation_preferences": {"total_candidates": self._extract_total_candidates(text)},
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
            "stiffener_type": "T",
            "design_targets": self._extract_design_targets(text),
        }

    def _build_prompt(self, text: str, hints: Dict[str, Any]) -> tuple[str, str]:
        system_prompt = (
            "你是复合材料加筋壁板任务解析助手。"
            "请把用户自然语言设计需求转成严格的 JSON。"
            "只输出 JSON，不要解释。"
            "load_conditions.type 只能是 axial_compression、in_plane_shear、compression_shear。"
            "boundary_conditions.type 只能是 SSSS、CCCC、SSCC。"
            "candidate_generation_preferences.total_candidates 表示初始候选池目标数量。"
            "screening_preferences.top_k_candidates 表示 DNN 初筛后希望保留的样本数。"
        )
        user_prompt = (
            "请根据用户需求和规则提示，输出一个任务 JSON 片段。"
            "允许只输出你能确定的字段。"
            "推荐字段：application、load_conditions、boundary_conditions、geometry_envelope、candidate_generation_preferences、screening_preferences、design_targets。"
            f"\n用户需求：{text}"
            f"\n规则提示：{json.dumps(hints, ensure_ascii=False, indent=2)}"
        )
        return system_prompt, user_prompt

    def _llm_hints(self, text: str, hints: Dict[str, Any]) -> Dict[str, Any]:
        if self.llm_backend is None:
            return {}

        try:
            system_prompt, user_prompt = self._build_prompt(text, hints)
            payload = self.llm_backend.generate_json(system_prompt, user_prompt)
            if isinstance(payload, dict):
                return payload
        except Exception:
            return {}
        return {}

    def _apply_locked_rule_hints(self, hints: Dict[str, Any], merged: Dict[str, Any]) -> Dict[str, Any]:
        """对规则高置信字段加锁，避免被 LLM 回填意外覆盖。"""

        normalized = dict(merged)
        normalized.setdefault("candidate_generation_preferences", {})
        normalized.setdefault("screening_preferences", {})

        normalized["candidate_generation_preferences"]["total_candidates"] = int(
            hints.get("candidate_generation_preferences", {}).get("total_candidates", self.default_total_candidates)
        )
        normalized["screening_preferences"]["top_k_candidates"] = int(
            hints.get("screening_preferences", {}).get("top_k_candidates", self.default_top_k)
        )

        hint_material = dict(hints.get("material_system", {}))
        if hint_material.get("is_user_specified", False):
            normalized["material_system"] = hint_material

        return normalized

    def parse_instruction(self, text: str) -> Dict[str, Any]:
        hints = self._rule_hints(text)
        llm_payload = self._llm_hints(text, hints)
        merged = _deep_merge(hints, llm_payload)
        merged = self._apply_locked_rule_hints(hints, merged)
        task = normalize_task_payload(merged, task_id=next_task_id())
        validate_or_raise("task.schema.json", task)
        return task

    def describe_parse_result(self, task: Dict[str, Any]) -> str:
        normalized = normalize_task_payload(task)
        return (
            f"已解析任务：{normalized['application']} | "
            f"{describe_load_conditions(normalized['load_conditions'])} | "
            f"{describe_boundary_conditions(normalized['boundary_conditions'])} | "
            f"候选池目标 {normalized['candidate_generation_preferences']['total_candidates']} 个 | "
            f"初筛保留 Top-{normalized['screening_preferences']['top_k_candidates']}"
        )
