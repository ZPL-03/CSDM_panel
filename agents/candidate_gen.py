from __future__ import annotations

import re
from typing import Any, Dict, List

from agents.base import BaseAgent
from core.case_retriever import CaseRetriever
from core.config_loader import load_app_config, load_llm_config, load_material_db
from core.doe_sampler import DOESampler
from core.id_utils import format_temp_candidate_id
from core.domain_knowledge import DomainKnowledgeBase
from core.llm_backend import LLMBackend, auto_llm_enabled
from core.rule_checker import RuleChecker
from core.schema_validator import SchemaValidationError, validate_or_raise
from core.stiffener_profile import (
    GEOMETRY_LABELS,
    TYPE_DISPLAY_NAMES,
    load_param_ranges_for_type,
    normalize_geometry,
    required_geometry_params,
    resolve_stiffener_type,
)
from core.task_contract import (
    describe_boundary_conditions,
    describe_load_conditions,
    requested_candidate_pool_size,
    task_payload_from_request,
)

class CandidateGenAgent(BaseAgent):
    """候选生成编排器，统一调度 LLM、案例迁移与 DOE 三条路径。"""

    agent_name = "CANDIDATE_GEN"

    def __init__(self, progress_callback=None) -> None:
        super().__init__(progress_callback=progress_callback)
        self.app_config = load_app_config()
        self.llm_config = load_llm_config()
        self.material_db = load_material_db()
        self.doe_sampler = DOESampler()
        self.rule_checker = RuleChecker()
        self.case_retriever = CaseRetriever()
        self.knowledge_base = DomainKnowledgeBase()
        self.material_catalog = self._build_material_catalog()
        self.llm_backend: LLMBackend | None = None
        if auto_llm_enabled():
            try:
                self.llm_backend = LLMBackend(self.llm_config)
            except Exception as exc:
                self.emit(f"LLM 后端初始化失败，将自动退回案例迁移和 DOE：{exc}")

    def _normalize_layup_string(self, layup_value: Any) -> str:
        if isinstance(layup_value, str):
            text = layup_value.strip()
            return text or "[45/-45/0/90/0/-45/45]s"
        if isinstance(layup_value, (list, tuple)):
            items = [str(item).strip() for item in layup_value if str(item).strip()]
            if not items:
                return "[45/-45/0/90/0/-45/45]s"
            suffix = ""
            if items[-1].lower() == "s":
                suffix = "s"
                items = items[:-1]
            return f"[{'/'.join(items)}]{suffix}"
        return "[45/-45/0/90/0/-45/45]s"

    def _normalize_ratio(self, value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _source_ratio_from_config(self) -> Dict[str, float]:
        pipeline = dict(self.app_config.get("pipeline", {}))
        ratio = pipeline.get("candidate_source_ratio")
        if not isinstance(ratio, dict):
            ratio = {"llm": 2.0, "case_transfer": 1.0, "doe": 1.0}
        return self._sanitize_source_ratio(ratio)

    def _sanitize_source_ratio(self, ratio: Dict[str, Any] | None) -> Dict[str, float]:
        data = ratio if isinstance(ratio, dict) else {}
        normalized = {
            "llm": max(self._normalize_ratio(data.get("llm"), 2.0), 0.0),
            "case_transfer": max(self._normalize_ratio(data.get("case_transfer"), 1.0), 0.0),
            "doe": max(self._normalize_ratio(data.get("doe"), 1.0), 0.0),
        }
        if sum(normalized.values()) <= 0.0:
            return {"llm": 2.0, "case_transfer": 1.0, "doe": 1.0}
        return normalized

    def _allocate_by_ratio(self, total: int, ratio: Dict[str, float]) -> Dict[str, int]:
        sources = ["llm", "case_transfer", "doe"]
        ratio_sum = max(sum(ratio.values()), 1e-9)
        raw_targets = {source: total * ratio[source] / ratio_sum for source in sources}
        targets = {source: int(raw_targets[source]) for source in sources}
        remainder = max(total - sum(targets.values()), 0)
        ranked_sources = sorted(
            sources,
            key=lambda source: (raw_targets[source] - targets[source], ratio[source]),
            reverse=True,
        )
        for source in ranked_sources[:remainder]:
            targets[source] += 1
        return targets

    def _build_material_catalog(self) -> List[Dict[str, Any]]:
        catalog: List[Dict[str, Any]] = []
        for material_key, payload in self.material_db.items():
            catalog.append(
                {
                    "name": payload.get("display_name", material_key),
                    "density_kg_per_m3": float(payload.get("density_kg_per_m3", 1600.0)),
                    "E1_GPa": float(payload.get("E1_GPa", 181.0)),
                    "E2_GPa": float(payload.get("E2_GPa", 10.3)),
                    "G12_GPa": float(payload.get("G12_GPa", 7.17)),
                    "nu12": float(payload.get("nu12", 0.28)),
                    "material_key": material_key,
                }
            )
        return catalog

    def _task_payload(self, task: Dict) -> Dict[str, Any]:
        return task_payload_from_request(task)

    def _material_options(self, task: Dict) -> List[Dict[str, Any]]:
        task_payload = self._task_payload(task)
        task_material = dict(task_payload.get("material_system", {}))
        if task_material.get("is_user_specified", False):
            return [task_material]
        return [dict(item) for item in self.material_catalog] or [task_material]

    def _resolve_material_system(self, task: Dict, raw_material: Any, index: int) -> Dict[str, Any]:
        task_payload = self._task_payload(task)
        task_material = dict(task_payload.get("material_system", {}))
        if task_material.get("is_user_specified", False):
            return task_material

        if isinstance(raw_material, dict) and raw_material:
            raw_name = str(raw_material.get("name") or raw_material.get("display_name") or "").strip().lower()
            raw_key = str(raw_material.get("material_key") or "").strip()
            for material in self.material_catalog:
                if raw_key and raw_key == material.get("material_key"):
                    return dict(material)
                if raw_name and raw_name == str(material.get("name", "")).strip().lower():
                    return dict(material)
            merged = dict(task_material)
            merged.update({key: value for key, value in raw_material.items() if value is not None})
            return merged

        options = self._material_options(task)
        if not options:
            return task_material
        return dict(options[(max(index, 1) - 1) % len(options)])

    def _knowledge_guidance(self, task: Dict, top_k: int) -> List[str]:
        """仅为 LLM 路径检索外部知识库/知识图谱片段，避免与历史案例迁移职责混用。"""
        return self.knowledge_base.format_snippets(task, top_k=max(1, min(3, top_k)))

    def _user_fact_lines(self, facts: Dict[str, Any]) -> List[str]:
        lines: List[str] = []
        candidate_generation = facts.get("candidate_generation", {})
        if isinstance(candidate_generation, dict):
            if candidate_generation.get("total_candidates") is not None:
                lines.append(f"候选池总数：{candidate_generation['total_candidates']}")
            if candidate_generation.get("top_k_candidates") is not None:
                lines.append(f"初筛保留：{candidate_generation['top_k_candidates']}")
        if facts.get("application"):
            lines.append(f"设计对象：{facts['application']}")
        load_conditions = facts.get("load_conditions", {})
        if isinstance(load_conditions, dict) and load_conditions:
            load_parts = [str(load_conditions.get("type") or "")]
            if load_conditions.get("Nx_kN_per_m") is not None:
                load_parts.append(f"Nx={load_conditions['Nx_kN_per_m']} kN/m")
            if load_conditions.get("Nxy_kN_per_m") is not None:
                load_parts.append(f"Nxy={load_conditions['Nxy_kN_per_m']} kN/m")
            lines.append("工况：" + "，".join(part for part in load_parts if part))
        boundary_conditions = facts.get("boundary_conditions", {})
        if isinstance(boundary_conditions, dict) and boundary_conditions.get("label"):
            lines.append(f"边界条件：{boundary_conditions['label']}")
        geometry = facts.get("geometry", {})
        if isinstance(geometry, dict) and geometry:
            labels = {
                "panel_length_mm": "壁板长度",
                "panel_width_mm": "壁板宽度",
                "max_stiffener_height_mm": "最大筋高",
            }
            for key in ["panel_length_mm", "panel_width_mm", "max_stiffener_height_mm"]:
                if geometry.get(key) is not None:
                    lines.append(f"{labels[key]}：{geometry[key]} mm")
        material = facts.get("material_system", {})
        if isinstance(material, dict) and material.get("name"):
            lines.append(f"材料：{material['name']}")
        if facts.get("stiffener_type"):
            lines.append(f"筋条类型：{facts['stiffener_type']}")
        design_targets = facts.get("design_targets", {})
        if isinstance(design_targets, dict):
            if design_targets.get("BLF_min") is not None:
                lines.append(f"BLF 目标：不低于 {design_targets['BLF_min']}")
            if design_targets.get("primary_objective"):
                lines.append(f"优化目标：{design_targets['primary_objective']}")
        return lines

    def _candidate_field_constraint_lines(self, stype: str) -> List[str]:
        ranges = load_param_ranges_for_type(stype)

        def range_text(key: str) -> str:
            value = ranges.get(key) or {}
            return f"{float(value.get('min', 0.0)):g}-{float(value.get('max', 0.0)):g}"

        material_names = "、".join(str(item.get("name")) for item in self.material_catalog if item.get("name"))
        lines = [f"材料必须从项目材料库选择：{material_names}"]
        for key in required_geometry_params(stype):
            label = GEOMETRY_LABELS.get(key, key)
            lines.append(f"{label} 必须给出数值，范围 {range_text(key)}")
        lines.extend(
            [
                "铺层形式必须给出 skin_layup，例如 [45/-45/0/90/0/-45/45]s",
                "铺层比例必须给出 f0、f45、f90，三者用小数表示，建议和为 1.0",
                "推荐理由应说明结构性能依据和制造风险依据，不要只写泛化结论",
            ]
        )
        return lines

    def _build_prompt(self, task: Dict, knowledge_guidance: List[str], desired_count: int) -> tuple[str, str]:
        """构造 LLM 候选生成 Prompt，要求输出工程自然语言，系统再解析成候选契约。"""
        task_payload = self._task_payload(task)
        material_options = [item.get("name", "") for item in self._material_options(task)]
        stype = task_payload.get("stiffener_type", "T")
        type_display = TYPE_DISPLAY_NAMES.get(stype, stype)
        user_facts = dict(task_payload.get("user_input_facts") or {})
        fact_lines = self._user_fact_lines(user_facts)
        constraint_lines = self._candidate_field_constraint_lines(stype)

        system_prompt = (
            "你是复合材料加筋壁板结构设计专家。"
            "请输出自然语言工程回答，不要输出 JSON、XML 或代码块。"
            "候选方案必须使用 Markdown 表格表达，表格列名必须清晰包含材料、几何参数、铺层比例和推荐理由。"
            "不要输出 case_id、task、abaqus_results、verdict、created_at 等历史字段。"
            f"方案必须为 {type_display} 加筋壁板，线性屈曲场景，与任务中的工况和边界一致。"
            "用户已给事实只能来自 input 的“用户已给信息”；系统约束用于生成候选，不得写成用户事实。"
        )
        knowledge_text = "\n\n".join(knowledge_guidance) if knowledge_guidance else "当前没有可用外部知识库/知识图谱片段，请仅依据任务约束生成。"
        user_prompt = (
            f"instruction: 请为{type_display}生成 {desired_count} 个可进入代理模型初筛的初始候选方案，"
            "用自然语言工程回答给出候选方案表和推荐理由。\n\n"
            "input:\n"
            "任务类型：批量候选方案生成\n"
            f"候选数量：{desired_count}\n"
            f"用户已给信息：\n{chr(10).join(f'- {line}' for line in fact_lines) if fact_lines else '- 用户未显式给出更多设计事实'}\n\n"
            "当前规范化任务约束：\n"
            f"- 设计对象：{task_payload['application']}\n"
            f"- 筋条类型：{type_display}\n"
            f"- 工况：{describe_load_conditions(task_payload['load_conditions'])}\n"
            f"- 边界：{describe_boundary_conditions(task_payload['boundary_conditions'])}\n"
            f"- BLF 目标：不低于 {task_payload['design_targets']['BLF_min']}\n"
            f"- 优化目标：{task_payload['design_targets']['primary_objective']}\n"
            f"可选材料体系：{', '.join(material_options)}。\n"
            f"系统候选字段约束：\n{chr(10).join(f'- {line}' for line in constraint_lines)}\n\n"
            f"外部知识库/知识图谱依据：\n{knowledge_text}\n"
            "\noutput 要求：\n"
            f"1. 候选表给出 {desired_count} 行。\n"
            "2. 表格列使用：编号 | 材料 | 壁板长度(mm) | 壁板宽度(mm) | 蒙皮厚度(mm) | 筋距(mm) | 筋高(mm) | 腹板厚度(mm) | 翼缘宽度(mm) | 翼缘厚度(mm) | 帽顶宽度(mm) | 帽顶厚度(mm) | 铺层 | f0 | f45 | f90 | 推荐理由。\n"
            "3. 对 BLADE、T、L 不需要的几何列可以填“-”，但该筋型必需参数不得缺失。\n"
            "4. 推荐理由必须同时包含结构性能依据和制造风险依据。\n"
            "5. 不要输出 JSON。"
        )
        return system_prompt, user_prompt

    def _split_markdown_row(self, line: str) -> List[str]:
        return [cell.strip() for cell in line.strip().strip("|").split("|")]

    def _is_markdown_separator(self, line: str) -> bool:
        cells = self._split_markdown_row(line)
        return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) for cell in cells)

    def _first_number(self, text: Any) -> float | None:
        match = re.search(r"[-+]?[0-9]+(?:\.[0-9]+)?", str(text))
        if not match:
            return None
        try:
            return float(match.group(0))
        except ValueError:
            return None

    def _parse_ratio_cell(self, text: Any) -> float | None:
        value = self._first_number(text)
        if value is None:
            return None
        lowered = str(text).lower()
        if "%" in lowered or value > 1.0:
            return value / 100.0
        return value

    def _looks_like_material_text(self, text: str) -> bool:
        stripped = str(text or "").strip()
        if not stripped or stripped in {"-", "--", "无"}:
            return False
        lowered = stripped.lower()
        return any(
            token in lowered
            for token in ["t300", "as4", "im7", "t700", "5208", "3501", "8552", "epoxy"]
        )

    def _material_from_text(self, text: str) -> Dict[str, Any]:
        lowered = str(text or "").strip().lower()
        for material in self.material_catalog:
            name = str(material.get("name") or "")
            key = str(material.get("material_key") or "")
            if (name and name.lower() in lowered) or (key and key.lower() in lowered):
                return dict(material)
        if self._looks_like_material_text(text):
            return {"name": str(text).strip()}
        return {}

    def _geometry_key_from_header(self, header: str) -> str | None:
        lowered = header.lower()
        if "翼缘宽" in header or "flange_width" in lowered:
            return "flange_width_mm"
        if "翼缘厚" in header or "flange_thickness" in lowered:
            return "flange_thickness_mm"
        if "帽顶宽" in header or "cap_width" in lowered:
            return "cap_width_mm"
        if "帽顶厚" in header or "cap_thickness" in lowered:
            return "cap_thickness_mm"
        if "蒙皮" in header or "skin" in lowered:
            return "skin_thickness_mm"
        if "筋距" in header or "pitch" in lowered:
            return "pitch_mm"
        if "筋高" in header or "stiffener_height" in lowered:
            return "stiffener_height_mm"
        if "腹板" in header or "web" in lowered:
            return "web_thickness_mm"
        if "长度" in header or "panel_length" in lowered:
            return "panel_length_mm"
        if "宽度" in header or "panel_width" in lowered:
            return "panel_width_mm"
        return None

    def _parse_candidate_text(self, text: str, headers: List[str] | None = None) -> Dict[str, Any] | None:
        headers = headers or []
        cells = self._split_markdown_row(text) if headers else [str(text)]
        if headers and len(cells) < len(headers):
            return None
        raw: Dict[str, Any] = {
            "geometry": {},
            "layup": {},
            "material_system": {},
            "rationale": "",
            "origin_summary": str(text).strip(),
        }
        rationale_parts: List[str] = []

        for header, cell in (zip(headers, cells) if headers else [("", cells[0])]):
            header_text = str(header or "")
            cell_text = str(cell or "").strip()
            lowered = header_text.lower()
            geometry_key = self._geometry_key_from_header(header_text)
            if geometry_key:
                value = self._first_number(cell_text)
                if value is not None:
                    raw["geometry"][geometry_key] = value
                continue
            if "材料" in header_text or "material" in lowered:
                raw["material_system"] = self._material_from_text(cell_text)
                continue
            if "铺层" in header_text or "layup" in lowered:
                if cell_text and cell_text not in {"-", "--"}:
                    raw["layup"]["skin_layup"] = cell_text
                continue
            if lowered in {"f0", "skin_f0"} or "0°" in header_text:
                value = self._parse_ratio_cell(cell_text)
                if value is not None:
                    raw["layup"]["skin_f0"] = value
                continue
            if lowered in {"f45", "skin_f45"} or "45" in header_text:
                value = self._parse_ratio_cell(cell_text)
                if value is not None:
                    raw["layup"]["skin_f45"] = value
                continue
            if lowered in {"f90", "skin_f90"} or "90" in header_text:
                value = self._parse_ratio_cell(cell_text)
                if value is not None:
                    raw["layup"]["skin_f90"] = value
                continue
            if any(keyword in header_text for keyword in ["理由", "依据", "风险", "说明", "推荐"]):
                rationale_parts.append(cell_text)

        if rationale_parts:
            raw["rationale"] = "；".join(part for part in rationale_parts if part)
        elif not headers:
            raw["rationale"] = str(text).strip()
        return raw if raw["geometry"] or raw["material_system"] or raw["layup"] else None

    def _parse_markdown_tables(self, text: str) -> List[Dict[str, Any]]:
        lines = str(text or "").splitlines()
        candidates: List[Dict[str, Any]] = []
        index = 0
        while index < len(lines):
            if "|" not in lines[index] or index + 1 >= len(lines) or not self._is_markdown_separator(lines[index + 1]):
                index += 1
                continue
            headers = self._split_markdown_row(lines[index])
            index += 2
            while index < len(lines) and "|" in lines[index]:
                if self._is_markdown_separator(lines[index]):
                    index += 1
                    continue
                raw = self._parse_candidate_text(lines[index], headers)
                if raw is not None:
                    candidates.append(raw)
                index += 1
        return candidates

    def _extract_candidates_from_natural_answer(self, text: str) -> List[Dict[str, Any]]:
        return self._parse_markdown_tables(text)

    def _merge_user_facts_into_raw_candidate(self, task: Dict, raw: Dict[str, Any]) -> Dict[str, Any]:
        task_payload = self._task_payload(task)
        facts = dict(task_payload.get("user_input_facts") or {})
        merged = dict(raw)
        geometry = dict(raw.get("geometry") or {})
        user_geometry = facts.get("geometry") if isinstance(facts.get("geometry"), dict) else {}
        for key in ["panel_length_mm", "panel_width_mm"]:
            if user_geometry.get(key) is not None:
                geometry[key] = user_geometry[key]
        if geometry:
            merged["geometry"] = geometry
        if isinstance(facts.get("material_system"), dict) and facts["material_system"].get("name"):
            merged["material_system"] = dict(facts["material_system"])
        return merged

    def _missing_required_geometry(self, raw_geometry: Any, stype: str) -> List[str]:
        geometry = raw_geometry if isinstance(raw_geometry, dict) else {}
        return [key for key in required_geometry_params(stype) if geometry.get(key) is None]

    def _llm_raw_candidate_is_usable(self, task: Dict, raw: Dict[str, Any]) -> bool:
        stype = resolve_stiffener_type(raw.get("stiffener_type") or self._task_payload(task).get("stiffener_type"))
        if self._missing_required_geometry(raw.get("geometry"), stype):
            return False
        layup = raw.get("layup", {})
        if not isinstance(layup, dict) or not layup.get("skin_layup"):
            return False
        return True

    def _normalize_candidate(self, task: Dict, raw: Dict[str, Any], index: int, source: str) -> Dict:
        task_payload = self._task_payload(task)
        stype = resolve_stiffener_type(
            raw.get("stiffener_type") or task_payload.get("stiffener_type", "T")
        )
        geometry = normalize_geometry(stype, raw.get("geometry"))
        layup = raw.get("layup", {})
        if not isinstance(layup, dict):
            layup = {}
        material_system = self._resolve_material_system(task, raw.get("material_system"), index)
        session_candidate_id = format_temp_candidate_id(index)

        candidate = {
            "candidate_id": session_candidate_id,
            "source": source,
            "stiffener_type": stype,
            "geometry": geometry,
            "layup": {
                "skin_layup": self._normalize_layup_string(layup.get("skin_layup", "[45/-45/0/90/0/-45/45]s")),
                "skin_f0": self._normalize_ratio(layup.get("skin_f0"), 0.286),
                "skin_f45": self._normalize_ratio(layup.get("skin_f45"), 0.571),
                "skin_f90": self._normalize_ratio(layup.get("skin_f90"), 0.143),
            },
            "rule_check": {},
            "surrogate_BLF": None,
            "rank_score": None,
            "rationale": str(raw.get("rationale", f"{source} 生成候选")),
            "origin_summary": str(raw.get("origin_summary") or ""),
            "screening_summary": None,
            "selection_reason": None,
            "display_name": str(raw.get("display_name") or session_candidate_id),
            "material_system": material_system,
            "load_conditions": task_payload["load_conditions"],
            "boundary_conditions": task_payload["boundary_conditions"],
            "design_targets": task_payload["design_targets"],
        }
        candidate["rule_check"] = self.rule_checker.run(
            candidate, strict_solver_window=True, stiffener_type=stype,
        )
        validate_or_raise("candidate.schema.json", candidate)
        return candidate

    def _finalize_candidate_identity(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        updated = dict(candidate)
        candidate_id = str(updated.get("candidate_id") or "").strip()
        if not candidate_id:
            raise SchemaValidationError("候选缺少 candidate_id")
        updated["candidate_id"] = candidate_id
        updated["display_name"] = str(updated.get("display_name") or candidate_id)
        if not updated.get("persistent_candidate_id"):
            updated.pop("persistent_candidate_id", None)
        validate_or_raise("candidate.schema.json", updated)
        return updated

    def _llm_candidates(self, task: Dict, start_index: int, desired_count: int) -> List[Dict]:
        if self.llm_backend is None or desired_count <= 0:
            return []

        knowledge_guidance = self._knowledge_guidance(task, top_k=max(3, desired_count))
        system_prompt, user_prompt = self._build_prompt(task, knowledge_guidance, desired_count)

        for _ in range(int(self.llm_config["fallback"]["max_format_retries"])):
            try:
                answer = self.llm_backend.chat(
                    system_prompt,
                    user_prompt,
                    max_tokens_override=max(int(self.llm_backend.max_tokens), 4096),
                )
                items = self._extract_candidates_from_natural_answer(answer)
                if not items:
                    raise SchemaValidationError("LLM 自然语言回答中没有可解析的候选表")
                usable_items = []
                for raw in items:
                    merged_raw = self._merge_user_facts_into_raw_candidate(task, raw)
                    if self._llm_raw_candidate_is_usable(task, merged_raw):
                        usable_items.append(merged_raw)
                if not usable_items:
                    raise SchemaValidationError("LLM 自然语言回答中没有完整候选参数")
                normalized = [
                    self._normalize_candidate(task, raw, start_index + offset, "LLM")
                    for offset, raw in enumerate(usable_items)
                ]
                return normalized[:desired_count]
            except Exception as exc:
                self.emit(f"LLM 生成失败，准备重试：{exc}")
        return []

    def _case_transfer_candidates(self, task: Dict, start_index: int, desired_count: int) -> List[Dict]:
        """从结构化历史案例中迁移候选，不向 LLM 提供历史案例原文。"""
        if desired_count <= 0:
            return []
        task_payload = self._task_payload(task)
        transferred: List[Dict] = []
        for offset, case in enumerate(self.case_retriever.retrieve_transferable_cases(task_payload, top_k=desired_count)):
            raw_design = case.get("design", {})
            if not isinstance(raw_design, dict) or not raw_design:
                continue
            candidate = self._normalize_candidate(task, raw_design, start_index + offset, "CASE_TRANSFER")
            candidate["rationale"] = f"参考历史案例 {case.get('case_id', 'UNKNOWN')} 微调生成"
            transferred.append(candidate)
            if len(transferred) >= desired_count:
                break
        return transferred

    def _resolve_source_targets(self, task: Dict) -> Dict[str, Any]:
        target_total = requested_candidate_pool_size(task)
        if target_total <= 0:
            target_total = int(
                self.app_config.get("pipeline", {}).get("default_total_candidates", 10)
            )

        preferences = self._task_payload(task).get("candidate_generation_preferences", {})
        ratio = self._sanitize_source_ratio(preferences.get("source_ratio") or self._source_ratio_from_config())
        allocated = self._allocate_by_ratio(target_total, ratio)
        return {
            "total": target_total,
            "llm": allocated["llm"],
            "case_transfer": allocated["case_transfer"],
            "doe": allocated["doe"],
            "source_ratio": ratio,
        }

    def run(self, task: Dict) -> List[Dict]:
        candidates: List[Dict] = []
        next_index = 1
        source_targets = self._resolve_source_targets(task)

        llm_candidates = self._llm_candidates(task, next_index, source_targets["llm"])
        valid_llm_candidates = [candidate for candidate in llm_candidates if candidate["rule_check"]["is_valid"]]
        candidates.extend(valid_llm_candidates)
        next_index += len(llm_candidates)

        transfer_candidates = self._case_transfer_candidates(task, next_index, source_targets["case_transfer"])
        valid_transfer_candidates = [candidate for candidate in transfer_candidates if candidate["rule_check"]["is_valid"]]
        candidates.extend(valid_transfer_candidates)
        next_index += len(transfer_candidates)

        doe_count = max(source_targets["total"] - len(candidates), 0)
        stype = task_payload_from_request(task).get("stiffener_type", "T")
        doe_candidates = self.doe_sampler.sample_candidates(
            task,
            n_samples=doe_count,
            start_index=next_index,
            strict_solver_window=True,
            stiffener_type=stype,
            id_factory=format_temp_candidate_id,
        )
        candidates.extend(doe_candidates)
        candidates = [
            self._finalize_candidate_identity(candidate)
            for candidate in candidates[: source_targets["total"]]
        ]
        self.emit(
            "候选生成完成："
            f"目标总数 {source_targets['total']}，"
            f"来源比例 LLM:CASE_TRANSFER:DOE = "
            f"{source_targets['source_ratio']['llm']:g}:"
            f"{source_targets['source_ratio']['case_transfer']:g}:"
            f"{source_targets['source_ratio']['doe']:g}，"
            f"LLM 原始 {len(llm_candidates)} / 保留 {len(valid_llm_candidates)}，"
            f"案例迁移 原始 {len(transfer_candidates)} / 保留 {len(valid_transfer_candidates)}，"
            f"DOE {len(doe_candidates)}"
        )
        return candidates
