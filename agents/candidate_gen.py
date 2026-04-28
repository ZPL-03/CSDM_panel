from __future__ import annotations

import json
from typing import Any, Dict, List

from agents.base import BaseAgent
from core.case_retriever import CaseRetriever
from core.config_loader import load_app_config, load_llm_config, load_material_db
from core.doe_sampler import DOESampler
from core.id_utils import format_temp_candidate_id
from core.literature_corpus import LiteratureCorpus
from core.llm_backend import LLMBackend
from core.rule_checker import RuleChecker
from core.schema_validator import SchemaValidationError, validate_or_raise
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
        self.literature_corpus = LiteratureCorpus()
        self.material_catalog = self._build_material_catalog()
        self.llm_backend: LLMBackend | None = None
        try:
            self.llm_backend = LLMBackend(self.llm_config)
        except Exception as exc:
            self.emit(f"LLM 后端初始化失败，将自动退回案例迁移和 DOE：{exc}")

    def _coerce_dict(self, value: Any) -> Dict[str, Any] | None:
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                return None
            if isinstance(parsed, dict):
                return parsed
        return None

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

    def _literature_guidance(self, task: Dict, top_k: int) -> List[str]:
        """仅为 LLM 路径检索文献片段，避免与历史案例迁移职责混用。"""
        return self.literature_corpus.format_snippets(task, top_k=max(1, top_k))

    def _build_prompt(self, task: Dict, literature_guidance: List[str], desired_count: int) -> tuple[str, str]:
        """构造 LLM 候选生成 Prompt，只注入任务约束和文献依据。"""
        task_payload = self._task_payload(task)
        material_options = [item.get("name", "") for item in self._material_options(task)]
        system_prompt = (
            "你是复合材料加筋壁板结构设计专家。"
            "只输出合法 JSON，不要输出解释。"
            "顶层格式必须是 {\"candidates\": [...]}。"
            "每个 candidate 必须包含 geometry、layup、rationale。"
            "当任务未固定材料时，可以为 candidate 指定 material_system。"
            "不要输出 case_id、task、abaqus_results、verdict、created_at 等历史字段。"
            "方案必须满足 T 形筋、线性屈曲场景，并与任务中的工况和边界一致。"
        )
        literature_text = "\n\n".join(literature_guidance) if literature_guidance else "当前没有可用文献片段，请仅依据任务约束生成。"
        user_prompt = (
            f"请基于以下任务和文献依据生成 {desired_count} 个候选方案，只输出 JSON。\n"
            "geometry 必须包含 panel_length_mm、panel_width_mm、skin_thickness_mm、pitch_mm、"
            "stiffener_height_mm、web_thickness_mm、flange_width_mm、flange_thickness_mm。\n"
            "layup 必须包含 skin_layup、skin_f0、skin_f45、skin_f90。\n"
            f"可选材料体系：{', '.join(material_options)}。\n"
            f"工况说明：{describe_load_conditions(task_payload['load_conditions'])}\n"
            f"边界说明：{describe_boundary_conditions(task_payload['boundary_conditions'])}\n"
            f"任务：\n{json.dumps(task_payload, ensure_ascii=False, indent=2)}\n"
            f"文献依据：\n{literature_text}"
        )
        return system_prompt, user_prompt

    def _normalize_candidate(self, task: Dict, raw: Dict[str, Any], index: int, source: str) -> Dict:
        task_payload = self._task_payload(task)
        geometry = raw.get("geometry", {})
        layup = raw.get("layup", {})
        if not isinstance(geometry, dict):
            geometry = {}
        if not isinstance(layup, dict):
            layup = {}
        material_system = self._resolve_material_system(task, raw.get("material_system"), index)

        candidate = {
            "candidate_id": format_temp_candidate_id(index),
            "source": source,
            "stiffener_type": "T",
            "geometry": {
                "panel_length_mm": float(geometry.get("panel_length_mm", 700.0)),
                "panel_width_mm": float(geometry.get("panel_width_mm", 600.0)),
                "skin_thickness_mm": float(geometry.get("skin_thickness_mm", 2.5)),
                "pitch_mm": float(geometry.get("pitch_mm", 120.0)),
                "stiffener_height_mm": float(geometry.get("stiffener_height_mm", 28.0)),
                "web_thickness_mm": float(geometry.get("web_thickness_mm", 2.0)),
                "flange_width_mm": float(geometry.get("flange_width_mm", 16.0)),
                "flange_thickness_mm": float(geometry.get("flange_thickness_mm", 2.0)),
            },
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
            "material_system": material_system,
            "load_conditions": task_payload["load_conditions"],
            "boundary_conditions": task_payload["boundary_conditions"],
            "design_targets": task_payload["design_targets"],
        }
        candidate["rule_check"] = self.rule_checker.run(candidate, strict_solver_window=True)
        validate_or_raise("candidate.schema.json", candidate)
        return candidate

    def _extract_llm_items(self, payload: Any) -> List[Dict[str, Any]]:
        if isinstance(payload, dict):
            payload = (
                payload.get("candidates")
                or payload.get("items")
                or payload.get("results")
                or payload.get("data")
                or ([payload["design"]] if isinstance(payload.get("design"), dict) else None)
                or ([payload] if isinstance(payload.get("geometry"), dict) else None)
            )
        elif isinstance(payload, str):
            parsed = self._coerce_dict(payload)
            if parsed is not None:
                return self._extract_llm_items(parsed)
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError as exc:
                raise SchemaValidationError(f"LLM 输出不是合法 JSON: {exc}") from exc

        if not isinstance(payload, list):
            raise SchemaValidationError("LLM 输出不是候选数组")

        items: List[Dict[str, Any]] = []
        for raw in payload:
            record = self._coerce_dict(raw)
            if record is None:
                continue
            items.append(record)
        return items

    def _llm_candidates(self, task: Dict, start_index: int, desired_count: int) -> List[Dict]:
        if self.llm_backend is None or desired_count <= 0:
            return []

        literature_guidance = self._literature_guidance(task, top_k=max(3, desired_count))
        system_prompt, user_prompt = self._build_prompt(task, literature_guidance, desired_count)

        for _ in range(int(self.llm_config["fallback"]["max_format_retries"])):
            try:
                payload = self.llm_backend.generate_json(system_prompt, user_prompt)
                items = self._extract_llm_items(payload)
                if not items:
                    raise SchemaValidationError("LLM 返回为空或无法解析为候选字典")
                normalized = [
                    self._normalize_candidate(task, raw, start_index + offset, "LLM")
                    for offset, raw in enumerate(items)
                ]
                return normalized[:desired_count]
            except Exception as exc:
                self.emit(f"LLM 生成失败，准备重试：{exc}")
        return []

    def _case_transfer_candidates(self, task: Dict, start_index: int, desired_count: int) -> List[Dict]:
        """从结构化历史案例中迁移候选，不向 LLM 提供历史案例原文。"""
        if desired_count <= 0:
            return []
        transferred: List[Dict] = []
        for offset, case in enumerate(self.case_retriever.retrieve_transferable_cases(task, top_k=desired_count)):
            raw_design = case.get("design", {})
            if not isinstance(raw_design, dict) or not raw_design:
                continue
            abaqus_results = case.get("abaqus_results", {})
            candidate = self._normalize_candidate(task, raw_design, start_index + offset, "CASE_TRANSFER")
            candidate["rationale"] = f"参考历史案例 {case.get('case_id', 'UNKNOWN')} 微调生成"
            transferred.append(candidate)
            if len(transferred) >= desired_count:
                break
        return transferred

    def _resolve_source_targets(self, task: Dict) -> Dict[str, int]:
        target_total = requested_candidate_pool_size(task)
        if target_total <= 0:
            target_total = int(
                self.app_config["pipeline"]["llm_candidates"]
                + self.app_config["pipeline"]["case_transfer_candidates"]
                + self.app_config["pipeline"]["doe_candidates"]
            )

        llm_default = int(self.app_config["pipeline"]["llm_candidates"])
        case_default = int(self.app_config["pipeline"]["case_transfer_candidates"])
        doe_default = int(self.app_config["pipeline"]["doe_candidates"])
        total_default = max(llm_default + case_default + doe_default, 1)

        llm_target = max(0, round(target_total * llm_default / total_default))
        case_target = max(0, round(target_total * case_default / total_default))
        if llm_default > 0 and llm_target == 0:
            llm_target = 1
        if case_default > 0 and case_target == 0 and target_total >= 3:
            case_target = 1

        overflow = max(llm_target + case_target - target_total, 0)
        if overflow > 0:
            reduce_case = min(case_target, overflow)
            case_target -= reduce_case
            overflow -= reduce_case
            llm_target = max(llm_target - overflow, 0)

        doe_target = max(target_total - llm_target - case_target, 0)
        return {
            "total": target_total,
            "llm": llm_target,
            "case_transfer": case_target,
            "doe": doe_target,
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
        doe_candidates = self.doe_sampler.sample_candidates(
            task,
            n_samples=doe_count,
            start_index=next_index,
            strict_solver_window=True,
            id_factory=format_temp_candidate_id,
        )
        candidates.extend(doe_candidates)
        candidates = candidates[: source_targets["total"]]
        self.emit(
            "候选生成完成："
            f"目标总数 {source_targets['total']}，"
            f"LLM 原始 {len(llm_candidates)} / 保留 {len(valid_llm_candidates)}，"
            f"案例迁移 原始 {len(transfer_candidates)} / 保留 {len(valid_transfer_candidates)}，"
            f"DOE {len(doe_candidates)}"
        )
        return candidates
