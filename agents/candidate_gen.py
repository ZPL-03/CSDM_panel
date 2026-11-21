"""程序控制的候选池：案例迁移、DOE、物理去重和精确数量约束。"""

from __future__ import annotations

from copy import deepcopy
from typing import Dict, List

from agents.base import BaseAgent
from core.case_retriever import CaseRetriever
from core.doe_sampler import DOESampler
from core.id_utils import format_temp_candidate_id
from core.laminate import fingerprint, lamina_material, section_layers
from core.layup import normalize_layup
from core.rule_checker import RuleChecker
from core.schema_validator import validate_or_raise
from core.stiffener_profile import normalize_geometry, resolve_stiffener_type
from core.task_contract import build_design_constraints, requested_candidate_pool_size, task_payload_from_request


class CandidateGenAgent(BaseAgent):
    """LLM 不提供候选数值；生成器对每个筋型分别分配配额并校验实际物理输入。"""

    agent_name = "CANDIDATE_GEN"

    def __init__(self, progress_callback=None) -> None:
        super().__init__(progress_callback=progress_callback)
        self.doe_sampler = DOESampler()
        self.rule_checker = RuleChecker()
        self.case_retriever = CaseRetriever()

    def _requested_stiffener_types(self, task: Dict) -> List[str]:
        payload = task_payload_from_request(task)
        types = payload["candidate_generation_preferences"].get("stiffener_types") or [payload["stiffener_type"]]
        return list(dict.fromkeys(resolve_stiffener_type(kind) for kind in types))

    def _task_for_stiffener_type(self, task: Dict, stiffener_type: str) -> Dict:
        record = deepcopy(task)
        payload = task_payload_from_request(record)
        payload["stiffener_type"] = stiffener_type
        payload["candidate_generation_preferences"]["stiffener_types"] = [stiffener_type]
        facts = payload.get("user_input_facts", {})
        if facts.get("stiffener_type") or facts.get("stiffener_types"):
            facts.update(stiffener_type=stiffener_type, stiffener_types=[stiffener_type])
        if isinstance(record.get("task"), dict):
            record["task"] = payload
            return record
        return payload

    @staticmethod
    def _allocate_by_stiffener_types(total: int, stiffener_types: List[str]) -> Dict[str, int]:
        return {kind: total // len(stiffener_types) + int(index < total % len(stiffener_types))
                for index, kind in enumerate(stiffener_types)}

    def _normalize_candidate(self, task: Dict, raw: Dict, index: int, source: str) -> Dict:
        """迁移只保留设计输入，不复制历史身份、工况、代理分数或求解结论。"""
        payload = task_payload_from_request(task)
        kind = resolve_stiffener_type(raw.get("stiffener_type") or payload["stiffener_type"])
        constraints = build_design_constraints(task, kind)
        geometry = normalize_geometry(kind, raw["geometry"])
        for key, bounds in constraints["geometry_ranges"].items():
            if bounds["min"] == bounds["max"]:
                geometry[key] = bounds["min"]
        layup = deepcopy(raw["layup"])
        layup.update(normalize_layup(layup["skin_layup"]))
        material = payload["material_system"] if payload["material_system"].get("is_user_specified") else raw["material_system"]
        candidate_id = format_temp_candidate_id(index)
        candidate = dict(candidate_id=candidate_id, display_name=candidate_id, source=source,
                         stiffener_type=kind, geometry=geometry, layup=layup, task_constraints=constraints,
                         material_system=deepcopy(material), load_conditions=deepcopy(payload["load_conditions"]),
                         boundary_conditions=deepcopy(payload["boundary_conditions"]),
                         design_targets=deepcopy(payload["design_targets"]),
                         surrogate_BLF=None, surrogate_weight=None, rank_score=None,
                         rationale="历史通过案例的设计输入；当前工况须独立求解。")
        candidate["rule_check"] = self.rule_checker.run(candidate, stiffener_type=kind, task=task)
        validate_or_raise("candidate.schema.json", candidate)
        return candidate

    def _case_transfer_candidates(self, task: Dict, start_index: int, desired_count: int) -> List[Dict]:
        if desired_count == 0:
            return []
        cases = self.case_retriever.retrieve_transferable_cases(task_payload_from_request(task), top_k=desired_count)
        candidates = []
        for case in cases:
            candidate = self._normalize_candidate(task, case["design"], start_index + len(candidates), "CASE_TRANSFER")
            candidate["rationale"] = f"参考案例 {case['case_id']} 的设计输入；当前方案须独立求解。"
            if candidate["rule_check"]["is_valid"]:
                candidates.append(candidate)
        return candidates

    @staticmethod
    def _candidate_signature(candidate: Dict) -> tuple:
        return (candidate["stiffener_type"], fingerprint(lamina_material(candidate)),
                fingerprint(candidate["geometry"]), fingerprint(section_layers(candidate)))

    def run(self, task: Dict) -> List[Dict]:
        total = requested_candidate_pool_size(task)
        types = self._requested_stiffener_types(task)
        if total < len(types):
            raise ValueError("候选数不能少于指定筋型数，无法完整覆盖任务筋型")
        ratio = task_payload_from_request(task)["candidate_generation_preferences"]["source_ratio"]
        candidates, seen = [], set()
        source_counts = {"CASE_TRANSFER": 0, "DOE": 0}
        for kind, quota in self._allocate_by_stiffener_types(total, types).items():
            typed_task = self._task_for_stiffener_type(task, kind)
            transfer_quota = int(quota * ratio["case_transfer"] / sum(ratio.values()))
            pool = []

            def accept(batch):
                for candidate in batch:
                    validate_or_raise("candidate.schema.json", candidate)
                    check = self.rule_checker.run(candidate, stiffener_type=kind, task=typed_task)
                    if not check["is_valid"]:
                        raise ValueError(f"生成器返回无效候选：{check}")
                    signature = self._candidate_signature(candidate)
                    if signature not in seen and len(pool) < quota:
                        seen.add(signature)
                        candidate["rule_check"] = check
                        pool.append(candidate)
                        source_counts[candidate["source"]] += 1

            accept(self._case_transfer_candidates(typed_task, len(candidates) + 1, transfer_quota))
            # 采样预算有明确上界；不足时报告失败，不复制候选凑数或放宽设计约束。
            for batch_index in range(8):
                deficit = quota - len(pool)
                if deficit == 0:
                    break
                accept(self.doe_sampler.sample_candidates(
                    typed_task, n_samples=deficit, start_index=len(candidates) + len(pool) + 1 + batch_index * total,
                    stiffener_type=kind, id_factory=format_temp_candidate_id))
            if len(pool) != quota:
                raise RuntimeError(f"{kind} 约束内唯一候选不足：需要 {quota}，实际 {len(pool)}")
            candidates.extend(pool)
        for index, candidate in enumerate(candidates, 1):
            candidate["candidate_id"] = candidate["display_name"] = format_temp_candidate_id(index)
            candidate.pop("persistent_candidate_id", None)
        if len(candidates) != total or len(seen) != total:
            raise RuntimeError("候选池数量或物理唯一性校验失败")
        self.emit(f"候选池 {total} 个；筋型 {' / '.join(types)}；案例迁移 {source_counts['CASE_TRANSFER']}，DOE {source_counts['DOE']}。")
        return candidates
