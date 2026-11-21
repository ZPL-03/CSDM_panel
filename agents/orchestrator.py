"""主控智能体，支持整链路运行与分阶段调用。"""

from __future__ import annotations

from typing import Callable, Dict, List
import re

from agents.base import BaseAgent
from agents.candidate_gen import CandidateGenAgent
from agents.fem_agent import FEMAgent
from agents.knowledge_agent import KnowledgeAgent
from agents.report_gen import ReportGenAgent
from agents.screener import ScreenerAgent
from core.id_utils import format_candidate_id, next_candidate_index, task_file_name
from core.io_utils import write_json
from core.paths import TASKS_DIR
from core.laminate import analysis_contract
from core.task_contract import (
    build_design_constraints,
    validate_candidate_constraints,
    describe_boundary_conditions,
    describe_load_conditions,
    effective_screen_top_k,
    requested_candidate_pool_size,
    requested_screen_top_k,
    summarize_task,
    task_payload_from_request,
)
from core.task_parser import TaskParser
from core.surrogate_model import SurrogateUnavailable


class OrchestratorAgent(BaseAgent):
    agent_name = "ORCHESTRATOR"

    def __init__(self, progress_callback: Callable[[str, str], None] | None = None) -> None:
        super().__init__(progress_callback=progress_callback)
        self.task_parser = TaskParser()
        self.candidate_gen = CandidateGenAgent(progress_callback)
        self.screener = ScreenerAgent(progress_callback)
        self.fem_agent = FEMAgent(progress_callback)
        self.knowledge_agent = KnowledgeAgent(progress_callback)
        self.report_gen = ReportGenAgent(progress_callback)

    def _task_payload(self, task: Dict) -> Dict:
        return task_payload_from_request(task)

    def _task_event_context(self, task: Dict) -> Dict:
        return {
            "task_id": task.get("task_id"),
            "created_at": task.get("created_at"),
            "source": task.get("source"),
        }

    def _attach_task_context(self, task: Dict, candidate: Dict) -> Dict:
        task_payload = self._task_payload(task)
        enriched = dict(candidate)
        enriched.pop("task_id", None)
        candidate_id = str(enriched.get("candidate_id") or "").strip()
        if not candidate_id:
            raise ValueError("候选缺少 candidate_id，无法进入主流程")
        enriched["candidate_id"] = candidate_id
        enriched["display_name"] = str(enriched.get("display_name") or candidate_id)
        if not enriched.get("persistent_candidate_id"):
            enriched.pop("persistent_candidate_id", None)
        enriched["design_targets"] = dict(candidate.get("design_targets") or task_payload["design_targets"])
        enriched["load_conditions"] = dict(candidate.get("load_conditions") or task_payload["load_conditions"])
        enriched["boundary_conditions"] = dict(candidate.get("boundary_conditions") or task_payload["boundary_conditions"])
        enriched["material_system"] = dict(candidate.get("material_system") or task_payload["material_system"])
        constraints = build_design_constraints(task, enriched["stiffener_type"])
        validate_candidate_constraints(enriched, constraints)
        enriched["task_constraints"] = constraints
        return enriched

    def _promote_candidate_for_fem(self, task: Dict, candidate: Dict) -> Dict:
        enriched_candidate = self._attach_task_context(task, candidate)
        persistent_candidate_id = enriched_candidate.get("persistent_candidate_id")
        if not persistent_candidate_id:
            persistent_candidate_id = format_candidate_id(next_candidate_index())
            candidate["persistent_candidate_id"] = persistent_candidate_id
            enriched_candidate["persistent_candidate_id"] = persistent_candidate_id

        session_candidate_id = enriched_candidate.get("candidate_id")
        promoted = dict(enriched_candidate)
        promoted["session_candidate_id"] = session_candidate_id
        promoted["candidate_id"] = persistent_candidate_id
        promoted["display_name"] = persistent_candidate_id
        promoted.pop("persistent_candidate_id", None)
        return promoted

    def parse_instruction(self, text: str) -> Dict:
        task = self.task_parser.parse_instruction(text)

        # 写入任务台账
        task_path = TASKS_DIR / task_file_name(task["task_id"])
        write_json(task_path, task)

        summary = summarize_task(task)
        self.emit_event(
            "task_parsed",
            (
                f"任务已结构化：{summary['application']} | "
                f"{summary['load_conditions']} | "
                f"{summary['boundary_conditions']} | "
                f"{summary['candidate_pool']} | "
                f"{summary['top_k']}"
            ),
            {"task": task, "summary": summary, **self._task_event_context(task)},
        )
        return task

    def generate_candidates(self, task: Dict) -> List[Dict]:
        task_payload = self._task_payload(task)
        target_total = requested_candidate_pool_size(task)
        self.emit_event(
            "candidate_generation_started",
            (
                f"开始生成候选方案：{describe_load_conditions(task_payload['load_conditions'])} / "
                f"{describe_boundary_conditions(task_payload['boundary_conditions'])} / "
                f"候选池目标 {target_total} 个"
            ),
            {**self._task_event_context(task), "target_total_candidates": target_total},
        )
        candidates = [self._attach_task_context(task, candidate) for candidate in self.candidate_gen.run(task)]
        self._validate_pool(candidates, expected_count=target_total)
        source_counter: Dict[str, int] = {}
        for candidate in candidates:
            source = str(candidate.get("source", "UNKNOWN"))
            source_counter[source] = source_counter.get(source, 0) + 1
        self.emit_event(
            "candidate_generation_completed",
            f"候选生成完成，目标 {target_total} 个，实际 {len(candidates)} 个："
            + " / ".join(f"{key}={value}" for key, value in sorted(source_counter.items())),
            {
                **self._task_event_context(task),
                "candidate_count": len(candidates),
                "target_total_candidates": target_total,
                "source_counter": source_counter,
            },
        )
        return candidates

    def _validate_pool(self, candidates: List[Dict], expected_count: int | None = None) -> None:
        """公开阶段入口核对身份和物理唯一性，不信任上游返回的数量或编号。"""
        ids = [candidate.get("candidate_id") for candidate in candidates]
        if any(not isinstance(cid, str) or not re.fullmatch(r"(?:TMP_|C)[1-9][0-9]*", cid) for cid in ids):
            raise ValueError("候选池包含无效编号")
        signatures = [self.candidate_gen._candidate_signature(candidate) for candidate in candidates]
        if len(set(ids)) != len(ids) or len(set(signatures)) != len(signatures):
            raise ValueError("候选池存在重复编号或物理设计")
        if expected_count is not None and len(candidates) != expected_count:
            raise ValueError("候选池数量与任务要求不一致")

    def screen_candidates(self, task: Dict, candidates: List[Dict]) -> List[Dict]:
        score_formula = self.screener.score_formula_text
        requested_top_k = requested_screen_top_k(task)
        effective_top_k = effective_screen_top_k(task, len(candidates))
        self.emit_event(
            "screening_started",
            (
                f"开始执行代理模型初筛，输入候选 {len(candidates)} 个，"
                f"目标保留 Top-{requested_top_k}，"
                f"当前最多保留 {effective_top_k} 个，"
                f"评分公式：{score_formula}"
            ),
            {
                **self._task_event_context(task),
                "input_count": len(candidates),
                "score_formula": score_formula,
                "requested_top_k_candidates": requested_top_k,
                "effective_top_k_candidates": effective_top_k,
            },
        )
        candidates = [self._attach_task_context(task, candidate) for candidate in candidates]
        self._validate_pool(candidates)
        try:
            screened = self.screener.run({"task": task, "candidates": candidates})
            decision = {"status": "available", "action": "surrogate_ranking"}
        except SurrogateUnavailable as exc:
            policy = self.screener.config["pipeline"]["unavailable_surrogate_policy"]
            decision = {**exc.to_dict(), "action": policy, "selection_basis": "generation_order_not_performance"}
            if policy not in ("direct_fem", "stop"):
                raise ValueError(f"无效代理不可用策略: {policy}") from exc
            screened = [] if policy == "stop" else [dict(candidate) for candidate in candidates[:effective_top_k]]
            for candidate in screened:
                candidate.update({"surrogate_BLF": None, "rank_score": None, "surrogate_weight": None,
                                  "surrogate_status": exc.status,
                                  "screening_summary": exc.reason,
                                  "selection_reason": "代理不可用，按生成顺序和任务求解名额选择直接 FEM；不是性能排序。"})
            self.emit_event("surrogate_unavailable", exc.reason, decision)
        self._validate_pool(screened, expected_count=0 if decision["action"] == "stop" else effective_top_k)
        source_contracts = {candidate["candidate_id"]: analysis_contract(candidate) for candidate in candidates}
        for candidate in screened:
            if source_contracts.get(candidate["candidate_id"]) != analysis_contract(candidate):
                raise ValueError("筛选结果不属于输入候选或改变了物理输入")
        decision.update({"input_ids": [c["candidate_id"] for c in candidates],
                         "selected_ids": [c["candidate_id"] for c in screened],
                         "requested_top_k": requested_top_k})
        write_json(TASKS_DIR / "screening" / f"{task['task_id']}.json", decision)
        if decision["action"] == "stop":
            raise RuntimeError(f"预筛终止: {decision['reason']}")
        for candidate in screened:
            candidate["screening_decision"] = decision
        self.emit_event(
            "screening_completed",
            f"候选选择完成，方式 {decision['action']}，请求 {requested_top_k}，实际输出 {len(screened)} 个。",
            {
                **self._task_event_context(task),
                "input_count": len(candidates),
                "output_count": len(screened),
                "score_formula": score_formula,
                "requested_top_k_candidates": requested_top_k,
                "effective_top_k_candidates": effective_top_k,
                "selected_candidates": screened,
                "decision": decision,
            },
        )
        return [self._attach_task_context(task, candidate) for candidate in screened]

    def evaluate_candidate(self, task: Dict, candidate: Dict) -> Dict:
        fem_candidate = self._promote_candidate_for_fem(task, candidate)
        self.emit_event(
            "fem_started",
            f"开始校核 {candidate.get('display_name', fem_candidate.get('session_candidate_id', '候选样本'))} "
            f"-> {fem_candidate['candidate_id']}",
            {**self._task_event_context(task), "candidate": fem_candidate},
        )
        result = self.fem_agent.run(fem_candidate)
        if result.get("candidate_id") != fem_candidate["candidate_id"]:
            raise ValueError("有限元返回结果编号与提交候选不一致")
        result["session_candidate_id"] = fem_candidate.get("session_candidate_id")
        result["display_name"] = candidate.get("display_name") or fem_candidate.get("session_candidate_id")
        self.knowledge_agent.run({"task": task, "design": fem_candidate, "abaqus_results": result})
        return result

    def generate_report(self, task: Dict, results: List[Dict], candidates: List[Dict] | None = None,
                        screened_candidates: List[Dict] | None = None) -> Dict:
        self.emit_event(
            "report_started",
            "开始生成设计报告",
            {**self._task_event_context(task), "result_count": len(results)},
        )
        return self.report_gen.run({"task": task, "results": results, "candidates": candidates or [],
                                    "screened_candidates": screened_candidates})

    def run(self, user_instruction: str) -> Dict:
        self.emit("正在解析用户需求")
        task = self.parse_instruction(user_instruction)
        candidates = self.generate_candidates(task)
        top_candidates = self.screen_candidates(task, candidates)

        results: List[Dict] = []
        for candidate in top_candidates:
            results.append(self.evaluate_candidate(task, candidate))

        report = self.generate_report(task, results, candidates, top_candidates)
        return {
            "task": task,
            "candidates": candidates,
            "top_candidates": top_candidates,
            "results": results,
            "report": report,
        }
