"""UI 无关的对话流程控制层。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from agents.orchestrator import OrchestratorAgent
from core.task_contract import (
    describe_boundary_conditions,
    describe_load_conditions,
    effective_screen_top_k,
    requested_candidate_pool_size,
    requested_screen_top_k,
    task_payload_from_request,
)


ConversationEventCallback = Optional[Callable[[str, str, Dict], None]]


def _source_counter(candidates: List[Dict]) -> Dict[str, int]:
    counter: Dict[str, int] = {}
    for candidate in candidates:
        source = str(candidate.get("source", "UNKNOWN"))
        counter[source] = counter.get(source, 0) + 1
    return counter


@dataclass
class ConversationState:
    instruction: str = ""
    task: Dict | None = None
    candidates: List[Dict] = field(default_factory=list)
    screened_candidates: List[Dict] = field(default_factory=list)
    evaluated_candidates: List[Dict] = field(default_factory=list)
    results: List[Dict] = field(default_factory=list)
    report: Dict | None = None
    pending_confirmation: str | None = None
    stage: str = "idle"
    screen_skipped: bool = False


class ConversationFlowController:
    """把主流程组织成带关键确认节点的对话状态机。"""

    def __init__(self, orchestrator: OrchestratorAgent, event_callback: ConversationEventCallback = None) -> None:
        self.orchestrator = orchestrator
        self.event_callback = event_callback

    def _emit(self, event_type: str, message: str, payload: Dict | None = None) -> None:
        if self.event_callback:
            self.event_callback(event_type, message, payload or {})

    def _target_counts(self, task: Dict | None, candidate_count: int) -> Dict[str, int]:
        return {
            "candidate_pool_target": requested_candidate_pool_size(task),
            "requested_top_k": requested_screen_top_k(task),
            "effective_top_k": effective_screen_top_k(task, candidate_count),
        }

    def _render_commentary(self, stage: str, payload: Dict) -> str:
        if stage == "task_summary":
            task = payload.get("task", {})
            task_payload = task_payload_from_request(task)
            target_total = requested_candidate_pool_size(task)
            top_k = requested_screen_top_k(task)
            return (
                f"需求已解析为结构化任务：工况为 {describe_load_conditions(task_payload.get('load_conditions', {}))}，"
                f"边界为 {describe_boundary_conditions(task_payload.get('boundary_conditions', {}))}；"
                f"候选池目标为 {target_total} 个，代理初筛目标为 Top-{top_k}。"
            )

        if stage == "candidate_summary":
            return (
                f"候选池已生成并通过数量、物理唯一性和任务约束校验，共 {payload.get('candidate_count', 0)} 个；"
                "代理初筛是否执行由下一步确认决定。"
            )
        if stage == "screening_summary":
            decision = payload.get("decision", {})
            if decision.get("action") == "direct_fem":
                return "代理不可用，候选按生成顺序和求解名额进入 FEM，未作性能排序。原因：" + decision["reason"]
            if payload.get("screen_skipped"):
                return "代理初筛已跳过，全部候选等待 FEM 校核。"
            return (
                f"代理初筛已完成，输入 {payload.get('input_count', 0)} 个候选，"
                f"选择 {payload.get('output_count', 0)} 个进入 FEM；排序依据为当前任务的代理筛选契约。"
            )
        if stage == "fem_summary":
            return (
                f"有限元校核已完成，共返回 {payload.get('result_count', 0)} 条结果，"
                f"其中 {payload.get('passed_count', 0)} 条满足任务阈值；下一步为导出或跳过报告。"
            )
        if stage == "report_summary":
            report = payload.get("report")
            if report:
                return f"报告已导出：{report.get('markdown_path')}；PDF：{report.get('pdf_path')}。"
            return "报告导出已跳过，当前会话没有报告工件。"
        if stage == "conversation_paused":
            return "流程已暂停在有限元校核前；当前候选和已生成的筛选结果仍保留在会话状态中。"
        return ""

    def _emit_commentary(self, stage: str, payload: Dict) -> None:
        note = self._render_commentary(stage, payload)
        if note:
            self._emit("assistant_commentary", note, {"stage": stage, **payload})

    def start(self, instruction: str) -> ConversationState:
        state = ConversationState(instruction=instruction, stage="parsing")
        self._emit("conversation_started", "已接收设计需求，正在解析任务并生成初始候选。", {"instruction": instruction})

        task = self.orchestrator.parse_instruction(instruction)
        task_payload = task_payload_from_request(task)
        candidates = self.orchestrator.generate_candidates(task)

        state.task = task
        state.candidates = candidates
        state.stage = "awaiting_screen_confirmation"
        state.pending_confirmation = "screen_candidates"

        source_counter = _source_counter(candidates)
        target_counts = self._target_counts(task, len(candidates))
        self._emit(
            "task_summary",
            (
                f"任务摘要：{task_payload['application']} | "
                f"{describe_load_conditions(task_payload['load_conditions'])} | "
                f"{describe_boundary_conditions(task_payload['boundary_conditions'])} | "
                f"候选池目标 {target_counts['candidate_pool_target']} 个 | "
                f"初筛保留 Top-{target_counts['requested_top_k']}"
            ),
            {"task": task},
        )
        self._emit_commentary("task_summary", {"task": task})
        self._emit(
            "candidate_summary",
            (
                f"初始候选目标 {target_counts['candidate_pool_target']} 个，实际生成 {len(candidates)} 个，来源拆分："
                + " / ".join(f"{key}={value}" for key, value in sorted(source_counter.items()))
            ),
            {
                "candidates": candidates,
                "source_counter": source_counter,
                **target_counts,
            },
        )
        self._emit_commentary(
            "candidate_summary",
            {
                "candidate_count": len(candidates),
                "source_counter": source_counter,
                **target_counts,
            },
        )
        screen_note = ""
        if target_counts["effective_top_k"] < target_counts["requested_top_k"]:
            screen_note = f" 当前只有 {len(candidates)} 个候选，因此最多保留 {target_counts['effective_top_k']} 个。"
        self._emit(
            "confirmation_requested",
            (
                f"是否进行代理模型初筛？系统将按 {self.orchestrator.screener.score_formula_text} 对候选排序，"
                f"目标保留 Top-{target_counts['requested_top_k']}。"
                f"{screen_note}"
            ),
            {
                "confirmation_id": "screen_candidates",
                "default": True,
                "score_formula": self.orchestrator.screener.score_formula_text,
                **target_counts,
            },
        )
        return state

    def continue_after_confirmation(self, state: ConversationState, approved: bool) -> ConversationState:
        if state.pending_confirmation == "screen_candidates":
            return self._handle_screen_confirmation(state, approved)
        if state.pending_confirmation == "fem_evaluation":
            return self._handle_fem_confirmation(state, approved)
        if state.pending_confirmation == "export_report":
            return self._handle_report_confirmation(state, approved)
        return state

    def _handle_screen_confirmation(self, state: ConversationState, approved: bool) -> ConversationState:
        if state.task is None:
            return state
        target_counts = self._target_counts(state.task, len(state.candidates))

        if approved:
            screened_candidates = self.orchestrator.screen_candidates(state.task, state.candidates)
            state.screened_candidates = screened_candidates
            state.evaluated_candidates = screened_candidates
            decision = screened_candidates[0].get("screening_decision", {}) if screened_candidates else {}
            self._emit(
                "screening_summary",
                (
                    f"候选选择方式 {decision.get('action', 'surrogate_ranking')}：{len(state.candidates)} -> {len(screened_candidates)}，"
                    f"请求 Top-{target_counts['requested_top_k']}。"
                ),
                {
                    "screened_candidates": screened_candidates,
                    "decision": decision,
                    **target_counts,
                },
            )
            self._emit_commentary(
                "screening_summary",
                {
                    "input_count": len(state.candidates),
                    "output_count": len(screened_candidates),
                    "selected_candidates": screened_candidates[:3],
                    "decision": decision,
                    **target_counts,
                },
            )
        else:
            state.screen_skipped = True
            state.screened_candidates = []
            state.evaluated_candidates = list(state.candidates)
            self._emit(
                "screening_summary",
                f"已跳过代理模型初筛，将直接对全部 {len(state.candidates)} 个候选进入有限元校核。",
                {"screened_candidates": state.candidates, "screen_skipped": True, **target_counts},
            )
            self._emit_commentary(
                "screening_summary",
                {
                    "input_count": len(state.candidates),
                    "output_count": len(state.candidates),
                    "screen_skipped": True,
                    **target_counts,
                },
            )

        state.pending_confirmation = "fem_evaluation"
        state.stage = "awaiting_fem_confirmation"
        fem_targets = state.evaluated_candidates
        preview_reasons = [candidate.get("selection_reason") for candidate in fem_targets[:3] if candidate.get("selection_reason")]
        detail_suffix = f" 重点入选原因：{' | '.join(preview_reasons)}" if preview_reasons else ""
        self._emit(
            "confirmation_requested",
            f"是否进行有限元校核？当前待校核样本 {len(fem_targets)} 个。{detail_suffix}",
            {"confirmation_id": "fem_evaluation", "default": True, "candidate_count": len(fem_targets)},
        )
        return state

    def _handle_fem_confirmation(self, state: ConversationState, approved: bool) -> ConversationState:
        if state.task is None:
            return state

        if not approved:
            state.pending_confirmation = None
            state.stage = "paused_before_fem"
            self._emit(
                "conversation_paused",
                "已暂停在有限元校核前。当前候选和代理模型结果已保留，可稍后继续。",
                {"stage": state.stage},
            )
            self._emit_commentary("conversation_paused", {"stage": state.stage})
            return state

        results = [self.orchestrator.evaluate_candidate(state.task, candidate) for candidate in state.evaluated_candidates]
        state.results = results
        passed_count = sum(1 for result in results if result.get("verdict") == "通过")
        state.pending_confirmation = "export_report"
        state.stage = "awaiting_report_confirmation"

        self._emit(
            "fem_summary",
            f"有限元校核完成：共 {len(results)} 个样本，其中通过 {passed_count} 个。",
            {"results": results, "passed_count": passed_count},
        )
        self._emit_commentary(
            "fem_summary",
            {
                "result_count": len(results),
                "passed_count": passed_count,
                "results": results[:3],
            },
        )
        self._emit(
            "confirmation_requested",
            "是否导出设计报告？报告将包含工况摘要、代理模型选择理由、有限元结果解读和工程建议。",
            {"confirmation_id": "export_report", "default": True},
        )
        return state

    def _handle_report_confirmation(self, state: ConversationState, approved: bool) -> ConversationState:
        if state.task is None:
            return state

        if approved:
            state.report = self.orchestrator.generate_report(state.task, state.results, state.candidates, state.evaluated_candidates)
            self._emit(
                "report_summary",
                (
                    f"报告已导出：{state.report.get('markdown_path')} / "
                    f"{state.report.get('pdf_path')}"
                ),
                {"report": state.report},
            )
            self._emit_commentary("report_summary", {"report": state.report})
        else:
            self._emit("report_summary", "已跳过报告导出。", {"report": None})
            self._emit_commentary("report_summary", {"report": None, "skipped": True})

        state.pending_confirmation = None
        state.stage = "completed"
        return state
