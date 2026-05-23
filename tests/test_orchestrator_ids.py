from __future__ import annotations

import shutil
from typing import Dict, Optional

from agents.orchestrator import OrchestratorAgent
from core.id_utils import format_candidate_id, next_candidate_index, next_case_id
from core.io_utils import write_json
from core.paths import ABAQUS_RUNS_DIR, IO_DIR

import pytest


class _FakeLLMBackend:
    def __init__(self, sink: Optional[Dict[str, str]] = None) -> None:
        self.sink = sink if sink is not None else {}
        self.max_tokens = 1800

    def chat(self, system_prompt: str, user_prompt: str, max_tokens_override: int | None = None) -> str:
        self.sink["system_prompt"] = system_prompt
        self.sink["user_prompt"] = user_prompt
        return "\n".join(
            [
                "以下为候选方案表：",
                "",
                "| 编号 | 材料 | 壁板长度(mm) | 壁板宽度(mm) | 蒙皮厚度(mm) | 筋距(mm) | 筋高(mm) | 腹板厚度(mm) | 翼缘宽度(mm) | 翼缘厚度(mm) | 帽顶宽度(mm) | 帽顶厚度(mm) | 铺层 | f0 | f45 | f90 | 推荐理由 |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | ---: | ---: | ---: | --- |",
                "| A1 | T300/5208 | 700 | 600 | 2.5 | 120 | 28 | 2.0 | 16 | 2.0 | - | - | [45/-45/0/90/0/-45/45]s | 0.286 | 0.571 | 0.143 | 结构性能与制造风险均衡 |",
            ]
        )


class _FakeKnowledge:
    def __init__(self, snippets: list[str]) -> None:
        self.snippets = snippets

    def format_snippets(self, _task, top_k: int = 5) -> list[str]:
        return self.snippets[:top_k]


def test_temporary_candidates_do_not_consume_persistent_ids() -> None:
    agent = OrchestratorAgent()
    agent.candidate_gen.llm_backend = None
    task = agent.parse_instruction("请设计一个T形加筋方案，压缩荷载为1000kN/m，生成 4 个候选，初筛保留 2 个候选")
    next_before = next_candidate_index()

    candidates = agent.generate_candidates(task)

    assert candidates
    assert all(candidate["candidate_id"].startswith("TMP_") for candidate in candidates)
    assert next_candidate_index() == next_before


def test_orchestrator_candidate_generation_uses_knowledge_base_guidance() -> None:
    captured: dict[str, str] = {}
    agent = OrchestratorAgent()
    agent.candidate_gen.knowledge_base = _FakeKnowledge(
        ["[外部知识库 1] Composite knowledge\nComposite buckling guidance"]
    )
    agent.candidate_gen.llm_backend = _FakeLLMBackend(captured)

    task = agent.parse_instruction("请设计一个T形加筋方案，压缩荷载为1000kN/m，剪切180kN/m，边界SSCC，生成 4 个候选，初筛保留 2 个候选")
    candidates = agent.generate_candidates(task)

    assert candidates
    assert "外部知识库/知识图谱依据" in captured["user_prompt"]
    assert "Composite knowledge" in captured["user_prompt"]
    assert "参考案例" not in captured["user_prompt"]
    assert candidates[0]["source"] == "LLM"
    assert candidates[0]["candidate_id"].startswith("TMP_")


def test_orchestrator_candidate_generation_without_knowledge_base_still_runs() -> None:
    agent = OrchestratorAgent()
    agent.candidate_gen.knowledge_base = _FakeKnowledge([])
    agent.candidate_gen.llm_backend = _FakeLLMBackend()

    task = agent.parse_instruction("请设计一个T形加筋方案，压缩荷载为1000kN/m，生成 4 个候选，初筛保留 2 个候选")
    candidates = agent.generate_candidates(task)

    assert candidates
    assert candidates[0]["source"] == "LLM"
    assert candidates[0]["candidate_id"].startswith("TMP_")


def test_orchestrator_candidates_keep_session_ids_before_fem() -> None:
    agent = OrchestratorAgent()
    agent.candidate_gen.knowledge_base = _FakeKnowledge([])
    agent.candidate_gen.llm_backend = _FakeLLMBackend()

    task = agent.parse_instruction("请设计一个T形加筋方案，压缩荷载为1000kN/m，生成 4 个候选，初筛保留 2 个候选")
    candidates = agent.generate_candidates(task)

    assert candidates[0]["candidate_id"].startswith("TMP_")
    assert candidates[0]["display_name"] == candidates[0]["candidate_id"]
    assert "task_id" not in candidates[0]
    assert candidates[0]["source"] == "LLM"
    assert candidates[0]["load_conditions"]["type"] == task["task"]["load_conditions"]["type"]
    assert candidates[0]["boundary_conditions"]["type"] == task["task"]["boundary_conditions"]["type"]
    assert candidates[0]["design_targets"] == task["task"]["design_targets"]
    assert candidates[0]["material_system"]
    assert candidates[0]["rule_check"]["is_valid"] is True

    promoted = agent._promote_candidate_for_fem(task, candidates[0])

    assert promoted["session_candidate_id"] == candidates[0]["candidate_id"]
    assert promoted["candidate_id"].startswith("C")
    assert promoted["display_name"] == promoted["candidate_id"]
    assert candidates[0]["persistent_candidate_id"] == promoted["candidate_id"]
    assert "persistent_candidate_id" not in promoted


def test_orphan_solver_artifacts_do_not_advance_case_numbering() -> None:
    next_before = next_candidate_index()
    orphan_run_dir = ABAQUS_RUNS_DIR / "C901"
    orphan_io_path = IO_DIR / "result_C901.json"
    created_run_dir = not orphan_run_dir.exists()
    created_io_path = not orphan_io_path.exists()
    marker_path = orphan_run_dir / "pytest_orphan_marker.tmp"

    try:
        orphan_run_dir.mkdir(parents=True, exist_ok=True)
        marker_path.write_text("orphan solver artifact", encoding="utf-8")
        if created_io_path:
            write_json(orphan_io_path, {"candidate_id": "C901", "status": "failed"})

        assert next_candidate_index() == next_before
        assert next_case_id(format_candidate_id(next_before)) == f"CASE_{next_before}"
        with pytest.raises(ValueError):
            next_case_id("C901")
    finally:
        marker_path.unlink(missing_ok=True)
        if created_io_path:
            orphan_io_path.unlink(missing_ok=True)
        if created_run_dir:
            shutil.rmtree(orphan_run_dir, ignore_errors=True)
