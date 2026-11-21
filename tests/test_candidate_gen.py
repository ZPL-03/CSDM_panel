"""程序候选池的数量、筋型、物理唯一性和失败行为。"""

from copy import deepcopy

import pytest

from agents.candidate_gen import CandidateGenAgent
from core.task_parser import TaskParser
from core.task_contract import validate_candidate_constraints
from tests.test_fem_agent import _real_fem_candidate


@pytest.mark.parametrize("kinds,total", [(["T"], 12), (["BLADE", "T", "L", "HAT"], 9)])
def test_harness_generates_exact_unique_valid_pool(kinds, total, monkeypatch):
    task = TaskParser().parse_instruction(f"长度360mm，宽度300mm，压缩120kN/m，生成{total}个候选，初筛1个")
    task["task"]["stiffener_type"] = kinds[0]
    task["task"]["candidate_generation_preferences"]["stiffener_types"] = kinds
    agent = CandidateGenAgent()
    monkeypatch.setattr(agent.case_retriever, "retrieve_transferable_cases", lambda *_a, **_k: [])
    pool = agent.run(task)
    assert len(pool) == total
    assert len({agent._candidate_signature(candidate) for candidate in pool}) == total
    assert {candidate["stiffener_type"] for candidate in pool} == set(kinds)
    assert [candidate["candidate_id"] for candidate in pool] == [f"TMP_{index}" for index in range(1, total + 1)]
    for candidate in pool:
        validate_candidate_constraints(candidate, candidate["task_constraints"])
        assert candidate["rule_check"]["is_valid"] and candidate["source"] == "DOE"
        assert candidate["surrogate_BLF"] is None and candidate["rank_score"] is None


def test_transfer_drops_historical_identity_scores_and_conclusions():
    agent = CandidateGenAgent()
    task = TaskParser().parse_instruction("长度360mm，宽度300mm，压缩120kN/m，生成2个候选，初筛1个")
    raw = _real_fem_candidate("C91")
    raw.update(surrogate_BLF=99, rank_score=98, persistent_candidate_id="C91", verdict="通过")
    before = deepcopy(raw)
    candidate = agent._normalize_candidate(task, raw, 1, "CASE_TRANSFER")
    assert raw == before
    assert candidate["candidate_id"] == "TMP_1" and "persistent_candidate_id" not in candidate
    assert "verdict" not in candidate and candidate["surrogate_BLF"] is None and candidate["rank_score"] is None
    assert candidate["geometry"]["panel_length_mm"] == 360


def test_duplicate_sampling_fails_instead_of_padding(monkeypatch):
    agent = CandidateGenAgent()
    task = TaskParser().parse_instruction("长度360mm，宽度300mm，压缩120kN/m，生成3个候选，初筛1个")
    raw = _real_fem_candidate("C1")
    candidate = agent._normalize_candidate(task, raw, 1, "DOE")
    monkeypatch.setattr(agent.case_retriever, "retrieve_transferable_cases", lambda *_a, **_k: [])
    monkeypatch.setattr(agent.doe_sampler, "sample_candidates", lambda *_a, **_k: [deepcopy(candidate)] * 3)
    with pytest.raises(RuntimeError, match="唯一候选不足"):
        agent.run(task)


def test_requested_types_must_fit_candidate_budget():
    task = TaskParser().parse_instruction("T型与帽形壁板，生成1个候选，初筛1个")
    with pytest.raises(ValueError, match="不能少于指定筋型数"):
        CandidateGenAgent().run(task)
