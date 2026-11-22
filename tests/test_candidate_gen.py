"""程序候选池的数量、筋型、物理唯一性和失败行为。"""

from copy import deepcopy
import json

import pytest

from agents.candidate_gen import CandidateGenAgent
from core.task_parser import TaskParser
from core.task_contract import validate_candidate_constraints
from tests.test_fem_agent import _real_fem_candidate


def _program_only_sources(task):
    task["task"]["candidate_generation_preferences"]["source_ratio"] = {
        "llm": 0.0,
        "case_transfer": 0.0,
        "doe": 1.0,
    }
    return task


def _llm_task(total=20):
    task = TaskParser().parse_instruction(
        f"T型加筋壁板，长度360mm，宽度300mm，压缩120kN/m，生成{total}个候选，初筛1个"
    )
    task["task"]["candidate_generation_preferences"]["source_ratio"] = {
        "llm": 1.0,
        "case_transfer": 0.0,
        "doe": 0.0,
    }
    return task


def _llm_raw_candidate(index):
    return {
        "stiffener_type": "T",
        "geometry": {
            "panel_length_mm": 360.0,
            "panel_width_mm": 300.0,
            "skin_thickness_mm": 1.8 + 0.05 * index,
            "pitch_mm": 90.0 + 2.0 * (index % 10),
            "stiffener_height_mm": 20.0 + 0.5 * (index % 10),
            "web_thickness_mm": 1.4 + 0.05 * (index % 8),
            "flange_width_mm": 14.0 + 0.5 * (index % 8),
            "flange_thickness_mm": 1.4 + 0.05 * (index % 6),
        },
        "layup": {"skin_layup": "[45/-45/0/90/0/-45/45]s"},
        "material_key": "T300_5208",
        "rationale": f"LLM 方案 {index}",
    }


class _BatchLLM:
    def __init__(self, responses=None):
        self.responses = list(responses or [])
        self.calls = []

    def chat(self, system_prompt, user_prompt, **kwargs):
        payload = json.loads(user_prompt.split("\n\n", 1)[0])
        self.calls.append({"count": payload["requested_candidate_count"], **kwargs})
        if self.responses:
            response = self.responses.pop(0)
            if isinstance(response, Exception):
                raise response
            return response
        start = sum(call["count"] for call in self.calls[:-1]) + 1
        return json.dumps({
            "candidates": [_llm_raw_candidate(start + offset) for offset in range(payload["requested_candidate_count"])]
        }, ensure_ascii=False)


@pytest.mark.parametrize("kinds,total", [(["T"], 12), (["BLADE", "T", "L", "HAT"], 9)])
def test_harness_generates_exact_unique_valid_pool(kinds, total, monkeypatch):
    task = _program_only_sources(TaskParser().parse_instruction(
        f"长度360mm，宽度300mm，压缩120kN/m，生成{total}个候选，初筛1个"
    ))
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
    task = _program_only_sources(TaskParser().parse_instruction(
        "长度360mm，宽度300mm，压缩120kN/m，生成3个候选，初筛1个"
    ))
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


def test_llm_generates_exact_twenty_candidates_in_strict_batches():
    agent = CandidateGenAgent()
    backend = _BatchLLM()
    agent.llm_backend = backend

    pool = agent.run(_llm_task(20))

    assert len(pool) == 20
    assert all(candidate["source"] == "LLM" for candidate in pool)
    assert len({agent._candidate_signature(candidate) for candidate in pool}) == 20
    assert [item["count"] for item in backend.calls] == [5, 5, 5, 5]
    assert all(item["json_mode"] is True and item["max_tokens_override"] >= 4750 for item in backend.calls)
    assert agent.last_generation_audit["accepted_counts"] == {
        "LLM": 20,
        "CASE_TRANSFER": 0,
        "DOE": 0,
    }


def test_llm_batch_count_mismatch_is_retried_without_partial_acceptance():
    valid = json.dumps({"candidates": [_llm_raw_candidate(1)]}, ensure_ascii=False)
    extra = json.dumps({"candidates": [_llm_raw_candidate(1), _llm_raw_candidate(2)]}, ensure_ascii=False)
    agent = CandidateGenAgent()
    backend = _BatchLLM([extra, valid])
    agent.llm_backend = backend

    pool = agent.run(_llm_task(1))

    assert len(pool) == 1 and pool[0]["source"] == "LLM"
    assert len(backend.calls) == 2
    assert any(item["stage"] == "request_or_parse" for item in agent.last_generation_audit["llm_diagnostics"])


def test_llm_failure_does_not_fall_back_to_doe():
    agent = CandidateGenAgent()
    agent.llm_backend = _BatchLLM([RuntimeError("provider unavailable") for _ in range(3)])

    with pytest.raises(RuntimeError, match="LLM 候选生成无法满足数量"):
        agent.run(_llm_task(1))


def test_llm_cannot_silently_rewrite_fixed_geometry_or_material():
    agent = CandidateGenAgent()
    task = _llm_task(1)

    geometry_mismatch = _llm_raw_candidate(1)
    geometry_mismatch["geometry"]["panel_length_mm"] = 361.0
    with pytest.raises(ValueError, match="panel_length_mm"):
        agent._normalize_candidate(task, geometry_mismatch, 1, "LLM")

    material_task = TaskParser().parse_instruction(
        "T型加筋壁板，长度360mm，宽度300mm，压缩120kN/m，材料IM7/8552，生成1个候选，初筛1个"
    )
    material_task["task"]["candidate_generation_preferences"]["source_ratio"] = {
        "llm": 1.0, "case_transfer": 0.0, "doe": 0.0,
    }
    material_mismatch = _llm_raw_candidate(1)
    with pytest.raises(ValueError, match="material_key"):
        agent._normalize_candidate(material_task, material_mismatch, 1, "LLM")


def test_source_matrix_preserves_global_row_and_column_totals():
    type_targets = {"BLADE": 5, "T": 5, "L": 5, "HAT": 5}
    source_targets = {"llm": 12, "case_transfer": 4, "doe": 4}

    matrix = CandidateGenAgent._allocate_source_matrix(type_targets, source_targets)

    assert all(sum(row.values()) == 5 for row in matrix.values())
    assert {key: sum(matrix[kind][key] for kind in matrix) for key in source_targets} == source_targets


@pytest.mark.parametrize("total", [1, 2])
def test_small_default_pool_keeps_llm_as_a_source(total):
    counts = CandidateGenAgent._allocate_source_counts(total, {"llm": 3, "case_transfer": 1, "doe": 1})
    assert counts["llm"] >= 1
    assert sum(counts.values()) == total
