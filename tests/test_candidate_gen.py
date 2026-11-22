"""程序候选池的数量、筋型、物理唯一性和失败行为。"""

from copy import deepcopy
import json

import pytest

from agents.candidate_gen import CandidateGenAgent
from core.llm_backend import LLMResponseError
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
        payload = json.loads(user_prompt)
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


def test_default_llm_generates_exact_twenty_candidates_in_one_request():
    agent = CandidateGenAgent()
    backend = _BatchLLM()
    agent.llm_backend = backend

    pool = agent.run(_llm_task(20))

    assert len(pool) == 20
    assert all(candidate["source"] == "LLM" for candidate in pool)
    assert len({agent._candidate_signature(candidate) for candidate in pool}) == 20
    assert backend.calls == [{"count": 20, "max_tokens_override": 8192, "json_mode": True}]
    assert agent.last_generation_audit["accepted_counts"] == {
        "LLM": 20,
        "CASE_TRANSFER": 0,
        "DOE": 0,
    }


def test_candidate_limit_caps_each_request_without_changing_the_total():
    agent = CandidateGenAgent()
    agent.llm_config = {**agent.llm_config, "candidate_generation": {
        "max_candidates_per_request": 5, "max_attempts": 1, "max_tokens": 4096,
    }}
    agent.llm_backend = _BatchLLM()
    assert len(agent.run(_llm_task(20))) == 20
    assert agent.llm_backend.calls == [{"count": 5, "max_tokens_override": 4096, "json_mode": True}] * 4


def test_truncation_reduces_batch_count_without_increasing_budget():
    agent = CandidateGenAgent()
    agent.llm_config = {**agent.llm_config, "candidate_generation": {
        "max_candidates_per_request": 6, "max_attempts": 1, "max_tokens": 1024,
    }}
    agent.llm_backend = _BatchLLM([LLMResponseError("输出截断", finish_reason="length")])
    pool = agent.run(_llm_task(6))
    assert len(pool) == 6 and all(candidate["source"] == "LLM" for candidate in pool)
    assert [row["count"] for row in agent.llm_backend.calls] == [6, 3, 3]
    assert {row["max_tokens_override"] for row in agent.llm_backend.calls} == {1024}
    assert agent.last_generation_audit["llm_diagnostics"][0]["finish_reason"] == "length"


def test_single_truncated_candidate_fails_without_repeating_same_request():
    agent = CandidateGenAgent()
    agent.llm_backend = _BatchLLM([LLMResponseError("输出截断", finish_reason="length")])
    with pytest.raises(RuntimeError, match="输出截断"):
        agent.run(_llm_task(1))
    assert len(agent.llm_backend.calls) == 1


@pytest.mark.parametrize("invalid", [0, True, "8192"])
def test_candidate_output_budget_must_be_a_positive_integer(invalid, monkeypatch):
    agent = CandidateGenAgent()
    monkeypatch.setitem(agent.llm_config["candidate_generation"], "max_tokens", invalid)
    agent.llm_backend = _BatchLLM()
    with pytest.raises(ValueError, match=r"candidate_generation\.max_tokens"):
        agent.run(_llm_task(1))
    assert agent.llm_backend.calls == []


@pytest.mark.parametrize("fault", ["short", "empty"])
def test_llm_batch_with_no_candidates_is_retried(fault):
    valid = json.dumps({"candidates": [_llm_raw_candidate(1)]}, ensure_ascii=False)
    rejected = {
        "short": '{"candidates":[]}',
        "empty": LLMResponseError("LLM 返回空正文"),
    }[fault]
    agent = CandidateGenAgent()
    backend = _BatchLLM([rejected, valid])
    agent.llm_backend = backend

    pool = agent.run(_llm_task(1))

    assert len(pool) == 1 and pool[0]["source"] == "LLM"
    assert len(backend.calls) == 2
    assert any(item["stage"] in {"request_or_parse", "rejected"} for item in agent.last_generation_audit["llm_diagnostics"])


def test_llm_overflow_is_audited_after_accepting_exact_quota():
    agent = CandidateGenAgent()
    agent.llm_backend = _BatchLLM([json.dumps({"candidates": [_llm_raw_candidate(1), {"rationale": "额外提案"}]})])
    assert [candidate["candidate_id"] for candidate in agent.run(_llm_task(1))] == ["TMP_1"]
    assert len(agent.llm_backend.calls) == 1
    assert agent.last_generation_audit["llm_diagnostics"][0]["surplus"] == 1


@pytest.mark.parametrize("rejection", ["duplicate", "layup_notation", "missing_field", "invalid_item"])
def test_llm_partial_valid_batch_only_requests_remaining_quota(rejection):
    agent = CandidateGenAgent()
    invalid = _llm_raw_candidate(1 if rejection == "duplicate" else 3)
    if rejection == "layup_notation":
        invalid["layup"]["skin_layup"] = "[0/45/-45/90]2s"
    elif rejection == "missing_field":
        invalid.pop("rationale")
    elif rejection == "invalid_item":
        invalid = None
    agent.llm_backend = _BatchLLM([
        json.dumps({"candidates": [_llm_raw_candidate(1), invalid]}),
        json.dumps({"candidates": [_llm_raw_candidate(2)]}),
    ])
    assert [candidate["candidate_id"] for candidate in agent.run(_llm_task(2))] == ["TMP_1", "TMP_2"]
    assert [row["count"] for row in agent.llm_backend.calls] == [2, 1]
    details = agent.last_generation_audit["llm_diagnostics"][0]
    assert (details["requested"], details["accepted"], details["rejected"], details["surplus"]) == (2, 1, 1, 0)


def test_llm_failure_does_not_fall_back_to_doe():
    agent = CandidateGenAgent()
    backend = _BatchLLM([RuntimeError("provider unavailable")])
    agent.llm_backend = backend

    with pytest.raises(RuntimeError, match="API 请求失败"):
        agent.run(_llm_task(1))
    assert len(backend.calls) == 1


def test_case_shortfall_preserves_doe_quota_and_increases_llm_share(monkeypatch):
    task = TaskParser().parse_instruction("T型加筋壁板，长度360mm，宽度300mm，压缩120kN/m，生成5个候选，初筛1个")
    agent = CandidateGenAgent()
    agent.llm_backend = _BatchLLM()
    monkeypatch.setattr(agent.case_retriever, "retrieve_transferable_cases", lambda *_a, **_k: [])
    pool = agent.run(task)
    assert [candidate["source"] for candidate in pool] == ["LLM"] * 4 + ["DOE"]
    assert agent.last_generation_audit["per_stiffener_type"]["T"]["case_transfer_shortfall"] == 1


def test_case_only_shortfall_is_explicit(monkeypatch):
    task = _llm_task(1)
    task["task"]["candidate_generation_preferences"]["source_ratio"] = {"case_transfer": 1}
    agent = CandidateGenAgent()
    monkeypatch.setattr(agent.case_retriever, "retrieve_transferable_cases", lambda *_a, **_k: [])
    with pytest.raises(RuntimeError, match="可迁移唯一案例不足"):
        agent.run(task)


def test_llm_rejects_repeated_physical_inputs_across_batches():
    def response(items):
        return json.dumps({"candidates": items}, ensure_ascii=False)
    duplicate = _llm_raw_candidate(1)
    duplicate["rationale"] = "不同理由"
    agent = CandidateGenAgent()
    agent.llm_config["candidate_generation"]["max_candidates_per_request"] = 5
    agent.llm_backend = _BatchLLM([
        response([_llm_raw_candidate(index) for index in range(1, 6)]),
        response([duplicate]), response([_llm_raw_candidate(6)]),
    ])
    pool = agent.run(_llm_task(6))
    assert len({agent._candidate_signature(candidate) for candidate in pool}) == 6
    assert [item["count"] for item in agent.llm_backend.calls] == [5, 1, 1]
    assert any("物理重复" in item["message"] for item in agent.last_generation_audit["llm_diagnostics"])


@pytest.mark.parametrize("field,value", [("panel_width_mm", "300"), ("web_thickness_mm", True), ("cap_width_mm", 12)])
def test_llm_does_not_coerce_geometry_or_drop_other_type_fields(field, value):
    raw = _llm_raw_candidate(1)
    raw["geometry"][field] = value
    with pytest.raises(ValueError, match="几何"):
        CandidateGenAgent()._normalize_candidate(_llm_task(1), raw, 1, "LLM")


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
