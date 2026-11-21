import pytest

from core.layup import normalize_layup
from agents.candidate_gen import CandidateGenAgent
from core.schema_validator import SchemaValidationError
from core.task_parser import TaskParser
from tests.test_fem_agent import _real_fem_candidate


def test_layup_normalizes_field_prefix_and_computes_ratios() -> None:
    result = normalize_layup("skin_layup: [45/-45/0/90/0/-45/45]s")
    assert result == {"skin_layup": "[45/-45/0/90/0/-45/45]s", "skin_f0": 0.286,
                      "skin_f45": 0.571, "skin_f90": 0.143, "ply_count": 14}


@pytest.mark.parametrize("value", ["[0/30/90]s", "[0/NaN/90]s", "0/45/-45/90", "[]s", "[0/45/90]s"])
def test_layup_rejects_unsupported_or_malformed_angles(value) -> None:
    with pytest.raises(ValueError):
        normalize_layup(value)


def test_candidate_source_must_be_program_controlled() -> None:
    agent = CandidateGenAgent()
    raw = _real_fem_candidate("C1")
    task = TaskParser().parse_instruction("T型壁板，生成1个候选，初筛1个")
    with pytest.raises(SchemaValidationError, match="source"):
        agent._normalize_candidate(task, raw, 1, "LLM")


def test_transferred_candidate_derives_ratios_without_rewriting_source() -> None:
    agent = CandidateGenAgent()
    raw = _real_fem_candidate("C1")
    raw["layup"]["skin_f0"] = 0.25
    task = TaskParser().parse_instruction("T型壁板，生成1个候选，初筛1个")
    result = agent._normalize_candidate(task, raw, 1, "CASE_TRANSFER")
    assert result["layup"]["skin_f0"] == 0.286
    assert raw["layup"]["skin_f0"] == 0.25
