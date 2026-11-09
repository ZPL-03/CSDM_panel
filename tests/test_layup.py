import pytest

from core.layup import normalize_layup
from agents.candidate_gen import CandidateGenAgent
from core.schema_validator import SchemaValidationError
from core.stiffener_profile import default_geometry


def test_layup_normalizes_field_prefix_and_computes_ratios() -> None:
    result = normalize_layup("skin_layup: [45/-45/0/90/0/-45/45]s")
    assert result == {"skin_layup": "[45/-45/0/90/0/-45/45]s", "skin_f0": 0.286,
                      "skin_f45": 0.571, "skin_f90": 0.143, "ply_count": 14}


@pytest.mark.parametrize("value", ["[0/30/90]s", "[0/NaN/90]s", "0/45/-45/90", "[]s", "[0/45/90]s"])
def test_layup_rejects_unsupported_or_malformed_angles(value) -> None:
    with pytest.raises(ValueError):
        normalize_layup(value)


def test_llm_layup_ratio_disagreement_is_rejected() -> None:
    agent = CandidateGenAgent()
    raw = {"geometry": default_geometry("T"), "layup": {
        "skin_layup": "[45/-45/0/90/0/-45/45]s", "skin_f0": 0.25}}
    with pytest.raises(SchemaValidationError, match="skin_f0"):
        agent._normalize_candidate({"task": {}}, raw, 1, "LLM")


def test_transferred_candidate_derives_ratios_without_rewriting_source() -> None:
    agent = CandidateGenAgent()
    raw = {"geometry": default_geometry("T"), "layup": {
        "skin_layup": "[45/-45/0/90/0/-45/45]s", "skin_f0": 0.25}}
    task = {"task": {"load_conditions": {}, "design_targets": {}, "boundary_conditions": {}}}
    result = agent._normalize_candidate(task, raw, 1, "CASE_TRANSFER")
    assert result["layup"]["skin_f0"] == 0.286
    assert raw["layup"]["skin_f0"] == 0.25
