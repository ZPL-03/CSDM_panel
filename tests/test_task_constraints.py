"""任务固定值在生成、预筛、提交和回流链路中的约束。"""

from copy import deepcopy

import pytest

from agents.candidate_gen import CandidateGenAgent
from agents.fem_agent import FEMAgent
from agents.orchestrator import OrchestratorAgent
from core.doe_sampler import DOESampler
from core.io_utils import read_json
from core.laminate import analysis_contract
from core.rule_checker import RuleChecker
from core.stiffener_profile import default_geometry
from core.task_contract import build_design_constraints, validate_candidate_constraints
from core.task_parser import TaskParser
from core import paths
from tests.test_fem_agent import _real_fem_candidate


@pytest.mark.parametrize("stype", ["T", "BLADE", "L", "HAT"])
def test_fixed_geometry_survives_all_sources_and_rules(stype):
    task = TaskParser().parse_instruction("长度360mm，宽度300mm，压缩120kN/m，生成2个候选，初筛1个")
    task["task"]["stiffener_type"] = stype
    task["task"]["candidate_generation_preferences"]["stiffener_types"] = [stype]
    constraints = build_design_constraints(task, stype)
    assert constraints["geometry_ranges"]["panel_length_mm"] == {"min": 360, "max": 360}
    generated = DOESampler().sample_candidates(task, 2, stiffener_type=stype)
    generator = CandidateGenAgent()
    for source in ("DOE", "CASE_TRANSFER"):
        raw = _real_fem_candidate("C1", stype)
        raw["geometry"]["panel_length_mm"] = 700
        generated.append(generator._normalize_candidate(task, raw, 3, source))
    assert len(generated) == 4
    for candidate in generated:
        assert candidate["geometry"]["panel_length_mm"] == 360
        assert candidate["geometry"]["panel_width_mm"] == 300
        assert RuleChecker().run(candidate, task=task)["is_valid"]
        validate_candidate_constraints(candidate, constraints)


def test_fem_rejects_changed_task_geometry_before_launch(monkeypatch):
    task = TaskParser().parse_instruction("长度360mm，宽度300mm，压缩120kN/m，生成1个候选，初筛1个")
    candidate = DOESampler().sample_candidates(task, 1)[0]
    candidate["candidate_id"] = "C1"
    candidate["geometry"]["panel_length_mm"] = 700
    monkeypatch.setattr("agents.fem_agent.is_abaqus_available", lambda *_: pytest.fail("不得启动求解"))
    with pytest.raises(ValueError, match="panel_length_mm"):
        FEMAgent().run(candidate)


def test_invalid_ranges_and_outside_candidates_are_not_silently_adjusted():
    task = {"geometry_envelope": {"panel_length_mm": [400, 300]}}
    with pytest.raises(ValueError):
        build_design_constraints(task)
    task = {"geometry_envelope": {"panel_length_mm": [300, 400], "panel_width_mm": [200, 300]}}
    candidate = _real_fem_candidate("C1")
    candidate["geometry"]["panel_length_mm"] = 450
    assert not RuleChecker().run(candidate, task=task)["is_valid"]


def fixed_geometry_task(stype, geometry):
    task = TaskParser().parse_instruction("长度360mm，宽度300mm，压缩120kN/m，生成1个候选，初筛1个")
    payload = task["task"]
    payload["stiffener_type"] = stype
    payload["candidate_generation_preferences"]["stiffener_types"] = [stype]
    payload["fixed_geometry"] = geometry
    payload["geometry_envelope"]["max_stiffener_height_mm"] = geometry["stiffener_height_mm"]
    return task


def test_positive_buildable_fixed_geometry_is_not_restricted_by_empirical_ratios():
    geometry = {**default_geometry("T"), "panel_length_mm": 360., "panel_width_mm": 300.,
                "skin_thickness_mm": 0.8, "stiffener_height_mm": 80., "pitch_mm": 100.}
    task = fixed_geometry_task("T", geometry)
    candidates = DOESampler().sample_candidates(task, 1)
    assert len(candidates) == 1 and candidates[0]["geometry"] == geometry
    before = deepcopy(candidates[0])
    assert RuleChecker().run(candidates[0], task=task)["is_valid"]
    assert candidates[0] == before


def test_hat_design_limits_are_configuration_values_not_geometry_constants():
    geometry = {**default_geometry("HAT"), "panel_length_mm": 360., "panel_width_mm": 300.}
    task = fixed_geometry_task("HAT", geometry)
    candidate = DOESampler().sample_candidates(task, 1, stiffener_type="HAT")[0]
    checker = RuleChecker()
    checker.param_ranges = deepcopy(checker.param_ranges)
    checker.param_ranges["HAT_constraints"] = {"cap_to_flange_width_max": 0.4, "web_incline_min_deg": 0}
    assert not checker.run(candidate, task=task)["is_valid"]
    checker.param_ranges["HAT_constraints"] = {"cap_to_flange_width_max": 0.6, "web_incline_min_deg": 0}
    assert checker.run(candidate, task=task)["is_valid"]
    checker.param_ranges["HAT_constraints"]["web_incline_min_deg"] = 85
    assert not checker.run(candidate, task=task)["is_valid"]


def test_unavailable_surrogate_is_audited_and_pipeline_keeps_fixed_geometry(monkeypatch):
    from core.surrogate_model import SurrogateModelManager

    agent = OrchestratorAgent()
    agent.screener.model_manager = SurrogateModelManager(models_dir=paths.DATA_DIR / "empty-models")
    monkeypatch.setattr("agents.fem_agent.is_abaqus_available", lambda *_: True)
    def solve(candidate, _):
        return {"candidate_id": candidate["candidate_id"], "status": "success", "retry_count": 0,
                "BLF_global": 0.9, "BLF_local": 1.5, "weight_kg_per_m2": 7.2, "verdict": "不通过",
                "analysis_contract": analysis_contract(candidate)}
    monkeypatch.setattr(agent.fem_agent, "_run_real", solve)
    monkeypatch.setattr(agent.knowledge_agent.case_memory, "upsert_cases", lambda *_a, **_k: None)
    result = agent.run("长度360mm，宽度300mm，压缩120kN/m，生成2个候选，初筛1个")
    selected = result["top_candidates"][0]
    assert selected["surrogate_BLF"] is None and selected["rank_score"] is None
    assert selected["screening_decision"]["status"] == "missing"
    assert selected["screening_decision"]["action"] == "direct_fem"
    assert selected["candidate_id"] == result["candidates"][0]["candidate_id"]
    decision = read_json(paths.TASKS_DIR / "screening" / f"{result['task']['task_id']}.json")
    assert decision["selected_ids"] == [selected["candidate_id"]]
    fem_input = read_json(paths.IO_DIR / f"input_{selected['persistent_candidate_id']}.json")
    assert fem_input["geometry"]["panel_length_mm"] == 360
    assert fem_input["geometry"]["panel_width_mm"] == 300
    case = read_json(next(paths.CASES_DIR.glob("CASE_*.json")))
    assert case["abaqus_results"]["analysis_contract"] == analysis_contract(case["design"])
    assert case["design"]["screening_decision"] == decision


def test_unavailable_surrogate_stop_policy_records_terminal_decision(monkeypatch):
    from core.surrogate_model import SurrogateModelManager

    agent = OrchestratorAgent()
    agent.screener.config = deepcopy(agent.screener.config)
    agent.screener.config["pipeline"]["unavailable_surrogate_policy"] = "stop"
    agent.screener.model_manager = SurrogateModelManager(models_dir=paths.DATA_DIR / "absent")
    task = agent.parse_instruction("压缩120kN/m，生成1个候选，初筛1个")
    with pytest.raises(RuntimeError, match="预筛终止"):
        agent.screen_candidates(task, agent.generate_candidates(task))
    assert read_json(paths.TASKS_DIR / "screening" / f"{task['task_id']}.json")["action"] == "stop"
