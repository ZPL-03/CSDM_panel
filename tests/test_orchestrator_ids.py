"""跨阶段候选身份、物理唯一性及求解工件保护。"""

from copy import deepcopy

import pytest

from agents.orchestrator import OrchestratorAgent
from agents.fem_agent import FEMAgent
from core.id_utils import next_candidate_index, next_case_id
from core.paths import ABAQUS_RUNS_DIR
from tests.test_fem_agent import _real_fem_candidate


def test_temporary_candidates_do_not_consume_persistent_ids(monkeypatch):
    agent = OrchestratorAgent()
    monkeypatch.setattr(agent.candidate_gen.case_retriever, "retrieve_transferable_cases", lambda *_a, **_k: [])
    task = agent.parse_instruction("长度360mm，宽度300mm，压缩120kN/m，生成4个候选，初筛2个")
    task["task"]["candidate_generation_preferences"]["source_ratio"] = {
        "llm": 0.0, "case_transfer": 0.0, "doe": 1.0,
    }
    next_before = next_candidate_index()
    candidates = agent.generate_candidates(task)
    assert next_candidate_index() == next_before
    promoted = agent._promote_candidate_for_fem(task, candidates[0])
    assert promoted["session_candidate_id"] == "TMP_1"
    assert promoted["candidate_id"] == f"C{next_before}"
    assert candidates[0]["persistent_candidate_id"] == promoted["candidate_id"]


def test_orphan_solver_artifacts_reserve_candidate_identity():
    (ABAQUS_RUNS_DIR / "C901").mkdir()
    assert next_candidate_index() == 902
    assert next_case_id("C902") == "CASE_902"


def test_existing_formal_input_is_never_overwritten():
    folder = ABAQUS_RUNS_DIR / "C1"
    folder.mkdir()
    marker = folder / "C1.inp"
    marker.write_text("formal source", encoding="utf-8")
    with pytest.raises(FileExistsError):
        FEMAgent().run(_real_fem_candidate("C1"))
    assert marker.read_text(encoding="utf-8") == "formal source"


@pytest.mark.parametrize("change_id", [False, True])
def test_stage_boundary_rejects_duplicate_physical_designs(change_id):
    agent = OrchestratorAgent()
    first = _real_fem_candidate("C1")
    other = deepcopy(first)
    if change_id:
        other["candidate_id"] = "C2"
    with pytest.raises(ValueError, match="重复"):
        agent._validate_pool([first, other])
