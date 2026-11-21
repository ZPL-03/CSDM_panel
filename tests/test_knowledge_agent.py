from agents.knowledge_agent import KnowledgeAgent
from copy import deepcopy
from tests.test_fem_agent import _real_fem_candidate


def test_knowledge_agent_preserves_physical_design_without_mutating_input() -> None:
    agent = KnowledgeAgent.__new__(KnowledgeAgent)

    design = _real_fem_candidate("C1", stiffener_type="HAT")
    original = deepcopy(design)

    clean_design = agent._sanitize_design(design)

    for key in ("candidate_id", "source", "stiffener_type", "geometry", "layup", "material_system",
                "load_conditions", "boundary_conditions", "design_targets", "analysis"):
        assert clean_design[key] == design.get(key, {})
    assert design == original
