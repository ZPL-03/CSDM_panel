from agents.knowledge_agent import KnowledgeAgent


def test_knowledge_agent_preserves_llm_candidate_trace_fields() -> None:
    agent = KnowledgeAgent.__new__(KnowledgeAgent)

    design = {
        "candidate_id": "C1",
        "source": "LLM",
        "stiffener_type": "HAT",
        "geometry": {},
        "layup": {},
        "material_system": {},
        "load_conditions": {"type": "axial_compression"},
        "boundary_conditions": {"type": "SSSS"},
        "design_targets": {},
        "rule_check": {},
        "rationale": "LLM 推荐",
        "origin_summary": "| A1 | HAT |",
        "llm_output_excerpt": "## 候选方案\n| A1 | HAT |",
    }

    clean_design = agent._sanitize_design(design)

    assert clean_design["origin_summary"] == "| A1 | HAT |"
    assert "候选方案" in clean_design["llm_output_excerpt"]
