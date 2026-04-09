from agents.orchestrator import OrchestratorAgent
from core.id_utils import next_candidate_index


def test_temporary_candidates_do_not_consume_persistent_ids() -> None:
    agent = OrchestratorAgent()
    task = agent.parse_instruction("请设计一个T形加筋方案，压缩荷载为1000kN/m")
    next_before = next_candidate_index()

    candidates = agent.generate_candidates(task)

    assert candidates
    assert all(candidate["candidate_id"].startswith("TMP_") for candidate in candidates)
    assert next_candidate_index() == next_before
