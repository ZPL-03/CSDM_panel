from agents.orchestrator import OrchestratorAgent


def test_parse_instruction_returns_session_task_record() -> None:
    agent = OrchestratorAgent()

    task = agent.parse_instruction("请设计一个T形加筋方案，压缩荷载为1200kN/m")

    assert task["task_id"].startswith("TASK_")
    assert task["source"] == "gui_instruction"
    assert task["task"]["load_conditions"]["Nx_kN_per_m"] == 1200.0
