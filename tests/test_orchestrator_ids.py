from typing import Dict, Optional

from agents.orchestrator import OrchestratorAgent
from core.id_utils import next_candidate_index


class _FakeLLMBackend:
    def __init__(self, sink: Optional[Dict[str, str]] = None) -> None:
        self.sink = sink if sink is not None else {}

    def generate_json(self, system_prompt: str, user_prompt: str) -> dict:
        self.sink["system_prompt"] = system_prompt
        self.sink["user_prompt"] = user_prompt
        return {
            "candidates": [
                {
                    "geometry": {
                        "panel_length_mm": 700,
                        "panel_width_mm": 600,
                        "skin_thickness_mm": 2.5,
                        "pitch_mm": 120,
                        "stiffener_height_mm": 28,
                        "web_thickness_mm": 2.0,
                        "flange_width_mm": 16,
                        "flange_thickness_mm": 2.0,
                    },
                    "layup": {
                        "skin_layup": "[45/-45/0/90/0/-45/45]s",
                        "skin_f0": 0.286,
                        "skin_f45": 0.571,
                        "skin_f90": 0.143,
                    },
                    "rationale": "llm candidate",
                }
            ]
        }


class _FakeLiterature:
    def __init__(self, snippets: list[str]) -> None:
        self.snippets = snippets

    def format_snippets(self, _task, top_k: int = 5) -> list[str]:
        return self.snippets[:top_k]


def test_temporary_candidates_do_not_consume_persistent_ids() -> None:
    agent = OrchestratorAgent()
    agent.task_parser.llm_backend = None
    agent.candidate_gen.llm_backend = None
    task = agent.parse_instruction("请设计一个T形加筋方案，压缩荷载为1000kN/m")
    next_before = next_candidate_index()

    candidates = agent.generate_candidates(task)

    assert candidates
    assert all(candidate["candidate_id"].startswith("TMP_") for candidate in candidates)
    assert next_candidate_index() == next_before


def test_orchestrator_candidate_generation_uses_literature_guidance() -> None:
    captured: dict[str, str] = {}
    agent = OrchestratorAgent()
    agent.task_parser.llm_backend = None
    agent.candidate_gen.literature_corpus = _FakeLiterature(
        ["[1] Composite literature | source=openalex\nComposite buckling guidance"]
    )
    agent.candidate_gen.llm_backend = _FakeLLMBackend(captured)

    task = agent.parse_instruction("请设计一个T形加筋方案，压缩荷载为1000kN/m，剪切180kN/m，边界SSCC")
    candidates = agent.generate_candidates(task)

    assert candidates
    assert "文献依据" in captured["user_prompt"]
    assert "Composite literature" in captured["user_prompt"]
    assert "参考案例" not in captured["user_prompt"]
    assert candidates[0]["source"] == "LLM"
    assert candidates[0]["candidate_id"].startswith("TMP_")


def test_orchestrator_candidate_generation_without_literature_still_runs() -> None:
    agent = OrchestratorAgent()
    agent.task_parser.llm_backend = None
    agent.candidate_gen.literature_corpus = _FakeLiterature([])
    agent.candidate_gen.llm_backend = _FakeLLMBackend()

    task = agent.parse_instruction("请设计一个T形加筋方案，压缩荷载为1000kN/m")
    candidates = agent.generate_candidates(task)

    assert candidates
    assert candidates[0]["source"] == "LLM"
    assert candidates[0]["candidate_id"].startswith("TMP_")


def test_orchestrator_candidates_keep_session_ids_before_fem() -> None:
    agent = OrchestratorAgent()
    agent.task_parser.llm_backend = None
    agent.candidate_gen.literature_corpus = _FakeLiterature([])
    agent.candidate_gen.llm_backend = _FakeLLMBackend()

    task = agent.parse_instruction("请设计一个T形加筋方案，压缩荷载为1000kN/m")
    candidates = agent.generate_candidates(task)

    assert candidates[0]["candidate_id"].startswith("TMP_")
    assert "task_id" not in candidates[0]
    assert candidates[0]["source"] == "LLM"
    assert candidates[0]["load_conditions"]["type"] == task["task"]["load_conditions"]["type"]
    assert candidates[0]["boundary_conditions"]["type"] == task["task"]["boundary_conditions"]["type"]
    assert candidates[0]["design_targets"] == task["task"]["design_targets"]
    assert candidates[0]["material_system"]
    assert candidates[0]["rule_check"]["is_valid"] is True
