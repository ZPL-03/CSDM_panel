from agents.screener import ScreenerAgent


def _candidate(index: int) -> dict:
    return {
        "candidate_id": f"TMP_{index}",
        "task_id": "TASK_1",
        "source": "MANUAL",
        "stiffener_type": "T",
        "geometry": {
            "panel_length_mm": 700,
            "panel_width_mm": 600,
            "skin_thickness_mm": 2.4 + index * 0.05,
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
        "material_system": {
            "density_kg_per_m3": 1600,
        },
    }


def test_screener_uses_task_top_k_candidates() -> None:
    agent = ScreenerAgent()
    agent.model_manager = type(
        "FakeModelManager",
        (),
        {"predict_candidates": staticmethod(lambda candidates, task=None: [1.8, 1.6, 1.4, 1.2])},
    )()

    task = {
        "screening_preferences": {"top_k_candidates": 2},
    }
    candidates = [_candidate(1), _candidate(2), _candidate(3), _candidate(4)]

    selected = agent.run({"task": task, "candidates": candidates})

    assert len(selected) == 2
    assert selected[0]["selection_reason"]
