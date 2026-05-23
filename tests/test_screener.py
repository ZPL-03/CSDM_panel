from agents.screener import ScreenerAgent


def _candidate(index: int) -> dict:
    return {
        "candidate_id": f"TMP_{index}",
        "display_name": f"TMP_{index}",
        "task_id": "TASK_1",
        "source": "DOE",
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
        "task_id": "TASK_1",
        "source": "test",
        "task": {
            "application": "复合材料加筋壁板",
            "load_conditions": {"type": "axial_compression", "Nx_kN_per_m": 900.0},
            "boundary_conditions": {"type": "SSSS"},
            "geometry_envelope": {
                "panel_length_mm": [600, 800],
                "panel_width_mm": [500, 700],
                "max_stiffener_height_mm": 50,
            },
            "material_system": {"name": "T300/5208"},
            "layup_constraints": {"allowed_angles": [0, 45, -45, 90], "symmetric": True, "balanced": True, "min_ratio_per_angle": 0.1},
            "stiffener_type": "T",
            "design_targets": {"BLF_min": 1.2, "primary_objective": "最小重量"},
            "screening_preferences": {"top_k_candidates": 2},
        },
    }
    candidates = [_candidate(1), _candidate(2), _candidate(3), _candidate(4)]

    selected = agent.run({"task": task, "candidates": candidates})

    assert len(selected) == 2
    assert selected[0]["selection_reason"]
