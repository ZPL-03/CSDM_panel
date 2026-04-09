from agents.fem_agent import FEMAgent


def test_fem_agent_mock_run() -> None:
    agent = FEMAgent()
    candidate = {
        "candidate_id": "C1",
        "task_id": "TASK_1",
        "source": "MANUAL",
        "stiffener_type": "T",
        "geometry": {
            "panel_length_mm": 700,
            "panel_width_mm": 600,
            "skin_thickness_mm": 2.4,
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
        "rule_check": {},
        "surrogate_BLF": None,
        "surrogate_weight": None,
        "rank_score": None,
        "rationale": "test",
        "design_targets": {"BLF_min": 1.2},
        "load_conditions": {"type": "compression_shear", "label": "压剪组合", "Nx_kN_per_m": 820.0, "Nxy_kN_per_m": 160.0},
        "boundary_conditions": {
            "type": "SSCC",
            "label": "X 向简支 + Y 向固支（SSCC）",
            "description": "X0/X1 简支，Y0/Y1 固支",
            "simply_supported_edges": ["X0", "X1"],
            "clamped_edges": ["Y0", "Y1"],
        },
        "mock_mode": True,
    }
    result = agent.run(candidate)
    assert result["status"] == "success"
    assert result["BLF_global"] is not None
    assert result["load_summary"]
    assert result["boundary_summary"]
    assert result["diagnosis_summary"]
