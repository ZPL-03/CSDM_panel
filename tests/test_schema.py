from core.schema_validator import validate_or_raise


def test_task_schema_passes() -> None:
    payload = {
        "application": "机翼下翼面壁板",
        "load_conditions": {"type": "compression_shear", "label": "压剪组合", "Nx_kN_per_m": 850, "Nxy_kN_per_m": 180},
        "boundary_conditions": {
            "type": "SSCC",
            "label": "X 向简支 + Y 向固支（SSCC）",
            "description": "X0/X1 简支，Y0/Y1 固支",
            "simply_supported_edges": ["X0", "X1"],
            "clamped_edges": ["Y0", "Y1"],
        },
        "geometry_envelope": {"panel_length_mm": [600, 800], "panel_width_mm": [500, 700], "max_stiffener_height_mm": 50},
        "material_system": {"name": "T300/5208", "E1_GPa": 181},
        "layup_constraints": {"allowed_angles": [0, 45, -45, 90], "symmetric": True, "balanced": True, "min_ratio_per_angle": 0.1},
        "stiffener_type": "T",
        "design_targets": {"BLF_min": 1.2, "primary_objective": "最小重量"},
    }
    validate_or_raise("task.schema.json", payload)


def test_candidate_schema_passes_without_task_identity() -> None:
    payload = {
        "candidate_id": "TMP_1",
        "display_name": "TMP_1",
        "source": "LLM",
        "stiffener_type": "T",
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
    }
    validate_or_raise("candidate.schema.json", payload)
