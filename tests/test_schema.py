from core.schema_validator import validate_or_raise


def test_task_schema_passes() -> None:
    payload = {
        "task_id": "TASK_1",
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
