from core.doe_sampler import DOESampler


def _task_payload(is_user_specified: bool) -> dict:
    return {
        "task_id": "TASK_1",
        "source": "test",
        "task": {
            "application": "复合材料加筋壁板",
            "load_conditions": {"type": "compression_shear", "label": "压剪组合", "Nx_kN_per_m": 900.0, "Nxy_kN_per_m": 180.0},
            "boundary_conditions": {
                "type": "SSCC",
                "label": "X 向简支 + Y 向固支（SSCC）",
                "description": "X0/X1 简支，Y0/Y1 固支",
                "simply_supported_edges": ["X0", "X1"],
                "clamped_edges": ["Y0", "Y1"],
            },
            "geometry_envelope": {
                "panel_length_mm": [600, 800],
                "panel_width_mm": [500, 700],
                "max_stiffener_height_mm": 50,
            },
            "design_targets": {"BLF_min": 1.2, "primary_objective": "最小重量"},
            "material_system": {
                "name": "T300/5208",
                "density_kg_per_m3": 1600,
                "E1_GPa": 181,
                "E2_GPa": 10.3,
                "G12_GPa": 7.17,
                "nu12": 0.28,
                "material_key": "T300_5208",
                "is_user_specified": is_user_specified,
            },
            "layup_constraints": {"allowed_angles": [0, 45, -45, 90], "symmetric": True, "balanced": True, "min_ratio_per_angle": 0.1},
            "stiffener_type": "T",
        },
    }


def test_doe_sampler_varies_material_when_task_is_not_fixed() -> None:
    sampler = DOESampler()
    candidates = sampler.sample_candidates(
        _task_payload(is_user_specified=False),
        n_samples=4,
        start_index=1,
        strict_solver_window=True,
        id_factory=lambda index: f"TMP_{index}",
    )

    materials = {candidate["material_system"]["name"] for candidate in candidates}

    assert len(candidates) == 4
    assert len(materials) >= 2


def test_doe_sampler_keeps_fixed_material_when_user_specifies_material() -> None:
    sampler = DOESampler()
    candidates = sampler.sample_candidates(
        _task_payload(is_user_specified=True),
        n_samples=3,
        start_index=1,
        strict_solver_window=True,
        id_factory=lambda index: f"TMP_{index}",
    )

    materials = {candidate["material_system"]["name"] for candidate in candidates}

    assert len(candidates) == 3
    assert materials == {"T300/5208"}
