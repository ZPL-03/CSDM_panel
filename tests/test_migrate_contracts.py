from scripts.migrate_contracts import normalize_case_record, normalize_candidate_document, normalize_result_document


def test_normalize_candidate_document_upgrades_legacy_boundary_and_load() -> None:
    candidate = {
        "candidate_id": "C1",
        "task_id": "TASK_1",
        "source": "MANUAL",
        "stiffener_type": "T",
        "geometry": {},
        "layup": {},
        "load_conditions": {"type": "单轴压缩", "Nx_kN_per_m": 880},
        "boundary_conditions": "四边简支（SSSS）",
    }
    normalized = normalize_candidate_document(candidate)
    assert normalized["load_conditions"]["type"] == "axial_compression"
    assert normalized["boundary_conditions"]["type"] == "SSSS"


def test_normalize_result_document_adds_human_readable_summaries() -> None:
    result = {"candidate_id": "C1", "status": "success", "retry_count": 0, "verdict": "通过"}
    design = {
        "load_conditions": {"type": "compression_shear", "Nx_kN_per_m": 900, "Nxy_kN_per_m": 180},
        "boundary_conditions": {"type": "SSCC", "simply_supported_edges": ["X0", "X1"], "clamped_edges": ["Y0", "Y1"]},
    }
    normalized = normalize_result_document(result, design)
    assert "压剪组合" in normalized["load_summary"]
    assert "SSCC" in normalized["boundary_summary"]
    assert normalized["diagnosis_summary"]


def test_normalize_case_record_preserves_ids_and_upgrades_nested_contract() -> None:
    record = {
        "case_id": "CASE_9",
        "created_at": "2026-01-01T00:00:00",
        "source": "abaqus_auto",
        "task": {
            "task_id": "TASK_7",
            "application": "复合材料加筋壁板",
            "load_conditions": {"type": "单轴压缩", "Nx_kN_per_m": 850},
            "boundary_conditions": "四边简支（SSSS）",
            "geometry_envelope": {"panel_length_mm": [600, 800], "panel_width_mm": [500, 700], "max_stiffener_height_mm": 50},
            "material_system": {"name": "T300/5208", "E1_GPa": 181},
            "layup_constraints": {"allowed_angles": [0, 45, -45, 90], "symmetric": True, "balanced": True, "min_ratio_per_angle": 0.1},
            "stiffener_type": "T",
            "design_targets": {"BLF_min": 1.2, "primary_objective": "最小重量"},
        },
        "design": {
            "candidate_id": "C9",
            "task_id": "TASK_7",
            "source": "MANUAL",
            "stiffener_type": "T",
            "geometry": {"panel_length_mm": 700, "panel_width_mm": 600, "skin_thickness_mm": 2.5, "pitch_mm": 120, "stiffener_height_mm": 28, "web_thickness_mm": 2.0, "flange_width_mm": 16, "flange_thickness_mm": 2.0},
            "layup": {"skin_layup": "[45/-45/0/90/0/-45/45]s", "skin_f0": 0.286, "skin_f45": 0.571, "skin_f90": 0.143},
            "boundary_conditions": "四边简支（SSSS）",
        },
        "abaqus_results": {"candidate_id": "C9", "status": "success", "retry_count": 0, "verdict": "通过"},
        "verdict": "通过",
    }
    normalized = normalize_case_record(record)
    assert normalized["case_id"] == "CASE_9"
    assert normalized["task"]["task_id"] == "TASK_7"
    assert normalized["design"]["boundary_conditions"]["type"] == "SSSS"
    assert normalized["abaqus_results"]["diagnosis_summary"]
