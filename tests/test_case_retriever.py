from core.case_retriever import CaseRetriever, _is_pass_verdict


def _task() -> dict:
    return {
        "application": "复合材料加筋壁板",
        "load_conditions": {"type": "compression_shear", "Nx_kN_per_m": 1200.0, "Nxy_kN_per_m": 800.0},
        "boundary_conditions": {"type": "SSSS"},
        "geometry_envelope": {
            "panel_length_mm": [600, 800],
            "panel_width_mm": [500, 700],
            "max_stiffener_height_mm": 50,
        },
        "material_system": {"name": "T300/5208"},
        "design_targets": {"BLF_min": 1.2, "primary_objective": "最小重量"},
        "stiffener_type": "T",
    }


def test_case_retriever_returns_matching_cases() -> None:
    retriever = CaseRetriever(include_archive=True, include_formal=True)
    matches = retriever.retrieve_similar_cases(_task(), top_k=3)

    assert matches
    assert all(match["design"]["stiffener_type"] == "T" for match in matches)
    assert all(match["design"]["load_conditions"]["type"] == "compression_shear" for match in matches)
    assert all(match["design"]["boundary_conditions"]["type"] == "SSSS" for match in matches)



def test_is_pass_verdict_only_accepts_exact_pass() -> None:
    assert _is_pass_verdict("通过") is True
    assert _is_pass_verdict("不通过") is False
    assert _is_pass_verdict("失败") is False
    assert _is_pass_verdict(None) is False



def test_case_retriever_transfer_cases_are_all_passed() -> None:
    retriever = CaseRetriever(include_archive=True, include_formal=True)
    task = _task()
    task["load_conditions"] = {"type": "axial_compression", "Nx_kN_per_m": 1000.0}
    task["boundary_conditions"] = {"type": "SSSS"}
    matches = retriever.retrieve_transferable_cases(task, top_k=5)

    assert matches
    assert all(match.get("abaqus_results", {}).get("status") == "success" for match in matches)
    assert all(match.get("abaqus_results", {}).get("verdict") == "通过" for match in matches)
    assert all(match["design"]["load_conditions"]["type"] == "axial_compression" for match in matches)
    assert all(match["design"]["boundary_conditions"]["type"] == "SSSS" for match in matches)
