from pathlib import Path

from abaqus.build_panel import build_panel
from abaqus.extract_blf import extract_blf
from core.io_utils import read_json, write_json


def _candidate_payload() -> dict:
    return {
        "candidate_id": "C901",
        "task_id": "TASK_1",
        "source": "DOE",
        "stiffener_type": "T",
        "geometry": {
            "panel_length_mm": 720,
            "panel_width_mm": 600,
            "skin_thickness_mm": 2.4,
            "pitch_mm": 120,
            "stiffener_height_mm": 30,
            "web_thickness_mm": 2.0,
            "flange_width_mm": 18,
            "flange_thickness_mm": 2.0,
        },
        "layup": {
            "skin_layup": "[45/-45/0/90/0/-45/45]s",
            "skin_f0": 0.286,
            "skin_f45": 0.571,
            "skin_f90": 0.143,
        },
        "design_targets": {"BLF_min": 1.2},
        "load_conditions": {"type": "compression_shear", "label": "压剪组合", "Nx_kN_per_m": 850, "Nxy_kN_per_m": 150},
        "boundary_conditions": {
            "type": "CCCC",
            "label": "四边固支（CCCC）",
            "description": "四条边均固支",
            "simply_supported_edges": [],
            "clamped_edges": ["X0", "X1", "Y0", "Y1"],
        },
        "material_system": {
            "name": "T300/5208",
            "density_kg_per_m3": 1600,
            "E1_GPa": 181,
            "E2_GPa": 10.3,
            "G12_GPa": 7.17,
            "nu12": 0.28,
        },
    }


def test_build_panel_mock_generates_result(tmp_path: Path) -> None:
    input_json = tmp_path / "input_C901.json"
    result_json = tmp_path / "result_C901.json"
    write_json(input_json, _candidate_payload())

    build_panel(input_json=input_json, result_json=result_json, mock=True)

    payload = read_json(result_json)
    assert payload["status"] == "success"
    assert payload["BLF_global"] >= 1.0
    assert payload["weight_kg_per_m2"] > 0
    assert payload["model_summary"]["stiffener_count"] >= 1
    assert payload["load_summary"]
    assert payload["boundary_summary"]


def test_extract_blf_mock_generates_result(tmp_path: Path) -> None:
    input_json = tmp_path / "input_C901.json"
    result_json = tmp_path / "result_C901.json"
    odb_path = tmp_path / "C901.odb"
    write_json(input_json, _candidate_payload())
    odb_path.write_text("mock", encoding="utf-8")

    extract_blf(odb_path=odb_path, result_json=result_json, input_json=input_json, mock=True)

    payload = read_json(result_json)
    assert payload["status"] == "success"
    assert payload["candidate_id"] == "C901"
    assert payload["mode_eigenvalues"]
    assert payload["diagnosis_summary"]
