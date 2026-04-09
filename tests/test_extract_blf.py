import sys
import types
from pathlib import Path

from abaqus.extract_blf import extract_blf
from core.io_utils import read_json, write_json


class _DummyFrame:
    def __init__(self, description: str, frame_value: float = 0.0) -> None:
        self.description = description
        self.frameValue = frame_value
        self.fieldOutputs = {}


class _DummyStep:
    def __init__(self, eigenvalues: list[float]) -> None:
        self.frames = [_DummyFrame("Base frame", 0.0)] + [
            _DummyFrame(f"EigenValue = {value}", value) for value in eigenvalues
        ]


class _DummyAssembly:
    def __init__(self) -> None:
        self.instances = {}


class _DummyOdb:
    def __init__(self, eigenvalues: list[float]) -> None:
        self.steps = {"Buckling": _DummyStep(eigenvalues)}
        self.rootAssembly = _DummyAssembly()

    def close(self) -> None:
        return None


def _candidate_payload() -> dict:
    return {
        "candidate_id": "C950",
        "task_id": "TASK_1",
        "source": "MANUAL",
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
        "load_conditions": {"type": "compression_shear", "label": "压剪组合", "Nx_kN_per_m": 1200, "Nxy_kN_per_m": 800},
        "boundary_conditions": {
            "type": "SSSS",
            "label": "四边简支（SSSS）",
            "description": "四边简支",
            "simply_supported_edges": ["X0", "X1", "Y0", "Y1"],
            "clamped_edges": [],
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


def test_extract_blf_uses_first_positive_eigenvalue(monkeypatch, tmp_path: Path) -> None:
    odb_path = tmp_path / "C950.odb"
    result_json = tmp_path / "result_C950.json"
    input_json = tmp_path / "input_C950.json"
    odb_path.write_text("placeholder", encoding="utf-8")
    write_json(input_json, _candidate_payload())

    fake_module = types.SimpleNamespace(openOdb=lambda path: _DummyOdb([-0.21, -0.08, 0.035, 0.072, 0.11]))
    monkeypatch.setitem(sys.modules, "odbAccess", fake_module)

    extract_blf(odb_path=odb_path, result_json=result_json, input_json=input_json, mock=False)

    payload = read_json(result_json)
    assert payload["status"] == "success"
    assert payload["BLF_global"] == 0.035
    assert payload["BLF_local"] == 0.072
    assert payload["analysis_flags"]["negative_modes_skipped"] == 2
    assert payload["analysis_flags"]["first_positive_mode_index"] == 3
    assert payload["mode_eigenvalues"][:3] == [-0.21, -0.08, 0.035]
    assert payload["effective_mode_eigenvalues"][:2] == [0.035, 0.072]
