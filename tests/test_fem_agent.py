from pathlib import Path
from copy import deepcopy
import builtins
import json
import os
from types import SimpleNamespace

import pytest

from abaqus.job_utils import is_abaqus_available, job_transient_paths
from agents.fem_agent import FEMAgent
from core.id_utils import format_candidate_id, next_candidate_index
from core.config_loader import load_app_config
from core.io_utils import read_json, write_json
from core.laminate import ANALYSIS_VERSION
from core.stiffener_profile import default_geometry


@pytest.mark.parametrize("python_major", [2, 3])
def test_launcher_preserves_json_float_precision_before_runtime_import(python_major):
    """模拟 CAE 的 15 位显示格式，核对独立启动器的 JSON 精度及运行参数。"""
    script = FEMAgent().generate_script({"candidate_id": "C1"})
    def display_repr(value):
        return format(value, ".15g")
    encoder = SimpleNamespace(FLOAT_REPR=display_repr)
    fake_sys = SimpleNamespace(version_info=(python_major, 7, 3), path=[])
    calls = []

    def load_runtime(name, path):
        assert name == "csdm_runtime_build_panel"
        assert Path(path).name == "runtime_build_panel.py"
        if python_major == 2:
            values = [1.0 / 7.0, 10300.0 / 2.7, 1.0000000000000002]
            assert json.loads("[" + ",".join(encoder.FLOAT_REPR(value) for value in values) + "]") == values
        else:
            assert encoder.FLOAT_REPR is display_repr
        assert fake_sys.path == [str(Path(path).parents[1])]
        return SimpleNamespace(build_panel=lambda **kwargs: calls.append(kwargs))

    modules = {"imp": SimpleNamespace(load_source=load_runtime), "json": SimpleNamespace(encoder=encoder),
               "os": os, "sys": fake_sys}
    namespace = {"__builtins__": {**vars(builtins), "__import__": lambda name, *args: modules[name]}}
    exec(compile(script.read_text(encoding="utf-8"), str(script), "exec"), namespace)

    assert calls == [{"input_json": str(FEMAgent()._input_path("C1")),
                      "result_json": str(FEMAgent()._result_path("C1"))}]


def _real_fem_candidate(candidate_id: str, stiffener_type: str = "T") -> dict:
    return {
        "candidate_id": candidate_id,
        "display_name": candidate_id,
        "source": "DOE",
        "stiffener_type": stiffener_type,
        "geometry": {
            **default_geometry(stiffener_type),
            "panel_length_mm": 360,
            "panel_width_mm": 240,
            "skin_thickness_mm": 2.4,
            "pitch_mm": 120,
            "stiffener_height_mm": 24,
            "web_thickness_mm": 2.0,
        },
        "layup": {
            "skin_layup": "[45/-45/0/90/0/-45/45]s",
            "skin_f0": 0.286,
            "skin_f45": 0.571,
            "skin_f90": 0.143,
        },
        "rule_check": {"is_valid": True},
        "surrogate_BLF": None,
        "surrogate_weight": None,
        "rank_score": None,
        "rationale": "真实 Abaqus 回归样本",
        "design_targets": {"BLF_min": 1.2, "primary_objective": "最小重量"},
        "load_conditions": {"type": "axial_compression", "label": "单轴压缩", "Nx_kN_per_m": 120.0, "Nxy_kN_per_m": 0.0},
        "boundary_conditions": {
            "type": "SSSS",
            "label": "四边简支（SSSS）",
            "description": "四条边均按简支处理。",
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
        "analysis": {"buckling_modes": 4},
    }


@pytest.mark.parametrize("stiffener_type", ["T", "BLADE", "L", "HAT"])
@pytest.mark.real_fem
def test_fem_agent_runs_real_abaqus(stiffener_type: str) -> None:
    command = load_app_config()["abaqus"]["command"]
    assert is_abaqus_available(command), "当前环境未找到配置的 Abaqus 命令，真实有限元测试不能运行"

    candidate_id = format_candidate_id(next_candidate_index())
    agent = FEMAgent(
        config={
            "abaqus": {
                "command": command,
                "job_timeout_seconds": 900,
            }
        }
    )

    result = agent.run(_real_fem_candidate(candidate_id, stiffener_type))

    assert result["status"] == "success"
    assert result["error_type"] is None
    assert result["BLF_global"] is not None
    assert result["BLF_global"] > 0
    assert result["mode_eigenvalues"]
    assert result["load_summary"]
    assert result["boundary_summary"]
    assert result["diagnosis_summary"]
    assert result["analysis_contract"]["version"] == ANALYSIS_VERSION
    inp_text = Path(result["abaqus_inp"]).read_text(encoding="utf-8", errors="replace").upper()
    assert "COMPOSITE" in inp_text
    assert "*ORIENTATION" in inp_text
    assert Path(result["abaqus_inp"]).exists()
    assert Path(result["abaqus_odb"]).exists()
    assert read_json(agent._result_path(candidate_id)) == result
    assert not job_transient_paths(agent._run_dir(candidate_id), [candidate_id])


@pytest.mark.real_fem
@pytest.mark.parametrize("stiffener_type", ["T", "BLADE", "L", "HAT"])
def test_real_fem_responds_to_physical_stacking_order(stiffener_type):
    """固定几何/材料/载荷，仅交换等厚等比例层序，保留输入和两组求解结果。"""
    from core.layup import normalize_layup

    agent = FEMAgent()
    results = []
    for index, text in enumerate(("[0/45/-45/90]s", "[90/45/-45/0]s"), 1):
        candidate = _real_fem_candidate(f"C{index}", stiffener_type)
        candidate["layup"] = normalize_layup(text)
        result = agent.run(candidate)
        assert result["status"] == "success"
        assert result["BLF_global"] > 0
        results.append(result)
    assert results[0]["analysis_contract"]["fingerprint"] != results[1]["analysis_contract"]["fingerprint"]
    assert abs(results[0]["BLF_global"] - results[1]["BLF_global"]) > 1e-6 * max(r["BLF_global"] for r in results)


@pytest.mark.parametrize("message", ["solver failed", "mesh distortion", "numerical singularity", "geometry error"])
def test_failed_job_preserves_design_and_diagnosis_before_cleanup(monkeypatch, message):
    agent = FEMAgent(config={"abaqus": {"command": "abaqus", "job_timeout_seconds": 10}})
    run_dir = agent._run_dir("C1")
    (run_dir / "C1.msg").write_text(message, encoding="utf-8")
    monkeypatch.setattr("agents.fem_agent.is_abaqus_available", lambda command: True)
    calls = []
    def failed_command(*args, **kwargs):
        calls.append(read_json(agent._input_path("C1")))
        (run_dir / "C1.inp").write_text("input", encoding="utf-8")
        write_json(agent._result_path("C1"), {"status": "success"})
        return 1, "", "process failed"
    monkeypatch.setattr("agents.fem_agent.run_command", failed_command)

    candidate = _real_fem_candidate("C1")
    original = deepcopy(candidate)
    result = agent.run(candidate)

    assert result["status"] == "failed"
    assert result["retry_count"] == 0
    assert len(calls) == 1
    assert candidate == original
    assert calls[0] == original
    assert result["error_type"]
    assert result["error_log"]["return_code"] == 1
    assert result["error_log"]["solver_logs"] == {"C1.msg": message}
    assert read_json(agent._result_path("C1")) == result
    assert (run_dir / "C1.inp").exists()
    assert not (run_dir / "C1.msg").exists()
