from pathlib import Path

import pytest

from abaqus.job_utils import is_abaqus_available, job_transient_paths
from agents.fem_agent import FEMAgent
from core.id_utils import format_candidate_id, next_candidate_index
from core.config_loader import load_app_config
from core.io_utils import read_json, write_json
from core.stiffener_profile import default_geometry


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
def test_fem_agent_runs_real_abaqus(stiffener_type: str) -> None:
    command = load_app_config()["abaqus"]["command"]
    assert is_abaqus_available(command), "当前环境未找到配置的 Abaqus 命令，真实有限元测试不能运行"

    candidate_id = format_candidate_id(next_candidate_index())
    agent = FEMAgent(
        config={
            "abaqus": {
                "command": command,
                "max_retries": 1,
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
    assert Path(result["abaqus_inp"]).exists()
    assert Path(result["abaqus_odb"]).exists()
    assert read_json(agent._result_path(candidate_id)) == result
    assert not job_transient_paths(agent._run_dir(candidate_id), [candidate_id])


def test_failed_job_persists_diagnosis_before_cleanup(monkeypatch):
    agent = FEMAgent(config={"abaqus": {"command": "abaqus", "max_retries": 1, "job_timeout_seconds": 10}})
    run_dir = agent._run_dir("C1")
    (run_dir / "C1.msg").write_text("solver failed", encoding="utf-8")
    (run_dir / "C1.inp").write_text("input", encoding="utf-8")
    monkeypatch.setattr("agents.fem_agent.is_abaqus_available", lambda command: True)
    def failed_command(*args, **kwargs):
        write_json(agent._result_path("C1"), {"status": "success"})
        return 1, "", "process failed"
    monkeypatch.setattr("agents.fem_agent.run_command", failed_command)

    result = agent.run(_real_fem_candidate("C1"))

    assert result["status"] == "max_retries_exceeded"
    assert result["error_type"] == "process_crash"
    assert result["error_log"]["return_code"] == 1
    assert result["error_log"]["solver_logs"] == {"C1.msg": "solver failed"}
    assert read_json(agent._result_path("C1")) == result
    assert (run_dir / "C1.inp").exists()
    assert not (run_dir / "C1.msg").exists()
