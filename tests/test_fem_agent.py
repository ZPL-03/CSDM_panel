import shutil
from pathlib import Path

from abaqus.job_utils import is_abaqus_available
from agents.fem_agent import FEMAgent
from core.id_utils import format_candidate_id, next_candidate_index
from core.paths import ABAQUS_RUNS_DIR, IO_DIR


def _real_fem_candidate(candidate_id: str) -> dict:
    return {
        "candidate_id": candidate_id,
        "source": "DOE",
        "stiffener_type": "T",
        "geometry": {
            "panel_length_mm": 360,
            "panel_width_mm": 240,
            "skin_thickness_mm": 2.4,
            "pitch_mm": 120,
            "stiffener_height_mm": 24,
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


def _cleanup_candidate_artifacts(candidate_id: str) -> None:
    for path in [
        IO_DIR / f"input_{candidate_id}.json",
        IO_DIR / f"result_{candidate_id}.json",
    ]:
        path.unlink(missing_ok=True)
    shutil.rmtree(ABAQUS_RUNS_DIR / candidate_id, ignore_errors=True)


def test_fem_agent_runs_real_abaqus() -> None:
    assert is_abaqus_available("abaqus"), "当前环境未找到 abaqus 命令，真实有限元测试不能运行"

    candidate_id = format_candidate_id(next_candidate_index())
    _cleanup_candidate_artifacts(candidate_id)

    agent = FEMAgent(
        config={
            "abaqus": {
                "command": "abaqus",
                "max_retries": 1,
                "job_timeout_seconds": 900,
                "poll_interval_seconds": 2,
            }
        }
    )

    try:
        result = agent.run(_real_fem_candidate(candidate_id))

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
    finally:
        _cleanup_candidate_artifacts(candidate_id)
