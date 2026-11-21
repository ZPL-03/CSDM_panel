from abaqus.job_utils import is_abaqus_available
from agents.orchestrator import OrchestratorAgent
from core.config_loader import load_app_config
import pytest


@pytest.mark.real_fem
def test_orchestrator_end_to_end_with_real_abaqus() -> None:
    assert is_abaqus_available(load_app_config()["abaqus"]["command"]), "当前环境未找到配置的 Abaqus 命令，端到端真实有限元测试不能运行"

    result = OrchestratorAgent().run(
        "请为机翼下蒙皮壁板设计T型筋方案，压缩载荷120kN/m，生成1个候选，"
        "长度360mm，宽度300mm，BLF目标0.1，初筛保留1个"
    )

    assert result["task"]["task_id"].startswith("TASK_")
    assert len(result["candidates"]) == 1
    assert all("task_id" not in candidate for candidate in result["candidates"])
    assert len(result["top_candidates"]) == 1
    assert len(result["results"]) == 1
    assert result["results"][0]["status"] == "success"
    assert result["results"][0]["error_type"] is None
    assert result["results"][0]["BLF_global"] is not None
    assert all(c["geometry"]["panel_length_mm"] == 360 and c["geometry"]["panel_width_mm"] == 300
               for c in result["candidates"])
