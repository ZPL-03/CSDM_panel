"""ABAQUS 求解智能体。"""

from __future__ import annotations

from pathlib import Path
from typing import Dict
from copy import deepcopy
import re

from jinja2 import Template

from agents.base import BaseAgent
from abaqus.job_utils import (
    cleanup_job_artifacts, diagnose_failure, ensure_job_unlocked, is_abaqus_available,
    read_tail_text, run_command, solver_log_tails,
)
from core.config_loader import load_app_config
from core.io_utils import read_json, write_json
from core.paths import ABAQUS_RUNS_DIR, ABAQUS_TEMPLATE_DIR, IO_DIR, ROOT_DIR
from core.schema_validator import validate_or_raise
from core.task_contract import (
    describe_boundary_conditions,
    describe_load_conditions,
)


class FEMAgent(BaseAgent):
    agent_name = "FEM_AGENT"

    def __init__(self, progress_callback=None, config: Dict | None = None) -> None:
        super().__init__(progress_callback=progress_callback)
        self.config = config or load_app_config()
        self.abaqus_config = self.config["abaqus"]

    def _input_path(self, candidate_id: str) -> Path:
        return IO_DIR / f"input_{candidate_id}.json"

    def _result_path(self, candidate_id: str) -> Path:
        return IO_DIR / f"result_{candidate_id}.json"

    def _run_dir(self, candidate_id: str) -> Path:
        if not re.fullmatch(r"C[1-9][0-9]*", candidate_id):
            raise ValueError(f"无效求解编号：{candidate_id}")
        run_dir = ABAQUS_RUNS_DIR / candidate_id
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _script_path(self, candidate_id: str) -> Path:
        return self._run_dir(candidate_id) / f"build_{candidate_id}.py"

    def _inp_path(self, candidate_id: str) -> Path:
        return self._run_dir(candidate_id) / f"{candidate_id}.inp"

    def _odb_path(self, candidate_id: str) -> Path:
        return self._run_dir(candidate_id) / f"{candidate_id}.odb"

    def generate_script(self, candidate: Dict) -> Path:
        template_path = ABAQUS_TEMPLATE_DIR / "t_stiffener_buckle.py.j2"
        template = Template(template_path.read_text(encoding="utf-8"))
        script_content = template.render(
            project_root=str(ROOT_DIR),
            input_json=str(self._input_path(candidate["candidate_id"])),
            result_json=str(self._result_path(candidate["candidate_id"])),
        )
        script_path = self._script_path(candidate["candidate_id"])
        script_path.write_text(script_content, encoding="utf-8")
        return script_path

    def _diagnosis_summary(self, result: Dict) -> str:
        if result.get("status") == "success":
            verdict = result.get("verdict", "未判定")
            analysis_flags = dict(result.get("analysis_flags", {}))
            negative_modes_skipped = int(analysis_flags.get("negative_modes_skipped", 0) or 0)
            first_positive_mode_index = int(analysis_flags.get("first_positive_mode_index", 1) or 1)
            if negative_modes_skipped > 0:
                return (
                    f"线性屈曲校核已完成，前 {negative_modes_skipped} 个负特征值模态已跳过，"
                    f"按第 {first_positive_mode_index} 阶正特征值判定，当前结论为“{verdict}”。"
                )
            return f"线性屈曲校核已完成，当前结论为“{verdict}”。"

        error_type = str(result.get("error_type") or "failed")
        mapping = {
            "mesh_error": "网格划分失败，需根据网格诊断检查几何拓扑、单元质量与网格设置。",
            "geometry_issue": "几何装配失败，需检查输入尺寸、连接关系及装配容差。",
            "convergence_fail": "特征值求解未收敛，需核对求解日志、载荷、约束与特征值搜索设置。",
            "blf_negative": "求得负特征值，通常意味着载荷方向或边界设置需要复核。",
            "process_crash": "ABAQUS 进程异常退出，请查看结果中的 error_log 并检查运行环境。",
            "abaqus_unavailable": "未找到 ABAQUS 命令，无法执行真实有限元求解。",
        }
        return mapping.get(error_type, "求解未完成，请查看结果中的 error_log。")

    def _annotate_result(self, candidate: Dict, result: Dict) -> Dict:
        annotated = dict(result)
        annotated["load_summary"] = describe_load_conditions(candidate.get("load_conditions", {}))
        annotated["boundary_summary"] = describe_boundary_conditions(candidate.get("boundary_conditions", {}))
        annotated["diagnosis_summary"] = self._diagnosis_summary(annotated)
        return annotated

    def _run_real(self, candidate: Dict, result_path: Path) -> Dict:
        candidate_id = candidate["candidate_id"]
        run_dir = self._run_dir(candidate_id)
        ensure_job_unlocked(run_dir)
        script_path = self.generate_script(candidate)
        self.emit(f"{candidate_id} 脚本已生成：{script_path.name}")

        command = [self.abaqus_config["command"], "cae", f"noGUI={script_path.name}"]
        return_code, stdout, stderr = run_command(
            command,
            workdir=run_dir,
            timeout=self.abaqus_config["job_timeout_seconds"],
        )
        self.emit(f"{candidate_id} ABAQUS 作业完成，返回码 {return_code}")

        if return_code == 0 and result_path.is_file():
            return read_json(result_path)

        diagnosis = diagnose_failure(
            msg_text=read_tail_text(run_dir / f"{candidate_id}.msg"),
            dat_text=read_tail_text(run_dir / f"{candidate_id}.dat"),
            return_code=return_code,
        )
        return {
            "candidate_id": candidate_id,
            "status": "failed",
            "retry_count": 0,
            "BLF_global": None,
            "BLF_local": None,
            "failure_mode": None,
            "max_displacement_mm": None,
            "weight_kg_per_m2": None,
            "verdict": None,
            "abaqus_odb": None,
            "abaqus_inp": str(self._inp_path(candidate_id)) if self._inp_path(candidate_id).exists() else None,
            "visualization_json": None,
            "artifact_dir": str(run_dir),
            "error_type": diagnosis["error_type"],
            "error_log": {
                "return_code": return_code,
                "stdout": stdout[-4000:],
                "stderr": stderr[-4000:],
                "reason": diagnosis["reason"],
            },
        }

    def _abaqus_unavailable_result(self, candidate: Dict) -> Dict:
        candidate_id = candidate["candidate_id"]
        result = self._annotate_result(
            candidate,
            {
                "candidate_id": candidate_id,
                "status": "failed",
                "retry_count": 0,
                "BLF_global": None,
                "BLF_local": None,
                "failure_mode": None,
                "max_displacement_mm": None,
                "weight_kg_per_m2": None,
                "verdict": None,
                "abaqus_odb": None,
                "abaqus_inp": None,
                "visualization_json": None,
                "artifact_dir": str(self._run_dir(candidate_id)),
                "error_type": "abaqus_unavailable",
                "error_log": f"未找到 ABAQUS 命令：{self.abaqus_config['command']}",
                "mode_eigenvalues": None,
            },
        )
        write_json(self._result_path(candidate_id), result)
        validate_or_raise("abaqus_result.schema.json", result)
        return result

    def run(self, candidate: Dict) -> Dict:
        validate_or_raise("candidate.schema.json", candidate)
        current = deepcopy(candidate)
        candidate_id = current["candidate_id"]
        result_path = self._result_path(candidate_id)
        run_dir = self._run_dir(candidate_id)
        ensure_job_unlocked(run_dir)

        if not is_abaqus_available(self.abaqus_config["command"]):
            self.emit(f"{candidate_id} 未找到 ABAQUS 命令，真实有限元求解无法启动")
            return self._abaqus_unavailable_result(current)

        if result_path.exists():
            result_path.unlink()
        write_json(self._input_path(candidate_id), current)
        write_json(run_dir / "candidate_input.json", current)
        self.emit(f"{candidate_id} 开始 ABAQUS 求解")
        result = self._annotate_result(current, self._run_real(current, result_path))
        result["retry_count"] = 0
        # 求解失败保留所提交的设计；诊断日志写入结果后再清理一次性工件。
        if result["status"] != "success":
            detail = result.get("error_log")
            result["error_log"] = dict(detail) if isinstance(detail, dict) else {"reason": detail}
            result["error_log"]["solver_logs"] = solver_log_tails(run_dir, [candidate_id])
            self.emit(f"{candidate_id} 失败类型：{result.get('error_type') or 'failed'}")
        validate_or_raise("abaqus_result.schema.json", result)
        write_json(result_path, result)
        cleanup_job_artifacts(run_dir, [candidate_id])
        return result
