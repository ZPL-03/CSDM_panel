"""ABAQUS 求解智能体。"""

from __future__ import annotations

from pathlib import Path
from typing import Dict
from copy import deepcopy
import re
import shutil
from datetime import datetime
from uuid import uuid4

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
from core.laminate import analysis_contract
from core.rule_checker import RuleChecker
from core.task_contract import (
    validate_candidate_constraints,
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
            "connection_error": "连接存在未关联节点，结果中的节点集给出位置；检查部件中面和 Tie 定义。",
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

    def run(self, candidate: Dict, *, archive_existing: bool = False) -> Dict:
        validate_or_raise("candidate.schema.json", candidate)
        if candidate.get("task_constraints"):
            validate_candidate_constraints(candidate, candidate["task_constraints"])
            check = RuleChecker().run(candidate)
            if not check["is_valid"]:
                raise ValueError("候选不满足任务物理约束: " + "; ".join(check["errors"]))
        expected_contract = analysis_contract(candidate)
        current = deepcopy(candidate)
        candidate_id = current["candidate_id"]
        result_path = self._result_path(candidate_id)
        run_dir = self._run_dir(candidate_id)
        ensure_job_unlocked(run_dir)
        if any((run_dir / name).exists() for name in ("candidate_input.json", candidate_id + ".inp",
                                                       candidate_id + ".odb", candidate_id + ".cae")):
            if not archive_existing:
                raise FileExistsError(f"{candidate_id} 已有正式输入或求解工件，拒绝覆盖，请分配独立编号")
            # 维护入口保留正式原件；归档失败时停止，不提交替代作业。
            archive = ROOT_DIR.parent / (ROOT_DIR.name + "_solver_archive") / (
                datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + candidate_id + "_" + uuid4().hex[:8])
            if run_dir.resolve().parent != ABAQUS_RUNS_DIR.resolve() or archive.resolve().is_relative_to(ROOT_DIR.resolve()):
                raise ValueError("求解工件归档路径越界")
            archive.mkdir(parents=True)
            shutil.move(str(run_dir), str(archive / "run"))
            for source in (self._input_path(candidate_id), result_path):
                if source.is_file():
                    shutil.move(str(source), str(archive / source.name))
            run_dir.mkdir()
            self.emit(f"{candidate_id} 正式原件归档：{archive}")

        if not is_abaqus_available(self.abaqus_config["command"]):
            self.emit(f"{candidate_id} 未找到 ABAQUS 命令，真实有限元求解无法启动")
            return self._abaqus_unavailable_result(current)

        if result_path.exists():
            result_path.unlink()
        write_json(self._input_path(candidate_id), current)
        write_json(run_dir / "candidate_input.json", current)
        self.emit(f"{candidate_id} 开始 ABAQUS 求解")
        result = self._annotate_result(current, self._run_real(current, result_path))
        if result.get("candidate_id") != candidate_id or (result["status"] == "success"
                and result.get("analysis_contract") != expected_contract):
            raise ValueError("求解结果未绑定当前候选的逐层物理截面契约；原始结果保留供核查")
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
