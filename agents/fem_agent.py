"""ABAQUS 求解智能体。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

from jinja2 import Template

from agents.base import BaseAgent
from abaqus.job_utils import diagnose_failure, is_abaqus_available, read_tail_text, run_command, wait_for_result_file
from core.config_loader import load_app_config
from core.io_utils import read_json, write_json, write_text
from core.paths import ABAQUS_RUNS_DIR, ABAQUS_TEMPLATE_DIR, IO_DIR, ROOT_DIR
from core.schema_validator import validate_or_raise
from core.task_contract import (
    boundary_stiffness_factor,
    describe_boundary_conditions,
    describe_load_conditions,
    equivalent_in_plane_load,
    normalize_boundary_conditions,
    normalize_load_conditions,
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
        run_dir = ABAQUS_RUNS_DIR / candidate_id
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _script_path(self, candidate_id: str) -> Path:
        return self._run_dir(candidate_id) / f"build_{candidate_id}.py"

    def _inp_path(self, candidate_id: str) -> Path:
        return self._run_dir(candidate_id) / f"{candidate_id}.inp"

    def _odb_path(self, candidate_id: str) -> Path:
        return self._run_dir(candidate_id) / f"{candidate_id}.odb"

    def _visualization_path(self, candidate_id: str) -> Path:
        return self._run_dir(candidate_id) / f"{candidate_id}_mode1.json"

    def _cleanup_run_artifacts(self, candidate_id: str, keep_logs: bool = True) -> None:
        run_dir = self._run_dir(candidate_id)
        removable = [".lck", ".023", ".com", ".jnl", ".sta", ".prt", ".sim", ".log", ".env", ".odb_f"]
        if not keep_logs:
            removable.extend([".msg", ".dat"])

        for suffix in removable:
            path = run_dir / f"{candidate_id}{suffix}"
            if path.exists():
                path.unlink(missing_ok=True)

        for path in run_dir.glob("abaqus.rpy*"):
            path.unlink(missing_ok=True)
        for path in run_dir.glob("abaqus*.rec"):
            path.unlink(missing_ok=True)
        for path in run_dir.glob("candidate_retry_*.json"):
            path.unlink(missing_ok=True)

        script_path = self._script_path(candidate_id)
        if script_path.exists():
            script_path.unlink(missing_ok=True)

    def generate_script(self, candidate: Dict, mock_mode: bool) -> Path:
        template_path = ABAQUS_TEMPLATE_DIR / "t_stiffener_buckle.py.j2"
        template = Template(template_path.read_text(encoding="utf-8"))
        script_content = template.render(
            project_root=str(ROOT_DIR),
            input_json=str(self._input_path(candidate["candidate_id"])),
            result_json=str(self._result_path(candidate["candidate_id"])),
            mock_mode=mock_mode,
        )
        script_path = self._script_path(candidate["candidate_id"])
        script_path.write_text(script_content, encoding="utf-8")
        return script_path

    def _write_mock_artifacts(self, candidate: Dict) -> None:
        candidate_id = candidate["candidate_id"]
        write_text(self._inp_path(candidate_id), f"*Heading\n** Mock input for {candidate_id}\n")
        write_text(self._odb_path(candidate_id), f"mock odb placeholder for {candidate_id}\n")

        geometry = candidate["geometry"]
        panel_length = float(geometry["panel_length_mm"])
        panel_width = float(geometry["panel_width_mm"])
        equivalent_load = equivalent_in_plane_load(candidate.get("load_conditions", {})) or 850.0
        stiffness_factor = boundary_stiffness_factor(candidate.get("boundary_conditions", {}))
        points = [
            [0.0, 0.0, 0.0],
            [panel_length, 0.0, 0.0],
            [panel_length, panel_width, 0.0],
            [0.0, panel_width, 0.0],
        ]
        faces = [[4, 0, 1, 2, 3]]
        amplitude = max(equivalent_load / 1800.0 / max(stiffness_factor, 0.1), 0.15)
        scalars = [0.0, round(amplitude * 0.8, 4), round(amplitude * 1.1, 4), round(amplitude * 0.6, 4)]
        write_json(
            self._visualization_path(candidate_id),
            {
                "candidate_id": candidate_id,
                "points": points,
                "faces": faces,
                "scalars": scalars,
                "scalar_name": "MockModeMagnitude",
                "title": "Mock First Buckling Mode",
            },
        )

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
            "mesh_error": "网格划分阶段出现异常，建议放宽网格尺寸后重试。",
            "geometry_issue": "几何装配阶段出现异常，建议调整筋高、翼缘宽度或容差设置。",
            "convergence_fail": "特征值求解未稳定收敛，建议增加搜索模态数并收敛到更稳健的几何区间。",
            "blf_negative": "求得负特征值，通常意味着载荷方向或边界设置需要复核。",
            "process_crash": "ABAQUS 进程异常退出，建议检查运行环境、日志和临时文件。",
        }
        return mapping.get(error_type, "求解未完成，建议检查日志后重试。")

    def _annotate_result(self, candidate: Dict, result: Dict) -> Dict:
        annotated = dict(result)
        annotated["load_summary"] = describe_load_conditions(candidate.get("load_conditions", {}))
        annotated["boundary_summary"] = describe_boundary_conditions(candidate.get("boundary_conditions", {}))
        annotated["diagnosis_summary"] = self._diagnosis_summary(annotated)
        return annotated

    def _run_mock(self, candidate: Dict, retry_count: int = 0) -> Dict:
        geometry = candidate["geometry"]
        load_conditions = normalize_load_conditions(candidate.get("load_conditions", {}))
        boundary_conditions = normalize_boundary_conditions(candidate.get("boundary_conditions", {}))
        equivalent_load = equivalent_in_plane_load(load_conditions) or 850.0
        stiffness_factor = boundary_stiffness_factor(boundary_conditions)
        blf = (
            1.02
            + geometry["skin_thickness_mm"] * 0.08
            + geometry["stiffener_height_mm"] * 0.006
            + geometry["web_thickness_mm"] * 0.03
            + geometry["flange_width_mm"] * 0.004
        ) * stiffness_factor / max(equivalent_load / 850.0, 0.2)
        weight = (
            3.55
            + geometry["skin_thickness_mm"] * 0.24
            + geometry["stiffener_height_mm"] * 0.026
            + geometry["web_thickness_mm"] * 0.15
            + geometry["flange_thickness_mm"] * 0.11
        )
        self._write_mock_artifacts(candidate)
        failure_mode = {
            "axial_compression": "整体屈曲",
            "in_plane_shear": "剪切诱导屈曲",
            "compression_shear": "压剪耦合屈曲",
        }.get(load_conditions["type"], "整体屈曲")
        result = {
            "candidate_id": candidate["candidate_id"],
            "status": "success",
            "retry_count": retry_count,
            "BLF_global": round(blf, 3),
            "BLF_local": round(blf * 1.18, 3),
            "failure_mode": failure_mode,
            "max_displacement_mm": round(max(0.35, 4.8 / max(blf, 0.45)), 3),
            "weight_kg_per_m2": round(weight, 3),
            "verdict": "通过" if blf >= candidate.get("design_targets", {}).get("BLF_min", 1.2) else "不通过",
            "abaqus_odb": str(self._odb_path(candidate["candidate_id"])),
            "abaqus_inp": str(self._inp_path(candidate["candidate_id"])),
            "visualization_json": str(self._visualization_path(candidate["candidate_id"])),
            "artifact_dir": str(self._run_dir(candidate["candidate_id"])),
            "error_type": None,
            "error_log": None,
            "mode_eigenvalues": [round(blf, 3), round(blf * 1.18, 3), round(blf * 1.32, 3)],
        }
        result = self._annotate_result(candidate, result)
        write_json(self._result_path(candidate["candidate_id"]), result)
        return result

    def apply_adjustment(self, candidate: Dict, failure_type: str, attempt: int) -> Dict:
        geometry = dict(candidate["geometry"])
        adjustment: Dict[str, object] = {"attempt": attempt + 1, "failure_type": failure_type}

        if failure_type == "mesh_error":
            current_mesh_size = float(candidate.get("analysis", {}).get("mesh_size_mm", 0.0))
            candidate.setdefault("analysis", {})
            candidate["analysis"]["mesh_size_mm"] = round(max(current_mesh_size * 1.1, 8.0) if current_mesh_size else 10.0, 3)
            geometry["pitch_mm"] = round(geometry["pitch_mm"] * 1.03, 3)
            adjustment["strategy"] = "增大网格尺寸并略微放宽筋距"
        elif failure_type == "geometry_issue":
            geometry["flange_width_mm"] = round(max(geometry["flange_width_mm"] - 1.5, 10.0), 3)
            geometry["stiffener_height_mm"] = round(max(geometry["stiffener_height_mm"] - 1.0, 12.0), 3)
            adjustment["strategy"] = "收缩翼缘并降低筋高以改善装配容差"
        elif failure_type == "convergence_fail":
            candidate.setdefault("analysis", {})
            candidate["analysis"]["buckling_modes"] = int(candidate["analysis"].get("buckling_modes", 10)) + 4
            geometry["skin_thickness_mm"] = round(min(geometry["skin_thickness_mm"] + 0.2, 4.0), 3)
            geometry["web_thickness_mm"] = round(min(geometry["web_thickness_mm"] + 0.15, 3.2), 3)
            geometry["pitch_mm"] = round(max(geometry["pitch_mm"] * 0.92, 90.0), 3)
            geometry["stiffener_height_mm"] = round(min(max(geometry["stiffener_height_mm"] * 0.9, 18.0), 38.0), 3)
            geometry["flange_width_mm"] = round(min(max(geometry["flange_width_mm"], 14.0), 22.0), 3)
            adjustment["strategy"] = "增加特征值搜索规模，并收敛到更稳的筋距、筋高和厚度组合"
        elif failure_type == "blf_negative":
            candidate.setdefault("analysis", {})
            candidate["analysis"]["buckling_modes"] = max(int(candidate["analysis"].get("buckling_modes", 8)), 12) + 4
            geometry["stiffener_height_mm"] = round(min(geometry["stiffener_height_mm"] + 2.0, 50.0), 3)
            geometry["skin_thickness_mm"] = round(min(geometry["skin_thickness_mm"] + 0.15, 4.0), 3)
            adjustment["strategy"] = "增加屈曲模态搜索数量，并略微提高筋高与蒙皮厚度"
        else:
            adjustment["strategy"] = "清理临时文件后重试"

        adjusted = dict(candidate)
        adjusted["geometry"] = geometry
        adjusted["last_adjustment"] = adjustment
        return adjusted

    def _run_real(self, candidate: Dict, result_path: Path) -> Dict:
        candidate_id = candidate["candidate_id"]
        run_dir = self._run_dir(candidate_id)
        self._cleanup_run_artifacts(candidate_id, keep_logs=False)
        script_path = self.generate_script(candidate, mock_mode=False)

        command = [self.abaqus_config["command"], "cae", f"noGUI={script_path.name}"]
        return_code, stdout, stderr = run_command(
            command,
            workdir=run_dir,
            timeout=self.abaqus_config["job_timeout_seconds"],
        )
        self.emit(f"{candidate_id} ABAQUS 作业完成，返回码 {return_code}")

        if wait_for_result_file(
            result_path=result_path,
            timeout_seconds=self.abaqus_config["job_timeout_seconds"],
            poll_interval_seconds=self.abaqus_config["poll_interval_seconds"],
        ):
            result = self._annotate_result(candidate, read_json(result_path))
            write_json(result_path, result)
            self._cleanup_run_artifacts(candidate_id, keep_logs=result["status"] != "success")
            return result

        diagnosis = diagnose_failure(
            msg_text=read_tail_text(run_dir / f"{candidate_id}.msg"),
            dat_text=read_tail_text(run_dir / f"{candidate_id}.dat"),
            return_code=return_code,
        )
        return self._annotate_result(
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
                "abaqus_inp": str(self._inp_path(candidate_id)) if self._inp_path(candidate_id).exists() else None,
                "visualization_json": None,
                "artifact_dir": str(run_dir),
                "error_type": diagnosis["error_type"],
                "error_log": {
                    "stdout": stdout[-4000:],
                    "stderr": stderr[-4000:],
                    "reason": diagnosis["reason"],
                },
            },
        )

    def run(self, candidate: Dict) -> Dict:
        validate_or_raise("candidate.schema.json", candidate)
        retries = int(self.abaqus_config["max_retries"])
        current = dict(candidate)
        candidate_id = current["candidate_id"]
        result_path = self._result_path(candidate_id)
        run_dir = self._run_dir(candidate_id)
        write_json(run_dir / "candidate_input.json", current)

        for attempt in range(retries):
            if result_path.exists():
                result_path.unlink()

            write_json(self._input_path(candidate_id), current)
            force_mock = os.getenv("CSDM_USE_MOCK_ABAQUS", "0") == "1"
            mock_mode = bool(current.get("mock_mode", False) or force_mock)
            if not mock_mode:
                mock_mode = bool(
                    self.abaqus_config.get("use_mock_when_unavailable", True)
                    and not is_abaqus_available(self.abaqus_config["command"])
                )

            script_path = self.generate_script(current, mock_mode=mock_mode)
            self.emit(f"{candidate_id} 第 {attempt + 1} 次尝试，脚本已生成：{script_path.name}")
            result = self._run_mock(current, retry_count=attempt) if mock_mode else self._run_real(current, result_path)
            result["retry_count"] = attempt

            if result["status"] == "success":
                self._cleanup_run_artifacts(candidate_id, keep_logs=True)
                result = self._annotate_result(current, result)
                validate_or_raise("abaqus_result.schema.json", result)
                return result

            failure_type = result.get("error_type") or "failed"
            self.emit(f"{candidate_id} 失败类型：{failure_type}")
            if attempt == retries - 1:
                final_result = self._annotate_result(current, dict(result))
                final_result["status"] = "max_retries_exceeded"
                validate_or_raise("abaqus_result.schema.json", final_result)
                return final_result

            current = self.apply_adjustment(current, failure_type, attempt)
            write_json(run_dir / f"candidate_retry_{attempt + 1}.json", current)
            strategy = current.get("last_adjustment", {}).get("strategy", "默认重试")
            self.emit(f"{candidate_id} 准备重试，调整策略：{strategy}")

        raise RuntimeError("FEM_AGENT 未按预期返回结果")
