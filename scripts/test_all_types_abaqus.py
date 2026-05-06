"""对 BLADE/T/HAT/L 四种筋型各运行一次 ABAQUS 屈曲分析，验证参数化建模。"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from jinja2 import Template
from core.stiffener_profile import default_geometry, STIFFENER_TYPES, TYPE_DISPLAY_NAMES

ABAQUS_CMD = "abaqus"
RUNS_DIR = ROOT_DIR / "data" / "abaqus_runs"
TEMPLATE_PATH = ROOT_DIR / "abaqus" / "templates" / "t_stiffener_buckle.py.j2"

# 每种类型的轻量测试参数（小面板 + 粗网格加速）
FAST_OVERRIDES = {
    "mesh_size_mm": 25.0,
    "buckling_modes": 4,
}

LAYUP = {
    "skin_layup": "[45/-45/0/90/0/-45/45]s",
    "skin_f0": 0.286,
    "skin_f45": 0.428,
    "skin_f90": 0.286,
    "ply_count": 14,
}

MATERIAL = {
    "name": "T300/5208",
    "density_kg_per_m3": 1600.0,
    "E1_GPa": 181.0,
    "E2_GPa": 10.3,
    "G12_GPa": 7.17,
    "nu12": 0.28,
}

LOAD = {
    "type": "axial_compression",
    "Nx_kN_per_m": -850.0,
    "Nxy_kN_per_m": 0.0,
}

BOUNDARY = {"type": "SSCC"}
TARGETS = {"BLF_min": 1.2}


def build_candidate(stype: str) -> dict:
    geom = default_geometry(stype)
    # 小面板 + 大筋距 = 快速求解
    geom["panel_length_mm"] = 400.0
    geom["panel_width_mm"] = 300.0
    geom["pitch_mm"] = 150.0
    return {
        "candidate_id": f"TEST_{stype}",
        "stiffener_type": stype,
        "geometry": geom,
        "layup": dict(LAYUP),
        "material_system": dict(MATERIAL),
        "load_conditions": dict(LOAD),
        "boundary_conditions": dict(BOUNDARY),
        "design_targets": dict(TARGETS),
        "analysis": dict(FAST_OVERRIDES),
    }


def main():
    template = Template(TEMPLATE_PATH.read_text(encoding="utf-8"))
    results = {}

    for stype in STIFFENER_TYPES:
        display = TYPE_DISPLAY_NAMES.get(stype, stype)
        print(f"\n{'='*60}")
        print(f"测试 {stype} ({display})")
        print("=" * 60)

        candidate = build_candidate(stype)
        candidate_id = candidate["candidate_id"]
        run_dir = RUNS_DIR / candidate_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # 写入输入 JSON
        input_json = run_dir / f"input_{candidate_id}.json"
        result_json = run_dir / f"result_{candidate_id}.json"
        input_json.write_text(json.dumps(candidate, ensure_ascii=False, indent=2), encoding="utf-8")

        # 清理旧结果
        for old in run_dir.glob("*"):
            if old.suffix in (".odb", ".inp", ".msg", ".dat", ".sta", ".prt",
                              ".sim", ".com", ".lck", ".023", ".log", ".env", ".odb_f"):
                old.unlink()
        if result_json.exists():
            result_json.unlink()

        # 从模板生成 ABAQUS 启动脚本
        script_path = run_dir / f"build_{candidate_id}.py"
        script_content = template.render(
            project_root=str(ROOT_DIR),
            input_json=str(input_json),
            result_json=str(result_json),
            mock_mode=False,
        )
        script_path.write_text(script_content, encoding="utf-8")

        # 运行 ABAQUS
        cmd_str = f'abaqus cae noGUI={script_path.name}'
        print(f"  执行: {cmd_str}")
        print(f"  工作目录: {run_dir}")
        t0 = time.time()

        try:
            proc = subprocess.run(
                cmd_str,
                cwd=str(run_dir),
                capture_output=True,
                text=True,
                timeout=600,
                encoding="utf-8",
                errors="replace",
                shell=True,
            )
            elapsed = time.time() - t0
            print(f"  耗时: {elapsed:.1f}s, 返回码: {proc.returncode}")

            if result_json.exists():
                result = json.loads(result_json.read_text(encoding="utf-8"))
                status = result.get("status", "unknown")
                blf = result.get("BLF_global")
                error = result.get("error_type")
                verdict = result.get("verdict", "-")
                print(f"  状态: {status}, BLF={blf}, 结论: {verdict}, 错误: {error}")
                results[stype] = {"status": status, "BLF": blf, "error": error, "verdict": verdict}
            else:
                print("  结果文件未生成！")
                stderr_tail = proc.stderr[-600:] if proc.stderr else "(空)"
                stdout_tail = proc.stdout[-600:] if proc.stdout else "(空)"
                print(f"  stderr: {stderr_tail}")
                results[stype] = {"status": "no_result", "BLF": None, "error": "result_not_found"}
        except subprocess.TimeoutExpired:
            print("  超时！")
            results[stype] = {"status": "timeout", "BLF": None, "error": "timeout"}
        except Exception as e:
            print(f"  异常: {e}")
            results[stype] = {"status": "exception", "BLF": None, "error": str(e)}

    # 汇总
    print(f"\n{'='*60}")
    print("ABAQUS 多筋型验证结果")
    print("=" * 60)
    all_ok = True
    for stype in STIFFENER_TYPES:
        r = results.get(stype, {})
        ok = r.get("status") == "success"
        if not ok:
            all_ok = False
        flag = "PASS" if ok else "FAIL"
        print(f"  [{flag}] {stype}: {r}")

    if all_ok:
        print("\n全部 4 种筋型 ABAQUS 求解成功。")
    else:
        print("\n部分筋型求解失败，详见上方日志。")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
