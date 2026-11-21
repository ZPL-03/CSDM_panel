"""按明确任务批量生成 DOE 候选、执行 FEM 并归档，不训练代理或重置业务数据。"""

from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.fem_agent import FEMAgent  # noqa: E402
from agents.knowledge_agent import KnowledgeAgent  # noqa: E402
from core.config_loader import load_material_db  # noqa: E402
from core.doe_sampler import DOESampler  # noqa: E402
from core.id_utils import format_task_id, next_candidate_index, next_task_index  # noqa: E402
from core.io_utils import read_json, write_json  # noqa: E402
from core.laminate import analysis_contract  # noqa: E402
from core.paths import TASKS_DIR  # noqa: E402
from core.rule_checker import RuleChecker  # noqa: E402
from core.stiffener_profile import STIFFENER_TYPES, load_param_ranges_for_type  # noqa: E402
from core.task_contract import boundary_condition_payload, load_condition_payload, normalize_task_payload  # noqa: E402
from scripts.case_records import validate_task  # noqa: E402


def input_tasks(args) -> list[dict]:
    """任务 JSON 校验后按其允许筋型拆分；参数模式显式指定筋型、工况及边界。"""
    if args.task_json:
        tasks = []
        for path in args.task_json:
            raw = read_json(path)
            task = raw["task"] if isinstance(raw, dict) and "task" in raw else raw
            validate_task(task)
            types = task.get("candidate_generation_preferences", {}).get("stiffener_types", [task["stiffener_type"]])
            if not types or task["stiffener_type"] not in types:
                raise ValueError("任务筋型必须属于非空的允许筋型集合")
            for stype in types:
                typed = deepcopy(task)
                typed["stiffener_type"] = stype
                typed.setdefault("candidate_generation_preferences", {})["stiffener_types"] = [stype]
                facts = typed.get("user_input_facts", {})
                if "stiffener_type" in facts or "stiffener_types" in facts:
                    facts.update(stiffener_type=stype, stiffener_types=[stype])
                tasks.append(typed)
        return tasks
    if not args.load_type or not args.boundary or args.nx is None or args.nxy is None:
        raise ValueError("参数模式要求 --load-type、--boundary、--nx 和 --nxy")
    if not all(math.isfinite(value) and value >= 0 for value in (args.nx, args.nxy)):
        raise ValueError("载荷分量必须为有限非负数")
    if len(set(args.stiffener_types)) != len(args.stiffener_types):
        raise ValueError("筋型不能重复")
    material = deepcopy(load_material_db()[args.material])
    material["name"] = material["display_name"]
    material["is_user_specified"] = True
    tasks = []
    for stype in args.stiffener_types:
        bounds = load_param_ranges_for_type(stype)
        envelope = {key: [bounds[key]["min"], bounds[key]["max"]]
                    for key in ("panel_length_mm", "panel_width_mm")}
        envelope["max_stiffener_height_mm"] = bounds["stiffener_height_mm"]["max"]
        task = normalize_task_payload({"application": "复合材料加筋壁板", "stiffener_type": stype,
                                       "geometry_envelope": envelope, "material_system": material})
        task["load_conditions"] = load_condition_payload(args.load_type, args.nx, args.nxy)
        task["boundary_conditions"] = boundary_condition_payload(args.boundary)
        validate_task(task)
        tasks.append(task)
    return tasks


def build_plan(tasks: list[dict], count: int, *, seed: int | None = None) -> list[dict]:
    """求解前校验全部任务和 DOE 物理输入，并预分配不冲突的任务/候选编号。"""
    if not tasks or count < len(tasks):
        raise ValueError("count 至少等于任务数，每个任务必须有候选")
    task_index, candidate_index = next_task_index(), next_candidate_index()
    plan = []
    for index, payload in enumerate(tasks):
        validate_task(payload)
        types = payload.get("candidate_generation_preferences", {}).get("stiffener_types", [payload["stiffener_type"]])
        if types != [payload["stiffener_type"]]:
            raise ValueError("每个任务 JSON 对应一种筋型，多筋型使用多个任务文件或 --stiffener-types")
        request = {"task_id": format_task_id(task_index + index), "created_at": datetime.now(timezone.utc).isoformat(),
                   "source": "initial_case_build", "task": deepcopy(payload)}
        size = count // len(tasks) + (index < count % len(tasks))
        sampler = DOESampler()
        if seed is not None:
            sampler.random_seed = seed + index
        request["sampling"] = {"method": "lhs", "seed": sampler.random_seed}
        candidates = sampler.sample_candidates(request, size, start_index=candidate_index,
                                                stiffener_type=payload["stiffener_type"])
        if len(candidates) != size:
            raise ValueError(f"{request['task_id']}: DOE 有效候选数 {len(candidates)}，要求 {size}")
        for candidate in candidates:
            if "analysis" in payload:
                candidate["analysis"] = deepcopy(payload["analysis"])
            check = RuleChecker().run(candidate, task=request)
            if not check["is_valid"]:
                raise ValueError(f"{candidate['candidate_id']}: " + "; ".join(check["errors"]))
            analysis_contract(candidate)
        plan.append({"task": request, "candidates": candidates})
        candidate_index += size
    return plan


def run_plan(plan: list[dict], workers: int) -> list[dict]:
    """按已校验输入求解，FEMAgent 保留物理工件与诊断并执行正式清理。"""
    if workers < 1:
        raise ValueError("workers 必须为正整数")
    for batch in plan:
        path = TASKS_DIR / (batch["task"]["task_id"] + ".json")
        if path.exists():
            raise FileExistsError("任务编号已占用: " + str(path))
    knowledge = KnowledgeAgent()
    knowledge.config = deepcopy(knowledge.config)
    knowledge.config["pipeline"]["auto_retrain_surrogate"] = False
    records = []
    for batch in plan:
        task = batch["task"]
        write_json(TASKS_DIR / (task["task_id"] + ".json"), task)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(FEMAgent().run, deepcopy(candidate)): candidate for candidate in batch["candidates"]}
            for future in as_completed(futures):
                candidate, result = futures[future], future.result()
                if result["candidate_id"] != candidate["candidate_id"]:
                    raise ValueError("求解结果候选编号与提交输入不一致")
                if result["status"] == "success" and result.get("analysis_contract") != analysis_contract(candidate):
                    raise ValueError("求解结果没有绑定所提交的物理输入")
                stored = knowledge.run({"task": task, "design": candidate, "abaqus_results": result})
                record = {"task_id": task["task_id"], "case_id": stored["case_id"], "candidate_id": candidate["candidate_id"],
                          "stiffener_type": candidate["stiffener_type"], "status": result["status"],
                          "BLF_global": result.get("BLF_global"), "verdict": result.get("verdict"),
                          "artifact_dir": result.get("artifact_dir")}
                records.append(record)
                print(json.dumps(record, ensure_ascii=False))
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--task-json", action="append", type=Path, help="完整任务 JSON，可重复指定")
    source.add_argument("--stiffener-types", nargs="+", choices=STIFFENER_TYPES)
    parser.add_argument("--load-type", choices=("axial_compression", "in_plane_shear", "compression_shear"))
    parser.add_argument("--boundary", choices=("SSSS", "SSCC", "CCCC"))
    parser.add_argument("--nx", type=float, help="轴向压缩 kN/m，无该分量时显式填 0")
    parser.add_argument("--nxy", type=float, help="剪切 kN/m，无该分量时显式填 0")
    parser.add_argument("--material", default=None, help="config/material_db.yaml 中的材料键，参数模式默认 T300_5208")
    parser.add_argument("--count", type=int, required=True, help="全部任务的候选总数，按任务均分")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, help="DOE 随机种子，各任务依次加一；未指定时读取项目配置")
    args = parser.parse_args()
    if args.count < 1 or args.workers < 1 or (args.seed is not None and args.seed < 0):
        parser.error("count 和 workers 必须为正整数，seed 必须非负")
    if args.task_json and any(value is not None for value in (args.load_type, args.boundary, args.nx, args.nxy, args.material)):
        parser.error("任务 JSON 与物理参数选项不能混用")
    args.material = args.material or "T300_5208"
    plan = build_plan(input_tasks(args), args.count, seed=args.seed)
    records = run_plan(plan, args.workers)
    success = sum(record["status"] == "success" for record in records)
    print(json.dumps({"task_count": len(plan), "sample_count": len(records), "success_count": success}, ensure_ascii=False))
    return 0 if success == len(records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
