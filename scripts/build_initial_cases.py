"""批量重建统一格式样本并训练代理模型。"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.fem_agent import FEMAgent
from agents.knowledge_agent import KnowledgeAgent
from core.config_loader import load_material_db
from core.doe_sampler import DOESampler
from core.id_utils import next_candidate_index, next_task_id
from core.paths import ABAQUS_RUNS_DIR, CASES_DIR, CASE_LIBRARY_DIR, CHROMA_DIR, IO_DIR, MODELS_DIR, RESULTS_DIR
from core.schema_validator import validate_or_raise
from core.surrogate_model import SurrogateModelManager
from core.task_contract import (
    boundary_condition_payload,
    build_task_request_record,
    load_condition_payload,
    normalize_task_payload,
    task_payload_from_request,
)


MODEL_FILES = [
    MODELS_DIR / "surrogate_rf.joblib",
    MODELS_DIR / "surrogate_scaler.joblib",
    MODELS_DIR / "surrogate_mlp.pt",
    MODELS_DIR / "surrogate_metrics.json",
]


def task_application(index: int) -> str:
    applications = [
        "复合材料加筋壁板",
        "机翼下蒙皮壁板",
        "舱段壁板",
        "尾翼壁板",
    ]
    return applications[index % len(applications)]


def default_task(material_key: str, nx_kN_per_m: float, task_index: int) -> Dict:
    material = load_material_db()[material_key]
    task_payload = normalize_task_payload(
        {
            "application": task_application(task_index),
            "load_conditions": load_condition_payload("axial_compression", nx_kN_per_m=nx_kN_per_m),
            "boundary_conditions": boundary_condition_payload("SSSS"),
            "geometry_envelope": {
                "panel_length_mm": [600, 900],
                "panel_width_mm": [480, 780],
                "max_stiffener_height_mm": 42,
            },
            "material_system": {
                "name": material["display_name"],
                "density_kg_per_m3": material["density_kg_per_m3"],
                "E1_GPa": material["E1_GPa"],
                "E2_GPa": material["E2_GPa"],
                "G12_GPa": material["G12_GPa"],
                "nu12": material["nu12"],
            },
            "layup_constraints": {
                "allowed_angles": [0, 45, -45, 90],
                "symmetric": True,
                "balanced": True,
                "min_ratio_per_angle": 0.1,
            },
            "stiffener_type": "T",
            "design_targets": {"BLF_min": 1.2, "primary_objective": "最小重量"},
        }
    )
    validate_or_raise("task.schema.json", task_payload)
    return build_task_request_record(
        task_payload,
        task_id=next_task_id(),
        source="initial_case_build",
    )


def clear_dir(path: Path) -> None:
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        return
    for child in path.iterdir():
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)


def reset_dataset() -> None:
    for directory in [IO_DIR, RESULTS_DIR, CASES_DIR, ABAQUS_RUNS_DIR, CASE_LIBRARY_DIR]:
        clear_dir(directory)
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR, ignore_errors=True)
    for model_path in MODEL_FILES:
        model_path.unlink(missing_ok=True)


def partition_count(total: int, buckets: int) -> List[int]:
    base = total // buckets
    remainder = total % buckets
    return [base + (1 if index < remainder else 0) for index in range(buckets)]


def build_candidates(task: Dict, count: int, start_index: int, strict_solver_window: bool) -> List[Dict]:
    sampler = DOESampler()
    return sampler.sample_candidates(
        task=task,
        n_samples=count,
        start_index=start_index,
        strict_solver_window=strict_solver_window,
    )


def solve_candidate(task: Dict, candidate: Dict) -> Tuple[Dict, Dict]:
    agent = FEMAgent()
    task_payload = task_payload_from_request(task)
    payload = dict(candidate)
    payload["design_targets"] = task_payload["design_targets"]
    payload["load_conditions"] = task_payload["load_conditions"]
    payload["boundary_conditions"] = task_payload["boundary_conditions"]
    payload["material_system"] = task_payload["material_system"]
    result = agent.run(payload)
    return payload, result


def clean_run_directory(run_dir: Path, candidate_id: str) -> None:
    keep_names = {
        "candidate_input.json",
        f"{candidate_id}.inp",
        f"{candidate_id}.odb",
        f"{candidate_id}_mode1.json",
    }
    for child in run_dir.iterdir():
        if child.name not in keep_names:
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)


def run_task_batch(task: Dict, count: int, workers: int, strict_solver_window: bool) -> List[Dict]:
    if count <= 0:
        return []

    start_index = next_candidate_index()
    candidates = build_candidates(
        task=task,
        count=count,
        start_index=start_index,
        strict_solver_window=strict_solver_window,
    )

    knowledge_agent = KnowledgeAgent()
    records: List[Dict] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        future_map = {
            executor.submit(solve_candidate, task, candidate): candidate["candidate_id"]
            for candidate in candidates
        }
        for future in as_completed(future_map):
            candidate, result = future.result()
            knowledge_agent.run({"task": task, "design": candidate, "abaqus_results": result})
            artifact_dir = Path(str(result.get("artifact_dir", ""))) if result.get("artifact_dir") else None
            if artifact_dir and artifact_dir.exists():
                clean_run_directory(artifact_dir, candidate["candidate_id"])
            task_payload = task_payload_from_request(task)
            records.append(
                {
                    "task_id": task.get("task_id"),
                    "candidate_id": candidate["candidate_id"],
                    "material": task_payload["material_system"].get("name"),
                    "status": result["status"],
                    "BLF_global": result.get("BLF_global"),
                    "BLF_local": result.get("BLF_local"),
                    "verdict": result.get("verdict"),
                    "retry_count": result.get("retry_count"),
                    "artifact_dir": result.get("artifact_dir"),
                }
            )
            print(json.dumps(records[-1], ensure_ascii=False))
    return records


def parse_loads(loads_text: str | None, task_count: int) -> List[float]:
    if loads_text:
        return [float(item.strip()) for item in loads_text.split(",") if item.strip()]
    base = [120.0, 160.0, 220.0, 300.0, 420.0, 580.0, 760.0, 980.0]
    if task_count <= len(base):
        return base[:task_count]
    extra = [base[-1] + 100.0 * index for index in range(task_count - len(base))]
    return base + extra


def task_specs(task_count: int, loads: Sequence[float]) -> List[Tuple[str, float, int]]:
    material_keys = list(load_material_db().keys())
    specs: List[Tuple[str, float, int]] = []
    for index in range(task_count):
        material_key = material_keys[index % len(material_keys)]
        load_value = loads[index % len(loads)]
        specs.append((material_key, load_value, index))
    return specs


def train_surrogate() -> Dict | None:
    manager = SurrogateModelManager()
    records = manager.load_training_records()
    if len(records) < 10:
        print(json.dumps({"warning": "成功案例不足 10 条，跳过模型训练", "success_records": len(records)}, ensure_ascii=False, indent=2))
        return None
    summary = manager.train_from_records(records)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--task-count", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--full-range", action="store_true")
    parser.add_argument("--loads", type=str, default="")
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    if args.reset:
        reset_dataset()

    loads = parse_loads(args.loads or None, args.task_count)
    specs = task_specs(args.task_count, loads)
    bucket_sizes = partition_count(args.count, len(specs))
    strict_solver_window = not args.full_range

    all_records: List[Dict] = []
    for (material_key, nx, task_index), bucket_count in zip(specs, bucket_sizes):
        task = default_task(material_key=material_key, nx_kN_per_m=nx, task_index=task_index)
        task_records = run_task_batch(
            task=task,
            count=bucket_count,
            workers=args.workers,
            strict_solver_window=strict_solver_window,
        )
        all_records.extend(task_records)

    summary = train_surrogate()
    print(
        json.dumps(
            {
                "task_count": len(specs),
                "sample_count": len(all_records),
                "success_count": sum(1 for item in all_records if item["status"] == "success"),
                "pass_count": sum(1 for item in all_records if item.get("verdict") == "通过"),
                "materials": sorted({item["material"] for item in all_records}),
                "selected_model": summary.get("selected_model") if summary else None,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
