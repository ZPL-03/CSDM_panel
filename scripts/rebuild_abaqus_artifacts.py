"""按当前编号体系重建缺失的 ABAQUS ODB/INP 工件。"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.fem_agent import FEMAgent
from core.io_utils import read_json, write_json
from core.paths import CASES_DIR, CASE_LIBRARY_DIR, IO_DIR
from scripts.migrate_contracts import normalize_case_record


def rebuild_case(case_path: Path) -> dict:
    """对单个案例重新执行 FEM 求解并回写结果。"""
    payload = normalize_case_record(read_json(case_path))
    candidate = payload["design"]
    candidate["design_targets"] = payload["task"]["design_targets"]
    candidate["load_conditions"] = payload["task"]["load_conditions"]
    candidate["boundary_conditions"] = payload["task"]["boundary_conditions"]
    candidate["material_system"] = payload["task"]["material_system"]

    result = FEMAgent().run(candidate)
    payload["abaqus_results"] = result

    write_json(case_path, payload)
    write_json(CASE_LIBRARY_DIR / case_path.name, payload)
    write_json(IO_DIR / f"result_{candidate['candidate_id']}.json", result)
    return {
        "case_id": payload["case_id"],
        "candidate_id": candidate["candidate_id"],
        "status": result["status"],
        "abaqus_odb": result.get("abaqus_odb"),
        "abaqus_inp": result.get("abaqus_inp"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="0 表示全部重建")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--case-id", action="append", default=[], help="只重建指定 case_id，可重复传入")
    parser.add_argument(
        "--candidate-id",
        action="append",
        default=[],
        help="只重建指定 candidate_id，可重复传入",
    )
    args = parser.parse_args()

    selected_case_ids = {item.strip() for item in args.case_id if item.strip()}
    selected_candidate_ids = {item.strip() for item in args.candidate_id if item.strip()}

    pending: list[Path] = []
    for case_path in sorted(CASES_DIR.glob("CASE_*.json")):
        payload = normalize_case_record(read_json(case_path))
        case_id = payload.get("case_id")
        candidate_id = payload.get("design", {}).get("candidate_id")
        if selected_case_ids and case_id not in selected_case_ids:
            continue
        if selected_candidate_ids and candidate_id not in selected_candidate_ids:
            continue

        odb_path = payload.get("abaqus_results", {}).get("abaqus_odb")
        if odb_path and Path(odb_path).exists():
            continue
        pending.append(case_path)

    if args.limit > 0:
        pending = pending[: args.limit]

    if not pending:
        print("没有需要重建的样本。")
        return 0

    results = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        future_map = {executor.submit(rebuild_case, case_path): case_path for case_path in pending}
        for future in as_completed(future_map):
            result = future.result()
            results.append(result)
            print(json.dumps(result, ensure_ascii=False))

    print(json.dumps({"rebuilt": len(results)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
