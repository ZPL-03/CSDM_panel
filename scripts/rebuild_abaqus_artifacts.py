"""按已保存的候选物理输入重建 ABAQUS 工件，记录求解结果与案例索引。"""

from __future__ import annotations

import argparse
import json
import re
import sys
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.fem_agent import FEMAgent  # noqa: E402
from core.case_memory import CaseMemoryIndex  # noqa: E402
from core.io_utils import write_json  # noqa: E402
from core.paths import CASES_DIR, CASE_LIBRARY_DIR, IO_DIR  # noqa: E402
from scripts.case_records import read_case_record, validate_case_record  # noqa: E402


def _should_store_case_library_record(payload: dict) -> bool:
    results = dict(payload.get("abaqus_results", {}))
    return results.get("status") == "success" and results.get("verdict") == "通过"


def _write_case_library_record(case_path: Path, payload: dict) -> None:
    library_path = CASE_LIBRARY_DIR / case_path.name
    if _should_store_case_library_record(payload):
        write_json(library_path, payload)
        return
    library_path.unlink(missing_ok=True)


def _upsert_case_memory(payload: dict) -> None:
    scope = "formal" if _should_store_case_library_record(payload) else "archive"
    CaseMemoryIndex().upsert_cases([payload], scope=scope)


def rebuild_case(case_path: Path, *, force: bool = False) -> dict:
    """校验案例后直接提交其设计；FEMAgent 管理作业锁与求解工件清理。"""
    payload = read_case_record(case_path)
    candidate = deepcopy(payload["design"])
    # 归档设计不要求 GUI 显示名；求解入口使用已有编号作为展示标识。
    candidate["display_name"] = candidate.get("display_name", candidate["candidate_id"])

    result = FEMAgent().run(candidate, archive_existing=True)
    payload["abaqus_results"] = result
    payload["candidate_id"] = candidate["candidate_id"]
    payload["verdict"] = result["verdict"] if result["status"] == "success" else "失败"
    payload["fem_agent_retry_count"] = result["retry_count"]

    validate_case_record(payload, case_path)
    write_json(case_path, payload)
    _write_case_library_record(case_path, payload)
    _upsert_case_memory(payload)
    write_json(IO_DIR / f"result_{candidate['candidate_id']}.json", result)
    return {
        "case_id": payload["case_id"],
        "candidate_id": candidate["candidate_id"],
        "status": result["status"],
        "abaqus_odb": result.get("abaqus_odb"),
        "abaqus_inp": result.get("abaqus_inp"),
        "force": force,
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
    parser.add_argument("--force", action="store_true", help="已有 INP/ODB 时也重新求解，不预先删除案例或索引")
    args = parser.parse_args()
    if args.limit < 0 or args.workers < 1:
        parser.error("limit 必须非负，workers 必须为正整数")
    if any(not re.fullmatch(r"CASE_[1-9][0-9]*", value) for value in args.case_id) or any(
        not re.fullmatch(r"C[1-9][0-9]*", value) for value in args.candidate_id
    ):
        parser.error("案例或候选编号格式错误")

    selected_case_ids, selected_candidate_ids = set(args.case_id), set(args.candidate_id)
    found_cases, found_candidates = set(), set()

    pending: list[Path] = []
    for case_path in sorted(CASES_DIR.glob("CASE_*.json")):
        if selected_case_ids and case_path.stem not in selected_case_ids:
            continue
        payload = read_case_record(case_path)
        case_id, candidate_id = payload["case_id"], payload["design"]["candidate_id"]
        found_cases.add(case_id)
        found_candidates.add(candidate_id)
        if selected_candidate_ids and candidate_id not in selected_candidate_ids:
            continue

        result = payload["abaqus_results"]
        complete = result["status"] == "success" and all(
            result.get(key) and Path(result[key]).is_file() for key in ("abaqus_inp", "abaqus_odb")
        )
        if not args.force and complete:
            continue
        pending.append(case_path)

    missing = (selected_case_ids - found_cases) | (selected_candidate_ids - found_candidates)
    if missing:
        raise ValueError("未找到指定记录: " + ", ".join(sorted(missing)))
    if args.limit > 0:
        pending = pending[: args.limit]

    if not pending:
        print("没有需要重建的样本。")
        return 0

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_map = {executor.submit(rebuild_case, case_path, force=args.force): case_path for case_path in pending}
        for future in as_completed(future_map):
            result = future.result()
            results.append(result)
            print(json.dumps(result, ensure_ascii=False))

    print(json.dumps({"rebuilt": len(results)}, ensure_ascii=False, indent=2))
    return 0 if all(result["status"] == "success" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
