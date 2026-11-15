"""默认仅清理 Python 缓存，按显式参数清理业务记录及关联工件。"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.paths import ABAQUS_DIR, ABAQUS_RUNS_DIR, CASES_DIR, CASE_LIBRARY_DIR, CHROMA_DIR, DATA_DIR, RESULTS_DIR


IO_DIR = DATA_DIR / "io"


def candidate_artifact_paths(candidate_id: str) -> list[Path]:
    normalized = str(candidate_id).strip()
    if not normalized:
        return []
    return [
        IO_DIR / f"input_{normalized}.json",
        IO_DIR / f"result_{normalized}.json",
        ABAQUS_RUNS_DIR / normalized,
        ABAQUS_DIR / f"{normalized}.inp",
        ABAQUS_DIR / f"{normalized}.odb",
        ABAQUS_DIR / f"{normalized}_mode1.json",
    ]


def case_artifact_paths(case_id: str) -> list[Path]:
    normalized = str(case_id).strip()
    if not normalized:
        return []
    return [
        CASES_DIR / f"{normalized}.json",
        CASE_LIBRARY_DIR / f"{normalized}.json",
    ]


def purge_business_records(candidate_ids: list[str], case_ids: list[str]) -> None:
    touched = False
    for candidate_id in candidate_ids:
        for path in candidate_artifact_paths(candidate_id):
            touched = touched or path.exists()
            remove_path(path)
    for case_id in case_ids:
        for path in case_artifact_paths(case_id):
            touched = touched or path.exists()
            remove_path(path)
    if touched and CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)


def purge_problem_prefixes(prefixes: list[str]) -> None:
    candidate_ids = [item.strip() for item in prefixes if item.strip() and item.strip().startswith("C") and not item.strip().startswith("CASE_")]
    case_ids = [item.strip() for item in prefixes if item.strip().startswith("CASE_")]
    purge_business_records(candidate_ids, case_ids)

    for path in RESULTS_DIR.glob("batch_summary_*.json"):
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        if any(prefix in text for prefix in prefixes):
            remove_path(path)

    if prefixes and not candidate_ids and not case_ids and CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)


def remove_path(path: Path) -> None:
    """删除文件或目录，不存在时跳过，其他删除失败直接抛出异常。"""
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def clean_python_caches() -> None:
    """清理 Python 缓存目录。"""
    for path in ROOT.rglob("__pycache__"):
        remove_path(path)
    remove_path(ROOT / ".pytest_cache")
    for path in ROOT.rglob("*.pyc"):
        remove_path(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--purge-prefix", action="append", default=[])
    parser.add_argument("--candidate-id", action="append", default=[])
    parser.add_argument("--case-id", action="append", default=[])
    args = parser.parse_args()

    clean_python_caches()
    if args.candidate_id or args.case_id:
        purge_business_records(args.candidate_id, args.case_id)
    if args.purge_prefix:
        purge_problem_prefixes(args.purge_prefix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
