"""清理 Python 缓存、Abaqus 会话残留和异常工件。"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.paths import ABAQUS_DIR, ABAQUS_RUNS_DIR, CASES_DIR, CASE_LIBRARY_DIR, CHROMA_DIR, DATA_DIR, RESULTS_DIR


def remove_path(path: Path) -> None:
    """删除文件或目录，不存在时直接跳过。"""
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        path.unlink(missing_ok=True)


def clean_python_caches() -> None:
    """清理 Python 缓存目录。"""
    for path in ROOT.rglob("__pycache__"):
        remove_path(path)
    remove_path(ROOT / ".pytest_cache")
    for path in ROOT.rglob("*.pyc"):
        remove_path(path)


def clean_abaqus_session_files() -> None:
    """清理 Abaqus 会话文件和临时 build 脚本。"""
    for pattern in ("abaqus.rpy*", "abaqus*.rec"):
        for path in ABAQUS_DIR.glob(pattern):
            remove_path(path)
        for path in ABAQUS_RUNS_DIR.rglob(pattern):
            remove_path(path)

    for path in ABAQUS_DIR.glob("build_*.py"):
        if path.name == "build_panel.py":
            continue
        remove_path(path)

    for path in ABAQUS_RUNS_DIR.rglob("build_*.py"):
        remove_path(path)


def purge_problem_prefixes(prefixes: list[str]) -> None:
    """按前缀联动清理案例、IO 和运行工件。"""
    targets = (CASES_DIR, CASE_LIBRARY_DIR, DATA_DIR / "io", ABAQUS_DIR, ABAQUS_RUNS_DIR)
    for folder in targets:
        for path in folder.glob("*"):
            if any(path.name.startswith(prefix) or prefix in path.name for prefix in prefixes):
                remove_path(path)

    for path in RESULTS_DIR.glob("batch_summary_*.json"):
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        if any(prefix in text for prefix in prefixes):
            remove_path(path)

    if prefixes and CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--purge-prefix", action="append", default=[])
    args = parser.parse_args()

    clean_python_caches()
    clean_abaqus_session_files()
    purge_problem_prefixes(args.purge_prefix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
