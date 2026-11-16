"""清理行为只在临时目录中验证，不访问正式工件或模型。"""

import json
import sys
from pathlib import Path

import pytest

from scripts import clean_artifacts


@pytest.fixture(autouse=True)
def isolated_runtime(tmp_path, monkeypatch):
    """替代公共运行隔离，避免复制模型，并限制所有删除操作的路径。"""
    targets = {
        "ROOT": tmp_path,
        "DATA_DIR": tmp_path / "data",
        "IO_DIR": tmp_path / "data/io",
        "ABAQUS_DIR": tmp_path / "abaqus",
        "ABAQUS_RUNS_DIR": tmp_path / "data/abaqus_runs",
        "CASES_DIR": tmp_path / "data/cases",
        "CASE_LIBRARY_DIR": tmp_path / "knowledge/case_library",
        "CHROMA_DIR": tmp_path / "knowledge/chroma_db",
    }
    for name, path in targets.items():
        path.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(clean_artifacts, name, path)

    original_unlink = Path.unlink
    original_rmtree = clean_artifacts.shutil.rmtree

    def guarded_unlink(path, *args, **kwargs):
        assert Path(path).resolve().is_relative_to(tmp_path.resolve())
        return original_unlink(path, *args, **kwargs)

    def guarded_rmtree(path, *args, **kwargs):
        assert Path(path).resolve().is_relative_to(tmp_path.resolve())
        return original_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", guarded_unlink)
    monkeypatch.setattr(clean_artifacts.shutil, "rmtree", guarded_rmtree)
    monkeypatch.setattr(sys, "argv", ["clean_artifacts.py"])
    monkeypatch.chdir(tmp_path)


def _write_file(path: Path, content: str = "retained") -> Path:
    assert path.resolve().is_relative_to(clean_artifacts.ROOT.resolve())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _snapshot_files() -> dict[Path, bytes]:
    return {path: path.read_bytes() for path in clean_artifacts.ROOT.rglob("*") if path.is_file()}


def _candidate_files(candidate_id: str) -> list[Path]:
    return [
        clean_artifacts.IO_DIR / f"input_{candidate_id}.json",
        clean_artifacts.IO_DIR / f"result_{candidate_id}.json",
        clean_artifacts.ABAQUS_RUNS_DIR / candidate_id / f"{candidate_id}.odb",
        clean_artifacts.ABAQUS_RUNS_DIR / candidate_id / f"{candidate_id}.sta",
        clean_artifacts.ABAQUS_DIR / f"{candidate_id}.inp",
        clean_artifacts.ABAQUS_DIR / f"{candidate_id}.odb",
        clean_artifacts.ABAQUS_DIR / f"{candidate_id}_mode1.json",
    ]


def test_default_only_removes_python_caches():
    """通用缓存命令不处理仍属于求解生命周期的文件。"""
    run_dir = clean_artifacts.ABAQUS_RUNS_DIR / "C1"
    artifact_names = [
        "C1.inp", "C1.odb", "C1.sta", "C1.sim", "C1.log", "C1.lck",
        "C1.023", "C1.com", "C1.jnl", "C1.prt", "C1.env", "C1.odb_f",
        "C1.msg", "C1.dat", "C1.fil", "C1.for", "C1.inc",
        "C1_mode1.json", "candidate_input.json", "candidate_retry_1.json",
        "build_C1.py", "abaqus.rpy", "abaqus.rpy.1", "abaqus1.rec",
        "C1.simdir/matrix.bin",
    ]
    for name in artifact_names:
        _write_file(run_dir / name)
    for name in ("build_C1.py", "abaqus.rpy", "abaqus.rpy.1", "abaqus1.rec"):
        _write_file(clean_artifacts.ABAQUS_DIR / name)
    record = {
        "case_id": "CASE_1",
        "abaqus_results": {
            "status": "running",
            "abaqus_inp": str(run_dir / "C1.inp"),
            "abaqus_odb": str(run_dir / "C1.odb"),
            "visualization_json": str(run_dir / "C1_mode1.json"),
            "artifact_dir": str(run_dir),
        },
    }
    for folder in (clean_artifacts.CASES_DIR, clean_artifacts.CASE_LIBRARY_DIR):
        _write_file(folder / "CASE_1.json", json.dumps(record))
    for path in (
        clean_artifacts.IO_DIR / "input_C1.json",
        clean_artifacts.IO_DIR / "result_C1.json",
        clean_artifacts.ROOT / "data/results/batch_summary_C1.json",
        clean_artifacts.CHROMA_DIR / "chroma.sqlite3",
        clean_artifacts.ROOT / "models/surrogate.pt",
        clean_artifacts.ROOT / "pkg/source.py",
        clean_artifacts.ROOT / "pkg/.pytest_cache/retained",
    ):
        _write_file(path)
    expected = _snapshot_files()
    cache_dirs = [
        clean_artifacts.ROOT / "__pycache__",
        clean_artifacts.ROOT / "pkg/__pycache__",
        run_dir / "__pycache__",
        clean_artifacts.ROOT / ".pytest_cache",
    ]
    for folder in cache_dirs:
        _write_file(folder / "cached.pyc")
    loose_pyc = _write_file(clean_artifacts.ROOT / "pkg/loose.pyc")

    assert clean_artifacts.main() == 0
    assert _snapshot_files() == expected
    assert all(not folder.exists() for folder in cache_dirs)
    assert not loose_pyc.exists()


def test_default_does_not_read_batch_summaries():
    """业务汇总无法按 UTF-8 解码时，默认缓存清理仍不读取或改动该文件。"""
    summary = _write_file(clean_artifacts.ROOT / "data/results/batch_summary_unreadable.json")
    summary.write_bytes(b"\xff")

    assert clean_artifacts.main() == 0
    assert summary.read_bytes() == b"\xff"


@pytest.mark.parametrize(
    "arguments,delete_candidate,delete_case",
    [
        (["--candidate-id", " C1 "], True, False),
        (["--case-id", " CASE_1 "], False, True),
        (["--candidate-id", "C1", "--case-id", "CASE_1"], True, True),
    ],
)
def test_explicit_purge_uses_exact_ids(monkeypatch, arguments, delete_candidate, delete_case):
    """指定 C1 不影响 C10 或包含相似文本的报告。"""
    selected_candidates = _candidate_files("C1")
    selected_cases = [folder / "CASE_1.json" for folder in (clean_artifacts.CASES_DIR, clean_artifacts.CASE_LIBRARY_DIR)]
    unrelated = _candidate_files("C10") + [
        clean_artifacts.CASES_DIR / "CASE_10.json",
        clean_artifacts.CASE_LIBRARY_DIR / "CASE_10.json",
    ]
    for path in selected_candidates + selected_cases + unrelated:
        _write_file(path)
    _write_file(clean_artifacts.ROOT / "data/results/batch_summary_C10.json", '{"candidate_id":"C10"}')
    index = _write_file(clean_artifacts.CHROMA_DIR / "chroma.sqlite3")
    expected = _snapshot_files()
    deleted = {index}
    if delete_candidate:
        deleted.update(selected_candidates)
    if delete_case:
        deleted.update(selected_cases)
    monkeypatch.setattr(sys, "argv", ["clean_artifacts.py", *arguments])

    assert clean_artifacts.main() == 0
    assert _snapshot_files() == {path: content for path, content in expected.items() if path not in deleted}
    assert not clean_artifacts.CHROMA_DIR.exists()
    assert (clean_artifacts.ABAQUS_RUNS_DIR / "C1").exists() == (not delete_candidate)


@pytest.mark.parametrize("arguments", [
    ["--candidate-id", "C404"],
    ["--case-id", "CASE_404"],
    ["--candidate-id", " ", "--case-id", " "],
])
def test_unmatched_business_ids_preserve_index_and_records(monkeypatch, arguments):
    for path in _candidate_files("C1") + [
        clean_artifacts.CASES_DIR / "CASE_1.json",
        clean_artifacts.CASE_LIBRARY_DIR / "CASE_1.json",
        clean_artifacts.CHROMA_DIR / "chroma.sqlite3",
        clean_artifacts.ROOT / "data/results/batch_summary_C1.json",
    ]:
        _write_file(path)
    expected = _snapshot_files()
    monkeypatch.setattr(sys, "argv", ["clean_artifacts.py", *arguments])

    assert clean_artifacts.main() == 0
    assert _snapshot_files() == expected


@pytest.mark.parametrize("mode", ["cache", "business_index"])
def test_directory_deletion_errors_propagate(monkeypatch, mode):
    """在底层 rmdir 注入错误，确保真实 rmtree 不会通过 ignore_errors 吞掉异常。"""
    target = clean_artifacts.ROOT / "__pycache__" if mode == "cache" else clean_artifacts.CHROMA_DIR
    target.mkdir(exist_ok=True)
    if mode == "business_index":
        _write_file(clean_artifacts.IO_DIR / "input_C1.json")
        monkeypatch.setattr(sys, "argv", ["clean_artifacts.py", "--candidate-id", "C1"])
    original_rmdir = clean_artifacts.shutil.os.rmdir

    def deny_target(path, *args, **kwargs):
        if Path(path).resolve() == target.resolve():
            raise PermissionError("locked directory")
        return original_rmdir(path, *args, **kwargs)

    monkeypatch.setattr(clean_artifacts.shutil.os, "rmdir", deny_target)
    with pytest.raises(PermissionError, match="locked directory"):
        clean_artifacts.main()
    assert target.exists()


def test_file_deletion_errors_propagate(monkeypatch):
    target = _write_file(clean_artifacts.ROOT / "locked.pyc")
    original_unlink = Path.unlink

    def deny_target(path, *args, **kwargs):
        if path == target:
            raise PermissionError("locked file")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", deny_target)
    with pytest.raises(PermissionError, match="locked file"):
        clean_artifacts.main()
    assert target.read_text(encoding="utf-8") == "retained"


def test_missing_path_is_ignored(tmp_path):
    assert clean_artifacts.remove_path(tmp_path / "missing") is None


def test_completed_job_cleanup_preserves_model_and_result_files():
    from abaqus.job_utils import cleanup_job_artifacts

    folder = clean_artifacts.ABAQUS_RUNS_DIR / "C1"
    keep = {"C1.inp", "C1.odb", "C1.cae", "C1_mode1.json", "candidate_input.json"}
    for name in keep | {"C1.sta", "C1.sim", "C1.msg", "C1.dat", "build_C1.py", "abaqus.rpy"}:
        _write_file(folder / name)
    _write_file(folder / "C1.simdir/matrix.bin")

    cleanup_job_artifacts(folder, ["C1"])

    assert {path.name for path in folder.iterdir()} == keep


def test_job_lock_preserves_all_files():
    from abaqus.job_utils import cleanup_job_artifacts

    folder = clean_artifacts.ABAQUS_RUNS_DIR / "C1"
    for name in ("C1.lck", "C1.sta", "build_C1.py"):
        _write_file(folder / name)
    before = _snapshot_files()
    with pytest.raises(RuntimeError, match="作业锁"):
        cleanup_job_artifacts(folder, ["C1"])
    assert _snapshot_files() == before


def test_cleanup_rejects_invalid_ids_and_project_root():
    for value in ("../config", "C1/..", "C0", "C1_suffix"):
        with pytest.raises(ValueError, match="无效求解编号"):
            clean_artifacts.candidate_artifact_paths(value)
    with pytest.raises(ValueError, match="越界"):
        clean_artifacts.remove_path(clean_artifacts.ROOT)
