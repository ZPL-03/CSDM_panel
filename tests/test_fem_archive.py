"""维护重算保留正式原件，普通调用不得覆盖。"""

from agents.fem_agent import FEMAgent
from core.paths import ABAQUS_RUNS_DIR, IO_DIR
from tests.test_fem_agent import _real_fem_candidate


def test_maintenance_archives_original_bytes_before_an_unavailable_solver(monkeypatch, tmp_path):
    monkeypatch.setattr("agents.fem_agent.ROOT_DIR", tmp_path / "panel")
    monkeypatch.setattr("agents.fem_agent.is_abaqus_available", lambda *_: False)
    run = ABAQUS_RUNS_DIR / "C1"
    run.mkdir()
    original = b"formal model bytes\r\n"
    (run / "C1.inp").write_bytes(original)
    (IO_DIR / "input_C1.json").write_bytes(b"original input")
    result = FEMAgent().run(_real_fem_candidate("C1"), archive_existing=True)
    archive = next((tmp_path / "panel_solver_archive").iterdir())
    assert (archive / "run/C1.inp").read_bytes() == original
    assert (archive / "input_C1.json").read_bytes() == b"original input"
    assert not (run / "C1.inp").exists()
    assert result["status"] == "failed" and result["BLF_global"] is None
