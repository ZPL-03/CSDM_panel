import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from gui.main_window import MainWindow


def _app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_pending_candidates_filter_out_evaluated_samples() -> None:
    app = _app()
    window = MainWindow()
    try:
        window.session.candidates = [
            {"candidate_id": "TMP_1"},
            {"candidate_id": "TMP_2"},
            {"candidate_id": "TMP_3"},
        ]
        window.session.results_by_session_id = {"TMP_2": {"candidate_id": "C2"}}

        pending = window._pending_candidates(window.session.candidates)

        assert [item["candidate_id"] for item in pending] == ["TMP_1", "TMP_3"]
    finally:
        window.close()
        app.processEvents()


def test_task_summary_uses_session_task_wording() -> None:
    app = _app()
    window = MainWindow()
    try:
        window.session.task = {
            "task_id": "TASK_9",
            "task": {
                "application": "复合材料加筋壁板",
                "load_conditions": {"type": "axial_compression", "Nx_kN_per_m": 850},
                "boundary_conditions": {"type": "SSSS"},
                "material_system": {"name": "T300/5208", "E1_GPa": 181, "density_kg_per_m3": 1600},
                "design_targets": {"BLF_min": 1.2, "primary_objective": "最小重量"},
            },
        }

        html = window._task_summary_html()

        assert "会话任务编号" in html
        assert "<b>任务编号：</b>TASK_9" not in html
    finally:
        window.close()
        app.processEvents()


def test_close_event_tolerates_deleted_worker_thread() -> None:
    from PyQt6 import sip
    from PyQt6.QtCore import QThread

    app = _app()
    window = MainWindow()
    try:
        thread = QThread()
        window.worker_thread = thread
        sip.delete(thread)

        window.close()
        app.processEvents()

        assert window.worker_thread is None
    finally:
        app.processEvents()
