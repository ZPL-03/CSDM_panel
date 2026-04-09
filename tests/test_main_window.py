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
