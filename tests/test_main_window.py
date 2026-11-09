import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from gui.main_window import MainWindow


def _app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _candidate(candidate_id: str = "TMP_1") -> dict:
    return {
        "candidate_id": candidate_id,
        "display_name": candidate_id,
        "source": "DOE",
        "stiffener_type": "T",
        "geometry": {},
        "layup": {},
        "rule_check": {},
    }


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


def test_candidate_table_uses_contract_display_name() -> None:
    app = _app()
    window = MainWindow()
    try:
        window.candidate_widget.update_candidates(
            [
                _candidate()
            ]
        )

        assert window.candidate_widget.table.item(0, 0).text() == "TMP_1"
    finally:
        window.close()
        app.processEvents()


def test_candidate_detail_shows_llm_answer_excerpt() -> None:
    app = _app()
    window = MainWindow()
    try:
        candidate = _candidate()
        candidate.update({
            "source": "LLM",
            "origin_summary": "| A1 | T300/5208 |",
            "llm_output_excerpt": "## 候选方案\n<raw table>",
        })

        window.candidate_widget.update_candidates([candidate])
        html = window.candidate_widget.detail_browser.toHtml()

        assert "LLM 回答原文" in html
        assert "&lt;raw table&gt;" in html
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


def test_chat_widget_escapes_and_wraps_messages() -> None:
    app = _app()
    window = MainWindow()
    try:
        window.chat_widget.add_message("USER", "<panel>\nline")

        html = window.chat_widget.toHtml()

        assert "&lt;panel&gt;" in html
        assert "line" in html
    finally:
        window.close()
        app.processEvents()


def test_flow_candidate_summary_updates_table_immediately() -> None:
    app = _app()
    window = MainWindow()
    try:
        window._handle_message(
            "FLOW",
            "candidate summary",
            {"event_type": "candidate_summary", "payload": {"candidates": [_candidate("TMP_9")]}},
        )

        assert window.session.candidates[0]["candidate_id"] == "TMP_9"
        assert window.session.pending_confirmation == "screen_candidates"
        assert window.candidate_widget.table.item(0, 0).text() == "TMP_9"
    finally:
        window.close()
        app.processEvents()


def test_flow_screening_summary_updates_table_immediately() -> None:
    app = _app()
    window = MainWindow()
    try:
        screened = _candidate("TMP_3")
        screened.update({"surrogate_BLF": 1.42, "rank_score": 0.91})

        window._handle_message(
            "FLOW",
            "screening summary",
            {"event_type": "screening_summary", "payload": {"screened_candidates": [screened]}},
        )

        assert window.session.screened_candidates[0]["candidate_id"] == "TMP_3"
        assert window.session.pending_confirmation == "fem_evaluation"
        assert window.candidate_widget.table.item(0, 0).text() == "TMP_3"
    finally:
        window.close()
        app.processEvents()


def test_flow_fem_summary_updates_results_immediately() -> None:
    app = _app()
    window = MainWindow()
    try:
        result = {
            "session_candidate_id": "TMP_4",
            "display_name": "TMP_4",
            "candidate_id": "CASE_351",
            "status": "completed",
            "BLF_global": 1.51,
            "verdict": "passed",
        }

        window._handle_message(
            "FLOW",
            "fem summary",
            {"event_type": "fem_summary", "payload": {"results": [result]}},
        )

        assert window.session.results_by_session_id["TMP_4"]["candidate_id"] == "CASE_351"
        assert window.session.pending_confirmation == "export_report"
        assert window.abaqus_widget.table.item(0, 0).text() == "TMP_4"
    finally:
        window.close()
        app.processEvents()


def test_report_button_allows_partial_evaluated_results() -> None:
    app = _app()
    window = MainWindow()
    try:
        window.session.task = {
            "task_id": "TASK_9",
            "task": {
                "application": "复合材料加筋壁板",
                "load_conditions": {"type": "axial_compression", "Nx_kN_per_m": 850},
                "boundary_conditions": {"type": "SSSS"},
                "material_system": {"name": "T300/5208"},
                "design_targets": {"BLF_min": 1.2, "primary_objective": "最小重量"},
            },
        }
        window.session.evaluated_candidates = [_candidate("TMP_1"), _candidate("TMP_2")]
        window.session.results_by_session_id = {"TMP_1": {"candidate_id": "C1", "session_candidate_id": "TMP_1"}}

        window._update_button_states()

        assert window.report_button.isEnabled() is True
        assert [item["candidate_id"] for item in window._report_candidate_set()] == ["TMP_1"]

        captured = {}
        window._run_action = lambda action, payload, status_text: captured.update(
            {"action": action, "payload": payload, "status_text": status_text}
        )
        window._start_report()

        assert captured["action"] == "report"
        assert [item["candidate_id"] for item in captured["payload"]["candidates"]] == ["TMP_1"]
        assert [item["candidate_id"] for item in captured["payload"]["results"]] == ["C1"]
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


def test_workbench_tables_use_full_right_pane_width() -> None:
    app = _app()
    window = MainWindow()
    try:
        window.resize(1280, 800)
        window.show()
        for panel in (window.candidate_widget, window.abaqus_widget):
            window.tabs.setCurrentWidget(panel)
            app.processEvents()
            assert panel.table.width() >= panel.width() - 30
            assert panel.table.horizontalScrollBar().maximum() == 0
            assert panel.preview_widget.width() > panel.detail_browser.width()
        assert window.workspace_splitter.sizes()[1] > 2 * window.workspace_splitter.sizes()[0]
    finally:
        window.close()
        app.processEvents()


def test_desktop_entry_opens_normal_window(monkeypatch) -> None:
    import main

    calls = []

    class Application:
        def exec(self):
            return 0

    class Window:
        def show(self):
            calls.append("normal")

    monkeypatch.setattr(main, "ensure_project_dirs", lambda: None)
    monkeypatch.setattr(main, "QApplication", lambda argv: Application())
    monkeypatch.setattr(main, "MainWindow", Window)
    assert main.main() == 0
    assert calls == ["normal"]


@pytest.mark.parametrize("size", [(1000, 640), (1280, 800), (1440, 900)])
def test_sidebar_controls_are_visible_without_scrolling(size) -> None:
    from PyQt6.QtCore import QPoint
    from PyQt6.QtWidgets import QPushButton, QScrollArea

    app = _app()
    window = MainWindow()
    try:
        window.resize(*size)
        window.show()
        for stage in ("idle", "awaiting_screen_confirmation", "awaiting_report_confirmation"):
            window.session.stage = stage
            window._update_overview_cards()
            window.task_browser.setPlainText("任务数据\n" * 100)
            window.chat_widget.setPlainText("对话消息\n" * 100)
            app.processEvents()
            assert window.width() <= size[0] and window.height() <= size[1]
            sidebar = window.workspace_splitter.widget(0)
            assert not sidebar.findChildren(QScrollArea)
            for button in sidebar.findChildren(QPushButton):
                top_left = button.mapTo(sidebar, QPoint())
                assert sidebar.rect().contains(top_left)
                assert sidebar.rect().contains(top_left + button.rect().bottomRight())
                assert button.fontMetrics().horizontalAdvance(button.text()) + 14 <= button.width()
            assert window.task_browser.height() >= 64
            assert window.chat_widget.height() >= 80
            assert window.task_browser.verticalScrollBar().maximum() > 0
            assert window.chat_widget.verticalScrollBar().maximum() > 0
    finally:
        window.close()
        app.processEvents()
