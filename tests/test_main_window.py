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


def test_candidate_table_uses_session_identifier() -> None:
    app = _app()
    window = MainWindow()
    try:
        window.candidate_widget.update_candidates(
            [
                {**_candidate(), "display_name": "候选样本1"}
            ]
        )

        assert window.candidate_widget.table.item(0, 0).text() == "TMP_1"
    finally:
        window.close()
        app.processEvents()


def test_candidate_detail_preserves_design_and_preview_binding(monkeypatch) -> None:
    app = _app()
    window = MainWindow()
    try:
        candidate = _candidate()
        candidate["geometry"] = {"panel_length_mm": 360, "skin_thickness_mm": 2.375}
        candidate["layup"] = {"skin_layup": "[0/45/-45/90]s", "skin_f0": 0.25,
                              "skin_f45": 0.5, "skin_f90": 0.25}
        shown = []
        monkeypatch.setattr(window.candidate_widget.preview_widget, "show_candidate", shown.append)

        window.candidate_widget.update_candidates([candidate])
        html = window.candidate_widget.detail_browser.toHtml()

        assert shown[-1] is candidate
        assert "2.375" in html and "360" in html
        assert candidate["layup"]["skin_layup"] in window.candidate_widget.detail_browser.toPlainText()
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
        window.session.candidates = [_candidate("TMP_1"), _candidate("TMP_2"), _candidate("TMP_3")]
        window.session.results_by_session_id = {"TMP_1": {"candidate_id": "C1", "session_candidate_id": "TMP_1"}}

        window._update_button_states()

        assert window.report_button.isEnabled() is True

        captured = {}
        window._run_action = lambda action, payload, status_text: captured.update(
            {"action": action, "payload": payload, "status_text": status_text}
        )
        window._start_report()

        assert captured["action"] == "report"
        assert [item["candidate_id"] for item in captured["payload"]["candidates"]] == ["TMP_1", "TMP_2", "TMP_3"]
        assert [item["candidate_id"] for item in captured["payload"]["screened_candidates"]] == ["TMP_1", "TMP_2"]
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


@pytest.mark.parametrize("size", [(1000, 640), (1280, 800), (1440, 900)])
def test_workbench_has_three_columns_with_stacked_preview(size) -> None:
    from PyQt6.QtCore import QPoint

    app = _app()
    window = MainWindow()
    try:
        window.resize(*size)
        window.show()
        window.candidate_widget.update_candidates([
            {**_candidate(f"TMP_{n}"), "stiffener_type": "BLADE"} for n in range(1, 21)
        ])
        for panel in (window.candidate_widget, window.abaqus_widget):
            window.tabs.setCurrentWidget(panel)
            app.processEvents()
            table, detail, preview = panel.table, panel.detail_browser, panel.preview_widget
            sidebar = window.workspace_splitter.widget(0)
            assert sidebar.mapTo(window, sidebar.rect().topRight()).x() < table.mapTo(window, QPoint()).x()
            assert table.mapTo(window, table.rect().topRight()).x() < detail.mapTo(window, QPoint()).x()
            assert detail.mapTo(window, QPoint()).x() == preview.mapTo(window, QPoint()).x()
            assert detail.width() == preview.width()
            assert detail.mapTo(window, detail.rect().bottomLeft()).y() < preview.mapTo(window, QPoint()).y()
            assert panel.table.horizontalScrollBar().maximum() == 0
            header = table.horizontalHeader()
            for col in range(table.columnCount()):
                if not table.isColumnHidden(col):
                    label = table.horizontalHeaderItem(col).text()
                    assert header.fontMetrics().horizontalAdvance(label) + 6 <= header.sectionSize(col)
        assert window.candidate_widget.table.rowCount() == 20
        table = window.candidate_widget.table
        assert table.fontMetrics().horizontalAdvance(table.item(0, 1).text()) + 6 <= table.columnWidth(1)
    finally:
        window.close()
        app.processEvents()


def test_candidate_table_keeps_selection_and_only_displays_available_predictions() -> None:
    from PyQt6.QtCore import QItemSelectionModel

    app = _app()
    window = MainWindow()
    try:
        panel = window.candidate_widget
        first, second, third = (_candidate(f"TMP_{n}") for n in range(1, 4))
        panel.update_candidates([first, second, third])
        assert all(panel.table.isColumnHidden(col) for col in (3, 4, 5))
        select = QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows
        panel.table.selectionModel().select(panel.table.model().index(2, 0), select)

        second.update(surrogate_BLF=1.8, surrogate_weight=3.2, rank_score=1.5)
        panel.update_candidates([third, second, first])

        assert {item["candidate_id"] for item in panel.selected_candidates()} == {"TMP_1", "TMP_3"}
        assert all(not panel.table.isColumnHidden(col) for col in (3, 4, 5))
        assert [panel.table.item(1, col).text() for col in (3, 4, 5)] == ["1.8", "3.2", "1.5"]
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
