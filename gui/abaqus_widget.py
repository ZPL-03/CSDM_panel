"""ABAQUS 结果展示组件。"""

from __future__ import annotations

from typing import Iterable

from core.display import format_number as _fmt

from PyQt6.QtCore import QSignalBlocker, Qt
from PyQt6.QtWidgets import (
    QLabel,
    QSplitter,
    QTableWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from gui.interactive_view import InteractivePlotWidget
from gui.theme import FEM_STATUS_LABELS, configure_design_table, design_cell, verdict_color


class AbaqusWidget(QWidget):
    """展示有限元校核结果与交互式模态视图。"""

    HEADERS = ["样本", "工件", "BLF", "面密度", "校核"]

    def __init__(self) -> None:
        super().__init__()
        self.results: list[dict] = []

        self.table = QTableWidget(0, len(self.HEADERS))
        self.table.setHorizontalHeaderLabels(self.HEADERS)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self._refresh_detail)
        configure_design_table(self.table)
        self.table.horizontalHeaderItem(2).setToolTip("首个正特征值对应的线性屈曲载荷因子")
        self.table.horizontalHeaderItem(3).setToolTip("面密度（kg/m²）")

        self.summary_label = QLabel("暂无有限元校核结果")
        self.summary_label.setObjectName("tableSummary")

        self.detail_browser = QTextBrowser()
        self.preview_widget = InteractivePlotWidget("完成 ABAQUS 校核后，这里会显示可旋转的模态云图。")

        self.detail_splitter = QSplitter(Qt.Orientation.Vertical)
        self.detail_splitter.setChildrenCollapsible(False)
        self.detail_splitter.addWidget(self.detail_browser)
        self.detail_splitter.addWidget(self.preview_widget)
        self.detail_splitter.setSizes([255, 390])
        self.detail_splitter.setStretchFactor(0, 0)
        self.detail_splitter.setStretchFactor(1, 1)

        self.content_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.content_splitter.setChildrenCollapsible(False)
        self.content_splitter.addWidget(self.table)
        self.content_splitter.addWidget(self.detail_splitter)
        self.content_splitter.setSizes([475, 525])
        self.content_splitter.setStretchFactor(0, 1)
        self.content_splitter.setStretchFactor(1, 1)

        layout = QVBoxLayout(self)
        layout.addWidget(self.summary_label)
        layout.addWidget(self.content_splitter)

    def update_results(self, results: Iterable[dict]) -> None:
        selection = self.table.selectionModel().selectedRows()
        selected_id = self.results[selection[0].row()].get("candidate_id") if selection else None
        self.results = list(results)
        with QSignalBlocker(self.table):
            self.table.setRowCount(len(self.results))
            for row, result in enumerate(self.results):
                display_label = result.get("session_candidate_id") or "-"
                status = result.get("status") or "-"
                values = [
                    display_label,
                    result.get("candidate_id", "-"),
                    _fmt(result.get("BLF_global")),
                    _fmt(result.get("weight_kg_per_m2")),
                    result.get("verdict") or FEM_STATUS_LABELS.get(status, status),
                ]
                for col, value in enumerate(values):
                    color = verdict_color(value) if col == 4 else None
                    self.table.setItem(row, col, design_cell(value, color=color))
        succeeded = sum(result.get("status") == "success" for result in self.results)
        passed = sum(result.get("verdict") == "通过" for result in self.results)
        self.summary_label.setText(f"求解完成：{succeeded} / {len(self.results)}  |  达标：{passed}" if self.results else "暂无有限元校核结果")
        if self.results:
            row = next((i for i, r in enumerate(self.results) if r.get("candidate_id") == selected_id), 0)
            with QSignalBlocker(self.table):
                self.table.selectRow(row)
            self._refresh_detail()
        else:
            self.detail_browser.setHtml("<p>暂无 ABAQUS 结果。</p>")
            self.preview_widget.clear_scene("暂无结果预览。")

    def reset_view(self) -> None:
        if hasattr(self, "detail_browser"):
            self.detail_browser.clear()
        self.preview_widget.reset_plotter()

    def append_or_update_result(self, result: dict) -> None:
        updated = False
        for index, current in enumerate(self.results):
            if current.get("session_candidate_id") == result.get("session_candidate_id"):
                self.results[index] = result
                updated = True
                break
        if not updated:
            self.results.append(result)
        self.update_results(self.results)

    def _refresh_detail(self) -> None:
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            self.detail_browser.setHtml("<p>请选择结果查看详情。</p>")
            self.preview_widget.clear_scene("请选择结果查看模态云图。")
            return

        result = self.results[rows[0].row()]
        candidate_id = result.get("session_candidate_id") or "-"
        html = (
            f"<h3>{candidate_id} / {result.get('candidate_id')}</h3>"
            f"<p><b>状态：</b>{FEM_STATUS_LABELS.get(result.get('status'), '-')}<br>"
            f"<b>首个正 BLF：</b>{_fmt(result.get('BLF_global'))}<br>"
            f"<b>面密度：</b>{_fmt(result.get('weight_kg_per_m2'))} kg/m²<br>"
            f"<b>校核结论：</b>{result.get('verdict', '-')}</p>"
            f"<p>"
            f"<b>工况：</b>{result.get('load_summary', '-')}<br>"
            f"<b>边界条件：</b>{result.get('boundary_summary', '-')}<br>"
            f"<b>局部 BLF：</b>{_fmt(result.get('BLF_local'))}<br>"
            f"<b>模态向量最大幅值：</b>{_fmt(result.get('max_displacement_mm'))}（归一化模态，非实际加载位移）<br>"
            f"<b>失效模式：</b>{result.get('failure_mode') or '未识别'}</p>"
            f"<p><b>工程说明：</b>{result.get('diagnosis_summary', '-')}</p>"
            f"<p><b>ODB：</b>{result.get('abaqus_odb', '-')}</p>"
            f"<p><b>INP：</b>{result.get('abaqus_inp', '-')}</p>"
            f"<p><b>模态数据：</b>{result.get('visualization_json', '-')}</p>"
            f"<p><b>工件目录：</b>{result.get('artifact_dir', '-')}</p>"
            f"<p><b>特征值：</b>{'、'.join(_fmt(value) for value in result.get('mode_eigenvalues', []))}</p>"
        )
        if result.get("error_log"):
            html += f"<h4>错误日志</h4><pre>{result.get('error_log')}</pre>"
        self.detail_browser.setHtml(html)
        self.preview_widget.show_mode_shape(result)
