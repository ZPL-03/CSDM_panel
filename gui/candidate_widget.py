"""候选方案展示组件。"""

from __future__ import annotations

from html import escape
from typing import Iterable

from PyQt6.QtCore import QItemSelectionModel, QSignalBlocker, Qt
from PyQt6.QtWidgets import (
    QLabel,
    QSplitter,
    QTableWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from gui.interactive_view import InteractivePlotWidget
from gui.theme import ACCENT, FEM_STATUS_LABELS, WARNING, configure_design_table, design_cell, verdict_color
from core.display import format_number as _format_number, format_parameter
from core.stiffener_profile import GEOMETRY_LABELS, TYPE_DISPLAY_NAMES
from core.task_contract import describe_boundary_conditions, describe_load_conditions


SOURCE_LABELS = {
    "LLM": "LLM 方案生成",
    "CASE_TRANSFER": "历史案例迁移",
    "DOE": "DOE 参数采样",
}
SOURCE_SHORT_LABELS = {"LLM": "LLM", "CASE_TRANSFER": "案例迁移", "DOE": "DOE"}
SOURCE_COLORS = {"LLM": ACCENT, "CASE_TRANSFER": "#c6b5eb", "DOE": WARNING}
TYPE_SHORT_LABELS = {"T": "T 型筋", "BLADE": "板式筋", "L": "L 型筋", "HAT": "帽型筋"}
PREDICTION_FIELDS = ("surrogate_BLF", "surrogate_weight", "rank_score")


def _format_generation_label(source: object) -> str:
    key = str(source or "UNKNOWN")
    return SOURCE_LABELS.get(key, key)


def _parse_layup_sequence(layup_text: str) -> list[str]:
    text = (layup_text or "").strip()
    if not text:
        return []
    symmetric = text.endswith("s")
    if symmetric:
        text = text[:-1]
    base = [item.strip() for item in text.strip("[] ").split("/") if item.strip()]
    return base + list(reversed(base)) if symmetric else base


class CandidateWidget(QWidget):
    """展示候选方案表格、设计细节与交互式几何视图。"""

    HEADERS = ["样本", "筋型", "来源", "预测BLF", "面密度", "评分", "校核"]

    def __init__(self) -> None:
        super().__init__()
        self.candidates: list[dict] = []
        self.results_by_session_id: dict[str, dict] = {}

        self.table = QTableWidget(0, len(self.HEADERS))
        self.table.setHorizontalHeaderLabels(self.HEADERS)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self._refresh_detail)
        configure_design_table(self.table)
        self.table.horizontalHeader().setMinimumSectionSize(58)
        self.table.horizontalHeaderItem(3).setToolTip("代理预测屈曲载荷因子")
        self.table.horizontalHeaderItem(4).setToolTip("代理预测面密度（kg/m²）")
        self.table.horizontalHeaderItem(5).setToolTip("代理排序评分")
        self._update_prediction_columns()

        self.summary_label = QLabel("当前还没有候选方案。")
        self.summary_label.setObjectName("tableSummary")
        self.summary_label.setWordWrap(True)
        self.summary_label.setTextFormat(Qt.TextFormat.PlainText)

        self.detail_browser = QTextBrowser()
        self.preview_widget = InteractivePlotWidget("选中候选方案后，这里会显示可旋转的三维几何视图。")

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
        layout.addWidget(self.content_splitter, 1)

    def _result_for_candidate(self, candidate: dict) -> dict | None:
        return self.results_by_session_id.get(candidate.get("candidate_id", ""))

    def _update_prediction_columns(self) -> None:
        # 只有返回预测值时才显示代理列，直接送检不占用比较区。
        for col, field in enumerate(PREDICTION_FIELDS, start=3):
            self.table.setColumnHidden(col, not any(c.get(field) is not None for c in self.candidates))

    def update_candidates(self, candidates: Iterable[dict], results_by_session_id: dict[str, dict] | None = None) -> None:
        selected_ids = {c.get("candidate_id") for c in self.selected_candidates()}
        self.candidates = list(candidates)
        self.results_by_session_id = dict(results_by_session_id or {})

        source_counter: dict[str, int] = {}

        with QSignalBlocker(self.table):
            self.table.setRowCount(len(self.candidates))
            self._update_prediction_columns()
            for row, candidate in enumerate(self.candidates):
                source = str(candidate.get("source", "UNKNOWN"))
                generation_label = _format_generation_label(source)
                source_label = SOURCE_SHORT_LABELS.get(source, source)
                source_counter[source_label] = source_counter.get(source_label, 0) + 1
                result = self._result_for_candidate(candidate) or {}
                archive_id = candidate.get("persistent_candidate_id") or result.get("candidate_id", "-")
                candidate_id = candidate.get("candidate_id") or "-"
                stype = candidate.get("stiffener_type", "-")
                status = result.get("status")
                verdict = result.get("verdict")
                state = verdict or FEM_STATUS_LABELS.get(status, status) or "未校核"
                cells = [
                    design_cell(candidate_id),
                    design_cell(TYPE_SHORT_LABELS.get(stype, stype)),
                    design_cell(source_label, color=SOURCE_COLORS.get(source)),
                    *(design_cell(_format_number(candidate.get(field))) for field in PREDICTION_FIELDS),
                    design_cell(state, color=verdict_color(verdict)),
                ]
                cells[0].setToolTip(f"会话编号：{candidate_id} | 正式编号：{archive_id}")
                cells[1].setToolTip(TYPE_DISPLAY_NAMES.get(stype, stype))
                cells[2].setToolTip(generation_label)
                for col, item in enumerate(cells):
                    self.table.setItem(row, col, item)
            if self.candidates:
                rows = [i for i, c in enumerate(self.candidates) if c.get("candidate_id") in selected_ids] or [0]
                self.table.setCurrentCell(rows[0], 0)
                self.table.clearSelection()
                select_row = QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows
                for row in rows:
                    self.table.selectionModel().select(self.table.model().index(row, 0), select_row)

        summary = "  /  ".join(f"{key} {value}" for key, value in source_counter.items())
        decision = next((c["screening_decision"] for c in self.candidates if c.get("screening_decision")), {})
        pool_size = len(decision.get("input_ids") or [])
        action = decision.get("action")
        if action == "direct_fem":
            text = f"按筋型送检：{len(self.candidates)} / {pool_size}（无代理排名）"
        elif pool_size:
            text = f"代理初筛：{len(self.candidates)} / {pool_size}"
        else:
            text = f"候选方案：{len(self.candidates)}"
        self.summary_label.setText("  |  ".join(part for part in (text, summary) if part))
        self.summary_label.setToolTip(str(decision.get("reason") or ""))

        if self.candidates:
            self._refresh_detail()
        else:
            self.detail_browser.setHtml("<p>暂无候选方案。</p>")
            self.preview_widget.clear_scene("暂无几何预览。")

    def reset_view(self) -> None:
        if hasattr(self, "detail_browser"):
            self.detail_browser.clear()
        self.preview_widget.reset_plotter()

    def selected_candidates(self) -> list[dict]:
        rows = sorted({index.row() for index in self.table.selectionModel().selectedRows()})
        return [self.candidates[row] for row in rows if 0 <= row < len(self.candidates)]

    def _refresh_detail(self) -> None:
        selected = self.selected_candidates()
        if not selected:
            self.detail_browser.setHtml("<p>请选择候选方案查看详细信息。</p>")
            self.preview_widget.clear_scene("请选择候选方案查看三维几何视图。")
            return

        candidate = selected[0]
        result = self._result_for_candidate(candidate)
        geometry = candidate.get("geometry", {})
        layup = candidate.get("layup", {})
        material = candidate.get("material_system", {})
        load_conditions = candidate.get("load_conditions", {})
        design_targets = candidate.get("design_targets", {})
        rule_check = candidate.get("rule_check", {})
        selection_html = ""
        if candidate.get("selection_reason"):
            selection_html = f"<p><b>送检：</b>{escape(str(candidate['selection_reason']))}</p>"
        if candidate.get("screening_summary"):
            selection_html += f"<p><b>代理检查：</b>{escape(str(candidate['screening_summary']))}</p>"
        if candidate.get("surrogate_BLF") is not None:
            selection_html += (
                f"<p><b>代理预测：</b>BLF {_format_number(candidate.get('surrogate_BLF'))}，"
                f"面密度 {_format_number(candidate.get('surrogate_weight'))} kg/m²，"
                f"评分 {_format_number(candidate.get('rank_score'))}</p>"
            )

        ply_items = "".join(
            f"<li>第 {index + 1} 层：{angle}&deg;</li>"
            for index, angle in enumerate(_parse_layup_sequence(layup.get("skin_layup", "")))
        )
        geometry_items = "".join(
            f"<li>{GEOMETRY_LABELS.get(key, key)}: {format_parameter(key, value)}</li>"
            for key, value in geometry.items()
        )
        material_items = "".join(f"<li>{key}: {value}</li>" for key, value in material.items())
        target_items = "".join(f"<li>{key}: {format_parameter(key, value)}</li>" for key, value in design_targets.items())
        rule_errors = rule_check.get("errors") or ["无"]
        rule_suggestions = rule_check.get("suggestions") or ["无"]
        archive_id = candidate.get("persistent_candidate_id") or (result or {}).get("candidate_id", "-")

        result_html = ""
        if result:
            result_html = (
                "<h4>真实 ABAQUS 结果</h4>"
                f"<p>正式样本编号：{archive_id}<br>"
                f"状态：{FEM_STATUS_LABELS.get(result.get('status'), '-')}<br>"
                f"首个正 BLF：{_format_number(result.get('BLF_global'))}<br>"
                f"局部 BLF：{_format_number(result.get('BLF_local'))}<br>"
                f"面密度：{_format_number(result.get('weight_kg_per_m2'))} kg/m²<br>"
                f"失效模式：{result.get('failure_mode') or '未识别'}<br>"
                f"结论：{result.get('verdict', '-')}</p>"
            )

        stype = str(candidate.get("stiffener_type", "T"))
        stype_display = TYPE_DISPLAY_NAMES.get(stype, stype)
        candidate_id = candidate.get("candidate_id") or "-"
        identity = f"{candidate_id} / {archive_id}" if archive_id != "-" else candidate_id

        html = (
            f"<h3>{identity} | {stype_display}</h3>"
            f"<p><b>来源：</b>{_format_generation_label(candidate.get('source'))}<br>"
            f"<b>蒙皮铺层：</b>{layup.get('skin_layup', '-')}</p>"
            f"{result_html}"
            f"{selection_html}"
            f"<p><b>生成说明：</b>{escape(str(candidate.get('rationale') or '-'))}</p>"
            "<h4>几何设计参数</h4><ul>"
            f"{geometry_items}</ul>"
            "<h4>材料系统</h4><ul>"
            f"{material_items}</ul>"
            "<h4>载荷与边界</h4><ul>"
            f"<li>{describe_load_conditions(load_conditions, display=True)}</li>"
            f"<li>{describe_boundary_conditions(candidate.get('boundary_conditions', {}))}</li></ul>"
            "<h4>设计目标</h4><ul>"
            f"{target_items}</ul>"
            "<h4>铺层明细</h4>"
            f"<p>比例：0°={_format_number(layup.get('skin_f0'), None)}，±45°={_format_number(layup.get('skin_f45'), None)}，90°={_format_number(layup.get('skin_f90'), None)}</p>"
            f"<ul>{ply_items or '<li>暂无铺层明细</li>'}</ul>"
            "<h4>规则检查</h4>"
            f"<p>是否通过校验：{rule_check.get('is_valid', False)}</p>"
            f"<p>问题：{'；'.join(rule_errors)}</p>"
            f"<p>建议：{'；'.join(rule_suggestions)}</p>"
        )
        self.detail_browser.setHtml(html)
        self.preview_widget.show_candidate(candidate)
