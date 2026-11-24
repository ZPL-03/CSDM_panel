"""桌面工作台的黑灰主题、文本与三维视口配色。"""

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont, QPalette
from PyQt6.QtWidgets import QApplication, QHeaderView, QTableWidget, QTableWidgetItem, QWidget

BACKGROUND = "#101114"
SURFACE = "#181a1e"
TEXT = "#ededf0"
MUTED = "#a8adb5"
ACCENT = "#83d6d5"
SUCCESS = "#94d7aa"
WARNING = "#f1c783"
FEM_STATUS_LABELS = {"success": "求解完成", "failed": "求解失败"}


def configure_design_table(table: QTableWidget) -> None:
    """使编号、工程量和校核状态可在同一行比较。"""
    header = table.horizontalHeader()
    header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    header.setMinimumSectionSize(52)
    header.setDefaultAlignment(Qt.AlignmentFlag.AlignCenter)
    table.verticalHeader().hide()
    table.verticalHeader().setDefaultSectionSize(28)
    table.setAlternatingRowColors(True)
    table.setShowGrid(False)
    table.setMinimumHeight(114)
    table.setWordWrap(False)


def design_cell(value: object, *, color: str | None = None) -> QTableWidgetItem:
    item = QTableWidgetItem(str(value))
    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
    if color is not None:
        item.setForeground(QColor(color))
    return item


def verdict_color(value: object) -> str | None:
    return {"通过": SUCCESS, "不通过": WARNING}.get(str(value))


def apply_dark_theme(window: QWidget) -> None:
    app = QApplication.instance()
    app.setStyle("Fusion")
    app.setFont(QFont("Microsoft YaHei UI", 9))
    palette = QPalette()
    for role, color in {
        QPalette.ColorRole.Window: BACKGROUND,
        QPalette.ColorRole.WindowText: TEXT,
        QPalette.ColorRole.Base: SURFACE,
        QPalette.ColorRole.AlternateBase: "#1b1b20",
        QPalette.ColorRole.Text: TEXT,
        QPalette.ColorRole.Button: "#232328",
        QPalette.ColorRole.ButtonText: TEXT,
        QPalette.ColorRole.ToolTipBase: "#24242a",
        QPalette.ColorRole.ToolTipText: TEXT,
        QPalette.ColorRole.Highlight: "#25525b",
        QPalette.ColorRole.HighlightedText: "#ffffff",
        QPalette.ColorRole.Link: ACCENT,
        QPalette.ColorRole.PlaceholderText: MUTED,
    }.items():
        palette.setColor(role, QColor(color))
    for role in (QPalette.ColorRole.Text, QPalette.ColorRole.ButtonText, QPalette.ColorRole.WindowText):
        palette.setColor(QPalette.ColorGroup.Disabled, role, QColor("#777782"))
    app.setPalette(palette)
    window.setStyleSheet(f"""
        QWidget {{ background: {BACKGROUND}; color: {TEXT}; font-size: 13px; }}
        QLabel {{ background: transparent; }}
        QLabel#appTitle {{ font-size: 21px; font-weight: 700; padding: 4px 0; }}
        QLabel#appSubtitle {{ color: {MUTED}; }}
        QLabel#sectionTitle {{ color: {MUTED}; font-weight: 600; padding-top: 3px; }}
        QLabel#statusLabel {{ background: #172729; border-left: 2px solid {ACCENT}; padding: 9px; }}
        QLabel[metric="true"] {{ background: {SURFACE}; border: 1px solid #30343b; border-radius: 4px; padding: 7px 10px; font-weight: 600; }}
        QLabel[metric="true"][active="true"] {{ color: {ACCENT}; }}
        QLabel#tableSummary {{ color: {MUTED}; padding: 4px 0; }}
        QWidget#sceneViewport {{ border: 1px solid #30343b; border-radius: 4px; }}
        QLabel#sceneTitle {{ font-weight: 600; color: {TEXT}; }}
        QToolButton {{ background: #24292e; border: 1px solid #3b4349; border-radius: 4px; padding: 4px; }}
        QToolButton:hover, QToolButton:focus {{ border-color: {ACCENT}; }}
        QTextBrowser, QPlainTextEdit, QTextEdit, QTableWidget, QLineEdit {{
            background: {SURFACE}; color: {TEXT}; border: 1px solid #303037;
            border-radius: 4px; selection-background-color: #25525b; selection-color: white;
        }}
        QTextBrowser {{ padding: 7px; }}
        QLineEdit {{ padding: 10px; }}
        QLineEdit:focus {{ border-color: {ACCENT}; }}
        QPushButton {{
            background: #232328; color: {TEXT}; border: 1px solid #3b3b43;
            border-radius: 4px; padding: 8px 7px; min-height: 20px; font-weight: 600;
        }}
        QPushButton:hover {{ background: #303037; border-color: #777782; }}
        QPushButton:pressed {{ background: #3b3b43; }}
        QPushButton:focus {{ border-color: {ACCENT}; }}
        QPushButton[role="primary"] {{ background: #176b75; border-color: #318e97; color: white; }}
        QPushButton[role="primary"]:hover {{ background: #207e88; }}
        QPushButton[role="success"] {{ background: #214335; border-color: #417a5d; color: #d4f4dd; }}
        QPushButton[role="success"]:hover {{ background: #2c5945; }}
        QPushButton[role="warning"] {{ color: #e5c28b; border-color: #766045; }}
        QPushButton[role="danger"] {{ color: #efaab8; border-color: #754652; }}
        QPushButton:disabled {{ background: #19191d; color: #777782; border-color: #303037; }}
        QWidget#sidebarControls QPushButton {{ padding: 5px 7px; }}
        QWidget#sidebarControls QLineEdit {{ padding: 7px; }}
        QWidget#sidebarControls QLabel#statusLabel {{ padding: 6px; }}
        QTabWidget::pane {{ border: 1px solid #303037; }}
        QTabBar::tab {{ background: #19191d; color: {MUTED}; padding: 10px 16px; border-bottom: 2px solid transparent; }}
        QTabBar::tab:selected {{ color: {TEXT}; background: #232328; border-bottom-color: {ACCENT}; }}
        QTabBar::tab:hover {{ background: #29292f; }}
        QHeaderView::section, QTableCornerButton::section {{
            background: #232328; color: #cdcdd4; border: 0; border-bottom: 1px solid #3b3b43; padding: 7px 3px;
        }}
        QTableWidget {{ alternate-background-color: #1e2126; }}
        QTableWidget::item {{ padding: 2px 3px; }}
        QTableWidget::item:selected {{ background: #25525b; color: white; }}
        QScrollBar:vertical {{ background: #151518; width: 10px; margin: 0; }}
        QScrollBar::handle:vertical {{ background: #484850; min-height: 24px; border-radius: 4px; }}
        QScrollBar:horizontal {{ background: #151518; height: 10px; margin: 0; }}
        QScrollBar::handle:horizontal {{ background: #484850; min-width: 24px; border-radius: 4px; }}
        QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
        QScrollBar::add-page, QScrollBar::sub-page {{ background: none; }}
        QSplitter::handle {{ background: #303037; width: 3px; height: 3px; }}
        QToolTip {{ background: #24242a; color: {TEXT}; border: 1px solid #55555f; padding: 5px; }}
    """)
