"""桌面工作台的黑灰主题、文本与三维视口配色。"""

from PyQt6.QtGui import QColor, QFont, QPalette
from PyQt6.QtWidgets import QApplication, QWidget

BACKGROUND = "#0c0c0f"
SURFACE = "#151518"
TEXT = "#ededf0"
MUTED = "#aaaab4"
ACCENT = "#66c8ce"


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
        QLabel#appSubtitle {{ color: {MUTED}; padding-bottom: 6px; }}
        QLabel#sectionTitle {{ color: {MUTED}; font-weight: 600; padding-top: 3px; }}
        QLabel#statusLabel {{ background: #192326; border-left: 3px solid {ACCENT}; padding: 9px; }}
        QLabel[metric="true"] {{ background: {SURFACE}; border-bottom: 1px solid #35353d; padding: 9px; font-weight: 600; }}
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
        QWidget#sidebarControls QLabel[metric="true"] {{ padding: 5px 6px; }}
        QTabWidget::pane {{ border: 1px solid #303037; }}
        QTabBar::tab {{ background: #19191d; color: {MUTED}; padding: 10px 16px; border-bottom: 2px solid transparent; }}
        QTabBar::tab:selected {{ color: {TEXT}; background: #232328; border-bottom-color: {ACCENT}; }}
        QTabBar::tab:hover {{ background: #29292f; }}
        QHeaderView::section, QTableCornerButton::section {{
            background: #232328; color: #cdcdd4; border: 0; border-bottom: 1px solid #3b3b43; padding: 8px;
        }}
        QTableWidget {{ gridline-color: #2b2b32; alternate-background-color: #1b1b20; }}
        QTableWidget::item {{ padding: 4px; }}
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
