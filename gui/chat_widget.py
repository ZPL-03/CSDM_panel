"""对话消息展示组件。"""

from __future__ import annotations

from html import escape

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QTextBrowser
from gui.theme import ACCENT, TEXT


class ChatWidget(QTextBrowser):
    """展示用户、系统和智能体消息，保留换行与 HTML 转义。"""

    def __init__(self) -> None:
        super().__init__()
        self.setReadOnly(True)
        self.setOpenExternalLinks(False)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.document().setDefaultStyleSheet(
            """
            body { margin: 0; font-size: 13px; }
            .message { margin: 0 0 10px 0; white-space: pre-wrap; }
            .sender { font-weight: 700; }
            .content { line-height: 1.45; }
            """
        )

    def add_message(self, sender: str, message: str) -> None:
        text = escape(str(message)).replace("\n", "<br>")
        sender_text = escape(str(sender))
        self.append(
            f'<div class="message"><span class="sender" style="color:{ACCENT}">[{sender_text}]</span> '
            f'<span class="content" style="color:{TEXT}">{text}</span></div>'
        )
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())
