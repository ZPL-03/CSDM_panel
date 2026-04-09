"""对话消息组件。"""

from __future__ import annotations

from PyQt6.QtWidgets import QListWidget, QListWidgetItem


class ChatWidget(QListWidget):
    """用于展示用户与智能体消息。"""

    def add_message(self, sender: str, message: str) -> None:
        item = QListWidgetItem(f"[{sender}] {message}")
        self.addItem(item)
        self.scrollToBottom()
