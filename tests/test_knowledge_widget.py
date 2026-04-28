import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from core.paths import MODELS_DIR
from gui.knowledge_widget import KnowledgeWidget


def _app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_knowledge_widget_renders_literature_status(monkeypatch) -> None:
    app = _app()
    metrics_path = MODELS_DIR / "surrogate_metrics.json"
    original_metrics = metrics_path.read_text(encoding="utf-8") if metrics_path.exists() else None
    try:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(
            json.dumps(
                {
                    "selected_model": "rf",
                    "training_size": 128,
                    "rf": {"mape": 0.12, "rmse": 0.08},
                    "mlp": {"mape": 0.15, "rmse": 0.11},
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(
            "gui.knowledge_widget.LiteratureCorpus",
            lambda: type(
                "FakeCorpus",
                (),
                {
                    "status": staticmethod(
                        lambda: {
                            "record_count": 42,
                            "chunk_count": 105,
                            "pdf_count": 3,
                            "last_ingested_at": "2026-04-27T12:00:00+00:00",
                        }
                    )
                },
            )(),
        )

        widget = KnowledgeWidget()
        try:
            widget.refresh()
            html = widget.toHtml()

            assert "文献知识库状态" in html
            assert "文献记录数" in html
            assert "42" in html
            assert "105" in html
            assert "2026-04-27T12:00:00+00:00" in html
            assert "代理模型指标" in html
            assert "training_size" not in html
            assert "128" in html
        finally:
            widget.close()
            app.processEvents()
    finally:
        if original_metrics is None:
            metrics_path.unlink(missing_ok=True)
        else:
            metrics_path.write_text(original_metrics, encoding="utf-8")
