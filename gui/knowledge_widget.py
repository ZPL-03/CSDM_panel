"""知识库展示组件。"""

from __future__ import annotations

import json

from PyQt6.QtWidgets import QTextBrowser

from core.case_memory import CaseMemoryIndex
from core.literature_corpus import LiteratureCorpus
from core.paths import ABAQUS_RUNS_DIR, CASES_DIR, CASE_LIBRARY_DIR, MODELS_DIR


class KnowledgeWidget(QTextBrowser):
    """展示案例知识资产、文献知识资产和代理模型指标。"""

    def refresh(self) -> None:
        archive_cases = sorted(CASES_DIR.glob("CASE_*.json"))
        formal_cases = sorted(CASE_LIBRARY_DIR.glob("CASE_*.json"))
        metrics_path = MODELS_DIR / "surrogate_metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
        literature_status = LiteratureCorpus().status()
        try:
            case_memory_count = CaseMemoryIndex().engine.count()
        except Exception:
            case_memory_count = 0

        odb_count = 0
        vis_count = 0
        for run_dir in ABAQUS_RUNS_DIR.glob("C*"):
            if (run_dir / f"{run_dir.name}.odb").exists():
                odb_count += 1
            if (run_dir / f"{run_dir.name}_mode1.json").exists():
                vis_count += 1

        lines = [
            "<h2>知识库状态</h2>",
            "<p><b>评估档案数：</b>"
            f"{len(archive_cases)}<br>"
            f"<b>正式知识库数：</b>{len(formal_cases)}<br>"
            f"<b>案例记忆向量块数：</b>{case_memory_count}<br>"
            f"<b>已归档 ODB 数：</b>{odb_count}<br>"
            f"<b>模态可视化数据数：</b>{vis_count}</p>",
            "<p>说明：候选方案只保存在当前会话中；完成 Abaqus 校核后会进入评估档案。"
            "只有校核结论为“通过”的样本才会进入正式知识库；案例记忆向量库用于相似案例召回和排序，"
            "不替代结构化迁移约束。</p>",
        ]

        if metrics:
            lines.extend(
                [
                    "<h3>代理模型指标</h3>",
                    "<p>",
                    f"<b>当前模型：</b>{metrics.get('selected_model', '-')}<br>",
                    f"<b>训练样本数：</b>{metrics.get('training_size', '-')}<br>",
                    f"<b>RF MAPE：</b>{metrics.get('rf', {}).get('mape', '-')}<br>",
                    f"<b>RF RMSE：</b>{metrics.get('rf', {}).get('rmse', '-')}<br>",
                    f"<b>MLP MAPE：</b>{metrics.get('mlp', {}).get('mape', '-')}<br>",
                    f"<b>MLP RMSE：</b>{metrics.get('mlp', {}).get('rmse', '-')}</p>",
                ]
            )

        recent_queries = literature_status.get("queries", [])
        if not isinstance(recent_queries, list):
            recent_queries = []
        recent_queries_text = "；".join(str(item) for item in recent_queries[:5]) or "-"
        lines.extend(
            [
                "<h3>文献知识库状态</h3>",
                "<p>",
                f"<b>文献记录数：</b>{literature_status.get('record_count', 0)}<br>",
                f"<b>文献向量块数：</b>{literature_status.get('chunk_count', 0)}<br>",
                f"<b>开放获取 PDF 数：</b>{literature_status.get('pdf_count', 0)}<br>",
                f"<b>最近同步时间：</b>{literature_status.get('last_ingested_at') or '-'}<br>",
                f"<b>最近来源：</b>{literature_status.get('source') or '-'}<br>",
                f"<b>最近查询：</b>{recent_queries_text}",
                "</p>",
            ]
        )

        lines.append("<p>说明：LLM 候选生成会优先引用文献片段；历史案例迁移使用案例记忆向量库辅助排序，但不依赖文献库。</p>")

        lines.append("<h3>最新正式案例</h3>")
        if formal_cases:
            for path in formal_cases[-10:]:
                lines.append(f"<p>{path.stem}</p>")
        else:
            lines.append("<p>当前还没有“通过”并进入正式知识库的样本。</p>")

        self.setHtml("".join(lines))
