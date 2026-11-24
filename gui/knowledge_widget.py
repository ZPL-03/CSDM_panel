"""知识库展示组件。"""

from __future__ import annotations

import json
from html import escape

from PyQt6.QtWidgets import QTextBrowser

from core.case_memory import CaseMemoryIndex
from core.domain_knowledge import DomainKnowledgeBase
from core.display import format_number
from core.paths import ABAQUS_RUNS_DIR, CASES_DIR, CASE_LIBRARY_DIR, MODELS_DIR
from core.stiffener_profile import STIFFENER_TYPES, TYPE_DISPLAY_NAMES


class KnowledgeWidget(QTextBrowser):
    """展示案例库、外部知识库/知识图谱和代理模型指标。"""

    def refresh(self) -> None:
        archive_cases = sorted(CASES_DIR.glob("CASE_*.json"))
        formal_cases = sorted(CASE_LIBRARY_DIR.glob("CASE_*.json"))
        knowledge_status = DomainKnowledgeBase().status()
        try:
            case_memory_count = CaseMemoryIndex().engine.count()
        except Exception as exc:
            case_memory_count = f"不可用：{escape(str(exc))}"

        odb_count = 0
        vis_count = 0
        for run_dir in ABAQUS_RUNS_DIR.glob("C*"):
            if (run_dir / f"{run_dir.name}.odb").exists():
                odb_count += 1
            if (run_dir / f"{run_dir.name}_mode1.json").exists():
                vis_count += 1

        lines = [
            "<h2>知识库状态</h2>",
            "<h3>设计案例与校核工件</h3>",
            f"<p><b>评估档案数：</b>{len(archive_cases)}&nbsp;&nbsp; | &nbsp;&nbsp;"
            f"<b>正式案例库数：</b>{len(formal_cases)}&nbsp;&nbsp; | &nbsp;&nbsp;"
            f"<b>案例记忆向量块数：</b>{case_memory_count}</p>",
            f"<p><b>ODB：</b>{odb_count}&nbsp;&nbsp; | &nbsp;&nbsp;<b>模态数据：</b>{vis_count}</p>",
        ]
        # 正式案例属于评估档案的达标子集，两处编号均可直接核对。
        for label, cases in (("评估档案", archive_cases), ("正式案例", formal_cases)):
            ids = " / ".join(escape(path.stem) for path in cases[-10:]) or "暂无"
            lines.append(f"<p><b>{label}：</b>{ids}</p>")

        knowledge_ready = "可用" if knowledge_status.get("ready") else "未就绪"
        lines.extend(
            [
                "<h3>外部知识库/知识图谱状态</h3>",
                "<p>",
                f"<b>运行状态：</b>{knowledge_ready}<br>",
                f"<b>知识库文本块数：</b>{knowledge_status.get('rag_chunk_count', 0)}<br>",
                f"<b>知识图谱实体数：</b>{knowledge_status.get('kg_entity_count', 0)}<br>",
                f"<b>知识图谱关系数：</b>{knowledge_status.get('kg_relation_count', 0)}<br>",
                f"<b>来源元数据数：</b>{knowledge_status.get('source_metadata_count', 0)}<br>",
                f"<b>检索引擎：</b>{escape(str(knowledge_status.get('retrieval_engine') or '-'))}<br>",
                f"<b>知识清单：</b>{escape(str(knowledge_status.get('manifest_path') or '-'))}<br>",
                f"<b>溯源资料目录：</b>{escape(str(knowledge_status.get('provenance_dir') or '-'))}<br>",
                f"<b>更新时间：</b>{escape(str(knowledge_status.get('updated_at') or '-'))}",
                "</p>",
            ]
        )

        lines.append("<h3>代理模型指标</h3>")
        for kind in STIFFENER_TYPES:
            metrics_path = MODELS_DIR / kind / "surrogate_metrics.json"
            if not metrics_path.is_file():
                lines.append(f"<p><b>{TYPE_DISPLAY_NAMES[kind]}：</b>无代理训练指标</p>")
                continue
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            lines.append(f"<p><b>{TYPE_DISPLAY_NAMES[kind]}：</b>训练样本 {metrics['training_size']}；"
                         f"维度 {len(metrics['feature_order'])}；RF MAPE {format_number(metrics['rf']['mape'])}</p>")

        self.setHtml("".join(lines))
