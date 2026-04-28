"""报告生成智能体。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List
from xml.sax.saxutils import escape

from agents.base import BaseAgent
from core.io_utils import write_text
from core.llm_backend import LLMBackend
from core.paths import RESULTS_DIR
from core.task_contract import describe_boundary_conditions, describe_load_conditions, task_payload_from_request


class ReportGenAgent(BaseAgent):
    agent_name = "REPORT_GEN"

    def __init__(self, progress_callback=None) -> None:
        super().__init__(progress_callback=progress_callback)
        self.llm_backend: LLMBackend | None = None
        try:
            self.llm_backend = LLMBackend()
        except Exception:
            self.llm_backend = None

    def _build_structured_summary(self, task: Dict, results: List[Dict], candidates: List[Dict]) -> Dict:
        task_payload = task_payload_from_request(task)
        passed = [result for result in results if result.get("verdict") == "通过"]
        best_blf = max(results, key=lambda item: float(item.get("BLF_global") or 0.0), default=None)
        lightest = min(results, key=lambda item: float(item.get("weight_kg_per_m2") or 1e9), default=None)
        return {
            "session_task_id": task.get("task_id"),
            "application": task_payload["application"],
            "load_conditions": describe_load_conditions(task_payload["load_conditions"]),
            "boundary_conditions": describe_boundary_conditions(task_payload["boundary_conditions"]),
            "result_count": len(results),
            "passed_count": len(passed),
            "best_blf_candidate": best_blf.get("candidate_id") if best_blf else None,
            "lightest_candidate": lightest.get("candidate_id") if lightest else None,
            "screened_candidates": [
                {
                    "candidate_id": candidate.get("persistent_candidate_id") or candidate.get("candidate_id"),
                    "display_name": candidate.get("display_name"),
                    "screening_summary": candidate.get("screening_summary"),
                    "selection_reason": candidate.get("selection_reason"),
                }
                for candidate in candidates
            ],
            "results": [
                {
                    "candidate_id": result.get("candidate_id"),
                    "display_name": result.get("display_name"),
                    "BLF_global": result.get("BLF_global"),
                    "BLF_local": result.get("BLF_local"),
                    "weight_kg_per_m2": result.get("weight_kg_per_m2"),
                    "verdict": result.get("verdict"),
                    "failure_mode": result.get("failure_mode"),
                }
                for result in results
            ],
        }

    def _render_narrative(self, summary: Dict) -> str:
        if self.llm_backend is not None:
            system_prompt = (
                "你是复合材料加筋壁板工程报告助手。"
                "请根据结构化摘要生成简洁专业的中文工程说明。"
                "输出 3 个短段落：总体判断、候选对比、建议动作。"
            )
            user_prompt = json.dumps(summary, ensure_ascii=False, indent=2)
            try:
                return self.llm_backend.chat(system_prompt, user_prompt).strip()
            except Exception:
                pass

        if summary["passed_count"] > 0:
            overall = f"本轮共完成 {summary['result_count']} 个样本校核，其中 {summary['passed_count']} 个满足当前 BLF 目标。"
        else:
            overall = f"本轮共完成 {summary['result_count']} 个样本校核，当前没有样本满足目标，需要继续迭代。"
        compare = (
            f"BLF 最优样本为 {summary.get('best_blf_candidate') or '-'}，"
            f"重量最优样本为 {summary.get('lightest_candidate') or '-'}。"
        )
        suggestion = "建议优先围绕已入选样本继续微调筋高、筋距和载荷工况匹配关系，并复核边界条件设定。"
        return "\n\n".join([overall, compare, suggestion])

    def _render_markdown(self, task: Dict, results: List[Dict], candidates: List[Dict]) -> str:
        task_payload = task_payload_from_request(task)
        summary = self._build_structured_summary(task, results, candidates)
        narrative = self._render_narrative(summary)
        lines = [
            "# CSDM 设计报告",
            "",
            f"- 会话任务编号：`{task.get('task_id') or '-'}`",
            f"- 应用场景：{task_payload['application']}",
            f"- 工况：{describe_load_conditions(task_payload['load_conditions'])}",
            f"- 边界条件：{describe_boundary_conditions(task_payload['boundary_conditions'])}",
            "",
            "## 工程摘要",
            "",
            narrative,
            "",
            "## DNN 初筛说明",
        ]
        if candidates:
            for candidate in candidates:
                lines.extend(
                    [
                        "",
                        f"### {candidate.get('display_name', candidate.get('candidate_id'))}",
                        f"- 会话编号：{candidate.get('candidate_id')}",
                        f"- 正式编号：{candidate.get('persistent_candidate_id') or '-'}",
                        f"- 初筛摘要：{candidate.get('screening_summary') or '暂无'}",
                        f"- 入选理由：{candidate.get('selection_reason') or '尚未进入 Top-K'}",
                    ]
                )
        else:
            lines.extend(["", "- 本轮未提供 DNN 初筛上下文。"])

        lines.extend(
            [
                "",
                "## 有限元校核结果",
            ]
        )
        for result in results:
            lines.extend(
                [
                    "",
                    f"### {result.get('display_name', result['candidate_id'])} / {result['candidate_id']}",
                    f"- 状态：{result['status']}",
                    f"- BLF_global：{result.get('BLF_global')}",
                    f"- BLF_local：{result.get('BLF_local')}",
                    f"- 面密度：{result.get('weight_kg_per_m2')}",
                    f"- 失效模式：{result.get('failure_mode')}",
                    f"- 结论：{result.get('verdict')}",
                    f"- 工程说明：{result.get('diagnosis_summary')}",
                ]
            )
        return "\n".join(lines)

    def _reportlab(self):
        from reportlab.lib.enums import TA_LEFT
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.cidfonts import UnicodeCIDFont
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

        return {
            "TA_LEFT": TA_LEFT,
            "A4": A4,
            "ParagraphStyle": ParagraphStyle,
            "getSampleStyleSheet": getSampleStyleSheet,
            "pdfmetrics": pdfmetrics,
            "UnicodeCIDFont": UnicodeCIDFont,
            "TTFont": TTFont,
            "Paragraph": Paragraph,
            "SimpleDocTemplate": SimpleDocTemplate,
            "Spacer": Spacer,
        }

    def _register_font(self) -> str:
        rl = self._reportlab()
        preferred_fonts = [
            ("CSDM_SimSun", Path("C:/Windows/Fonts/simsun.ttc")),
            ("CSDM_MSYH", Path("C:/Windows/Fonts/msyh.ttc")),
            ("CSDM_SimHei", Path("C:/Windows/Fonts/simhei.ttf")),
        ]
        for font_name, font_path in preferred_fonts:
            try:
                if font_path.exists():
                    rl["pdfmetrics"].registerFont(rl["TTFont"](font_name, str(font_path)))
                    return font_name
            except Exception:
                continue

        fallback_font = "STSong-Light"
        rl["pdfmetrics"].registerFont(rl["UnicodeCIDFont"](fallback_font))
        return fallback_font

    def _pdf_styles(self, font_name: str) -> Dict[str, object]:
        rl = self._reportlab()
        stylesheet = rl["getSampleStyleSheet"]()
        body = rl["ParagraphStyle"](
            name="CSDMBody",
            parent=stylesheet["BodyText"],
            fontName=font_name,
            fontSize=10.5,
            leading=15,
            alignment=rl["TA_LEFT"],
            wordWrap="CJK",
            spaceAfter=4,
        )
        return {
            "title": rl["ParagraphStyle"](
                name="CSDMTitle",
                parent=stylesheet["Title"],
                fontName=font_name,
                fontSize=18,
                leading=24,
                alignment=rl["TA_LEFT"],
                wordWrap="CJK",
                spaceAfter=12,
            ),
            "heading2": rl["ParagraphStyle"](
                name="CSDMHeading2",
                parent=stylesheet["Heading2"],
                fontName=font_name,
                fontSize=14,
                leading=20,
                alignment=rl["TA_LEFT"],
                wordWrap="CJK",
                spaceBefore=10,
                spaceAfter=6,
            ),
            "heading3": rl["ParagraphStyle"](
                name="CSDMHeading3",
                parent=stylesheet["Heading3"],
                fontName=font_name,
                fontSize=12,
                leading=18,
                alignment=rl["TA_LEFT"],
                wordWrap="CJK",
                spaceBefore=8,
                spaceAfter=4,
            ),
            "body": body,
            "bullet": rl["ParagraphStyle"](
                name="CSDMBullet",
                parent=body,
                leftIndent=14,
                firstLineIndent=0,
                bulletIndent=0,
                spaceAfter=3,
            ),
        }

    def _paragraph_text_for_pdf(self, text: str) -> str:
        normalized = text.replace("\t", "    ").strip()
        return escape(normalized)

    def _build_pdf_story(self, markdown_text: str, font_name: str) -> List[object]:
        rl = self._reportlab()
        styles = self._pdf_styles(font_name)
        story: List[object] = []

        for raw_line in markdown_text.splitlines():
            stripped = raw_line.strip()
            if not stripped:
                if story:
                    story.append(rl["Spacer"](1, 6))
                continue

            if stripped.startswith("# "):
                story.append(rl["Paragraph"](self._paragraph_text_for_pdf(stripped[2:]), styles["title"]))
                continue
            if stripped.startswith("## "):
                story.append(rl["Paragraph"](self._paragraph_text_for_pdf(stripped[3:]), styles["heading2"]))
                continue
            if stripped.startswith("### "):
                story.append(rl["Paragraph"](self._paragraph_text_for_pdf(stripped[4:]), styles["heading3"]))
                continue
            if stripped.startswith("- "):
                story.append(
                    rl["Paragraph"](
                        self._paragraph_text_for_pdf(stripped[2:]),
                        styles["bullet"],
                        bulletText="•",
                    )
                )
                continue

            story.append(rl["Paragraph"](self._paragraph_text_for_pdf(stripped), styles["body"]))

        if not story:
            story.append(rl["Paragraph"]("CSDM 设计报告内容为空。", styles["body"]))
        return story

    def _write_pdf(self, markdown_text: str, pdf_path: Path) -> None:
        rl = self._reportlab()
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        font_name = self._register_font()
        document = rl["SimpleDocTemplate"](
            str(pdf_path),
            pagesize=rl["A4"],
            leftMargin=40,
            rightMargin=40,
            topMargin=48,
            bottomMargin=48,
            title="CSDM 设计报告",
            author="CSDM",
        )
        document.build(self._build_pdf_story(markdown_text, font_name))

    def run(self, input_data: Dict) -> Dict:
        markdown_text = self._render_markdown(
            input_data["task"],
            input_data["results"],
            input_data.get("candidates", []),
        )
        markdown_path = RESULTS_DIR / "latest_report.md"
        pdf_path = RESULTS_DIR / "latest_report.pdf"
        write_text(markdown_path, markdown_text)
        pdf_generated = False
        try:
            self._write_pdf(markdown_text, pdf_path)
            pdf_generated = True
        except ModuleNotFoundError as exc:
            self.emit(f"PDF 依赖缺失，已跳过 PDF 导出：{exc}")
        self.emit("Markdown/PDF 报告已生成" if pdf_generated else "Markdown 报告已生成")
        return {
            "markdown_path": str(markdown_path),
            "pdf_path": str(pdf_path) if pdf_generated else None,
            "content": markdown_text,
        }
