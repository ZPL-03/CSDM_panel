from pathlib import Path

from agents.report_gen import ReportGenAgent


def test_report_pdf_generation_handles_long_lines(tmp_path: Path) -> None:
    agent = ReportGenAgent()
    markdown_text = "\n".join(
        [
            "# CSDM_panel 设计报告",
            "",
            "## 工程摘要",
            "",
            "这是一段用于验证 PDF 自动换行的超长说明文本，" * 8,
            "",
            "- 入选理由：" + "代理预测 BLF=0.238，面密度=5.298 kg/m^2，按 score = 1.00 × BLF - 0.08 × 面密度 得分 -0.1858。 " * 6,
            "- 工程说明：" + "线性屈曲校核已完成，前 13 个负特征值模态已跳过，按第 10 阶正特征值判定，当前结论为“不通过”。" * 5,
        ]
    )
    pdf_path = tmp_path / "long_report.pdf"

    agent._write_pdf(markdown_text, pdf_path)

    assert pdf_path.exists()
    assert pdf_path.stat().st_size > 0
