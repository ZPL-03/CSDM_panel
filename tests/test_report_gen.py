from pathlib import Path

from agents.report_gen import ReportGenAgent


def _report_sample() -> tuple[dict, list[dict], list[dict]]:
    task = {
        "task_id": "TASK_1",
        "task": {
            "application": "机翼下蒙皮壁板",
            "load_conditions": {
                "type": "compression_shear",
                "label": "压剪组合",
                "Nx_kN_per_m": 900.0,
                "Nxy_kN_per_m": 180.0,
            },
            "boundary_conditions": {
                "type": "SSCC",
                "label": "X 向简支 + Y 向固支（SSCC）",
                "description": "X0/X1 简支，Y0/Y1 固支",
                "simply_supported_edges": ["X0", "X1"],
                "clamped_edges": ["Y0", "Y1"],
            },
            "material_system": {"name": "T300/5208"},
            "stiffener_type": "帽型",
            "design_targets": {"BLF_min": 1.35, "primary_objective": "最小重量"},
            "candidate_generation_preferences": {
                "total_candidates": 6,
                "source_allocation_mode": "ratio",
                "source_ratio": {"llm": 2.0, "case_transfer": 1.0, "doe": 1.0},
            },
            "screening_preferences": {"top_k_candidates": 3},
        },
    }
    candidate = {
        "candidate_id": "TMP_1",
        "display_name": "TMP_1",
        "persistent_candidate_id": "C1",
        "source": "DOE",
        "stiffener_type": "帽型",
        "geometry": {
            "panel_length_mm": 700.0,
            "panel_width_mm": 600.0,
            "skin_thickness_mm": 2.5,
            "pitch_mm": 120.0,
            "stiffener_height_mm": 28.0,
            "web_thickness_mm": 2.0,
            "flange_width_mm": 16.0,
            "flange_thickness_mm": 2.0,
        },
        "layup": {
            "skin_layup": "[45/-45/0/90/0/-45/45]s",
            "skin_f0": 0.286,
            "skin_f45": 0.571,
            "skin_f90": 0.143,
        },
        "material_system": {"name": "T300/5208"},
        "rationale": "兼顾屈曲稳定性和制造风险。",
        "screening_summary": "代理模型初筛通过。",
        "selection_reason": "排序靠前。",
        "surrogate_BLF": 1.52,
        "surrogate_weight": 12.5,
    }
    result = {
        "candidate_id": "C1",
        "session_candidate_id": "TMP_1",
        "display_name": "C1",
        "status": "completed",
        "BLF_global": 1.46,
        "BLF_local": 1.62,
        "weight_kg_per_m2": 12.5,
        "failure_mode": "整体屈曲",
        "verdict": "通过",
        "diagnosis_summary": "线性屈曲校核达到目标。",
    }
    return task, [result], [candidate]


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


def test_report_falls_back_to_structured_engineering_explanation() -> None:
    task, results, candidates = _report_sample()
    agent = ReportGenAgent()

    markdown = agent._render_markdown(task, results, candidates)

    assert "## 工程解释与制造建议" in markdown
    assert "### 制造与装配关注点" in markdown
    assert "有限元结论中通过 1 个" in markdown
    assert agent._last_llm_explanation_used is False


def test_report_places_engineering_explanation_after_fem_results() -> None:
    task, results, candidates = _report_sample()
    agent = ReportGenAgent()

    markdown = agent._render_markdown(task, results, candidates)

    assert markdown.index("## 有限元校核结果") < markdown.index("## 工程解释与制造建议")


def test_report_uses_llm_only_for_grounded_engineering_explanation() -> None:
    task, results, candidates = _report_sample()
    agent = ReportGenAgent()

    class ControlledBackend:
        def chat(self, system_prompt, user_prompt, max_tokens_override=None):
            assert "不得新增候选编号、数值" in system_prompt
            assert "制造与装配关注点" in user_prompt
            assert "只做定性解释" in user_prompt
            return (
                "### 制造与装配关注点\n"
                "- 围绕候选筋条成形、胶接界面和边界夹持区域组织工艺评审。\n\n"
                "### 有限元结果解读\n"
                "- 结果解释沿用结构化校核结论。"
            )

    agent.llm_backend = ControlledBackend()

    markdown = agent._render_markdown(task, results, candidates)

    assert "胶接界面和边界夹持区域" in markdown
    assert "代理模型初筛通过" in markdown
    assert agent._last_llm_explanation_used is True


def test_report_cleans_llm_explanation_before_using_it() -> None:
    task, results, candidates = _report_sample()
    agent = ReportGenAgent()

    class ControlledBackend:
        def __init__(self):
            self.calls = 0

        def chat(self, system_prompt, user_prompt, max_tokens_override=None):
            self.calls += 1
            if self.calls == 1:
                return "### 屈曲建议\n- 建议增加 0.3 mm，并可考虑 T800 替代材料。"
            assert "报告解释文本约束清理器" in system_prompt
            return (
                "### 屈曲与重量权衡\n"
                "- 建议围绕当前材料、铺层和筋条几何开展权衡，不引入额外阈值或替代材料。"
            )

    backend = ControlledBackend()
    agent.llm_backend = backend

    markdown = agent._render_markdown(task, results, candidates)

    assert "额外阈值或替代材料" in markdown
    assert "0.3 mm" not in markdown
    assert "T800" not in markdown
    assert backend.calls == 2
    assert agent._last_llm_explanation_used is True
