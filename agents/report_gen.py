"""报告生成智能体。"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List
from xml.sax.saxutils import escape

from agents.base import BaseAgent
from core.io_utils import write_text
from core.llm_backend import LLMBackend, auto_llm_enabled
from core.paths import RESULTS_DIR
from core.stiffener_profile import TYPE_DISPLAY_NAMES
from core.task_contract import describe_boundary_conditions, describe_load_conditions, task_payload_from_request


class ReportGenAgent(BaseAgent):
    agent_name = "REPORT_GEN"

    def __init__(self, progress_callback=None) -> None:
        super().__init__(progress_callback=progress_callback)
        self.llm_backend: LLMBackend | None = None
        self._last_llm_explanation_used = False
        if auto_llm_enabled():
            try:
                self.llm_backend = LLMBackend()
            except Exception as exc:
                self.emit(f"报告解释 LLM 后端初始化失败，将使用确定性工程解释：{exc}")

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _compact_candidate_record(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        geometry = dict(candidate.get("geometry") or {})
        layup = dict(candidate.get("layup") or {})
        material = dict(candidate.get("material_system") or {})
        return {
            "candidate_id": candidate.get("persistent_candidate_id") or candidate.get("candidate_id"),
            "session_candidate_id": candidate.get("candidate_id"),
            "display_name": candidate.get("display_name") or candidate.get("candidate_id"),
            "source": candidate.get("source"),
            "stiffener_type": candidate.get("stiffener_type"),
            "material": material.get("name") or material.get("display_name") or material.get("material_key"),
            "geometry": geometry,
            "skin_layup": layup.get("skin_layup"),
            "skin_f0": layup.get("skin_f0"),
            "skin_f45": layup.get("skin_f45"),
            "skin_f90": layup.get("skin_f90"),
            "surrogate_BLF": candidate.get("surrogate_BLF"),
            "surrogate_weight": candidate.get("surrogate_weight") or candidate.get("weight_kg_per_m2"),
            "rank_score": candidate.get("rank_score"),
            "screening_summary": candidate.get("screening_summary"),
            "selection_reason": candidate.get("selection_reason"),
            "rationale": candidate.get("rationale"),
        }

    def _source_counts(self, candidates: List[Dict[str, Any]]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for candidate in candidates:
            source = str(candidate.get("source") or "UNKNOWN")
            counts[source] = counts.get(source, 0) + 1
        return counts

    def _stiffener_type_label(self, task_payload: Dict[str, Any]) -> str:
        preferences = dict(task_payload.get("candidate_generation_preferences") or {})
        raw_types = preferences.get("stiffener_types")
        if isinstance(raw_types, list) and raw_types:
            labels = [TYPE_DISPLAY_NAMES.get(str(stype), str(stype)) for stype in raw_types]
            return " / ".join(labels)
        stype = str(task_payload.get("stiffener_type") or "T")
        return TYPE_DISPLAY_NAMES.get(stype, stype)

    def _build_structured_summary(self, task: Dict, results: List[Dict], candidates: List[Dict]) -> Dict:
        task_payload = task_payload_from_request(task)
        design_targets = dict(task_payload.get("design_targets", {}))
        target_blf = float(design_targets.get("BLF_min") or 0.0)
        passed = [result for result in results if result.get("verdict") == "通过"]
        best_blf = max(results, key=lambda item: float(item.get("BLF_global") or 0.0), default=None)
        lightest = min(results, key=lambda item: float(item.get("weight_kg_per_m2") or 1e9), default=None)
        return {
            "session_task_id": task.get("task_id"),
            "application": task_payload["application"],
            "load_conditions": describe_load_conditions(task_payload["load_conditions"]),
            "boundary_conditions": describe_boundary_conditions(task_payload["boundary_conditions"]),
            "stiffener_type": self._stiffener_type_label(task_payload),
            "target_BLF": target_blf,
            "primary_objective": design_targets.get("primary_objective"),
            "candidate_source_ratio": task_payload.get("candidate_generation_preferences", {}).get("source_ratio", {}),
            "result_count": len(results),
            "passed_count": len(passed),
            "best_blf_candidate": best_blf.get("candidate_id") if best_blf else None,
            "best_blf_value": best_blf.get("BLF_global") if best_blf else None,
            "lightest_candidate": lightest.get("candidate_id") if lightest else None,
            "lightest_weight_kg_per_m2": lightest.get("weight_kg_per_m2") if lightest else None,
            "source_counts": self._source_counts(candidates),
            "screened_candidates": [
                self._compact_candidate_record(candidate)
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
                    "status": result.get("status"),
                    "diagnosis_summary": result.get("diagnosis_summary"),
                    "analysis_flags": result.get("analysis_flags"),
                }
                for result in results
            ],
        }

    def _render_narrative(self, summary: Dict) -> str:
        if summary["passed_count"] > 0:
            overall = (
                f"本轮共完成 {summary['result_count']} 个样本校核，其中 {summary['passed_count']} 个满足 "
                f"BLF 不低于 {summary.get('target_BLF')} 的目标。"
            )
        else:
            overall = (
                f"本轮共完成 {summary['result_count']} 个样本校核，当前没有样本满足 "
                f"BLF 不低于 {summary.get('target_BLF')} 的目标。"
            )
        compare = (
            f"BLF 最优样本为 {summary.get('best_blf_candidate') or '-'}，"
            f"BLF_global={summary.get('best_blf_value') or '-'}；"
            f"面密度最低样本为 {summary.get('lightest_candidate') or '-'}，"
            f"面密度={summary.get('lightest_weight_kg_per_m2') or '-'} kg/m^2。"
        )
        suggestion = "后续应优先复核通过样本的屈曲模态、边界约束和制造可达性；未通过样本先定位失效模式，再决定是否调整筋距、筋高、蒙皮厚度或铺层比例。"
        scope = "有限元使用均质正交各向异性壳截面，不逐层离散铺层，不能据此验证铺层顺序或比例对屈曲的影响。线性特征值分析只评价理想结构的屈曲载荷系数，不证明损伤、初始缺陷或非线性极限承载能力。"
        if any(result.get("BLF_global") is not None and float(result["BLF_global"]) < 1.0
               for result in summary.get("results", [])):
            scope += " 存在 BLF 小于 1 的样本，其线性临界载荷低于完整参考载荷；即使达到用户设定阈值，也不能据此认定可承受完整参考载荷。"
        return "\n\n".join([overall, compare, scope, suggestion])

    def _format_source_counts(self, counts: Dict[str, int]) -> str:
        ordered = ["LLM", "CASE_TRANSFER", "DOE"]
        parts = [f"{source}={counts.get(source, 0)}" for source in ordered if counts.get(source, 0)]
        extras = [f"{source}={count}" for source, count in sorted(counts.items()) if source not in ordered]
        return "，".join(parts + extras) if parts or extras else "无"

    def _geometry_summary(self, geometry: Dict[str, Any]) -> str:
        keys = [
            "panel_length_mm",
            "panel_width_mm",
            "skin_thickness_mm",
            "pitch_mm",
            "stiffener_height_mm",
            "web_thickness_mm",
            "flange_width_mm",
            "flange_thickness_mm",
            "cap_width_mm",
            "cap_thickness_mm",
        ]
        parts = [f"{key}={geometry.get(key)}" for key in keys if geometry.get(key) is not None]
        return "，".join(parts) if parts else "暂无几何字段"

    def _render_deterministic_engineering_explanation(self, summary: Dict[str, Any]) -> str:
        candidates = summary.get("screened_candidates") or []
        results = summary.get("results") or []
        materials = sorted({str(item.get("material")) for item in candidates if item.get("material")})
        layups = sorted({str(item.get("skin_layup")) for item in candidates if item.get("skin_layup")})
        failure_modes = sorted({str(item.get("failure_mode")) for item in results if item.get("failure_mode")})
        diagnosis = [str(item.get("diagnosis_summary")) for item in results if item.get("diagnosis_summary")]
        passed = [item for item in results if item.get("verdict") == "通过"]
        failed = [item for item in results if item.get("verdict") and item.get("verdict") != "通过"]

        material_text = "、".join(materials) if materials else "结构化候选未提供材料名称"
        layup_text = "；".join(layups[:4]) if layups else "结构化候选未提供铺层表达式"
        failure_text = "、".join(failure_modes) if failure_modes else "当前结果未给出明确失效模式"
        diagnosis_text = "；".join(diagnosis[:3]) if diagnosis else "当前结果未给出详细诊断文本"

        return "\n\n".join(
            [
                "### 制造与装配关注点\n"
                f"- 入选候选涉及材料体系：{material_text}。制造评审应围绕蒙皮厚度、筋高、筋距、腹板厚度和翼缘/帽顶尺寸的可达性展开。\n"
                "- 对帽型、T 型、L 型和板式筋，应分别核查筋条成形、胶接或共固化界面、边界夹持区域以及筋条端部过渡区，避免报告之外新增结构形式。",
                "### 铺层与刚度分配\n"
                f"- 入选候选的铺层表达式包括：{layup_text}。解释刚度贡献时应以 0/±45/90 比例和对称均衡约束为基础。\n"
                "- 有限元壳截面不逐层使用铺层序列；铺层对刚度的贡献属于待验证机理，不能解释为当前求解已证实的效果。",
                "### 屈曲与重量权衡\n"
                f"- 当前 BLF 最优样本为 {summary.get('best_blf_candidate') or '-'}，面密度最低样本为 {summary.get('lightest_candidate') or '-'}。两者不是同一样本时，应按任务目标决定优先级。\n"
                f"- 候选来源统计为：{self._format_source_counts(summary.get('source_counts', {}))}。LLM、案例迁移和 DOE 只说明来源，不代表工程可信度排序；最终仍以规则检查、代理模型初筛和 ABAQUS 结果为准。",
                "### 有限元结果解读\n"
                f"- 有限元结论中通过 {len(passed)} 个，未通过 {len(failed)} 个。失效模式概览：{failure_text}。\n"
                f"- 诊断摘要：{diagnosis_text}",
                "### 后续验证建议\n"
                "- 通过样本应继续核查首个正屈曲模态、边界约束实现、网格敏感性和铺层制造偏差。\n"
                "- 未通过样本应先区分全局屈曲、局部屈曲、求解失败或几何装配问题，再决定是否进入下一轮候选生成。",
            ]
        )

    def _material_codes(self, text: str) -> set[str]:
        codes = re.findall(r"\b(?:T|M|IM)[0-9]+[A-Z0-9/-]*\b", text, flags=re.IGNORECASE)
        return {code.upper() for code in codes}

    def _build_llm_explanation_payload(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "task": {
                "application": summary.get("application"),
                "load_case": summary.get("load_conditions"),
                "boundary_conditions": summary.get("boundary_conditions"),
                "stiffener_type": summary.get("stiffener_type"),
                "target_BLF": summary.get("target_BLF"),
                "primary_objective": summary.get("primary_objective"),
            },
            "screened_candidates": [
                {key: value for key, value in candidate.items()
                 if key not in {"rationale", "screening_summary", "selection_reason"}}
                for candidate in summary.get("screened_candidates", [])
            ],
            "results": summary.get("results", []),
            "analysis_scope": "仅线性屈曲。蒙皮、腹板、翼缘和帽顶使用均质正交各向异性壳截面，不逐层离散铺层；铺层序列与比例是设计输入，不进入当前有限元截面计算，不能据此验证铺层变化效果。代理预测不等于有限元结果；BLF 小于一时不能证明可承受完整参考载荷。工艺、缺陷、非线性和网格收敛均未单独验证。",
        }

    def _llm_text_uses_only_known_numbers(self, text: str, payload: Dict[str, Any]) -> bool:
        for code in self._material_codes(json.dumps(payload, ensure_ascii=False, default=str)):
            text = re.sub(re.escape(code), "", text, flags=re.IGNORECASE)
        return re.search(r"[0-9]", text) is None

    def _llm_text_uses_only_known_material_codes(self, text: str, payload: Dict[str, Any]) -> bool:
        known_codes = self._material_codes(json.dumps(payload, ensure_ascii=False, default=str))
        return self._material_codes(text).issubset(known_codes)

    def _validate_llm_engineering_text(self, text: str, payload: Dict[str, Any]) -> None:
        if not str(text or "").strip():
            raise ValueError("LLM 报告解释为空")
        if not self._llm_text_uses_only_known_numbers(text, payload):
            raise ValueError("LLM 报告解释包含结构化数据之外的数值")
        if not self._llm_text_uses_only_known_material_codes(text, payload):
            raise ValueError("LLM 报告解释包含结构化数据之外的材料牌号")
        forbidden_structure_terms = ["夹芯", "蜂窝", "金属衬套", "圆柱壳", "耐压壳", "缠绕成型"]
        if any(term in text for term in forbidden_structure_terms):
            raise ValueError("LLM 报告解释包含当前设计变量域之外的结构或工艺")

    def _render_llm_engineering_explanation(self, summary: Dict[str, Any]) -> str:
        if self.llm_backend is None:
            return ""
        payload = self._build_llm_explanation_payload(summary)
        system_prompt = (
            "你是复合材料加筋壁板设计报告解释助手。"
            "只能基于用户提供的完整结构化数据撰写中文工程解释。"
            "不得新增候选编号、数值、材料名、工况或有限元结论；不得改写 verdict、BLF、面密度和排序。"
            "可以从复合材料加筋壁板常用制造评审角度讨论铺放、共固化、胶接、筋条端部过渡、边界夹持和无损检测，"
            "但不能把 JSON 中没有的设备、厂家、具体固化温度、检验阈值或替代材料牌号写成事实。"
            "优化建议限定在当前候选变量域内，只讨论材料、铺层、厚度、筋条几何、边界、有限元和试验复核，"
            "不要引入夹芯、蜂窝、金属衬套、圆柱壳或其他未在输入中出现的结构型式。"
            "数值事实已经由报告模板输出，解释段禁止出现数字、角度、单位、阈值和候选编号，只做定性解释。"
            "每类内容最多两条，区分已计算事实、机理推断和待验证建议。不得把输入已有的铺层、比例和特征值写成未提供。"
            "没有对比试验或敏感性分析，不得声称某尺寸偏大偏小或某参数已经提高了承载能力。"
            "有限元采用均质正交各向异性壳截面，未逐层使用设计铺层；必须说明铺层作用尚未由该有限元模型验证，不得用铺层比例解释当前特征值。"
            "整体屈曲标签不能证明局部损伤未发生，也不能证明界面胶接可靠。"
            "BLF 小于一时必须说明不能证明可承受完整参考载荷，代理预测与有限元差距较大时应提示代理适用性需复核。"
            "输出 Markdown，直接从“### 制造与装配关注点”开始，五类内容各用一个三级标题和短横线项目；"
            "报告模板已有“工程解释与制造建议”总标题，不要重复总标题，不要输出表格、JSON、编号列表或代码块。"
        )
        user_prompt = (
            "请生成报告中的“工程解释与制造建议”段落，必须覆盖：制造与装配关注点、铺层与刚度分配、"
            "屈曲与重量权衡、有限元结果解读、后续验证建议。只做定性解释，不要复述 JSON 中的数字，"
            "不要给出新的数字、阈值、候选编号或替代材料牌号；如果结构化数据缺少某项，请明确说明当前结构化数据未提供。\n\n"
            f"结构化校核数据 JSON：\n{json.dumps(payload, ensure_ascii=False, indent=2, default=str)}"
        )
        answer = self.llm_backend.chat(
            system_prompt,
            user_prompt,
            max_tokens_override=2600,
        ).strip()
        if not answer:
            return ""
        self._validate_llm_engineering_text(answer, payload)
        return answer

    def _render_engineering_explanation(self, summary: Dict[str, Any]) -> str:
        self._last_llm_explanation_used = False
        if self.llm_backend is not None:
            try:
                llm_text = self._render_llm_engineering_explanation(summary)
                if llm_text:
                    self._last_llm_explanation_used = True
                    return llm_text
            except Exception as exc:
                self.emit(f"报告 LLM 工程解释生成失败，已使用确定性解释：{exc}")
        return self._render_deterministic_engineering_explanation(summary)

    def _render_markdown(self, task: Dict, results: List[Dict], candidates: List[Dict]) -> str:
        task_payload = task_payload_from_request(task)
        summary = self._build_structured_summary(task, results, candidates)
        narrative = self._render_narrative(summary)
        engineering_explanation = self._render_engineering_explanation(summary)
        lines = [
            "# CSDM_panel 设计报告",
            "",
            f"- 会话任务编号：`{task.get('task_id') or '-'}`",
            f"- 应用场景：{task_payload['application']}",
            f"- 筋条类型：{self._stiffener_type_label(task_payload)}",
            f"- 工况：{describe_load_conditions(task_payload['load_conditions'])}",
            f"- 边界条件：{describe_boundary_conditions(task_payload['boundary_conditions'])}",
            f"- BLF 目标：不低于 {task_payload['design_targets']['BLF_min']}",
            f"- 优化目标：{task_payload['design_targets']['primary_objective']}",
            f"- 候选来源比例：LLM:CASE_TRANSFER:DOE = "
            f"{task_payload['candidate_generation_preferences']['source_ratio']['llm']:g}:"
            f"{task_payload['candidate_generation_preferences']['source_ratio']['case_transfer']:g}:"
            f"{task_payload['candidate_generation_preferences']['source_ratio']['doe']:g}",
            "",
            "## 工程摘要",
            "",
            narrative,
            "",
            "## 候选来源与初筛说明",
            "",
            f"- 入选候选来源统计：{self._format_source_counts(summary['source_counts'])}",
        ]
        if candidates:
            for candidate in candidates:
                geometry = dict(candidate.get("geometry") or {})
                layup = dict(candidate.get("layup") or {})
                material = dict(candidate.get("material_system") or {})
                display_label = candidate.get("display_name") or candidate.get("candidate_id") or "-"
                lines.extend(
                    [
                        "",
                        f"### {display_label}",
                        f"- 会话编号：{candidate.get('candidate_id')}",
                        f"- 正式编号：{candidate.get('persistent_candidate_id') or '-'}",
                        f"- 来源：{candidate.get('source') or '-'}",
                        f"- 材料：{material.get('name') or '-'}",
                        f"- 几何摘要：{self._geometry_summary(geometry)}",
                        f"- 铺层设计输入：{layup.get('skin_layup') or '-'}；比例 f0={layup.get('skin_f0')}，f45={layup.get('skin_f45')}，f90={layup.get('skin_f90')}",
                        f"- 初筛摘要：{candidate.get('screening_summary') or '暂无'}",
                        f"- 入选理由：{candidate.get('selection_reason') or '尚未进入 Top-K'}",
                    ]
                )
        else:
            lines.extend(["", "- 本轮未提供代理模型初筛上下文。"])

        lines.extend(
            [
                "",
                "## 有限元校核结果",
            ]
        )
        for result in results:
            result_label = result.get("display_name") or result.get("session_candidate_id") or result.get("candidate_id")
            lines.extend(
                [
                    "",
                    f"### {result_label} / {result['candidate_id']}",
                    f"- 状态：{result['status']}",
                    f"- BLF_global：{result.get('BLF_global')}",
                    f"- BLF_local：{result.get('BLF_local')}",
                    f"- 面密度：{result.get('weight_kg_per_m2')}",
                    f"- 失效模式：{result.get('failure_mode')}",
                    f"- 结论：{result.get('verdict')}",
                    f"- 工程说明：{result.get('diagnosis_summary')}",
                ]
            )
        lines.extend(
            [
                "",
                "## 工程解释与制造建议",
                "",
                engineering_explanation,
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
        from reportlab.platypus import Paragraph, SimpleDocTemplate

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
        }

    def _register_font(self) -> str:
        rl = self._reportlab()
        preferred_fonts = [
            ("CSDM_panel_SimSun", Path("C:/Windows/Fonts/simsun.ttc")),
            ("CSDM_panel_MSYH", Path("C:/Windows/Fonts/msyh.ttc")),
            ("CSDM_panel_SimHei", Path("C:/Windows/Fonts/simhei.ttf")),
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
            name="CSDM_panelBody",
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
                name="CSDM_panelTitle",
                parent=stylesheet["Title"],
                fontName=font_name,
                fontSize=18,
                leading=24,
                alignment=rl["TA_LEFT"],
                wordWrap="CJK",
                spaceAfter=12,
                keepWithNext=True,
            ),
            "heading2": rl["ParagraphStyle"](
                name="CSDM_panelHeading2",
                parent=stylesheet["Heading2"],
                fontName=font_name,
                fontSize=14,
                leading=20,
                alignment=rl["TA_LEFT"],
                wordWrap="CJK",
                spaceBefore=10,
                spaceAfter=6,
                keepWithNext=True,
            ),
            "heading3": rl["ParagraphStyle"](
                name="CSDM_panelHeading3",
                parent=stylesheet["Heading3"],
                fontName=font_name,
                fontSize=12,
                leading=18,
                alignment=rl["TA_LEFT"],
                wordWrap="CJK",
                spaceBefore=8,
                spaceAfter=4,
                keepWithNext=True,
            ),
            "body": body,
            "bullet": rl["ParagraphStyle"](
                name="CSDM_panelBullet",
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
                continue

            heading = re.fullmatch(r"(#{1,6})[ \t]+(.*?)(?:[ \t]+#+)?", stripped)
            if heading:
                level = len(heading.group(1))
                style = styles["title" if level == 1 else "heading2" if level == 2 else "heading3"]
                story.append(rl["Paragraph"](self._paragraph_text_for_pdf(heading.group(2)), style))
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
            story.append(rl["Paragraph"]("CSDM_panel 设计报告内容为空。", styles["body"]))
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
            title="CSDM_panel 设计报告",
            author="CSDM_panel",
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
            "llm_explanation_used": self._last_llm_explanation_used,
        }
