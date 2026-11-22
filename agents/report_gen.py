"""总体设计方案报告：绑定任务、候选、求解结果与文献证据。"""

from __future__ import annotations

import json
import math
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Dict, List
from xml.sax.saxutils import escape

from agents.base import BaseAgent
from core.config_loader import load_llm_config
from core.display import format_number
from core.domain_knowledge import DomainKnowledgeBase
from core.io_utils import write_json, write_text
from core.laminate import ANALYSIS_VERSION, analysis_contract, section_layers
from core.llm_backend import LLMBackend, auto_llm_enabled
from core.paths import RESULTS_DIR
from core.report_facts import (
    METRIC_LABELS, build_comparison_facts, build_facts, format_fact_value, normalize_results,
)
from core.report_review import (
    parse_engineering_review, prepare_review_payload, render_engineering_review, request_engineering_review,
    review_engineering_content,
)
from core.stiffener_profile import TYPE_DISPLAY_NAMES
from core.task_contract import describe_boundary_conditions, describe_load_conditions, task_payload_from_request


class ReportGenAgent(BaseAgent):
    agent_name = "REPORT_GEN"
    REPORT_ARTIFACTS = {"overall": ("overall_design_report", "CSDM_panel 总体设计方案报告")}
    OBJECTIVE_LABELS = {"最小重量": "模型质量（面密度乘投影面积）", "最小面密度": "面密度"}
    REVIEW_RESULT_METRICS = frozenset({
        "BLF_global", "BLF_local", "BLF_margin_percent", "objective_value", "weight_kg_per_m2",
        "critical_Nx", "critical_Nxy", "stiffener_type", "material",
    })

    def __init__(self, progress_callback=None) -> None:
        super().__init__(progress_callback=progress_callback)
        llm_config = load_llm_config()
        settings = llm_config.get("report_explanation", {})
        budget = settings.get("max_output_tokens") if isinstance(settings, dict) else None
        if type(budget) is not int or budget <= 0:
            raise ValueError("report_explanation.max_output_tokens 必须为正整数")
        self._explanation_max_output_tokens = budget
        self.llm_backend = LLMBackend(llm_config) if auto_llm_enabled() else None
        self.knowledge_base = DomainKnowledgeBase()
        self._knowledge_context = None
        self._last_llm_explanation_used = False
        self._last_llm_explanation_audit = {}

    def _bind_candidates(self, candidates: List[Dict], results: List[Dict], selected: List[Dict] | None) -> List[Dict]:
        """以会话编号关联阶段快照，以正式编号关联结果，不借用显示名称。"""
        records = {}
        for candidate in candidates:
            cid = candidate.get("candidate_id")
            if not isinstance(cid, str) or not cid or cid in records:
                raise ValueError("报告候选的会话编号缺失或重复")
            records[cid] = dict(candidate)
        selected_ids = set()
        for candidate in selected or []:
            cid = candidate.get("candidate_id")
            if cid not in records or cid in selected_ids:
                raise ValueError("送检候选不属于报告候选池")
            selected_ids.add(cid)
            for field in ("stiffener_type", "geometry", "layup", "material_system", "load_conditions", "boundary_conditions"):
                if candidate.get(field) != records[cid].get(field):
                    raise ValueError(f"送检候选改变了设计输入：{cid}/{field}")
            records[cid].update(candidate)
        by_formal = {item["persistent_candidate_id"]: cid for cid, item in records.items() if item.get("persistent_candidate_id")}
        if len(by_formal) != sum(bool(item.get("persistent_candidate_id")) for item in records.values()):
            raise ValueError("报告候选的正式编号重复")
        linked_sessions = set()
        for result in results:
            formal_id = result["candidate_id"]
            cid = result.get("session_candidate_id") or by_formal.get(formal_id) or formal_id
            if records and cid not in records:
                raise ValueError("结果无法绑定到报告候选")
            if cid in linked_sessions:
                raise ValueError("一个会话候选对应了多条求解结果")
            linked_sessions.add(cid)
            if cid in records:
                candidate = records[cid]
                if candidate.get("persistent_candidate_id") not in (None, formal_id):
                    raise ValueError("结果正式编号与候选记录冲突")
                candidate["persistent_candidate_id"] = formal_id
                result["session_candidate_id"] = cid
                contract = result.get("analysis_contract") or {}
                if contract.get("version") == ANALYSIS_VERSION:
                    if contract != analysis_contract(candidate):
                        raise ValueError("实际求解物理输入与报告候选不一致")
        if selected is None:
            selected_ids = linked_sessions
        elif not linked_sessions.issubset(selected_ids):
            raise ValueError("结果不属于本报告送检清单")
        for cid, record in records.items():
            record["report_selected"] = cid in selected_ids
        return list(records.values())

    def _build_structured_summary(self, task: Dict, results: List[Dict], candidates: List[Dict], selected: List[Dict] | None = None) -> Dict:
        task_payload = task_payload_from_request(task)
        target = float(task_payload["design_targets"]["BLF_min"])
        results = normalize_results(results, target)
        candidates = self._bind_candidates(candidates, results, selected)
        successful = [item for item in results if item["status"] == "success"]
        facts = build_facts(candidates, results, target)
        summary = {
            "facts": facts, "target_BLF": target,
            "primary_objective": task_payload["design_targets"]["primary_objective"],
            "candidate_count": len(candidates), "result_count": len(results),
            "selected_count": sum(item["report_selected"] for item in candidates),
            "successful_count": len(successful), "passed_count": sum(item["verdict"] == "通过" for item in successful),
            "failed_count": len(results) - len(successful),
            "not_evaluated_count": len(candidates) - len(results) if candidates else 0,
            "physical_section_count": sum((item.get("analysis_contract") or {}).get("version") == ANALYSIS_VERSION for item in successful),
            "all_candidates": candidates, "results": results,
            "recommendation": self._recommendation_facts(task_payload, candidates, results),
        }
        summary["facts"].extend(build_comparison_facts(summary, task_payload))
        summary["task"] = task_payload
        return summary

    def _recommendation_facts(self, task_payload: Dict, candidates: List[Dict], results: List[Dict]) -> Dict:
        """只比较当前逐层模型的成功结果，质量与面密度使用各自的目标量。"""
        objective = task_payload["design_targets"]["primary_objective"]
        required_blf = max(1.0, float(task_payload["design_targets"]["BLF_min"]))
        by_id = {item.get("persistent_candidate_id") or item["candidate_id"]: item for item in candidates}
        ranked, excluded = [], []
        for result in results:
            cid = result["candidate_id"]
            identity = {"candidate_id": cid, "session_candidate_id": result.get("session_candidate_id")}
            if result["status"] != "success":
                excluded.append({**identity, "reasons": ["求解未成功"]})
                continue
            candidate = by_id.get(cid, {})
            geometry = candidate.get("geometry") or {}
            contract = result.get("analysis_contract") or {}
            reasons = []
            if result["BLF_global"] < required_blf:
                reasons.append("首个正特征值低于推荐 BLF 下限")
            if not candidate or contract.get("version") != ANALYSIS_VERSION or not contract.get("fingerprint"):
                reasons.append("没有当前逐层模型的求解证据")
            density = result.get("weight_kg_per_m2")
            value = None
            if objective in {"最小重量", "最小面密度"}:
                if isinstance(density, (int, float)) and not isinstance(density, bool) and math.isfinite(density) and density > 0:
                    if objective == "最小面密度":
                        value = density
                    elif all(isinstance(geometry.get(key), (int, float)) and geometry[key] > 0 for key in ("panel_length_mm", "panel_width_mm")):
                        value = density * geometry["panel_length_mm"] * geometry["panel_width_mm"] / 1_000_000
                if value is None or not math.isfinite(value):
                    reasons.append("缺少有效质量或面密度比较数据")
            else:
                reasons.append("没有该目标的直接比较量，BLF 不作为刚度测量值")
            if reasons:
                excluded.append({**identity, "reasons": reasons})
            else:
                ranked.append({**identity, "objective_value": value, "unit": "kg/m^2" if objective == "最小面密度" else "kg",
                               "BLF_global": result["BLF_global"]})
        ranked.sort(key=lambda item: (item["objective_value"], -item["BLF_global"], item["candidate_id"]))
        return {"objective": objective, "required_BLF": required_blf, "selected": ranked[0] if ranked else None,
                "ranked": ranked, "excluded": excluded}

    def _build_llm_explanation_payload(self, summary: Dict) -> Dict:
        recommendation = summary["recommendation"]
        best, ranked = recommendation["selected"], recommendation["ranked"]
        alternative = ranked[1] if len(ranked) > 1 else None
        if best:
            required = [(best["candidate_id"], "BLF_global"), (best["candidate_id"], "objective_value"),
                        (None, "required_BLF")]
        else:
            required = [(None, "result_count"), (None, "comparable_count")]
            if summary["successful_count"]:
                required.extend(((None, "max_result_BLF"), (None, "required_BLF")))
        objective = recommendation["objective"]
        objective_label = self.OBJECTIVE_LABELS.get(objective, "目标比较值")
        # 评议解释同一目标排序中的取舍，全量候选、校核记录和排序由正式报告承载。
        compared_ids = {row["candidate_id"] for row in (best, alternative) if row is not None}
        facts = [{**fact, "label": objective_label if fact["metric"] == "objective_value"
                  else METRIC_LABELS.get(fact["metric"], fact["metric"])} for fact in summary["facts"]
                 if fact["candidate_id"] is None or (fact["candidate_id"] in compared_ids
                                                    and fact["metric"] in self.REVIEW_RESULT_METRICS)]
        decision = {
            "objective": objective,
            "selected": {"candidate_id": best["candidate_id"]} if best else None,
            "alternative": {"candidate_id": alternative["candidate_id"]} if alternative else None,
            "exclusion_counts": dict(Counter(reason for row in recommendation["excluded"] for reason in row["reasons"])),
        }
        knowledge = self._knowledge_context or {}
        # 文献身份和定位用于报告审计，评议以同一证据编号读取正文与适用条件。
        evidence_fields = ("ref", "evidence_kind", "evidence_usage", "applicability_context", "text")
        evidence = [{key: item[key] for key in evidence_fields} for item in knowledge.get("evidence", [])]
        return prepare_review_payload(
            facts, decision, format_fact_value,
            results=(row for row in summary["results"] if row["candidate_id"] in compared_ids),
            comparison_count=len(ranked), evidence=evidence, relations=knowledge.get("relations", []),
            required_metrics=required,
        )

    def _render_engineering_explanation(self, summary: Dict) -> str:
        self._last_llm_explanation_used = False
        call_traces = []
        audit = {"status": "disabled" if self.llm_backend is None else "pending", "call_traces": call_traces}
        self._last_llm_explanation_audit = audit
        if self.llm_backend is None:
            return ""

        def chat_review(stage, system, inputs):
            user = json.dumps(inputs, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            trace = {"stage": stage, "max_output_tokens": self._explanation_max_output_tokens}
            call_traces.append(trace)
            try:
                answer = self.llm_backend.chat(
                    system, user, max_tokens_override=self._explanation_max_output_tokens, json_mode=True,
                )
            except Exception as exc:
                trace.update(status="failed", error_type=type(exc).__name__)
                if type(getattr(exc, "status_code", None)) is int:
                    trace["http_status"] = exc.status_code
                raise
            else:
                trace["status"] = "completed"
                return answer
            finally:
                # 逐次读取当前响应的元数据，调用链路不复制正文或外部错误详情。
                response = getattr(self.llm_backend, "last_response", None) or {}
                usage = response.get("usage") or {}
                trace.update(
                    model=response.get("model") or getattr(self.llm_backend, "model", None),
                    finish_reason=response.get("finish_reason"),
                    usage={key: usage[key] for key in ("prompt_tokens", "completion_tokens", "total_tokens")
                           if key in usage} or None,
                )

        try:
            payload = self._build_llm_explanation_payload(summary)
            audit["input"] = payload
            answer = request_engineering_review(chat_review, "draft", payload)
            audit["raw_response"] = answer
            review = parse_engineering_review(chat_review, answer, payload, audit)
            review = review_engineering_content(chat_review, review, payload, audit)
        except Exception as exc:
            if call_traces and call_traces[-1]["status"] == "failed":
                failed = call_traces[-1]
                reason = f"{failed['stage']} 请求失败：{failed['error_type']}"
                if "http_status" in failed:
                    reason += f"（HTTP {failed['http_status']}）"
                if failed["finish_reason"]:
                    reason += f"（finish_reason={failed['finish_reason']}）"
                status = "failed"
            else:
                reason = str(exc)
                status = "rejected" if isinstance(exc, ValueError) else "failed"
            audit.update(status=status, error=reason)
            self.emit_event(f"report_explanation_{status}", f"报告解释生成失败：{reason}",
                            {"reason": reason, "call_traces": call_traces})
            raise
        self._last_llm_explanation_used = True
        audit.update(status="accepted", review=review)
        return render_engineering_review(review, payload)

    def _geometry_summary(self, geometry: Dict) -> str:
        return "，".join(f"{METRIC_LABELS.get(key, key)} {format_number(value, 3)} mm" for key, value in geometry.items()) or "未提供几何"

    def _render_markdown(self, task: Dict, summary: Dict) -> str:
        payload = task_payload_from_request(task)
        preferences = payload["candidate_generation_preferences"]
        types = preferences.get("stiffener_types") or [payload["stiffener_type"]]
        candidates, results = summary["all_candidates"], summary["results"]
        by_id = {item.get("persistent_candidate_id") or item["candidate_id"]: item for item in candidates}
        recommendation = summary["recommendation"]
        lines = ["# CSDM_panel 总体设计方案报告", "", "## 设计需求与校核依据", "",
                 f"- 会话任务编号：`{task['task_id']}`",
                 f"- 设计对象：{payload['application']}；筋型：{' / '.join(TYPE_DISPLAY_NAMES.get(kind, kind) for kind in types)}",
                 f"- 参考工况：{describe_load_conditions(payload['load_conditions'], display=True)}",
                 f"- 边界条件：{describe_boundary_conditions(payload['boundary_conditions'])}",
                 f"- 任务 BLF 下限：{format_number(summary['target_BLF'], None)}（无量纲）；优化目标：{summary['primary_objective']}",
                 "- 计算模型：逐层复合壳、理想 Tie 连接及线性特征值屈曲。"]
        if "HAT" in types:
            lines.append("- 帽形筋底部的两块连接板位于腹板外侧。")
        lines.extend(["- 分析边界：未计算材料损伤、初始缺陷敏感性、非线性极限承载和制造验收。",
                 "- 显示精度：实算 BLF 和判据保留原值；几何尺寸最多保留 3 位小数，单层厚度最多 6 位，面密度和目标比较值最多 4 位。判定和排序使用事实 JSON 中的原值。",
                 "", "## 全部候选方案与生成依据", "",
                 f"- 生成 {summary['candidate_count']} 个候选，送检 {summary['selected_count']} 个，实际返回 {summary['result_count']} 条结果，未求解 {summary['not_evaluated_count']} 个。",
                 "- 会话编号对应生成顺序，正式编号关联有限元作业与案例。提案理由描述设计意图，性能结论见校核结果。"])
        for candidate in candidates:
            cid = candidate["candidate_id"]
            layup = candidate.get("layup") or {}
            material = candidate.get("material_system") or {}
            lines.extend(["", f"### 候选 {cid}：{TYPE_DISPLAY_NAMES.get(candidate.get('stiffener_type'), candidate.get('stiffener_type', '未记录'))}",
                          f"- 正式编号：{candidate.get('persistent_candidate_id') or '未分配'}；来源：{candidate.get('source') or '未记录'}；材料：{material.get('name') or '未记录'}",
                          f"- 几何：{self._geometry_summary(candidate.get('geometry') or {})}",
                          "- 提案理由：" + str(candidate.get("rationale") or "未记录"),
                          f"- 送检：{'是' if candidate['report_selected'] else '否'}。"])
            if candidate.get("geometry") and layup.get("skin_layup"):
                names = {"skin": "蒙皮", "web": "腹板", "flange": "翼缘或连接板", "cap": "帽顶"}
                groups = {}
                layers_by_part = section_layers(candidate)
                for component, layers in layers_by_part.items():
                    groups.setdefault(tuple(layer["angle_deg"] for layer in layers), []).append(names[component])
                for angles, components in groups.items():
                    expression = layup["skin_layup"] if angles == tuple(layer["angle_deg"] for layer in layers_by_part["skin"]) else "/".join(format_number(angle) for angle in angles)
                    lines.append(f"- {'、'.join(components)}层序：`{expression}`。")
                for component, layers in layers_by_part.items():
                    thicknesses = {layer["thickness_mm"] for layer in layers}
                    description = (f"单层厚度 {format_number(next(iter(thicknesses)), 6)} mm" if len(thicknesses) == 1
                                   else "逐层厚度 " + "/".join(format_number(layer["thickness_mm"], 6) for layer in layers) + " mm")
                    lines.append(f"- {names[component]}：{len(layers)} 层，{description}。")
        lines.extend(["", "## 预筛与送检依据", ""])
        selected = [item for item in candidates if item["report_selected"]]
        if len(types) > 1:
            pool_by_type = Counter(item.get("stiffener_type") for item in candidates)
            selected_by_type = Counter(item.get("stiffener_type") for item in selected)
            coverage = "；".join(f"{TYPE_DISPLAY_NAMES[kind]} {selected_by_type[kind]}/{pool_by_type[kind]}" for kind in types)
            lines.append(f"- 筋型分布（送检数/候选数）：{coverage}。")
        direct_ids = {item["candidate_id"] for item in selected
                      if (item.get("screening_decision") or {}).get("selection_basis") == "stiffener_type_round_robin"}
        if direct_ids:
            lines.append("- 直接 FEM 按筋型轮流选样，筋型内保持生成顺序。")
        if any(item.get("surrogate_status") == "available" for item in selected):
            lines.append("- 代理按筋型独立预测并筛选；代理估计与实际有限元结果分别记录。")
        for candidate in selected:
            if candidate["candidate_id"] in direct_ids:
                continue
            lines.append(f"- {candidate['candidate_id']}：{candidate.get('selection_reason') or '按实际送检记录列出，未记录额外筛选理由。'}")
            if candidate.get("surrogate_status") == "available":
                lines.append(f"  代理 BLF 估计 {format_number(candidate.get('surrogate_BLF'), 4)}，评分 {format_number(candidate.get('rank_score'), 4)}。")
        lines.extend(["", "## 有限元校核结果", "",
                      f"- 成功求解 {summary['successful_count']} 个，求解失败 {summary['failed_count']} 个；成功结果中满足任务阈值 {summary['passed_count']} 个。",
                      "- BLF 为首个正特征值；第二个正特征值不等同于局部屈曲。模态未分类时不作局部或整体失效分类。",
                      "", "| 会话 / 正式编号 | 筋型 | BLF | 面密度 kg/m^2 | 任务阈值判定 |",
                      "| --- | --- | --- | --- | --- |"])
        for result in results:
            candidate = by_id.get(result["candidate_id"], {})
            lines.append(f"| {result.get('session_candidate_id') or candidate.get('candidate_id') or '-'} / {result['candidate_id']} | "
                         f"{TYPE_DISPLAY_NAMES.get(candidate.get('stiffener_type'), '-')} | {format_number(result.get('BLF_global'), None)} | "
                         f"{format_number(result.get('weight_kg_per_m2'), 4)} | {result.get('verdict') or '未判定'} |")
        if not results:
            lines.append("\n没有有限元结果，不能给出承载判定或最终推荐。")
        for result in results:
            if result["status"] != "success":
                lines.append(f"\n- {result['candidate_id']} 求解失败：{result.get('error_log') or result.get('diagnosis_summary') or '未记录详细诊断'}")
        lines.extend(["", "## 最终推荐设计方案", ""])
        best = recommendation["selected"]
        objective_label = self.OBJECTIVE_LABELS.get(recommendation["objective"], "目标比较值")
        if best:
            lines.extend([f"- 推荐：{best.get('session_candidate_id') or '-'} / {best['candidate_id']}。",
                          f"- 依据：仅在当前逐层模型成功求解且 BLF 不低于 {format_number(recommendation['required_BLF'], None)} 的方案中，按“{recommendation['objective']}”比较。",
                          f"- {objective_label}：{format_number(best['objective_value'], 4)} {best['unit']}；BLF={format_number(best['BLF_global'], None)}。"])
        else:
            lines.append("- 没有同时具备完整通过证据和目标比较数据的方案，不给出最终推荐。")
        for item in recommendation["excluded"]:
            lines.append(f"- {item['candidate_id']} 未参与推荐：{'；'.join(item['reasons'])}。")
        if recommendation["ranked"]:
            unit = recommendation["ranked"][0]["unit"]
            lines.extend(["", f"| 校核方案 | {objective_label} {unit} | BLF | 相对比较下限余量 % |",
                          "| --- | --- | --- | --- |"])
            for item in recommendation["ranked"]:
                margin = (item["BLF_global"] / recommendation["required_BLF"] - 1) * 100
                lines.append(f"| {item.get('session_candidate_id') or '-'} / {item['candidate_id']} | "
                             f"{format_number(item['objective_value'], 4)} | {format_number(item['BLF_global'], None)} | {format_number(margin, 2)} |")
            lines.extend(["", "余量按“BLF / 比较下限 - 1”计算；比较下限同时覆盖任务要求和完整参考载荷。此值不是强度余量或规范安全系数。"])
        lines.extend(["", self._render_engineering_explanation(summary),
                      "", "## 外部知识依据", "",
                      "文献及图谱用于提出待验证问题，不替代本任务的数值校核；引用按文献身份合并。完整证据和适用条件保存在报告事实 JSON。"])
        for evidence in self._knowledge_context["evidence"]:
            locator = "；".join(str(evidence[key]) for key in ("section", "pages") if evidence.get(key)) or "未提供章节或页码"
            lines.append(f"- 证据 {evidence['ref']}：[{evidence['source_label']}] {locator}。")
        lines.extend(["", "### 参考文献", "", *["- " + source["citation"] for source in self._knowledge_context.get("sources", [])]])
        if not self._knowledge_context.get("sources"):
            lines.append("- 未采用外部文献证据。")
        return "\n".join(lines)

    def _reportlab(self):
        from reportlab.lib.enums import TA_LEFT
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.cidfonts import UnicodeCIDFont
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
        return dict(TA_LEFT=TA_LEFT, A4=A4, ParagraphStyle=ParagraphStyle,
                    pdfmetrics=pdfmetrics, UnicodeCIDFont=UnicodeCIDFont, TTFont=TTFont, Paragraph=Paragraph,
                    SimpleDocTemplate=SimpleDocTemplate, Spacer=Spacer, Table=Table, TableStyle=TableStyle)

    def _register_font(self) -> str:
        rl = self._reportlab()
        font = Path("C:/Windows/Fonts/simsun.ttc")
        if font.is_file():
            name = "CSDM_panel_SimSun"
            rl["pdfmetrics"].registerFont(rl["TTFont"](name, str(font)))
        else:
            name = "STSong-Light"
            rl["pdfmetrics"].registerFont(rl["UnicodeCIDFont"](name))
        return name

    def _pdf_styles(self, font_name: str) -> Dict:
        rl = self._reportlab()
        base = dict(fontName=font_name, alignment=rl["TA_LEFT"], wordWrap="CJK")
        styles = {"body": rl["ParagraphStyle"]("CSDM_panelBody", **base, fontSize=10, leading=15, spaceAfter=5)}
        for key, label, size in (("title", "Title", 18), ("heading2", "Heading2", 13), ("heading3", "Heading3", 11)):
            styles[key] = rl["ParagraphStyle"]("CSDM_panel" + label, **base, fontSize=size, leading=size + 6,
                                               spaceBefore=10, spaceAfter=6, keepWithNext=True)
        styles["bullet"] = rl["ParagraphStyle"]("CSDM_panelBullet", parent=styles["body"], leftIndent=12, bulletIndent=0)
        styles["cell"] = rl["ParagraphStyle"]("CSDM_panelCell", parent=styles["body"], fontSize=9, leading=13, spaceAfter=0)
        return styles

    def _pdf_inline_text(self, tokens) -> str:
        return "".join("<br/>" if token.type == "hardbreak" else " " if token.type == "softbreak"
                       else escape(token.content) if token.type in {"text", "code_inline", "html_inline"} else "" for token in tokens)

    def _build_pdf_story(self, markdown_text: str, font_name: str) -> List:
        from markdown_it import MarkdownIt
        rl, styles = self._reportlab(), self._pdf_styles(font_name)
        story, heading, rows, row, lists = [], None, None, None, []
        for token in MarkdownIt("commonmark").enable("table").parse(markdown_text):
            if token.type == "heading_open":
                heading = "title" if token.tag == "h1" else "heading2" if token.tag == "h2" else "heading3"
            elif token.type == "heading_close":
                heading = None
            elif token.type in {"bullet_list_open", "ordered_list_open"}:
                lists.append(int(token.attrGet("start") or 1) if token.type == "ordered_list_open" else None)
            elif token.type in {"bullet_list_close", "ordered_list_close"}:
                lists.pop()
            elif token.type == "table_open":
                rows = []
            elif token.type == "tr_open":
                row = []
            elif token.type == "tr_close":
                rows.append(row)
                row = None
            elif token.type == "table_close":
                count = max(len(item) for item in rows)
                table = rl["Table"](rows, colWidths=[(rl["A4"][0] - 88) / count] * count, repeatRows=1, hAlign="LEFT", splitInRow=1)
                table.setStyle(rl["TableStyle"]([
                    ("VALIGN", (0, 0), (-1, -1), "TOP"), ("BACKGROUND", (0, 0), (-1, 0), "#EEEEEE"),
                    ("LINEBELOW", (0, 0), (-1, 0), 0.5, "#888888"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                    ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ]))
                story.extend([table, rl["Spacer"](1, 6)])
                rows = None
            elif token.type == "inline":
                bullet = None
                if lists and row is None:
                    bullet = "-" if lists[-1] is None else f"{lists[-1]}."
                    if lists[-1] is not None:
                        lists[-1] += 1
                paragraph = rl["Paragraph"](self._pdf_inline_text(token.children or []),
                                             styles["cell" if row is not None else heading or ("bullet" if lists else "body")], bulletText=bullet)
                (row if row is not None else story).append(paragraph)
            elif token.type in {"fence", "code_block"}:
                story.append(rl["Paragraph"](escape(token.content).replace("\n", "<br/>"), styles["body"]))
        if not story:
            raise ValueError("报告正文为空，不能生成 PDF")
        return story

    def _write_pdf(self, markdown_text: str, pdf_path: Path) -> None:
        rl, font = self._reportlab(), self._register_font()
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        def page_footer(canvas, document):
            canvas.saveState()
            canvas.setFont(font, 9)
            canvas.setFillColorRGB(0.4, 0.4, 0.4)
            canvas.drawString(44, 28, "CSDM_panel 总体设计方案报告")
            canvas.drawRightString(rl["A4"][0] - 44, 28, str(document.page))
            canvas.restoreState()
        document = rl["SimpleDocTemplate"](str(pdf_path), pagesize=rl["A4"], leftMargin=44, rightMargin=44,
                                            topMargin=44, bottomMargin=48, title=self.REPORT_ARTIFACTS["overall"][1], author="CSDM_panel")
        document.build(self._build_pdf_story(markdown_text, font), onFirstPage=page_footer, onLaterPages=page_footer)

    def run(self, input_data: Dict) -> Dict:
        task = input_data["task"]
        task_id = str(task.get("task_id") or "")
        if not re.fullmatch(r"TASK_[1-9][0-9]*", task_id):
            raise ValueError("报告任务编号无效")
        summary = self._build_structured_summary(task, input_data["results"], input_data.get("candidates", []), input_data.get("screened_candidates"))
        self._knowledge_context = self.knowledge_base.prepare_context(task, top_k=3)
        markdown = self._render_markdown(task, summary)
        output = Path(input_data["output_dir"]) if input_data.get("output_dir") else RESULTS_DIR / task_id
        output.mkdir(parents=True, exist_ok=True)
        stem, title = self.REPORT_ARTIFACTS["overall"]
        markdown_path, pdf_path, facts_path = (output / f"{stem}{suffix}" for suffix in (".md", ".pdf", "_facts.json"))
        audit = {"summary": summary, "knowledge_context": self._knowledge_context,
                 "llm_explanation_used": self._last_llm_explanation_used, "llm_explanation_audit": self._last_llm_explanation_audit}
        # 三种工件全部生成后才替换正式文件，失败不会用旧 PDF 配对新正文。
        with tempfile.TemporaryDirectory(prefix="report_", dir=output) as folder:
            temporary = Path(folder)
            write_text(temporary / markdown_path.name, markdown)
            self._write_pdf(markdown, temporary / pdf_path.name)
            write_json(temporary / facts_path.name, audit)
            for path in (markdown_path, pdf_path, facts_path):
                (temporary / path.name).replace(path)
        artifact = {"markdown_path": str(markdown_path), "pdf_path": str(pdf_path), "facts_path": str(facts_path),
                    "title": title, "report_kind": "overall", "markdown_generated": True, "pdf_generated": True,
                    "llm_explanation_used": self._last_llm_explanation_used}
        self.emit("总体设计方案报告已生成")
        return {**artifact, "content": markdown, "output_dir": str(output), "report_outputs": {"overall": artifact},
                "generated_output_keys": ["overall"], "llm_explanation_audit": dict(self._last_llm_explanation_audit)}
