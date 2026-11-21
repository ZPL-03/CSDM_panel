"""报告事实绑定与受限解释契约；模型不能向事实段注入自由文本。"""

from __future__ import annotations

import json
import math

from core.display import format_parameter
from core.laminate import parse_angles


FACT_REFERENCE_KEYS = {"candidate_id", "metric"}
FOLLOW_UPS = {
    "review_mesh": "核查首个正屈曲模态、网格收敛和边界实现。",
    "review_manufacturing": "核查层厚、层序、筋条成形及连接工艺；制造可行性尚未验收。",
    "inspect_failed_job": "求解未成功，先检查作业日志，不作承载判定。",
    "check_reference_load": "BLF 小于 1，线性临界载荷低于完整参考载荷，不能据此认定可承受完整参考载荷。",
}
METRIC_LABELS = {
    "evaluation_status": "求解状态", "BLF_global": "首个正特征值 BLF", "BLF_local": "已分类局部屈曲 BLF",
    "second_positive_BLF": "第二个正特征值 BLF", "weight_kg_per_m2": "面密度",
    "target_BLF": "任务 BLF 下限", "target_met": "任务阈值判定", "skin_layup": "蒙皮逐层角度",
    "ply_count": "蒙皮层数", "skin_f0": "0 度层比例", "skin_f45": "正负 45 度层比例",
    "skin_f90": "90 度层比例", "surrogate_BLF": "代理 BLF 估计", "rank_score": "代理评分",
    "panel_length_mm": "壁板长度", "panel_width_mm": "壁板宽度", "skin_thickness_mm": "蒙皮厚度",
    "pitch_mm": "筋距", "stiffener_height_mm": "筋高", "web_thickness_mm": "腹板厚度",
    "flange_width_mm": "翼缘或帽形底部跨度", "flange_thickness_mm": "翼缘或连接板厚度",
    "cap_width_mm": "帽顶宽度", "cap_thickness_mm": "帽顶厚度",
}
STATUS_LABELS = {"success": "求解成功", "failed": "求解失败", "not_evaluated": "未求解",
                 "task_input": "任务输入", "design_input": "设计输入", "predicted": "代理估计"}


def normalize_results(results, target):
    """只对成功求解的有限正 BLF 判定阈值，失败和未求解保持未知。"""
    normalized, ids = [], set()
    if not math.isfinite(target) or target <= 0:
        raise ValueError("报告目标 BLF 必须为有限正数")
    for result in results:
        item = dict(result)
        cid = item.get("candidate_id")
        if not cid or cid in ids:
            raise ValueError("结果候选编号缺失或重复")
        ids.add(cid)
        if item.get("status") == "success":
            value = item.get("BLF_global")
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{cid} 成功结果缺少有限正 BLF")
            verdict = "通过" if value >= target else "不通过"
            if item.get("verdict") not in (None, verdict):
                raise ValueError(f"{cid} 结论与原始 BLF/任务目标冲突")
            item["verdict"] = verdict
        else:
            item["verdict"] = None
            for metric in ("BLF_global", "BLF_local", "second_positive_BLF", "weight_kg_per_m2"):
                item[metric] = None
        normalized.append(item)
    return normalized


def build_facts(candidates, results, target):
    by_result = {result["candidate_id"]: result for result in results}
    by_candidate = {str(c.get("persistent_candidate_id") or c["candidate_id"]): c for c in candidates}
    if len(by_candidate) != len(candidates):
        raise ValueError("报告候选编号重复")
    if candidates and not set(by_result).issubset(by_candidate):
        raise ValueError("结果无法绑定到报告候选")
    facts = []
    def add(cid, metric, value, unit, status):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("报告事实含非有限数值")
        facts.append(dict(candidate_id=cid, metric=metric, value=value, unit=unit, result_status=status))
    for cid in sorted(set(by_candidate) | set(by_result)):
        candidate = by_candidate.get(cid, {})
        result = by_result.get(cid, {})
        status = result.get("status", "not_evaluated")
        add(cid, "evaluation_status", status, "status", status)
        for metric, unit in (("BLF_global", "1"), ("BLF_local", "1"), ("second_positive_BLF", "1"), ("weight_kg_per_m2", "kg/m^2")):
            add(cid, metric, result.get(metric), unit, status)
        add(cid, "target_BLF", target, "1", "task_input")
        add(cid, "target_met", result["BLF_global"] >= target if status == "success" else None, "boolean", status)
        for metric, value in candidate.get("geometry", {}).items():
            add(cid, metric, value, "mm", "design_input")
        layup = candidate.get("layup", {})
        if layup.get("skin_layup"):
            angles = parse_angles(layup["skin_layup"])
            add(cid, "skin_layup", angles, "deg", "design_input")
            add(cid, "ply_count", len(angles), "count", "design_input")
            for name, family in (("skin_f0", [0]), ("skin_f45", [45, -45]), ("skin_f90", [90])):
                add(cid, name, sum(angles.count(angle) for angle in family) / len(angles), "1", "design_input")
        if candidate.get("surrogate_status") == "available":
            add(cid, "surrogate_BLF", candidate.get("surrogate_BLF"), "1", "predicted")
            add(cid, "rank_score", candidate.get("rank_score"), "score", "predicted")
    return facts


def render_fact(fact):
    value = fact["value"]
    if value is None:
        display = "未知"
    elif isinstance(value, bool):
        display = "满足" if value else "不满足"
    elif isinstance(value, (float, int)):
        display = format_parameter(fact["metric"], value)
    else:
        display = json.dumps(value, ensure_ascii=False) if isinstance(value, list) else STATUS_LABELS.get(str(value), str(value))
    unit = {"1": "（无量纲）", "boolean": "", "status": "", "count": " 层", "deg": " 度"}.get(fact["unit"], " " + fact["unit"])
    return f"{fact['candidate_id']}：{METRIC_LABELS.get(fact['metric'], fact['metric'])}={display}{unit}。"


def validate_report_selection(text, payload):
    """引用只读事实，推荐必须与程序排序一致，拒绝重复字段和自由文本。"""
    def unique_object(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("报告 JSON 含重复字段")
            value[key] = item
        return value

    try:
        answer = json.loads(text, object_pairs_hook=unique_object)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("解释必须是契约 JSON，不接收自由文本") from exc
    if not isinstance(answer, dict) or set(answer) != {"fact_refs", "recommendation_candidate_id", "follow_up_codes"}:
        raise ValueError("报告字段必须为 fact_refs、recommendation_candidate_id、follow_up_codes")
    references = answer["fact_refs"]
    if not isinstance(references, list) or not 0 < len(references) <= min(len(payload["facts"]), 12):
        raise ValueError("报告需要 1 至 12 项关键事实引用")
    known = {}
    for fact in payload["facts"]:
        key = (fact["candidate_id"], fact["metric"])
        if key in known:
            raise ValueError("事实索引包含重复候选/指标")
        known[key] = fact
    used = set()
    resolved = []
    for reference in references:
        if (not isinstance(reference, dict) or set(reference) != FACT_REFERENCE_KEYS
                or any(not isinstance(value, str) for value in reference.values())):
            raise ValueError("事实引用只允许 candidate_id 和 metric 字符串")
        key = (reference["candidate_id"], reference["metric"])
        if key not in known or key in used:
            raise ValueError("事实引用未知或重复")
        used.add(key)
        resolved.append(dict(known[key]))
    expected_id = (payload["recommendation"]["selected"] or {}).get("candidate_id")
    if answer["recommendation_candidate_id"] != expected_id:
        raise ValueError("报告推荐与任务目标及求解证据不一致")
    codes = answer["follow_up_codes"]
    if not isinstance(codes, list) or any(not isinstance(code, str) or code not in payload["follow_up_codes"] for code in codes):
        raise ValueError("验证建议代码不满足当前结果条件")
    if len(set(codes)) != len(codes):
        raise ValueError("验证建议重复")
    return {**answer, "facts": resolved}


def allowed_follow_ups(facts):
    codes = {"review_manufacturing"}
    for fact in facts:
        if fact["metric"] == "evaluation_status":
            if fact["value"] == "success":
                codes.add("review_mesh")
            elif fact["value"] != "not_evaluated":
                codes.add("inspect_failed_job")
        if fact["metric"] == "BLF_global" and fact["value"] is not None and fact["value"] < 1:
            codes.add("check_reference_load")
    return {code: FOLLOW_UPS[code] for code in FOLLOW_UPS if code in codes}


def render_report_selection(answer):
    lines = ["### 关键事实", *["- " + render_fact(fact) for fact in answer["facts"]], "", "### 验证建议"]
    lines.extend("- " + FOLLOW_UPS[code] for code in answer["follow_up_codes"])
    if not answer["follow_up_codes"]:
        lines.append("- 未选择额外验证建议。")
    return "\n".join(lines)
