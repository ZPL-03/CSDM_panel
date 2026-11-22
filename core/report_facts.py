"""报告事实、载荷比例及目标比较量。"""

from __future__ import annotations

import json
import math

from core.display import format_number
from core.laminate import parse_angles
from core.task_contract import describe_boundary_conditions


ANALYSIS_SCOPE = "理想逐层复合壳与理想 Tie 连接的线性屈曲；未计算损伤、初始缺陷敏感性或非线性极限承载，未提供制造验收。"
METRIC_LABELS = {
    "evaluation_status": "求解状态", "BLF_global": "首个正特征值 BLF", "BLF_local": "已分类局部屈曲 BLF",
    "second_positive_BLF": "第二个正特征值 BLF", "weight_kg_per_m2": "面密度",
    "target_BLF": "任务 BLF 下限", "target_met": "任务阈值判定", "skin_layup": "蒙皮逐层角度",
    "ply_count": "蒙皮层数", "skin_f0": "沿壁板长度方向的层比例", "skin_f45": "沿正负斜向的层比例",
    "skin_f90": "沿壁板宽度方向的层比例", "surrogate_BLF": "代理 BLF 估计", "rank_score": "代理评分",
    "panel_length_mm": "壁板长度", "panel_width_mm": "壁板宽度", "skin_thickness_mm": "蒙皮厚度",
    "pitch_mm": "筋距", "stiffener_height_mm": "筋高", "web_thickness_mm": "腹板厚度",
    "flange_width_mm": "翼缘或帽形底部跨度", "flange_thickness_mm": "翼缘或连接板厚度",
    "cap_width_mm": "帽顶宽度", "cap_thickness_mm": "帽顶厚度",
    "required_BLF": "推荐所需 BLF 下限", "BLF_margin_percent": "BLF 相对比较下限的余量",
    "objective_value": "优化目标比较值",
    "candidate_count": "候选总数", "result_count": "实际校核数", "comparable_count": "满足推荐条件的方案数",
    "analysis_scope": "分析范围", "recommendation_status": "推荐状态", "material": "材料",
    "reference_Nx": "参考压缩载荷", "reference_Nxy": "参考剪切载荷",
    "critical_Nx": "线性临界比例下的压缩载荷", "critical_Nxy": "线性临界比例下的剪切载荷",
    "stiffener_type": "筋型", "load_type": "载荷类型", "boundary_conditions": "边界条件",
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
            value = result.get(metric)
            metric_status = status if value is not None else "unavailable" if status == "success" else status
            add(cid, metric, value, unit, metric_status)
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


def format_fact_value(fact):
    """正文与结果表采用一致的显示精度，事实原值不参与舍入。"""
    value = fact["value"]
    if fact["metric"] == "analysis_scope":
        return "逐层壳与理想 Tie 连接的线性屈曲分析"
    if value is None:
        return "未确认"
    elif isinstance(value, bool):
        display = "满足" if value else "不满足"
    elif isinstance(value, (float, int)):
        digits = 2 if fact["unit"] == "%" else 3 if fact["unit"] == "mm" else 6 if fact["metric"].startswith("skin_f") else 4
        display = format_number(value, digits)
    else:
        display = json.dumps(value, ensure_ascii=False) if isinstance(value, list) else STATUS_LABELS.get(str(value), str(value))
    # 数量的量词由句子承担，物理量的单位由事实显示值承担。
    unit = {"1": "（无量纲）", "boolean": "", "status": "", "count": "", "个": "", "deg": " 度", "": ""}.get(fact["unit"], " " + fact["unit"])
    return display + unit


def build_comparison_facts(summary, task_payload):
    """比较当前模型的成功结果，组合工况按同一特征值等比例缩放。"""
    facts = []
    def add(cid, metric, value, unit, status="derived"):
        facts.append(dict(candidate_id=cid, metric=metric, value=value, unit=unit, result_status=status))
    recommendation = summary["recommendation"]
    ranked, best = recommendation["ranked"], recommendation["selected"]
    add(None, "required_BLF", recommendation["required_BLF"], "1", "task_input")
    for metric in ("candidate_count", "result_count"):
        add(None, metric, summary[metric], "个", "recorded")
    add(None, "comparable_count", len(ranked), "个")
    add(None, "analysis_scope", ANALYSIS_SCOPE, "", "scope")
    add(None, "recommendation_status", "有满足条件的方案" if best else "没有满足完整校核及目标比较条件的方案", "")
    for row in ranked:
        cid = row["candidate_id"]
        add(cid, "objective_value", row["objective_value"], row["unit"])
        add(cid, "BLF_margin_percent", (row["BLF_global"] / recommendation["required_BLF"] - 1) * 100, "%")
    for candidate in summary["all_candidates"]:
        cid = candidate.get("persistent_candidate_id") or candidate["candidate_id"]
        add(cid, "material", (candidate.get("material_system") or {}).get("name"), "", "design_input")
        add(cid, "stiffener_type", candidate.get("stiffener_type"), "", "design_input")
    load = task_payload["load_conditions"]
    add(None, "load_type", load["label"], "", "task_input")
    add(None, "boundary_conditions", describe_boundary_conditions(task_payload["boundary_conditions"]), "", "task_input")
    for component in ("Nx", "Nxy"):
        value = load.get(f"{component}_kN_per_m")
        if value is None:
            continue
        add(None, f"reference_{component}", value, "kN/m", "task_input")
        for row in summary["results"]:
            if row["status"] == "success":
                add(row["candidate_id"], f"critical_{component}", value * row["BLF_global"], "kN/m")
    return facts
