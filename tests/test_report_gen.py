"""报告的身份关联、事实选择、目标排序和实际 PDF 工件契约。"""

from copy import deepcopy
import json
from pathlib import Path

import fitz
import pytest

from agents.report_gen import ReportGenAgent
from core.laminate import analysis_contract
from core.llm_backend import LLMResponseError
from core.report_facts import format_fact_value


def _report_sample() -> tuple[dict, list[dict], list[dict]]:
    task = {"task_id": "TASK_1", "task": {
        "application": "机翼下蒙皮壁板",
        "load_conditions": {"type": "compression_shear", "Nx_kN_per_m": 900.0, "Nxy_kN_per_m": 180.0},
        "boundary_conditions": {"type": "SSCC"}, "material_system": {"name": "T300/5208"},
        "stiffener_type": "T", "design_targets": {"BLF_min": 1.35, "primary_objective": "最小重量"},
        "candidate_generation_preferences": {"total_candidates": 6, "source_allocation_mode": "ratio",
                                             "source_ratio": {"case_transfer": 1.0, "doe": 1.0}},
        "screening_preferences": {"top_k_candidates": 3},
    }}
    candidate = {"candidate_id": "TMP_1", "persistent_candidate_id": "C1", "source": "DOE", "stiffener_type": "T",
                 "geometry": {"panel_length_mm": 700.0, "panel_width_mm": 600.0, "skin_thickness_mm": 2.5,
                              "pitch_mm": 120.0, "stiffener_height_mm": 28.0, "web_thickness_mm": 2.0,
                              "flange_width_mm": 16.0, "flange_thickness_mm": 2.0},
                 "layup": {"skin_layup": "[45/-45/0/90/0/-45/45]s", "skin_f0": 0.286, "skin_f45": 0.571, "skin_f90": 0.143},
                 "material_system": {"name": "T300/5208", "density_kg_per_m3": 1600, "E1_GPa": 181,
                                     "E2_GPa": 10.3, "G12_GPa": 7.17, "nu12": 0.28},
                 "load_conditions": deepcopy(task["task"]["load_conditions"]),
                 "boundary_conditions": deepcopy(task["task"]["boundary_conditions"]),
                 "rationale": "DOE 采样", "surrogate_status": "missing",
                 "screening_decision": {"action": "direct_fem"}}
    result = {"candidate_id": "C1", "session_candidate_id": "TMP_1", "status": "success", "BLF_global": 1.46,
              "BLF_local": None, "second_positive_BLF": 1.62, "weight_kg_per_m2": 12.5, "verdict": "通过",
              "analysis_contract": analysis_contract(candidate)}
    return task, [result], [candidate]


def _review(payload):
    refs = "、".join("{{" + ref + "}}" for ref in payload["required_assessment_refs"])
    scope = next(fact["ref"] for fact in payload["facts"] if fact["metric"] == "analysis_scope")
    return json.dumps({"recommendation_candidate_id": (payload["recommendation"]["selected"] or {}).get("candidate_id"),
                       "assessment": "当前推荐判断依据为" + refs + "，比较仅覆盖完成校核的候选。",
                       "tradeoffs": [], "verification": ["模型边界为{{" + scope + "}}；需核查实物连接与该理想约束的对应关系。"]})


def _backend():
    class Backend:
        calls = 0
        last_response = {"finish_reason": "stop", "model": "test-model"}

        def chat(self, system_prompt, user_prompt, **kwargs):
            self.calls += 1
            payload = json.loads(user_prompt)
            if set(payload) == {"report", "evidence"}:
                return json.dumps({"issues": []})
            return _review(payload["engineering_context"])
    return Backend()


def test_structured_facts_restore_original_values_and_compute_mass():
    task, results, candidates = _report_sample()
    agent = ReportGenAgent()
    summary = agent._build_structured_summary(task, results, candidates)
    payload = agent._build_llm_explanation_payload(summary)
    fact = next(item for item in summary["facts"] if item["metric"] == "BLF_global")
    assert next(item for item in payload["facts"] if item["metric"] == "BLF_global")["value"] == fact["value"]
    assert fact["value"] == 1.46 and fact["unit"] == "1" and fact["result_status"] == "success"
    assert next(item for item in summary["facts"] if item["metric"] == "BLF_local")["result_status"] == "unavailable"
    assert next(item for item in payload["facts"] if item["metric"] == "objective_value")["label"] == "模型质量（面密度乘投影面积）"
    assert next(item for item in summary["facts"] if item["metric"] == "skin_f0")["value"] == pytest.approx(4 / 14)
    assert not {"skin_f0", "skin_layup", "surrogate_BLF"} & {item["metric"] for item in payload["facts"]}
    assert summary["recommendation"]["selected"]["objective_value"] == pytest.approx(12.5 * 0.7 * 0.6)


@pytest.mark.parametrize(("value", "display"), [(True, "达到任务 BLF 下限"), (False, "未达到任务 BLF 下限")])
def test_threshold_fact_names_the_limit_in_generated_prose(value, display):
    assert format_fact_value({"metric": "target_met", "unit": "boolean", "value": value}) == display


@pytest.mark.parametrize("response", ["C1 已通过，制造已认证。", '{"fact_refs":[],"text":"无依据结论"}'])
def test_rejected_model_response_fails_export_without_publishing(tmp_path, response):
    task, results, candidates = _report_sample()
    agent = ReportGenAgent()
    class Backend:
        def chat(self, *args, **kwargs):
            return response
    agent.llm_backend = Backend()
    with pytest.raises(ValueError):
        agent.run({"task": task, "results": results, "candidates": candidates, "output_dir": tmp_path})
    assert agent._last_llm_explanation_used is False
    assert agent._last_llm_explanation_audit["status"] == "rejected"
    assert agent._last_llm_explanation_audit["raw_response"] == response
    assert not list(tmp_path.rglob("overall_design_report*"))


def test_api_failure_is_not_replaced_by_deterministic_llm_success(tmp_path):
    task, results, candidates = _report_sample()
    agent = ReportGenAgent()
    class Backend:
        def chat(self, *args, **kwargs):
            raise RuntimeError("API unavailable")
    agent.llm_backend = Backend()
    with pytest.raises(RuntimeError, match="API unavailable"):
        agent.run({"task": task, "results": results, "candidates": candidates, "output_dir": tmp_path})
    assert agent._last_llm_explanation_audit["status"] == "failed"
    assert not list(tmp_path.rglob("overall_design_report*"))


@pytest.mark.parametrize("settings", [None, {}, {"max_output_tokens": 0}, {"max_output_tokens": -1},
                                     {"max_output_tokens": True}, {"max_output_tokens": 1.5},
                                     {"max_output_tokens": "2048"}])
def test_report_rejects_invalid_explanation_budget(monkeypatch, settings):
    monkeypatch.setattr("agents.report_gen.load_llm_config", lambda: {"report_explanation": settings})
    with pytest.raises(ValueError, match=r"report_explanation\.max_output_tokens"):
        ReportGenAgent()


def test_report_audits_each_call_with_the_configured_budget(monkeypatch):
    monkeypatch.setattr("agents.report_gen.load_llm_config", lambda: {"report_explanation": {"max_output_tokens": 400}})
    task, results, candidates = _report_sample()
    agent = ReportGenAgent()
    budgets = []

    class Backend:
        def chat(self, system, user, *, max_tokens_override, json_mode):
            assert json_mode
            budgets.append(max_tokens_override)
            position = len(budgets)
            self.last_response = {"model": "test-model", "finish_reason": "stop", "content": "response body",
                                  "usage": {"prompt_tokens": 100, "completion_tokens": position,
                                            "total_tokens": 100 + position, "provider_diagnostics": {"details": True}}}
            request = json.loads(user)
            if "report" in request:
                issues = [{"quote": "比较仅覆盖完成校核的候选", "issue": "说明证据范围", "fact_refs": ["C1.BLF_global"]}]
                return json.dumps({"issues": issues if position == 3 else []})
            assert set(request["response_schema"]) == {"recommendation_candidate_id", "assessment", "tradeoffs", "verification"}
            assert "comparison_count" not in request
            assert "unavailable_facts" not in request["engineering_context"]
            reply = json.loads(_review(request["engineering_context"]))
            if position == 1:
                reply["comparison_count"] = request["engineering_context"]["comparison_count"]
            elif position == 2:
                assert "不得输出 comparison_count" in request["validation_error"]
            return json.dumps(reply)

    agent.llm_backend = Backend()
    summary = agent._build_structured_summary(task, results, candidates)
    assert agent._render_engineering_explanation(summary)
    audit = agent._last_llm_explanation_audit
    assert audit["status"] == "accepted" and "api_response" not in audit
    assert budgets == [400] * 5
    stages = ["draft", "format_repair", "content_check", "revision", "content_check"]
    assert audit["call_traces"] == [
        {"stage": stage, "max_output_tokens": 400, "status": "completed", "model": "test-model",
         "finish_reason": "stop", "usage": {"prompt_tokens": 100, "completion_tokens": i, "total_tokens": 100 + i}}
        for i, stage in enumerate(stages, 1)
    ]


def test_truncated_content_check_stops_export_and_records_failure(tmp_path):
    task, results, candidates = _report_sample()
    events = []
    agent = ReportGenAgent(lambda _, __, event: events.append(event))

    class Backend:
        def chat(self, system, user, **kwargs):
            request = json.loads(user)
            if "report" in request:
                self.last_response = {"model": "test-model", "finish_reason": "length", "content": '{"issues":'}
                raise LLMResponseError("LLM 回答未完整结束（finish_reason=length）")
            self.last_response = {"model": "test-model", "finish_reason": "stop"}
            return _review(request["engineering_context"])

    agent.llm_backend = Backend()
    with pytest.raises(LLMResponseError, match="finish_reason=length"):
        agent.run({"task": task, "results": results, "candidates": candidates, "output_dir": tmp_path})
    audit = agent._last_llm_explanation_audit
    assert audit["status"] == "failed" and not agent._last_llm_explanation_used
    assert [(row["stage"], row["finish_reason"], row["status"]) for row in audit["call_traces"]] == [
        ("draft", "stop", "completed"), ("content_check", "length", "failed"),
    ]
    assert "content" not in audit["call_traces"][-1] and not list(tmp_path.rglob("overall_design_report*"))
    assert events[-1]["event_type"] == "report_explanation_failed"
    assert events[-1]["payload"]["call_traces"] == audit["call_traces"]


def test_context_rejection_preserves_http_status_without_retry_or_export(tmp_path):
    task, results, candidates = _report_sample()
    agent = ReportGenAgent()

    class RequestRejected(Exception):
        status_code = 400

    class Backend:
        def chat(self, *args, **kwargs):
            raise RequestRejected("input and output exceed context length")

    agent.llm_backend = Backend()
    with pytest.raises(RequestRejected, match="context length"):
        agent.run({"task": task, "results": results, "candidates": candidates, "output_dir": tmp_path})
    audit = agent._last_llm_explanation_audit
    assert [(row["stage"], row["http_status"], row["finish_reason"]) for row in audit["call_traces"]] == [
        ("draft", 400, None),
    ]
    assert audit["status"] == "failed" and not agent._last_llm_explanation_used
    assert "context length" not in audit["error"] and not list(tmp_path.rglob("overall_design_report*"))


def test_failed_and_uncomputed_results_remain_unknown():
    task, results, candidates = _report_sample()
    results[0].update(status="failed", verdict="通过", BLF_global=99)
    other = deepcopy(candidates[0])
    other.update(candidate_id="TMP_2", persistent_candidate_id="C2")
    summary = ReportGenAgent()._build_structured_summary(task, results, candidates + [other])
    assert summary["passed_count"] == 0 and summary["recommendation"]["selected"] is None
    assert summary["failed_count"] == 1 and summary["not_evaluated_count"] == 1
    values = {(fact["candidate_id"], fact["metric"]): fact["value"] for fact in summary["facts"]}
    assert values["C1", "BLF_global"] is None and values["C2", "target_met"] is None


def test_exact_threshold_and_negative_predicted_score():
    task, results, candidates = _report_sample()
    task["task"]["design_targets"]["BLF_min"] = 1.5
    results[0].update(BLF_global=1.4999999, verdict="不通过")
    candidates[0].update(surrogate_status="available", rank_score=-0.273)
    agent = ReportGenAgent()
    summary = agent._build_structured_summary(task, results, candidates)
    assert summary["passed_count"] == 0 and summary["recommendation"]["selected"] is None
    payload = agent._build_llm_explanation_payload(summary)
    assert next(row for row in summary["facts"] if row["metric"] == "rank_score")["value"] == -0.273
    assert [row["candidate_id"] for row in payload["facts"] if row["candidate_id"]] == []
    assert payload["recommendation"]["exclusion_counts"] == {"首个正特征值低于推荐 BLF 下限": 1}
    assert {"TASK.max_result_BLF", "TASK.required_BLF"} <= set(payload["required_assessment_refs"])
    highest = next(fact for fact in payload["facts"] if fact["ref"] == "TASK.max_result_BLF")
    assert highest["value"] == 1.4999999 and highest["display"] == "1.4999999（无量纲）"
    results[0]["verdict"] = "通过"
    with pytest.raises(ValueError, match="冲突"):
        agent._build_structured_summary(task, results, candidates)


def test_low_blf_does_not_qualify_as_full_reference_load_recommendation():
    task, results, candidates = _report_sample()
    task["task"]["design_targets"]["BLF_min"] = 0.1
    results[0]["BLF_global"] = 0.2
    agent = ReportGenAgent()
    summary = agent._build_structured_summary(task, results, candidates)
    assert summary["passed_count"] == 1 and summary["recommendation"]["selected"] is None
    assert summary["recommendation"]["required_BLF"] == 1.0


@pytest.mark.parametrize("mutation", ["duplicate_result", "unbound_result", "wrong_formal", "wrong_layup", "duplicate_selected"])
def test_candidate_and_result_binding_rejects_identity_or_physical_conflict(mutation):
    task, results, candidates = _report_sample()
    selected = deepcopy(candidates)
    if mutation == "duplicate_result":
        results += deepcopy(results)
    elif mutation == "unbound_result":
        results[0].update(session_candidate_id="TMP_99")
    elif mutation == "wrong_formal":
        results[0]["candidate_id"] = "C99"
    elif mutation == "wrong_layup":
        results[0]["analysis_contract"]["sections"]["skin"][0]["angle_deg"] = 90
    else:
        selected += deepcopy(selected)
    with pytest.raises(ValueError):
        ReportGenAgent()._build_structured_summary(task, results, candidates, selected)


def test_report_includes_all_candidates_and_preserves_stage_snapshots(tmp_path):
    task, results, candidates = _report_sample()
    results[0]["odb_diagnostics"] = {"trace": ["solver detail" * 512]}
    selected = deepcopy(candidates)
    candidates[0].pop("persistent_candidate_id")
    for index in range(2, 51):
        other = deepcopy(candidates[0])
        other["candidate_id"] = f"TMP_{index}"
        other["geometry"]["skin_thickness_mm"] += index / 100
        candidates.append(other)
    original = deepcopy(candidates)
    agent = ReportGenAgent()
    agent.llm_backend = _backend()
    response = agent.run({"task": task, "results": results, "candidates": candidates,
                          "screened_candidates": selected, "output_dir": tmp_path})
    audit = json.loads(Path(response["facts_path"]).read_text(encoding="utf-8"))
    summary = audit["summary"]
    assert summary["candidate_count"] == 50 and summary["selected_count"] == 1 and summary["not_evaluated_count"] == 49
    assert summary["all_candidates"][0]["persistent_candidate_id"] == "C1"
    assert summary["all_candidates"][1]["report_selected"] is False
    assert candidates == original and agent.llm_backend.calls == 2
    assert audit["llm_explanation_audit"]["status"] == "accepted"
    assert audit["llm_explanation_audit"]["review"]["recommendation_candidate_id"] == "C1"
    assert {fact["candidate_id"] for fact in summary["facts"] if fact["result_status"] == "not_evaluated"} == {
        f"TMP_{index}" for index in range(2, 51)}
    review_input = audit["llm_explanation_audit"]["input"]
    assert set(review_input["candidate_ids"]) == {"TMP_1", "C1"}
    assert all(fact["candidate_id"] in (None, "C1")
               for field in ("facts", "unavailable_facts") for fact in review_input[field])
    assert not {"all_candidates", "results", "task"} & review_input.keys()
    assert "odb_diagnostics" not in json.dumps(review_input)
    assert summary["results"][0]["odb_diagnostics"] == results[0]["odb_diagnostics"]
    assert {"TASK.load_type", "TASK.boundary_conditions"} <= {item["ref"] for item in review_input["facts"]}
    context = audit["knowledge_context"]
    assert review_input["relations"] == context["relations"]
    assert {item["ref"]: item["text"] for item in review_input["evidence"]} == {
        item["ref"]: item["text"] for item in context["evidence"]}
    assert all("source_label" not in item and "table_evidence" not in item for item in review_input["evidence"])
    assert response["generated_output_keys"] == ["overall"]
    assert response["report_outputs"]["overall"]["pdf_generated"] is True
    assert "任务 BLF 下限：1.35" in response["content"]
    assert "BLF 不低于 1.35" in response["content"]
    assert "两块连接板位于腹板外侧" not in response["content"]
    with fitz.open(response["pdf_path"]) as pdf:
        text = "".join(page.get_text() for page in pdf)
        assert all(f"候选 TMP_{index}：" in text for index in range(1, 51)) and "1.46" in text
        assert "证据契约" not in text and "`" not in text


def test_many_fem_results_use_true_objective_order_without_duplicating_rows():
    task, results, candidates = _report_sample()
    agent = ReportGenAgent()
    for number in range(2, 51):
        candidate = deepcopy(candidates[0])
        candidate.update(candidate_id=f"TMP_{number}", persistent_candidate_id=f"C{number}")
        candidate["geometry"]["stiffener_height_mm"] += number / 100
        candidates.append(candidate)
        result = {**results[0], "candidate_id": f"C{number}", "weight_kg_per_m2": 17 - number / 10,
                  "analysis_contract": analysis_contract(candidate), "odb_diagnostics": {"frame": "detail" * 500}}
        result.pop("session_candidate_id")
        if number == 2:
            result["status"] = "failed"
        results.append(result)
    summary = agent._build_structured_summary(task, list(reversed(results)), candidates)
    payload = agent._build_llm_explanation_payload(summary)
    assert payload["comparison_count"] == 49 and len(summary["results"]) == 50
    assert len(summary["recommendation"]["ranked"]) == 49
    assert payload["recommendation"]["selected"] == {"candidate_id": "C50"}
    assert payload["recommendation"]["alternative"] == {"candidate_id": "C49"}
    assert payload["recommendation"]["exclusion_counts"] == {"求解未成功": 1}
    assert set(payload["candidate_ids"]) == {"C50", "C49", "TMP_50", "TMP_49"}
    assert {(fact["ref"], fact["value"]) for fact in payload["facts"] if fact["metric"] == "target_BLF"} == {
        ("TASK.target_BLF", 1.35),
    }
    for fact in payload["facts"] + payload["unavailable_facts"]:
        assert fact["candidate_id"] is None or fact["metric"] in agent.REVIEW_RESULT_METRICS
    assert "odb_diagnostics" not in json.dumps(payload) and summary["results"][0]["odb_diagnostics"]
    pair = agent._build_structured_summary(task, results[-2:], candidates[-2:])
    pair_payload = agent._build_llm_explanation_payload(pair)
    assert len(payload["facts"]) == len(pair_payload["facts"])
    assert len(payload["unavailable_facts"]) == len(pair_payload["unavailable_facts"])


def test_mass_and_areal_density_objectives_do_not_use_same_ranking():
    task, results, candidates = _report_sample()
    second = deepcopy(candidates[0])
    second.update(candidate_id="TMP_2", persistent_candidate_id="C2")
    second["geometry"]["panel_length_mm"] *= 2
    second_result = deepcopy(results[0])
    second_result.update(candidate_id="C2", session_candidate_id="TMP_2", weight_kg_per_m2=10.0,
                         analysis_contract=analysis_contract(second))
    agent = ReportGenAgent()
    assert agent._build_structured_summary(task, results + [second_result], candidates + [second])["recommendation"]["selected"]["candidate_id"] == "C1"
    task["task"]["design_targets"]["primary_objective"] = "最小面密度"
    assert agent._build_structured_summary(task, results + [second_result], candidates + [second])["recommendation"]["selected"]["candidate_id"] == "C2"
    assert agent._build_structured_summary(task, results, [])["recommendation"]["selected"] is None
    task["task"]["design_targets"]["primary_objective"] = "刚度优先"
    assert agent._build_structured_summary(task, results, candidates)["recommendation"]["selected"] is None


def test_legacy_result_does_not_become_current_physical_model_recommendation():
    task, results, candidates = _report_sample()
    results[0].pop("analysis_contract")
    summary = ReportGenAgent()._build_structured_summary(task, results, candidates)
    assert summary["passed_count"] == 1 and summary["physical_section_count"] == 0
    assert summary["recommendation"]["selected"] is None


def test_task_report_directories_do_not_overwrite_each_other(tmp_path, monkeypatch):
    import agents.report_gen as module
    monkeypatch.setattr(module, "RESULTS_DIR", tmp_path)
    task, results, candidates = _report_sample()
    agent = ReportGenAgent()
    first = agent.run({"task": task, "results": results, "candidates": candidates})
    first_bytes = Path(first["pdf_path"]).read_bytes()
    task["task_id"] = "TASK_2"
    second = agent.run({"task": task, "results": results, "candidates": candidates})
    assert Path(first["pdf_path"]).parent != Path(second["pdf_path"]).parent
    assert Path(first["pdf_path"]).read_bytes() == first_bytes
    task["task_id"] = "../outside"
    with pytest.raises(ValueError):
        agent.run({"task": task, "results": results, "candidates": candidates})


def test_pdf_failure_preserves_existing_complete_report(tmp_path, monkeypatch):
    task, results, candidates = _report_sample()
    agent = ReportGenAgent()
    response = agent.run({"task": task, "results": results, "candidates": candidates, "output_dir": tmp_path})
    original = {path.name: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()}
    def fail(*args):
        raise RuntimeError("PDF failed")
    monkeypatch.setattr(agent, "_write_pdf", fail)
    with pytest.raises(RuntimeError, match="PDF failed"):
        agent.run({"task": task, "results": results, "candidates": candidates, "output_dir": tmp_path})
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()} == original
    assert Path(response["pdf_path"]).is_file()


def test_pdf_parser_renders_markdown_tables_inline_and_long_text(tmp_path):
    agent = ReportGenAgent()
    markdown = "# 标题\n\n## 结果\n\n| 对象 | 内容 |\n| --- | --- |\n| C1 | **通过** `1.46` |\n\n" + "自动换行检查，" * 150
    path = tmp_path / "report.pdf"
    agent._write_pdf(markdown, path)
    with fitz.open(path) as pdf:
        text = "".join(page.get_text() for page in pdf)
        assert "C1" in text and "1.46" in text
        assert "| ---" not in text and "**" not in text and "`" not in text
        for page in pdf:
            for word in page.get_text("words"):
                assert 0 <= word[0] < word[2] <= page.rect.width
                assert 0 <= word[1] < word[3] <= page.rect.height


def test_pdf_heading_levels_keep_with_following_content():
    agent = ReportGenAgent()
    headings = ["# 标题", "## 章节", "### 小节", "#### 关注点", "##### 细节", "###### 约束 ###"]
    story = agent._build_pdf_story("\n\n".join(headings + ["####### 普通文本", "正文含 # 字符"]), "Helvetica")
    assert [paragraph.getPlainText() for paragraph in story] == ["标题", "章节", "小节", "关注点", "细节", "约束", "####### 普通文本", "正文含 # 字符"]
    assert all(paragraph.style.keepWithNext for paragraph in story[:6])


@pytest.mark.parametrize("metric,value,unit,expected", [
    ("BLF_global", 1.3682, "1", "1.3682（无量纲）"),
    ("second_positive_BLF", 1.3841, "1", "1.3841（无量纲）"),
    ("weight_kg_per_m2", 4.457706984276072, "kg/m^2", "4.4577 kg/m^2"),
    ("skin_thickness_mm", 1.6293750334818733, "mm", "1.629 mm"),
    ("stiffener_height_mm", 35.6138545429407, "mm", "35.614 mm"),
    ("skin_f0", 0.375, "1", "0.375（无量纲）"),
    ("skin_f45", 0.3333333333333333, "1", "0.333333（无量纲）"),
    ("rank_score", -0.273, "score", "-0.273 score"),
    ("BLF_global", 0.00000001, "1", "1e-08（无量纲）"),
    ("candidate_count", 12, "个", "12"),
    ("comparable_count", 0, "个", "0"),
    ("ply_count", 16, "count", "16"),
])
def test_bound_fact_display_has_consistent_precision_without_changing_values(metric, value, unit, expected):
    fact = dict(candidate_id="C1", metric=metric, value=value, unit=unit, result_status="design_input")
    before = deepcopy(fact)
    assert format_fact_value(fact) == expected
    assert fact == before
