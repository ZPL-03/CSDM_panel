"""报告的身份关联、事实选择、目标排序和实际 PDF 工件契约。"""

from copy import deepcopy
import json
from pathlib import Path

import fitz
import pytest

from agents.report_gen import ReportGenAgent
from core.laminate import analysis_contract
from core.report_facts import render_fact, validate_report_selection


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


def _selection(payload, references=None, codes=None):
    return json.dumps({"fact_refs": references if references is not None else [{"candidate_id": "C1", "metric": "BLF_global"}],
                       "recommendation_candidate_id": (payload["recommendation"]["selected"] or {}).get("candidate_id"),
                       "follow_up_codes": codes if codes is not None else list(payload["follow_up_codes"])})


def _backend():
    class Backend:
        calls = 0
        last_response = {"finish_reason": "stop", "model": "test-model"}

        def chat(self, system_prompt, user_prompt, **kwargs):
            self.calls += 1
            return _selection(json.loads(user_prompt))
    return Backend()


def test_structured_facts_restore_original_values_and_compute_mass():
    task, results, candidates = _report_sample()
    agent = ReportGenAgent()
    summary = agent._build_structured_summary(task, results, candidates)
    payload = agent._build_llm_explanation_payload(summary)
    resolved = validate_report_selection(_selection(payload), payload)
    fact = next(item for item in summary["facts"] if item["metric"] == "BLF_global")
    assert resolved["facts"] == [fact]
    assert fact["value"] == 1.46 and fact["unit"] == "1" and fact["result_status"] == "success"
    assert next(item for item in summary["facts"] if item["metric"] == "skin_f0")["value"] == pytest.approx(4 / 14)
    assert summary["recommendation"]["selected"]["objective_value"] == pytest.approx(12.5 * 0.7 * 0.6)


@pytest.mark.parametrize("field,value", [("candidate_id", "C2"), ("metric", "unknown_metric"), ("value", 1.4601),
                                        ("unit", "kN/m"), ("result_status", "not_evaluated")])
def test_unknown_references_and_value_fields_are_rejected(field, value):
    task, results, candidates = _report_sample()
    agent = ReportGenAgent()
    payload = agent._build_llm_explanation_payload(agent._build_structured_summary(task, results, candidates))
    reference = {"candidate_id": "C1", "metric": "BLF_global", field: value}
    with pytest.raises(ValueError):
        validate_report_selection(_selection(payload, [reference]), payload)


@pytest.mark.parametrize("value", [1, 1.0, -0.273, None])
def test_fact_references_preserve_type_unit_and_status(value):
    fact = {"candidate_id": "C1", "metric": "metric", "value": value, "unit": "1", "result_status": "success"}
    payload = {"facts": [fact], "recommendation": {"selected": None}, "follow_up_codes": {}}
    reference = {"candidate_id": "C1", "metric": "metric"}
    resolved = validate_report_selection(_selection(payload, [reference]), payload)
    assert resolved["facts"] == [fact] and type(resolved["facts"][0]["value"]) is type(value)
    for refs in ([reference, reference], [fact]):
        with pytest.raises(ValueError):
            validate_report_selection(_selection(payload, refs), payload)


def test_model_cannot_override_recommendation_or_add_inapplicable_advice():
    task, results, candidates = _report_sample()
    agent = ReportGenAgent()
    payload = agent._build_llm_explanation_payload(agent._build_structured_summary(task, results, candidates))
    answer = json.loads(_selection(payload))
    answer["recommendation_candidate_id"] = "C2"
    with pytest.raises(ValueError):
        validate_report_selection(json.dumps(answer), payload)
    for codes in (["inspect_failed_job"], ["review_mesh", "review_mesh"], ["assert_manufacturing_safe"]):
        with pytest.raises(ValueError):
            validate_report_selection(_selection(payload, codes=codes), payload)
    duplicate = _selection(payload).replace('"recommendation_candidate_id":', '"fact_refs": [], "recommendation_candidate_id":')
    with pytest.raises(ValueError, match="重复字段"):
        validate_report_selection(duplicate, payload)


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
    resolved = validate_report_selection(_selection(payload, [{"candidate_id": "C1", "metric": "rank_score"}]), payload)
    assert resolved["facts"][0]["value"] == -0.273
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
    assert "check_reference_load" in agent._build_llm_explanation_payload(summary)["follow_up_codes"]


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
    selected = deepcopy(candidates)
    candidates[0].pop("persistent_candidate_id")
    other = deepcopy(candidates[0])
    other["candidate_id"] = "TMP_2"
    candidates.append(other)
    original = deepcopy(candidates)
    agent = ReportGenAgent()
    agent.llm_backend = _backend()
    response = agent.run({"task": task, "results": results, "candidates": candidates,
                          "screened_candidates": selected, "output_dir": tmp_path})
    audit = json.loads(Path(response["facts_path"]).read_text(encoding="utf-8"))
    summary = audit["summary"]
    assert summary["candidate_count"] == 2 and summary["selected_count"] == 1 and summary["not_evaluated_count"] == 1
    assert summary["all_candidates"][0]["persistent_candidate_id"] == "C1"
    assert summary["all_candidates"][1]["report_selected"] is False
    assert candidates == original and agent.llm_backend.calls == 1
    assert audit["llm_explanation_audit"]["status"] == "accepted"
    assert audit["llm_explanation_audit"]["selection"]["recommendation_candidate_id"] == "C1"
    assert response["generated_output_keys"] == ["overall"]
    assert response["report_outputs"]["overall"]["pdf_generated"] is True
    assert "任务 BLF 下限：1.35" in response["content"]
    assert "BLF 不低于 1.35" in response["content"]
    with fitz.open(response["pdf_path"]) as pdf:
        text = "".join(page.get_text() for page in pdf)
        assert "TMP_1" in text and "TMP_2" in text and "1.46" in text
        assert "证据契约" not in text and "`" not in text


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
    ("BLF_global", 1.3682, "1", "1.3682"),
    ("second_positive_BLF", 1.3841, "1", "1.3841"),
    ("weight_kg_per_m2", 4.457706984276072, "kg/m^2", "4.4577"),
    ("skin_thickness_mm", 1.6293750334818733, "mm", "1.629"),
    ("stiffener_height_mm", 35.6138545429407, "mm", "35.614"),
    ("skin_f0", 0.375, "1", "0.375"),
    ("skin_f45", 0.3333333333333333, "1", "0.333333"),
    ("rank_score", -0.273, "score", "-0.273"),
    ("BLF_global", 0.00000001, "1", "1.0000e-08"),
])
def test_bound_fact_display_has_consistent_precision_without_changing_values(metric, value, unit, expected):
    fact = dict(candidate_id="C1", metric=metric, value=value, unit=unit, result_status="design_input")
    before = deepcopy(fact)
    assert f"={expected}" in render_fact(fact)
    assert fact == before
