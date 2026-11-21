"""需求语义字段必须绑定原文，不能改变明确约束或调用模型补造数值。"""

import json
from copy import deepcopy

import pytest

from core.task_parser import TaskParser


TEXT = "T型壁板，长度360mm，宽度300mm，Nx120kN/m，四边简支，生成2个候选，初筛1个，BLF至少0.8"
FACTS = [dict(field=field, value=value, quote=quote) for field, value, quote in (
    ("stiffener_types", ["T"], "T型"), ("panel_length_mm", 360, "长度360mm"),
    ("panel_width_mm", 300, "宽度300mm"), ("Nx_kN_per_m", 120, "Nx120kN/m"),
    ("boundary_type", "SSSS", "四边简支"), ("total_candidates", 2, "生成2个候选"),
    ("top_k_candidates", 1, "初筛1个"), ("BLF_min", 0.8, "BLF至少0.8"))]


def invoke(monkeypatch, facts, text=TEXT):
    monkeypatch.setattr("core.task_parser.LLMBackend", lambda: type("Backend", (), {
        "chat": lambda *_a, **_k: json.dumps({"facts": facts}, ensure_ascii=False)})())
    return TaskParser()._interpret_requirements(text)


def test_grounded_requirements_keep_actual_counts_loads_and_target(monkeypatch):
    task = invoke(monkeypatch, FACTS)
    assert task["load_conditions"]["Nx_kN_per_m"] == 120
    assert task["geometry_envelope"]["panel_length_mm"] == [360, 360]
    assert task["design_targets"]["BLF_min"] == 0.8
    assert task["candidate_generation_preferences"]["total_candidates"] == 2
    assert task["screening_preferences"]["top_k_candidates"] == 1
    assert task["requirement_extraction"]["method"] == "llm_grounded"


@pytest.mark.parametrize("action", ["invent_number", "invent_quote", "omit_load", "duplicate", "wrong_type", "wrong_unit"])
def test_invalid_or_missing_requirement_evidence_is_rejected(monkeypatch, action):
    facts = deepcopy(FACTS)
    if action == "invent_number":
        facts[1]["value"] = 500
    elif action == "invent_quote":
        facts[1]["quote"] = "长度360mm，材料IM7"
    elif action == "omit_load":
        facts = [item for item in facts if item["field"] != "Nx_kN_per_m"]
    elif action == "duplicate":
        facts.append(deepcopy(facts[1]))
    elif action == "wrong_type":
        facts[0]["value"] = ["L"]
    elif action == "wrong_unit":
        with pytest.raises(ValueError):
            invoke(monkeypatch, facts, TEXT.replace("360mm", "360m"))
        return
    with pytest.raises(ValueError):
        invoke(monkeypatch, facts)


@pytest.mark.parametrize("dimension", ["长度0.36m", "蒙皮厚度2cm", "筋距120", "腹板厚度2毫米"])
def test_unsupported_geometry_units_cannot_be_silently_omitted(monkeypatch, dimension):
    text = TEXT + "，" + dimension
    with pytest.raises(ValueError, match="必须使用 mm|冲突数值"):
        invoke(monkeypatch, FACTS, text)


@pytest.mark.parametrize("field,label", [("skin_thickness_mm", "蒙皮厚度"), ("pitch_mm", "筋距"),
                                         ("web_thickness_mm", "腹板厚度"), ("cap_width_mm", "帽顶宽度")])
def test_explicit_fixed_geometry_cannot_be_omitted_or_changed(monkeypatch, field, label):
    text = TEXT + f"，{label}2mm"
    with pytest.raises(ValueError, match="遗漏或改变了明确几何"):
        invoke(monkeypatch, FACTS, text)
    task = invoke(monkeypatch, [*FACTS, dict(field=field, value=2, quote=f"{label}2mm")], text)
    assert task["fixed_geometry"][field] == 2
