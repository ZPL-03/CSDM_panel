from core.task_parser import TaskParser


def test_task_parser_supports_axial_compression_defaults() -> None:
    parser = TaskParser()
    task = parser.parse_instruction("请设计 T 形加筋壁板，压缩 980 kN/m，四边简支，生成 10 个候选，初筛保留 5 个候选")
    task_payload = task["task"]
    assert task["task_id"].startswith("TASK_")
    assert task_payload["load_conditions"]["type"] == "axial_compression"
    assert task_payload["load_conditions"]["Nx_kN_per_m"] == 980.0
    assert task_payload["boundary_conditions"]["type"] == "SSSS"
    assert task_payload["candidate_generation_preferences"]["total_candidates"] == 10
    assert task_payload["design_targets"]["primary_objective"]


def test_task_parser_supports_in_plane_shear() -> None:
    parser = TaskParser()
    task = parser.parse_instruction("请设计面内剪切壁板方案，Nxy 240 kN/m，四边固支，生成 10 个候选，初筛保留 5 个候选")
    task_payload = task["task"]
    assert task_payload["load_conditions"]["type"] == "in_plane_shear"
    assert task_payload["load_conditions"]["Nxy_kN_per_m"] == 240.0
    assert task_payload["boundary_conditions"]["type"] == "CCCC"


def test_task_parser_supports_compression_shear_and_sscc() -> None:
    parser = TaskParser()
    task = parser.parse_instruction(
        "请做机翼下蒙皮 T 形筋方案，压缩 900 kN/m，剪切 180 kN/m，边界 SSCC，"
        "生成 18 个候选，初筛保留 6 个候选"
    )
    task_payload = task["task"]
    assert task_payload["load_conditions"]["type"] == "compression_shear"
    assert task_payload["load_conditions"]["Nx_kN_per_m"] == 900.0
    assert task_payload["load_conditions"]["Nxy_kN_per_m"] == 180.0
    assert task_payload["boundary_conditions"]["type"] == "SSCC"
    assert task_payload["candidate_generation_preferences"]["total_candidates"] == 18
    assert task_payload["candidate_generation_preferences"]["source_ratio"] == {
        "llm": 2.0,
        "case_transfer": 1.0,
        "doe": 1.0,
    }
    assert task_payload["screening_preferences"]["top_k_candidates"] == 6
    assert task_payload["user_input_facts"]["load_conditions"] == {
        "type": "compression_shear",
        "Nx_kN_per_m": 900.0,
        "Nxy_kN_per_m": 180.0,
    }
    assert "candidate_generation_preferences.total_candidates" in task_payload["user_input_facts"]["explicit_fields"]
    assert "screening_preferences.top_k_candidates" in task_payload["user_input_facts"]["explicit_fields"]


def test_task_parser_recognizes_short_hat_stiffener_phrase() -> None:
    parser = TaskParser()
    task = parser.parse_instruction(
        "请为机翼下蒙皮壁板设计一个帽加筋方案，压缩 1500 kN/m，剪切 200 kN/m，"
        "边界 SSCC，生成 18 个候选，初筛保留 6 个候选"
    )
    task_payload = task["task"]

    assert task_payload["stiffener_type"] == "HAT"
    assert task_payload["user_input_facts"]["stiffener_type"] == "HAT"
    assert "stiffener_type" in task_payload["user_input_facts"]["explicit_fields"]
    assert task_payload["load_conditions"]["type"] == "compression_shear"
    assert task_payload["load_conditions"]["Nx_kN_per_m"] == 1500.0
    assert task_payload["load_conditions"]["Nxy_kN_per_m"] == 200.0


def test_task_parser_supports_short_screening_phrase() -> None:
    parser = TaskParser()
    task = parser.parse_instruction(
        "设计一个T形加筋壁板结构，压缩荷载1000kN/m，剪切荷载800kN/m，边界条件SSSS，"
        "生成50个候选方案，初筛10个"
    )
    task_payload = task["task"]
    assert task_payload["load_conditions"]["type"] == "compression_shear"
    assert task_payload["load_conditions"]["Nx_kN_per_m"] == 1000.0
    assert task_payload["load_conditions"]["Nxy_kN_per_m"] == 800.0
    assert task_payload["boundary_conditions"]["type"] == "SSSS"
    assert task_payload["candidate_generation_preferences"]["total_candidates"] == 50
    assert task_payload["screening_preferences"]["top_k_candidates"] == 10


def test_task_parser_supports_user_reported_short_prompt() -> None:
    parser = TaskParser()
    task = parser.parse_instruction("请设计T形加筋壁板， 压缩1000，剪切800,20个候选，10个初筛")
    task_payload = task["task"]
    assert task_payload["load_conditions"]["type"] == "compression_shear"
    assert task_payload["load_conditions"]["Nx_kN_per_m"] == 1000.0
    assert task_payload["load_conditions"]["Nxy_kN_per_m"] == 800.0
    assert task_payload["candidate_generation_preferences"]["total_candidates"] == 20
    assert task_payload["screening_preferences"]["top_k_candidates"] == 10


def test_task_parser_supports_short_filter_phrase() -> None:
    parser = TaskParser()
    task = parser.parse_instruction("设计 T 形加筋壁板，压缩 980 kN/m，四边简支，生成 20 个候选，筛8个候选")
    task_payload = task["task"]
    assert task_payload["candidate_generation_preferences"]["total_candidates"] == 20
    assert task_payload["screening_preferences"]["top_k_candidates"] == 8


def test_task_parser_keeps_explicit_material_and_candidate_count() -> None:
    parser = TaskParser()
    task = parser.parse_instruction(
        "请为机翼下蒙皮壁板设计一个 T 形加筋方案，压缩 900 kN/m，边界 SSCC，"
        "材料 IM7/8552，生成 12 个候选，初筛保留 5 个候选"
    )
    task_payload = task["task"]
    assert task_payload["material_system"]["name"] == "IM7/8552"
    assert task_payload["material_system"]["is_user_specified"] is True
    assert task_payload["candidate_generation_preferences"]["total_candidates"] == 12
    assert task_payload["user_input_facts"]["material_system"] == {
        "name": "IM7/8552",
        "material_key": "IM7_8552",
    }


def test_task_parser_requires_pool_and_screen_counts() -> None:
    parser = TaskParser()
    for instruction, message in [
        ("请设计 T 形加筋壁板，压缩 980 kN/m，初筛保留 5 个候选", "候选池总数"),
        ("请设计 T 形加筋壁板，压缩 980 kN/m，生成 10 个候选", "初筛保留数量"),
    ]:
        try:
            parser.parse_instruction(instruction)
        except ValueError as exc:
            assert message in str(exc)
        else:
            raise AssertionError("缺少流程控制数量时应拒绝启动任务解析")


def test_task_parser_supports_english_candidate_count_and_top_k() -> None:
    parser = TaskParser()
    task = parser.parse_instruction(
        "Design a composite stiffened panel with T stiffeners, axial compression Nx 700 kN/m, "
        "boundary SSCC, BLF at least 1.1, generate 5 candidates, screen top 2."
    )
    payload = task["task"]

    assert payload["stiffener_type"] == "T"
    assert payload["load_conditions"]["type"] == "axial_compression"
    assert payload["load_conditions"]["Nx_kN_per_m"] == 700.0
    assert payload["boundary_conditions"]["type"] == "SSCC"
    assert payload["candidate_generation_preferences"]["total_candidates"] == 5
    assert payload["screening_preferences"]["top_k_candidates"] == 2
