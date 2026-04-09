from core.task_parser import TaskParser


def test_task_parser_supports_axial_compression_defaults() -> None:
    parser = TaskParser()
    task = parser.parse_instruction("请设计 T 形加筋壁板，压缩 980 kN/m，四边简支")
    assert task["load_conditions"]["type"] == "axial_compression"
    assert task["load_conditions"]["Nx_kN_per_m"] == 980.0
    assert task["boundary_conditions"]["type"] == "SSSS"
    assert task["design_targets"]["primary_objective"]


def test_task_parser_supports_in_plane_shear() -> None:
    parser = TaskParser()
    task = parser.parse_instruction("请设计面内剪切壁板方案，Nxy 240 kN/m，四边固支")
    assert task["load_conditions"]["type"] == "in_plane_shear"
    assert task["load_conditions"]["Nxy_kN_per_m"] == 240.0
    assert task["boundary_conditions"]["type"] == "CCCC"


def test_task_parser_supports_compression_shear_and_sscc() -> None:
    parser = TaskParser()
    task = parser.parse_instruction(
        "请做机翼下蒙皮 T 形筋方案，压缩 900 kN/m，剪切 180 kN/m，边界 SSCC，"
        "生成 18 个候选，初筛保留 6 个候选"
    )
    assert task["load_conditions"]["type"] == "compression_shear"
    assert task["load_conditions"]["Nx_kN_per_m"] == 900.0
    assert task["load_conditions"]["Nxy_kN_per_m"] == 180.0
    assert task["boundary_conditions"]["type"] == "SSCC"
    assert task["candidate_generation_preferences"]["total_candidates"] == 18
    assert task["screening_preferences"]["top_k_candidates"] == 6


def test_task_parser_keeps_explicit_material_and_candidate_count() -> None:
    parser = TaskParser()
    task = parser.parse_instruction(
        "请为机翼下蒙皮壁板设计一个 T 形加筋方案，压缩 900 kN/m，边界 SSCC，"
        "材料 IM7/8552，生成 12 个候选"
    )
    assert task["material_system"]["name"] == "IM7/8552"
    assert task["material_system"]["is_user_specified"] is True
    assert task["candidate_generation_preferences"]["total_candidates"] == 12
