"""stiffener_profile 模块单元测试。"""
from __future__ import annotations

import pytest

from core.stiffener_profile import (
    ALL_GEOMETRY_PARAMS,
    CANONICAL_GEOMETRY_ORDER,
    REQUIRED_GEOMETRY_PARAMS,
    STIFFENER_TYPES,
    TYPE_DISPLAY_NAMES,
    build_stiffener_meshes,
    build_stiffener_part_specs,
    default_geometry,
    describe_geometry_text,
    geometry_to_feature_vector,
    hat_incline_angle_deg,
    load_param_ranges_for_type,
    normalize_geometry,
    required_geometry_params,
    resolve_stiffener_type,
    stiffener_footprints,
    rule_check_param_keys,
    stiffener_positions,
    validate_stiffener_type,
)


@pytest.mark.parametrize("value", ["plate", "flat", "steel", "invalid", "T/unknown", "unknown_xyz", 3])
def test_explicit_unknown_type_does_not_match_single_letter_alias(value):
    with pytest.raises(ValueError):
        validate_stiffener_type(value)
    with pytest.raises(ValueError):
        resolve_stiffener_type(value)


def test_unspecified_natural_language_uses_default_but_structured_type_is_strict():
    from core.task_contract import normalize_task_payload
    from core.task_parser import TaskParser

    task = TaskParser().parse_instruction("压缩120kN/m，生成1个候选，初筛1个")
    assert task["task"]["stiffener_type"] == "T"
    assert "stiffener_type" not in task["task"]["user_input_facts"]["explicit_fields"]
    with pytest.raises(ValueError):
        normalize_task_payload({"stiffener_type": "plate"})


@pytest.mark.parametrize("stype,expected", [("BLADE", (59, 61)), ("T", (52, 68)),
                                           ("L", (59, 76)), ("HAT", (30, 90))])
def test_section_footprint_includes_single_sided_flange_and_hat_outer_feet(stype, expected):
    footprints = stiffener_footprints(stype, default_geometry(stype))
    assert len(footprints) == 5
    assert footprints[0] == pytest.approx(expected)
    assert footprints[-1] == pytest.approx(tuple(value + 480 for value in expected))


@pytest.mark.parametrize("stype,overrides", [
    ("L", {"panel_width_mm": 300, "pitch_mm": 200, "flange_width_mm": 60}),
    ("HAT", {"panel_width_mm": 449, "pitch_mm": 100, "flange_width_mm": 80, "cap_width_mm": 40}),
    ("BLADE", {"panel_width_mm": 224.9, "pitch_mm": 50, "web_thickness_mm": 60}),
    ("HAT", {"flange_width_mm": 20, "cap_width_mm": 20}),
])
def test_unbuildable_section_footprints_are_rejected(stype, overrides):
    with pytest.raises(ValueError):
        stiffener_footprints(stype, {**default_geometry(stype), **overrides})


@pytest.mark.parametrize("value", [None, "bad", 0, -1, float("nan"), float("inf")])
def test_explicit_invalid_geometry_does_not_become_default(value):
    with pytest.raises(ValueError):
        normalize_geometry("T", {"skin_thickness_mm": value})


class TestStiffenerTypeValidation:
    def test_validate_uppercase(self):
        assert validate_stiffener_type("T") == "T"
        assert validate_stiffener_type("BLADE") == "BLADE"
        assert validate_stiffener_type("HAT") == "HAT"
        assert validate_stiffener_type("L") == "L"

    def test_validate_chinese(self):
        assert validate_stiffener_type("T型") == "T"
        assert validate_stiffener_type("帽型") == "HAT"
        assert validate_stiffener_type("帽加筋") == "HAT"
        assert validate_stiffener_type("板式") == "BLADE"
        assert validate_stiffener_type("角材") == "L"

    def test_validate_default_fallback(self):
        assert resolve_stiffener_type(None) == "T"
        assert resolve_stiffener_type("") == "T"
        with pytest.raises(ValueError):
            resolve_stiffener_type("unknown_xyz")

    def test_validate_raises_on_unknown(self):
        with pytest.raises(ValueError):
            validate_stiffener_type("unknown_shape")


class TestParameterManagement:
    def test_required_params_per_type(self):
        assert len(REQUIRED_GEOMETRY_PARAMS["BLADE"]) == 6
        assert len(REQUIRED_GEOMETRY_PARAMS["T"]) == 8
        assert len(REQUIRED_GEOMETRY_PARAMS["HAT"]) == 10
        assert len(REQUIRED_GEOMETRY_PARAMS["L"]) == 8

    def test_blade_params_no_flange(self):
        params = required_geometry_params("BLADE")
        assert "flange_width_mm" not in params
        assert "flange_thickness_mm" not in params

    def test_hat_params_include_cap(self):
        params = required_geometry_params("HAT")
        assert "cap_width_mm" in params
        assert "cap_thickness_mm" in params

    def test_default_geometry_is_complete(self):
        for stype in STIFFENER_TYPES:
            defaults = default_geometry(stype)
            required = required_geometry_params(stype)
            for key in required:
                assert key in defaults, f"{stype} missing {key}"

    def test_normalize_geometry_fills_defaults(self):
        geom = normalize_geometry("T", {"panel_length_mm": 800.0})
        assert geom["panel_length_mm"] == 800.0
        assert geom["panel_width_mm"] == 600.0  # From default

    def test_rule_check_keys_match_required_params(self):
        for stype in STIFFENER_TYPES:
            assert rule_check_param_keys(stype) == required_geometry_params(stype)


class TestFeatureVector:
    def test_canonical_order_covers_all_geometry(self):
        assert CANONICAL_GEOMETRY_ORDER == ALL_GEOMETRY_PARAMS

    def test_blade_feature_vector_zero_flange(self):
        geom = default_geometry("BLADE")
        features = geometry_to_feature_vector(geom)
        assert len(features) == 10
        assert features[0] == 700.0  # panel_length
        assert features[6] == 0.0  # flange_width → 0
        assert features[7] == 0.0  # flange_thickness → 0

    def test_t_feature_vector_has_flange(self):
        geom = default_geometry("T")
        features = geometry_to_feature_vector(geom)
        assert features[6] == 16.0  # flange_width
        assert features[7] == 2.0  # flange_thickness

    def test_hat_feature_vector_includes_cap(self):
        geometry = default_geometry("HAT")
        features = dict(zip(CANONICAL_GEOMETRY_ORDER, geometry_to_feature_vector(geometry)))
        assert features["cap_width_mm"] == geometry["cap_width_mm"]
        assert features["cap_thickness_mm"] == geometry["cap_thickness_mm"]

    def test_all_geometry_params_superset(self):
        assert len(ALL_GEOMETRY_PARAMS) == 10


class TestStiffenerPositions:
    def test_single_stiffener(self):
        positions = stiffener_positions(600.0, 800.0)
        assert positions == [300.0]

    def test_multiple_stiffeners(self):
        positions = stiffener_positions(600.0, 120.0)
        assert len(positions) == 5
        assert positions[0] == 60.0
        assert positions[-1] == 540.0

    def test_zero_pitch(self):
        with pytest.raises(ValueError):
            stiffener_positions(600.0, 0.0)

    def test_half_count_uses_the_same_rule_in_solver_and_display(self):
        assert stiffener_positions(600.0, 240.0) == [60.0, 300.0, 540.0]


class TestParamRanges:
    def test_hat_ranges_include_cap(self):
        ranges = load_param_ranges_for_type("HAT")
        assert "cap_width_mm" in ranges
        assert "cap_thickness_mm" in ranges
        assert ranges["cap_width_mm"]["min"] == 12.0
        assert ranges["flange_width_mm"]["min"] == 24.0  # HAT wider flange

    def test_blade_ranges_no_flange(self):
        ranges = load_param_ranges_for_type("BLADE")
        assert "flange_width_mm" not in ranges
        assert "panel_length_mm" in ranges

    def test_all_common_params_present(self):
        for stype in STIFFENER_TYPES:
            ranges = load_param_ranges_for_type(stype)
            assert "panel_length_mm" in ranges
            assert "stiffener_height_mm" in ranges


class TestGeometryDescription:
    def test_blade_description_no_flange(self):
        desc = describe_geometry_text("BLADE", default_geometry("BLADE"))
        assert "L=" in desc
        assert "b_flange" not in desc

    def test_hat_description_includes_cap(self):
        desc = describe_geometry_text("HAT", default_geometry("HAT"))
        assert "cap_w=" in desc
        assert "cap_t=" in desc

    def test_t_description(self):
        desc = describe_geometry_text("T", default_geometry("T"))
        assert "b_flange" in desc
        assert "t_flange" in desc


class TestPartSpecs:
    def test_blade_part_specs(self):
        geom = default_geometry("BLADE")
        specs = build_stiffener_part_specs("BLADE", geom, [300.0])
        assert len(specs) == 1
        assert specs[0]["part_type"] == "web"

    def test_t_part_specs(self):
        geom = default_geometry("T")
        specs = build_stiffener_part_specs("T", geom, [300.0])
        assert len(specs) == 3  # web + left flange + right flange

    def test_hat_part_specs(self):
        geom = default_geometry("HAT")
        specs = build_stiffener_part_specs("HAT", geom, [300.0])
        types = [s["part_type"] for s in specs]
        assert "web_left" in types
        assert "web_right" in types
        assert "cap" in types
        assert "flange_half" in types
        flanges = {s["side"]: s for s in specs if s["part_type"] == "flange_half"}
        assert flanges["left"]["y_offset_mm"] == pytest.approx(270.0)
        assert flanges["left"]["width_mm"] == pytest.approx(10.0)
        assert flanges["right"]["y_offset_mm"] == pytest.approx(320.0)
        assert flanges["right"]["width_mm"] == pytest.approx(10.0)

    def test_l_part_specs(self):
        geom = default_geometry("L")
        specs = build_stiffener_part_specs("L", geom, [300.0])
        assert len(specs) == 2  # web + 1 flange (single side)

    def test_multiple_positions(self):
        geom = default_geometry("T")
        specs = build_stiffener_part_specs("T", geom, [100.0, 300.0, 500.0])
        assert len(specs) == 9  # 3 positions × 3 parts each


class TestRenderMeshes:
    def test_hat_render_mesh_connects_flange_web_and_cap(self):
        geom = default_geometry("HAT")
        meshes = build_stiffener_meshes("HAT", geom, [300.0])
        by_name = {style["name"]: mesh for mesh, style in meshes}

        assert by_name["flange_L_1"].bounds[2:4] == pytest.approx((270.0, 280.0))
        assert by_name["web_L_1"].bounds[2:4] == pytest.approx((280.0, 290.0))
        assert by_name["cap_1"].bounds[2:4] == pytest.approx((290.0, 310.0))
        assert by_name["web_R_1"].bounds[2:4] == pytest.approx((310.0, 320.0))
        assert by_name["flange_R_1"].bounds[2:4] == pytest.approx((320.0, 330.0))


class TestHatInclineAngle:
    def test_angle_calculation(self):
        angle = hat_incline_angle_deg(40.0, 20.0, 28.0)
        assert 60.0 < angle < 75.0  # atan2(28, 10) ≈ 70.3°

    def test_angle_zero_diff(self):
        angle = hat_incline_angle_deg(20.0, 20.0, 28.0)
        assert angle == 0.0


class TestDisplayNames:
    def test_all_types_have_display_names(self):
        for stype in STIFFENER_TYPES:
            assert stype in TYPE_DISPLAY_NAMES
            assert isinstance(TYPE_DISPLAY_NAMES[stype], str)
            assert len(TYPE_DISPLAY_NAMES[stype]) > 0
