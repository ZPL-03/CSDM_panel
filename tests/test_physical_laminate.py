"""逐层积分和 Abaqus 截面调用的离线物理回归，不启动求解器。"""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from abaqus.runtime_build_panel import (
    apply_panel_boundary_and_load, assign_material_orientation, create_composite_sections,
    edge_nodal_forces, resolve_analysis_spec,
)
from core.laminate import analysis_contract, areal_density, laminate_abd, lamina_material, parse_angles, section_layers
from tests.test_fem_agent import _real_fem_candidate


def test_single_zero_ply_matches_closed_form():
    material = lamina_material(_real_fem_candidate("C1"))
    e1, e2 = material["E1_GPa"] * 1000, material["E2_GPa"] * 1000
    nu, g = material["nu12"], material["G12_GPa"] * 1000
    d = 1 - nu**2 * e2 / e1
    q = np.array([[e1 / d, nu * e2 / d, 0], [nu * e2 / d, e2 / d, 0], [0, 0, g]])
    a, b, bending = laminate_abd([{"angle_deg": 0, "thickness_mm": 2}], material)
    np.testing.assert_allclose(a, q * 2)
    np.testing.assert_allclose(b, 0, atol=1e-10)
    np.testing.assert_allclose(bending, q * 2**3 / 12)


def test_equal_fractions_different_order_change_bending_and_reverse_changes_b():
    material = lamina_material(_real_fem_candidate("C1"))
    def abd(text):
        return np.asarray(laminate_abd([{"angle_deg": a, "thickness_mm": 0.25} for a in parse_angles(text)], material))
    outer0, outer90 = abd("[0/90]s"), abd("[90/0]s")
    np.testing.assert_allclose(outer0[0], outer90[0], atol=1e-9)
    np.testing.assert_allclose(outer0[1], 0, atol=1e-9)
    assert outer0[2, 0, 0] > outer90[2, 0, 0]
    np.testing.assert_allclose(abd("[0/90]")[1], -abd("[90/0]")[1], atol=1e-9)
    balanced_symmetric = abd("[45/-45/0/90]s")
    assert abs(balanced_symmetric[0, 0, 2]) < 1e-9
    assert abs(balanced_symmetric[2, 0, 2]) > 1


@pytest.mark.parametrize("stype", ["T", "BLADE", "L", "HAT"])
def test_all_component_sections_use_physical_angles_thickness_and_direction(monkeypatch, stype):
    candidate = _real_fem_candidate("C1", stype)
    spec = resolve_analysis_spec(candidate)
    constants = {name: name for name in ("DEFAULT", "GRADIENT", "OFF", "SIMPSON", "NO_IDEALIZATION",
                                        "AXIS_3", "CARTESIAN", "ROTATION_NONE", "STACK_3", "SYSTEM")}
    monkeypatch.setitem(sys.modules, "abaqusConstants", SimpleNamespace(**constants))
    monkeypatch.setitem(sys.modules, "section", SimpleNamespace(SectionLayer=lambda **kwargs: kwargs))
    monkeypatch.setitem(sys.modules, "regionToolset", SimpleNamespace(Region=lambda **kwargs: kwargs))
    sections = {}
    model = SimpleNamespace(CompositeShellSection=lambda **kwargs: sections.update({kwargs["name"]: kwargs}))
    create_composite_sections(model, spec, "MATERIAL")
    layers = section_layers(candidate)
    assert set(sections) == {name.capitalize() + "Section" for name in layers}
    for name, plies in layers.items():
        actual = sections[name.capitalize() + "Section"]
        assert actual["symmetric"] == "OFF"
        assert [p["orientAngle"] for p in actual["layup"]] == parse_angles(candidate["layup"]["skin_layup"])
        assert sum(p["thickness"] for p in actual["layup"]) == pytest.approx(sum(p["thickness_mm"] for p in plies))
        assert all(p["material"] == "MATERIAL" for p in actual["layup"])
    orientation, axes = {}, {}
    part = SimpleNamespace(faces="faces", datums={1: "datum"},
                           DatumCsysByThreePoints=lambda **kw: (axes.update(kw) or SimpleNamespace(id=1)),
                           MaterialOrientation=lambda **kw: orientation.update(kw))
    assign_material_orientation(part)
    assert axes["point1"] == (1, 0, 0) and axes["point2"] == (0, 1, 0)
    assert orientation["localCsys"] == "datum" and orientation["stackDirection"] == "STACK_3"
    assert orientation["axis"] == "AXIS_3" and orientation["angle"] == 0


def test_component_override_and_thickness_conservation():
    candidate = _real_fem_candidate("C1", "HAT")
    candidate["layup"]["web_layup"] = "[0/90]s"
    candidate["layup"]["web_ply_thicknesses_mm"] = [0.25, 0.75, 0.75, 0.25]
    assert len(section_layers(candidate)["web"]) == 4
    original = analysis_contract(candidate)["fingerprint"]
    candidate["layup"]["web_layup"] = "[90/0]s"
    assert analysis_contract(candidate)["fingerprint"] != original
    candidate["layup"]["web_ply_thicknesses_mm"][0] = 0.5
    with pytest.raises(ValueError, match="厚度之和"):
        section_layers(candidate)


@pytest.mark.parametrize("text", ["", "[0/nan]", "[0/30]s", "garbage"])
def test_runtime_does_not_invent_layers(text):
    candidate = _real_fem_candidate("C1")
    candidate["layup"]["skin_layup"] = text
    with pytest.raises(ValueError):
        resolve_analysis_spec(candidate)


@pytest.mark.parametrize("stype", ["T", "BLADE", "L", "HAT"])
def test_areal_density_matches_model_part_areas(stype):
    from core.stiffener_profile import stiffener_positions
    from abaqus.runtime_extract_blf import estimate_weight_kg_per_m2
    from agents.screener import ScreenerAgent

    candidate = _real_fem_candidate("C1", stype)
    g = candidate["geometry"]
    width, height, web = g["panel_width_mm"], g["stiffener_height_mm"], g["web_thickness_mm"]
    count = len(stiffener_positions(width, g["pitch_mm"]))
    if stype == "BLADE":
        area = height * web
    elif stype == "HAT":
        foot = (g["flange_width_mm"] - g["cap_width_mm"]) / 2
        area = 2 * np.sqrt(height**2 + foot**2) * web + 2 * foot * g["flange_thickness_mm"] + g["cap_width_mm"] * g["cap_thickness_mm"]
    else:
        area = height * web + g["flange_width_mm"] * g["flange_thickness_mm"]
    expected = 1600 * (g["skin_thickness_mm"] + count * area / width) * 0.001
    assert areal_density(candidate) == pytest.approx(expected)
    assert estimate_weight_kg_per_m2(candidate) == pytest.approx(expected)
    assert ScreenerAgent()._estimate_weight(candidate) == pytest.approx(expected)


def rectangular_edge_nodes():
    """非均匀节点间距用于检查端点半权重与角点载荷叠加。"""
    nodes = {}
    for x in (0, 4, 13):
        for y in (0, 2, 5, 7):
            label = len(nodes) + 1
            nodes[label] = SimpleNamespace(label=label, coordinates=(x, y, 0))
    edges = {name: [node for node in nodes.values() if node.coordinates[axis] == value]
             for name, axis, value in (("X0", 0, 0), ("X1", 0, 13), ("Y0", 1, 0), ("Y1", 1, 7))}
    return nodes, edges


@pytest.mark.parametrize("nx,nxy", [(120, 0), (0, 40), (120, 40)])
def test_edge_tractions_preserve_force_moment_and_affine_work(nx, nxy):
    nodes, edges = rectangular_edge_nodes()
    forces = edge_nodal_forces(edges, nx, nxy)
    assert sum(value[0] for value in forces.values()) == pytest.approx(0, abs=1e-10)
    assert sum(value[1] for value in forces.values()) == pytest.approx(0, abs=1e-10)
    moment, work = 0.0, 0.0
    for label, (fx, fy) in forces.items():
        x, y, _ = nodes[label].coordinates
        moment += x * fy - y * fx
        work += fx * (0.002 * x + 0.004 * y) + fy * (0.001 * x - 0.003 * y)
    assert moment == pytest.approx(0, abs=1e-10)
    assert work == pytest.approx(13 * 7 * (-nx * 0.002 + nxy * 0.005))


@pytest.mark.parametrize("boundary", ["SSSS", "SSCC", "CCCC"])
def test_edge_supports_leave_membrane_deformation_free(monkeypatch, boundary):
    nodes, edges = rectangular_edge_nodes()
    monkeypatch.setitem(sys.modules, "regionToolset", SimpleNamespace(Region=lambda **kwargs: kwargs))
    monkeypatch.setitem(sys.modules, "abaqusConstants", SimpleNamespace(UNIFORM="UNIFORM"))
    constraints, loads = {}, []
    model = SimpleNamespace(DisplacementBC=lambda **kw: constraints.update({kw["name"]: kw}),
                            BuckleStep=lambda **kw: None, ConcentratedForce=lambda **kw: loads.append(kw))
    assembly = SimpleNamespace(Set=lambda **kw: kw)
    skin = SimpleNamespace(nodes=SimpleNamespace(sequenceFromLabels=lambda labels: [nodes[label] for label in labels]))
    candidate = _real_fem_candidate("C1")
    candidate["boundary_conditions"] = {"type": boundary}
    spec = resolve_analysis_spec(candidate)
    apply_panel_boundary_and_load(model, assembly, skin, spec, edges)
    assert len(constraints) == 6
    for edge in edges:
        actual = constraints["BC_" + edge]
        assert actual["u3"] == 0 and "u1" not in actual and "u2" not in actual
        clamped = boundary == "CCCC" or (boundary == "SSCC" and edge.startswith("Y"))
        assert ("ur1" in actual) == clamped and ("ur2" in actual) == clamped
        assert "ur3" not in actual
    assert constraints["BC_Anchor_Origin"]["u1"] == constraints["BC_Anchor_Origin"]["u2"] == 0
    assert "u1" not in constraints["BC_Anchor_Axis"] and constraints["BC_Anchor_Axis"]["u2"] == 0
    assert loads


def test_empty_load_edge_is_a_model_error():
    _, edges = rectangular_edge_nodes()
    edges["X0"] = []
    with pytest.raises(ValueError):
        edge_nodal_forces(edges, 120, 0)


@pytest.mark.parametrize("analysis", [{"mesh_size_mm": 0}, {"mesh_size_mm": float("nan")},
                                      {"buckling_modes": 2.5}, {"buckling_modes": -1}])
def test_invalid_mesh_or_mode_request_is_rejected(analysis):
    candidate = _real_fem_candidate("C1")
    candidate["analysis"] = analysis
    with pytest.raises(ValueError):
        resolve_analysis_spec(candidate)


def test_invalid_hat_span_is_rejected_before_building():
    candidate = _real_fem_candidate("C1", "HAT")
    candidate["geometry"]["cap_width_mm"] = candidate["geometry"]["flange_width_mm"]
    with pytest.raises(ValueError):
        resolve_analysis_spec(candidate)
