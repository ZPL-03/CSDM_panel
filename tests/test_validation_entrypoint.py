"""验证入口的输入检查与失败终态，使用独立测试数据。"""

import sys
import math
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.io_utils import read_json
from core.laminate import section_layers, lamina_material
from scripts import validate_physical_workflow as entry
from tests.test_fem_agent import _real_fem_candidate


def fixture_deck(candidate):
    material = lamina_material(candidate)
    sections = section_layers(candidate)
    names = {"skin": "SkinPanel", "web": "WebPart", "flange": "FlangeHalfPart", "cap": "CapPart"}
    lines = []
    for component, layers in sections.items():
        lines += [f"*Part, name={names[component]}", "*Orientation, name=Ori-1", "1,0,0,0,1,0", "3,0",
                  "*Shell Section, elset=E, composite, orientation=Ori-1"]
        lines += [f"{p['thickness_mm']},3,LAMINA_MATERIAL,{p['angle_deg']}" for p in layers]
        lines += ["*End Part"]
    lines += ["*Material, name=LAMINA_MATERIAL", "*Elastic, type=LAMINA", ",".join(str(material[key] * (1 if key == "nu12" else 1000))
              for key in ("E1_GPa", "E2_GPa", "nu12", "G12_GPa", "G13_GPa", "G23_GPa"))]
    return "\n".join(lines)


@pytest.mark.parametrize("stype", ["T", "BLADE", "L", "HAT"])
@pytest.mark.parametrize("comment_encoding", ["utf-8", "gbk"])
def test_inp_checks_bind_part_layers_material_and_orientation(tmp_path, stype, comment_encoding):
    candidate = _real_fem_candidate("C1", stype)
    path = tmp_path / "C1.inp"
    text = fixture_deck(candidate)
    path.write_bytes("** 边界类型：位移/转角\n".encode(comment_encoding) + text.encode("ascii"))
    evidence = entry.inspect_input_deck(path, candidate)
    assert len(evidence["composite_sections"]) == len(section_layers(candidate))
    for wrong in (text.replace("composite", "homogeneous"), text.replace("1,0,0,0,1,0", "0,1,0,1,0,0"),
                  text.replace("orientation=Ori-1", "orientation=Missing"), text.replace("3,0\n", "3,90\n"),
                  text.replace(",3,LAMINA_MATERIAL,45", ",3,LAMINA_MATERIAL,0"), text.replace("181000", "100000")):
        path.write_text(wrong, encoding="utf-8")
        with pytest.raises(ValueError):
            entry.inspect_input_deck(path, candidate)


def test_validation_entrypoint_records_failure_terminal_state(monkeypatch, tmp_path):
    output = tmp_path / "isolated"
    def setup(target):
        target.mkdir()
        (target / "models").mkdir()
    monkeypatch.setattr(entry, "isolate_runtime", setup)
    def fail(_types):
        raise RuntimeError("fixture failure")
    monkeypatch.setattr(entry, "run_layup_comparison", fail)
    monkeypatch.setattr(sys, "argv", ["validate_physical_workflow.py", "--mode", "layup", "--output-dir", str(output)])
    assert entry.main() == 1
    report = read_json(output / "acceptance.json")
    assert report["status"] == "failed"
    assert report["finished_at"] and report["code_sha256"]
    assert all("\\" not in name for name in report["code_sha256"])
    assert "abaqus/runtime_build_panel.py" in report["code_sha256"]
    assert report["model_sha256"] == {}
    assert report["error_type"] == "RuntimeError"


def test_api_acceptance_requires_reports_and_records_effective_request(monkeypatch, tmp_path):
    import agents.orchestrator
    from core.llm_backend import LLMBackend

    candidate = _real_fem_candidate("C1", "T")
    candidate["source"] = "DOE"
    deck = tmp_path / "C1.inp"
    deck.write_text(fixture_deck(candidate), encoding="utf-8")
    reports = {name: str(tmp_path / name) for name in ("markdown_path", "pdf_path", "facts_path")}
    for path in reports.values():
        Path(path).write_text("test artifact", encoding="utf-8")
    from core.io_utils import write_json
    write_json(Path(reports["facts_path"]), {"knowledge_context": {"included_chunk_ids": ["chunk_1"]}})
    response = {"task": {"task": {"requirement_extraction": {"method": "llm_grounded"}}},
                "candidates": [candidate], "top_candidates": [candidate],
                "results": [{"status": "success", "abaqus_inp": str(deck)}],
                "report": {**reports, "llm_explanation_used": True}}
    backend = object.__new__(LLMBackend)
    backend.model, backend.max_tokens, backend.temperature = "qwen-test", 1800, 0.2
    backend.extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
    backend.api_key = "test-secret-must-not-be-recorded"
    def reply(self, *args, **kwargs):
        self.last_response = {"finish_reason": "stop", "content": "{}", "usage": {"completion_tokens": 2}}
        return "{}"
    monkeypatch.setattr(LLMBackend, "chat", reply)
    def run(instruction):
        backend.chat("system", instruction, max_tokens_override=4096)
        return response
    fake_agent = SimpleNamespace(report_gen=SimpleNamespace(llm_backend=backend),
                                 knowledge_agent=SimpleNamespace(config={"pipeline": {}}), run=run)
    monkeypatch.setattr(agents.orchestrator, "OrchestratorAgent", lambda: fake_agent)
    args = SimpleNamespace(output_dir=tmp_path, instruction="task", require_llm_explanation=True)
    result = entry.run_api_workflow(args)
    assert result["passed"] and result["reports_complete"]
    evidence = read_json(tmp_path / "api_calls.json")
    assert evidence[0]["max_tokens"] == 4096
    assert evidence[0]["extra_body"] == backend.extra_body
    assert evidence[0]["response_metadata"]["usage"]["completion_tokens"] == 2
    assert backend.api_key not in (tmp_path / "api_calls.json").read_text(encoding="utf-8")
    response["report"]["pdf_path"] = None
    result = entry.run_api_workflow(args)
    assert result["passed"] is False and result["reports_complete"] is False


def mechanics_candidate(boundary="SSSS", load_type="compression_shear", stype="T"):
    candidate = _real_fem_candidate("C1", stype)
    candidate["boundary_conditions"]["type"] = boundary
    candidate["load_conditions"] = {"type": load_type, "Nx_kN_per_m": 0.0 if load_type == "in_plane_shear" else 120.0,
                                     "Nxy_kN_per_m": 0.0 if load_type == "axial_compression" else 60.0}
    return candidate


def mechanics_fixture(candidate):
    """九节点矩形的解析载荷，边段比例为 3:7 和 4:6，不调用生产载荷函数。"""
    length, width = [candidate["geometry"][key] for key in ("panel_length_mm", "panel_width_mm")]
    nx, shear = [candidate["load_conditions"][key] for key in ("Nx_kN_per_m", "Nxy_kN_per_m")]
    nodes = [f"{j * 3 + i + 1},{x},{y},0" for j, y in enumerate((0, 0.4 * width, width))
             for i, x in enumerate((0, 0.3 * length, length))]
    text = fixture_deck(candidate).replace("*Part, name=SkinPanel", "*Part, name=SkinPanel\n*Node\n" + "\n".join(nodes)
                + "\n*Element, type=S4R, elset=SkinElements\n1,1,2,5,4\n2,2,3,6,5\n3,4,5,8,7\n4,5,6,9,8")
    assembly = ["*Assembly, name=Assembly", "*Instance, name=Skin-1, part=SkinPanel", "*End Instance"]
    for edge, ids in {"X0": "1,4,7", "X1": "3,6,9", "Y0": "1,2,3", "Y1": "7,8,9"}.items():
        assembly += [f"*Nset, nset=Edge_{edge}, instance=Skin-1", ids]
    assembly += ["*Nset, nset=Anchor_Origin, instance=Skin-1", "1", "*Nset, nset=Anchor_Axis, instance=Skin-1", "3"]
    loads = {
        1: (nx * width * .2 - shear * length * .15, -shear * width * .2),
        2: (-shear * length * .5, 0),
        3: (-nx * width * .2 - shear * length * .35, shear * width * .2),
        4: (nx * width * .5, -shear * width * .5),
        6: (-nx * width * .5, shear * width * .5),
        7: (nx * width * .3 + shear * length * .15, -shear * width * .3),
        8: (shear * length * .5, 0),
        9: (-nx * width * .3 + shear * length * .35, shear * width * .3),
    }
    for node in loads:
        assembly += [f"*Nset, nset=Load_{node}, instance=Skin-1", str(node)]
    assembly += ["*End Assembly"]
    boundaries = ["Anchor_Origin,1,2", "Anchor_Axis,2,2"]
    clamped = {"SSSS": [], "SSCC": ["Y0", "Y1"], "CCCC": ["X0", "X1", "Y0", "Y1"]}[candidate["boundary_conditions"]["type"]]
    for edge in ("X0", "X1", "Y0", "Y1"):
        boundaries.append(f"Edge_{edge},3,3")
        if edge in clamped:
            boundaries.append(f"Edge_{edge},4,5")
    analysis = ["*Boundary", *boundaries, "*Step, name=Buckling", "*Buckle", "8, , 16, 200"]
    for load_case in (1, 2):
        analysis += [f"*Boundary, op=NEW, load case={load_case}", *boundaries]
    analysis += ["*Cload", *[f"Load_{node}, {dof}, {force:.12g}" for node, forces in loads.items() for dof, force in enumerate(forces, 1)], "*End Step"]
    return text + "\n" + "\n".join(assembly + analysis) + "\n"


@pytest.mark.parametrize("boundary", ["SSSS", "SSCC", "CCCC"])
@pytest.mark.parametrize("load_type", ["axial_compression", "in_plane_shear", "compression_shear"])
def test_mechanics_checks_nonuniform_weights_corners_and_both_load_cases(tmp_path, boundary, load_type):
    candidate = mechanics_candidate(boundary, load_type)
    path = tmp_path / "input.inp"
    path.write_text(mechanics_fixture(candidate), encoding="ascii")
    report = entry.inspect_mechanics_deck(path, candidate)
    assert report["passed"]
    assert report["total_force_N"] == pytest.approx([0, 0, 0], abs=1e-8)
    assert report["total_moment_Nmm"] == pytest.approx([0, 0, 0], abs=1e-8)
    expected_count = {"SSSS": 11, "SSCC": 23, "CCCC": 27}[boundary]
    assert all(item["count"] == expected_count for item in report["constraints"].values())
    assert report["edge_node_counts"] == dict.fromkeys(("X0", "X1", "Y0", "Y1"), 3)


def test_zero_resultant_does_not_accept_wrong_nodal_distribution(tmp_path):
    candidate = mechanics_candidate(load_type="axial_compression")
    text = mechanics_fixture(candidate).replace("Load_4, 1, 14400", "Load_4, 1, 14410").replace("Load_6, 1, -14400", "Load_6, 1, -14410")
    path = tmp_path / "wrong.inp"
    path.write_text(text, encoding="ascii")
    report = entry.inspect_mechanics_deck(path, candidate)
    assert report["passed"] is False
    assert len(report["nodal_force_mismatches"]) == 2
    assert report["total_force_N"] == pytest.approx([0, 0, 0])
    assert report["total_moment_Nmm"] == pytest.approx([0, 0, 0])


@pytest.mark.parametrize("mutation", ["missing_shear", "extra_in_plane", "missing_anchor", "drilling", "wrong_set", "load_case_2"])
def test_mechanics_rejects_load_and_support_errors(tmp_path, mutation):
    candidate = mechanics_candidate(load_type="in_plane_shear")
    text = mechanics_fixture(candidate)
    if mutation == "missing_shear":
        text = text.replace("Load_2, 1, -10800", "Load_2, 1, 0")
    elif mutation == "extra_in_plane":
        text = text.replace("Edge_Y0,3,3", "Edge_Y0,1,3")
    elif mutation == "missing_anchor":
        text = text.replace("Anchor_Axis,2,2\n", "")
    elif mutation == "drilling":
        text = text.replace("Edge_X0,3,3", "Edge_X0,3,6")
    elif mutation == "wrong_set":
        text = text.replace("nset=Edge_X0, instance=Skin-1\n1,4,7", "nset=Edge_X0, instance=Skin-1\n1,4,5,7")
    else:
        text = text.replace("*Boundary, op=NEW, load case=2\nAnchor_Origin,1,2", "*Boundary, op=NEW, load case=2\nAnchor_Origin,1,3\nSkin-1.5,1,1")
    path = tmp_path / "wrong.inp"
    path.write_text(text, encoding="ascii")
    assert entry.inspect_mechanics_deck(path, candidate)["passed"] is False


def test_inp_resolves_instance_transforms_and_generated_sets(tmp_path):
    text = mechanics_fixture(mechanics_candidate()).replace("*Instance, name=Skin-1, part=SkinPanel\n", "*Instance, name=Skin-1, part=SkinPanel\n5,6,7\n0,0,0,0,0,1,90\n")
    text = text.replace("*Nset, nset=Edge_Y0, instance=Skin-1\n1,2,3", "*Nset, nset=Edge_Y0, instance=Skin-1, generate\n1,3,1")
    path = tmp_path / "transformed.inp"
    path.write_text(text, encoding="ascii")
    deck = entry.MechanicsDeck(path)
    assert deck.nodes["SKIN-1", 1] == pytest.approx([-6, 5, 7])
    assert deck.resolve("EDGE_Y0") == {("SKIN-1", i) for i in (1, 2, 3)}
    assert len(deck.resolve("SKIN-1.SKINELEMENTS", kind="element")) == 4


def connected_fixture(candidate):
    """用单元中面和装配变换独立构造四筋型连接测试模型。"""
    candidate["geometry"]["pitch_mm"] = candidate["geometry"]["panel_width_mm"]
    geometry, stype = candidate["geometry"], candidate["stiffener_type"]
    length, height, center = geometry["panel_length_mm"], geometry["stiffener_height_mm"], geometry["panel_width_mm"] / 2
    flange_z = (geometry["skin_thickness_mm"] + geometry.get("flange_thickness_mm", 0)) / 2 if stype != "BLADE" else 0
    flange_width = geometry.get("flange_width_mm", 0)
    foot = (flange_width - geometry.get("cap_width_mm", 0)) / 2 if stype == "HAT" else 0
    dimensions = {"WebPart": math.hypot(height, foot) if stype == "HAT" else height}
    if stype != "BLADE":
        dimensions["FlangeHalfPart"] = foot if stype == "HAT" else flange_width / 2 if stype == "T" else flange_width
    if stype == "HAT":
        dimensions["CapPart"] = geometry["cap_width_mm"]
    text = mechanics_fixture(candidate)
    for name, span in dimensions.items():
        mesh = f"*Part, name={name}\n*Node\n1,0,0,0\n2,{length},0,0\n3,{length},{span},0\n4,0,{span},0\n*Element, type=S4R, elset=Face\n1,1,2,3,4"
        text = text.replace("*Part, name=" + name, mesh)
    assembly = []
    def instance(name, part, y, z, angle=None):
        assembly.extend([f"*Instance, name={name}, part={part}", f"0,{y},{z}"])
        if angle is not None:
            assembly.append(f"0,{y},{z},1,{y},{z},{angle}")
        assembly.append("*End Instance")
    def surface(name, members):
        assembly.extend([f"*Surface, type=ELEMENT, name={name}", *[f"{target},{side}" for target, side in members]])
    def tie(name, secondary, primary, gap=0):
        assembly.extend([f"*Tie, name={name}, adjust=no, no thickness, position tolerance={gap + .001:.12g}", f"{secondary},{primary}"])
    surface("SkinSurface", [("Skin-1.SkinElements", "SPOS")])
    if stype == "BLADE":
        instance("Web-01", "WebPart", center, 0, 90)
        surface("WebBase", [("Web-01.1", "E1")])
        tie("WebSkin", "WebBase", "SkinSurface")
    elif stype in ("T", "L"):
        instance("Web-01", "WebPart", center, flange_z, 90)
        surface("WebBase", [("Web-01.1", "E1")])
        members = []
        flanges = [("FlangeL-01", center - flange_width / 2, "E3"), ("FlangeR-01", center, "E1")] if stype == "T" else [("Flange-01", center, "E1")]
        for name, y, side in flanges:
            instance(name, "FlangeHalfPart", y, flange_z)
            surface(name + "Face", [(name + ".1", "SPOS")])
            tie(name + "Skin", name + "Face", "SkinSurface", flange_z)
            members.append((name + ".1", side))
        surface("FlangeInner", members)
        tie("WebFlange", "WebBase", "FlangeInner")
    else:
        cap = geometry["cap_width_mm"]
        instance("Cap-01", "CapPart", center - cap / 2, flange_z + height)
        angle = math.degrees(math.atan2(height, foot))
        for side, sign, rotation, cap_edge in (("L", -1, angle, "E1"), ("R", 1, 180 - angle, "E3")):
            web, flange = "Web" + side + "-01", "Flange" + side + "-01"
            instance(web, "WebPart", center + sign * flange_width / 2, flange_z, rotation)
            instance(flange, "FlangeHalfPart", center - flange_width / 2 - foot if side == "L" else center + flange_width / 2, flange_z)
            surface(flange + "Face", [(flange + ".1", "SPOS")])
            surface(flange + "Inner", [(flange + ".1", "E3" if side == "L" else "E1")])
            surface(web + "Base", [(web + ".1", "E1")])
            surface(web + "Top", [(web + ".1", "E3")])
            surface("Cap" + side, [("Cap-01.1", cap_edge)])
            tie(flange + "Skin", flange + "Face", "SkinSurface", flange_z)
            tie(web + "Flange", web + "Base", flange + "Inner")
            tie(web + "Cap", web + "Top", "Cap" + side)
    return text.replace("*End Assembly", "\n".join(assembly) + "\n*End Assembly")


@pytest.mark.parametrize("stype", ["T", "BLADE", "L", "HAT"])
def test_connection_geometry_and_tie_coverage_for_all_stiffeners(tmp_path, stype):
    candidate = mechanics_candidate(stype=stype)
    path = tmp_path / "connected.inp"
    path.write_text(connected_fixture(candidate), encoding="ascii")
    result = entry.inspect_connections(path, candidate)
    assert result["passed"], result["errors"]
    assert len(result["ties"]) == {"T": 3, "BLADE": 1, "L": 2, "HAT": 6}[stype]
    assert all(tie["secondary_node_count"] > 0 for tie in result["ties"])
    candidate["geometry"]["stiffener_height_mm"] += 1
    assert entry.inspect_connections(path, candidate)["passed"] is False


@pytest.mark.parametrize("wrong", ["adjust", "thickness", "computed", "oversized", "missing_surface"])
def test_connection_contract_rejects_mesh_dependent_or_missing_ties(tmp_path, wrong):
    candidate = mechanics_candidate()
    text = connected_fixture(candidate)
    replacements = {"adjust": ("adjust=no", "adjust=yes"), "thickness": ("no thickness", ""),
                    "computed": (", position tolerance=0.001", ""),
                    "oversized": ("position tolerance=0.001", "position tolerance=2"),
                    "missing_surface": ("WebBase,FlangeInner", "Missing,FlangeInner")}
    text = text.replace(*replacements[wrong])
    path = tmp_path / "wrong.inp"
    path.write_text(text, encoding="ascii")
    if wrong == "missing_surface":
        with pytest.raises(ValueError):
            entry.inspect_connections(path, candidate)
    else:
        assert entry.inspect_connections(path, candidate)["passed"] is False


def test_static_copy_preserves_model_and_loads_without_changing_source(tmp_path):
    candidate = mechanics_candidate("CCCC")
    source, target = tmp_path / "buckling.inp", tmp_path / "static.inp"
    source.write_text(connected_fixture(candidate), encoding="ascii")
    before = source.read_bytes()
    evidence = entry.derive_static_deck(source, target)
    assert source.read_bytes() == before
    blocks = entry.inp_blocks(target)
    assert sum(block["keyword"] == "*STATIC" for block in blocks) == 1
    assert not any(block["keyword"] == "*BUCKLE" for block in blocks)
    assert target.read_text().split("*Step")[0] == source.read_text().split("*Step")[0]
    assert [block["data"] for block in blocks if block["keyword"] == "*CLOAD"] == [block["data"] for block in entry.inp_blocks(source) if block["keyword"] == "*CLOAD"]
    assert evidence["source_inp_sha256"] == entry.digest(source)


def test_reaction_balance_includes_moments_and_rejects_missing_nodes(tmp_path):
    path = tmp_path / "buckling.inp"
    path.write_text(mechanics_fixture(mechanics_candidate("CCCC")), encoding="ascii")
    nodes = entry.MechanicsDeck(path).nodes
    audit = {"connection_passed": True, "frame_value": 1.0, "RF": [], "RM": []}
    for field in ("RF", "RM"):
        audit[field] = [{"instance": instance, "node": node, "value": [0, 0, 0]} for instance, node in nodes]
    kwargs = {"rtol": 1e-8, "force_atol": 1e-6, "moment_atol": 1e-6}
    assert entry.check_static_reactions(path, audit, **kwargs)["passed"]
    audit["RM"][0]["value"][2] = 100
    assert entry.check_static_reactions(path, audit, **kwargs)["passed"] is False
    audit["RF"] = []
    with pytest.raises(ValueError):
        entry.check_static_reactions(path, audit, **kwargs)


@pytest.mark.parametrize("name,count,fatal", [("WarnNodeMissMainIntersect", 2, True), ("WarnNodeUnconnected", 1, True),
                                               ("ErrNodeMissingDof", 1, True), ("WarnNodeOther", 1, False), ("WarnNodeMissMainIntersect", 0, False)])
def test_odb_reader_flags_connection_warnings_without_solver(monkeypatch, tmp_path, name, count, fatal):
    nodes = [SimpleNamespace(label=i + 1) for i in range(count)]
    owner = SimpleNamespace(nodeSets={name: SimpleNamespace(nodes=(nodes,))}, instances={})
    frame = SimpleNamespace(description="Mode 1: EigenValue = 2.64000E+00")
    closed, calls = [], []
    odb = SimpleNamespace(rootAssembly=owner, steps={"Buckling": SimpleNamespace(frames=[frame])}, close=lambda: closed.append(True))
    def open_odb(**kwargs):
        calls.append(kwargs)
        return odb
    monkeypatch.setitem(sys.modules, "odbAccess", SimpleNamespace(openOdb=open_odb))
    target = tmp_path / "audit.json"
    monkeypatch.setattr(sys, "argv", ["reader.py", "case.odb", str(target), "buckling"])
    exec(compile(entry.ODB_CHECK_SCRIPT, "reader.py", "exec"), {})
    report = read_json(target)
    assert report["connection_passed"] is not fatal
    assert report["warning_node_sets"][0]["node_count"] == count
    assert report["BLF_global"] == 2.64
    assert calls == [{"path": "case.odb", "readOnly": True}] and closed == [True]


def mechanics_args(output, **overrides):
    values = dict(output_dir=output, stiffener_types=["T", "BLADE", "L", "HAT"],
                           mesh_sizes_mm=[18.0, 12.0, 8.0], matrix_stiffener_types=["T"],
                           convergence_rtol=0.05, balance_rtol=1e-5, force_atol_n=1e-5, moment_atol_nmm=1e-3,
                           connection_atol_mm=0.01, nx_kn_per_m=120.0, nxy_kn_per_m=60.0, static_check=False,
                           reuse_run_dir=[], reuse_build_change_reason="")
    return SimpleNamespace(**{**values, **overrides})


def test_plan_shares_fine_t_baseline_and_only_changes_requested_fields(tmp_path):
    args = mechanics_args(tmp_path)
    plan = entry.build_mechanics_plan(args)
    assert len(plan["jobs"]) == 20 and len(plan["matrix"]) == 9
    assert plan["matrix"][0]["job_id"] == plan["convergence_jobs"]["T"][-1]
    by_id = {job["job_id"]: job for job in plan["jobs"]}
    for ids in plan["convergence_jobs"].values():
        candidates = [by_id[cid]["candidate"] for cid in ids]
        assert [candidate["analysis"]["mesh_size_mm"] for candidate in candidates] == args.mesh_sizes_mm
        for key in ("geometry", "layup", "material_system", "load_conditions", "boundary_conditions"):
            assert candidates[0][key] == candidates[1][key] == candidates[2][key]
    args.matrix_stiffener_types = args.stiffener_types
    assert len(entry.build_mechanics_plan(args)["jobs"]) == 44


def test_mechanics_baseline_preserves_recorded_designs_without_sampling(monkeypatch, tmp_path):
    from core.doe_sampler import DOESampler

    args = mechanics_args(tmp_path)
    baseline = entry.build_mechanics_plan(args)
    baseline["status"] = "passed"
    for job in baseline["jobs"]:
        job["status"] = "passed"
    source = tmp_path / "prior"
    source.mkdir()
    entry.save(source / "mechanics.json", baseline)
    before = (source / "mechanics.json").read_bytes()
    args.baseline_run_dir = source
    monkeypatch.setattr(DOESampler, "sample_candidates", lambda *a, **kw: pytest.fail("固定基准不应重新采样"))
    plan = entry.build_mechanics_plan(args)
    assert [job["identity"] for job in plan["jobs"]] == [job["identity"] for job in baseline["jobs"]]
    assert plan["baseline_source"]["sha256"] == entry.digest(source / "mechanics.json")
    assert (source / "mechanics.json").read_bytes() == before


@pytest.mark.parametrize("values,counts,passed", [([1, 1.02, 1.03], [10, 20, 40], True),
                                                   ([1, 1.1, 1.11], [10, 20, 40], True),
                                                   ([1, 1.1, 1.2], [10, 20, 40], False),
                                                   ([1, 1.02, 1.03], [10, 10, 40], False)])
def test_convergence_uses_raw_values_and_actual_mesh_counts(values, counts, passed):
    jobs = [{"BLF_global": value, "mechanics": {"skin_node_count": count}} for value, count in zip(values, counts)]
    report = entry.convergence_check(jobs, 0.05)
    assert report["passed"] is passed
    assert report["relative_changes"][0] == pytest.approx(abs(values[0] - values[1]) / values[1])
    assert report["acceptance_relative_change"] == report["relative_changes"][-1]


def test_mechanics_failure_keeps_terminal_state_and_unexecuted_jobs(monkeypatch, tmp_path):
    from agents.fem_agent import FEMAgent
    calls = []
    def fail(_self, candidate):
        calls.append(candidate["candidate_id"])
        return {"status": "failed", "error_type": "connection_error"}
    monkeypatch.setattr(FEMAgent, "run", fail)
    report = entry.run_mechanics(mechanics_args(tmp_path))
    assert report["status"] == "failed" and report["passed"] is False
    assert calls == ["C1"]
    assert report["jobs"][0]["status"] == "failed"
    assert all(job["status"] == "not_run" for job in report["jobs"][1:])
    assert read_json(tmp_path / "mechanics.json")["finished_at"]


@pytest.mark.parametrize("with_static", [False, True])
def test_mechanics_orchestration_reuses_matching_grid_and_shared_baseline(monkeypatch, tmp_path, with_static):
    from agents.fem_agent import FEMAgent
    from core.laminate import analysis_contract

    args = mechanics_args(tmp_path, static_check=with_static)
    plan = entry.build_mechanics_plan(args)
    fem_calls, static_calls = [], []
    def checks(candidate, inp="case.inp", odb="case.odb", *a):
        mesh = candidate["analysis"]["mesh_size_mm"]
        return {"passed": True, "sections": {}, "mechanics": {"skin_node_count": int(100000 / mesh**2)},
                "connections": {}, "odb_checks": {"connection_passed": True}, "BLF_global": 2 + mesh * .001,
                "abaqus_inp": str(inp), "abaqus_odb": str(odb)}
    reusable = {job["identity"]: {**checks(job["candidate"]), "origin": "archive/" + job["job_id"]}
                for job in plan["jobs"] if job["candidate"]["analysis"]["mesh_size_mm"] == 12}
    monkeypatch.setattr(entry, "audit_reusable_runs", lambda *a: reusable)
    monkeypatch.setattr(entry, "audit_mechanics_case", checks)
    def solve(_self, candidate):
        fem_calls.append(candidate["candidate_id"])
        return {"status": "success", "analysis_contract": analysis_contract(candidate),
                "BLF_global": checks(candidate)["BLF_global"], "abaqus_inp": candidate["candidate_id"] + ".inp", "abaqus_odb": "case.odb"}
    monkeypatch.setattr(FEMAgent, "run", solve)
    def static(source, target, _args):
        static_calls.append(str(target))
        return {"passed": True}
    monkeypatch.setattr(entry, "run_static_check", static)
    result = entry.run_mechanics(args)
    assert result["passed"], result.get("error")
    assert len(fem_calls) == result["new_fem_jobs"] == 16
    assert result["reused_jobs"] == 4
    assert len(static_calls) == len(set(static_calls)) == (9 if with_static else 0)
    assert result["static_status"] == ("passed" if with_static else "not_requested")
    assert all(item["acceptance_pair"] == "middle_to_fine" for item in result["convergence"].values())


def test_reuse_audit_is_read_only_and_requires_build_change_review(monkeypatch, tmp_path):
    args = mechanics_args(tmp_path / "output")
    args.output_dir.mkdir()
    plan = entry.build_mechanics_plan(args)
    candidate = plan["jobs"][2]["candidate"]
    root = tmp_path / "prior"
    folder = root / "data/abaqus_runs" / candidate["candidate_id"]
    folder.mkdir(parents=True)
    entry.save(folder / "candidate_input.json", candidate)
    entry.save(root / "acceptance.json", {"status": "passed", "code_sha256": {
        "abaqus\\runtime_build_panel.py": "0" * 64, "core\\laminate.py": "1" * 64}})
    before = {str(path): path.read_bytes() for path in root.rglob("*.json")}
    args.reuse_run_dir = [root]
    monkeypatch.setattr(entry, "audit_mechanics_case", lambda *a, **kw: {"passed": True, "BLF_global": 2.64})
    report = {**plan, "existing_runs": []}
    with pytest.raises(ValueError, match="reuse-build-change-reason"):
        entry.audit_reusable_runs(args, report, args.output_dir / "mechanics.json")
    args.reuse_build_change_reason = "几何工况与 INP 契约一致，输入校验实现已审查"
    report = {**plan, "existing_runs": []}
    available = entry.audit_reusable_runs(args, report, args.output_dir / "mechanics.json")
    assert len(available) == 1
    assert report["existing_runs"][0]["same_build_code"] is False
    assert report["existing_runs"][0]["build_change_review"] == args.reuse_build_change_reason
    assert before == {str(path): path.read_bytes() for path in root.rglob("*.json")}


def test_manifest_normalization_only_changes_keys_and_rejects_conflicts():
    original = {"code_sha256": {"core\\laminate.py": "a" * 64, "abaqus/runtime_build_panel.py": "b" * 64},
                "artifact": "D:\\runs\\C1.inp"}
    copied = deepcopy(original)
    result = entry.manifest_code_hashes(original)
    assert result == {"core/laminate.py": "a" * 64, "abaqus/runtime_build_panel.py": "b" * 64}
    assert original == copied
    original["code_sha256"]["core/laminate.py"] = "c" * 64
    with pytest.raises(ValueError):
        entry.manifest_code_hashes(original)


@pytest.mark.parametrize("extra", [[], ["--convergence-rtol", "nan"], ["--convergence-rtol", "0.05", "--mesh-sizes-mm", "12", "12", "6"]])
def test_mechanics_cli_rejects_missing_or_invalid_threshold_before_work(monkeypatch, tmp_path, extra):
    def forbidden(*args, **kwargs):
        pytest.fail("无效参数不应创建运行目录或提交求解")
    monkeypatch.setattr(entry, "isolate_runtime", forbidden)
    monkeypatch.setattr(sys, "argv", ["validate_physical_workflow.py", "--mode", "mechanics", "--output-dir", str(tmp_path / "run"), *extra])
    with pytest.raises(SystemExit) as caught:
        entry.main()
    assert caught.value.code == 2
