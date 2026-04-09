# -*- coding: utf-8 -*-
"""ABAQUS 运行时 T 形筋壁板建模脚本，兼容旧版 Python 解释器。"""

from __future__ import absolute_import

import codecs
import imp
import json
import os
import traceback


DEFAULT_MATERIAL = {
    "name": "T300/5208",
    "density_kg_per_m3": 1600.0,
    "E1_GPa": 181.0,
    "E2_GPa": 10.3,
    "G12_GPa": 7.17,
    "nu12": 0.28,
}

BOUNDARY_CONDITION_LIBRARY = {
    "SSSS": {
        "type": "SSSS",
        "simply_supported_edges": ["X0", "X1", "Y0", "Y1"],
        "clamped_edges": [],
    },
    "CCCC": {
        "type": "CCCC",
        "simply_supported_edges": [],
        "clamped_edges": ["X0", "X1", "Y0", "Y1"],
    },
    "SSCC": {
        "type": "SSCC",
        "simply_supported_edges": ["X0", "X1"],
        "clamped_edges": ["Y0", "Y1"],
    },
}

try:
    text_type = unicode
    binary_type = str
except NameError:
    text_type = str
    binary_type = bytes


def read_json(path):
    with codecs.open(path, "r", "utf-8") as file_handle:
        return json.load(file_handle)


def normalize_json_value(value):
    if isinstance(value, dict):
        normalized = {}
        for key, item in value.items():
            normalized[normalize_json_value(key)] = normalize_json_value(item)
        return normalized
    if isinstance(value, (list, tuple)):
        return [normalize_json_value(item) for item in value]
    if isinstance(value, text_type):
        return value
    if isinstance(value, binary_type):
        for encoding_name in ("utf-8", "mbcs", "latin-1"):
            try:
                return value.decode(encoding_name)
            except Exception:
                pass
        return value.decode("utf-8", "ignore")
    return value


def write_json(path, data):
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    with codecs.open(path, "w", "utf-8") as file_handle:
        json.dump(normalize_json_value(data), file_handle, ensure_ascii=True, indent=2)


def safe_float(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_load_conditions(load_conditions):
    if not isinstance(load_conditions, dict):
        load_conditions = {}
    raw_type = str(load_conditions.get("type", "")).strip()
    nx_value = max(safe_float(load_conditions.get("Nx_kN_per_m"), 0.0), 0.0)
    nxy_value = max(safe_float(load_conditions.get("Nxy_kN_per_m"), 0.0), 0.0)
    mapping = {
        "单轴压缩": "axial_compression",
        "axial_compression": "axial_compression",
        "AXIAL_COMPRESSION": "axial_compression",
        "面内剪切": "in_plane_shear",
        "剪切": "in_plane_shear",
        "in_plane_shear": "in_plane_shear",
        "IN_PLANE_SHEAR": "in_plane_shear",
        "压剪组合": "compression_shear",
        "压剪": "compression_shear",
        "compression_shear": "compression_shear",
        "COMPRESSION_SHEAR": "compression_shear",
    }
    load_type = mapping.get(raw_type, raw_type)
    if load_type not in ("axial_compression", "in_plane_shear", "compression_shear"):
        if nx_value > 0.0 and nxy_value > 0.0:
            load_type = "compression_shear"
        elif nxy_value > 0.0:
            load_type = "in_plane_shear"
        else:
            load_type = "axial_compression"

    if load_type == "axial_compression":
        nxy_value = 0.0
    elif load_type == "in_plane_shear":
        nx_value = 0.0

    return {
        "type": load_type,
        "Nx_kN_per_m": nx_value,
        "Nxy_kN_per_m": nxy_value,
    }


def normalize_boundary_conditions(boundary_conditions):
    if isinstance(boundary_conditions, dict):
        boundary_type = str(boundary_conditions.get("type", "SSSS")).upper()
    else:
        text = str(boundary_conditions or "").upper()
        if "SSCC" in text or ("简支" in text and "固支" in text):
            boundary_type = "SSCC"
        elif "CCCC" in text or "固支" in text:
            boundary_type = "CCCC"
        else:
            boundary_type = "SSSS"
    if boundary_type not in BOUNDARY_CONDITION_LIBRARY:
        boundary_type = "SSSS"
    return dict(BOUNDARY_CONDITION_LIBRARY[boundary_type])


def parse_layup_text(layup_text):
    text = (layup_text or "").strip()
    if not text:
        return [45.0, -45.0, 0.0, 90.0, 0.0, -45.0, 45.0] * 2

    is_symmetric = text.endswith("s")
    if is_symmetric:
        text = text[:-1]
    text = text.strip("[] ")
    angles = []
    for raw in text.split("/"):
        value = raw.strip().replace("deg", "").replace("°", "")
        if not value:
            continue
        if value.startswith("±"):
            angle = safe_float(value[1:], 45.0)
            angles.extend([angle, -angle])
            continue
        angles.append(safe_float(value, 0.0))
    if not angles:
        angles = [45.0, -45.0, 0.0, 90.0, 0.0, -45.0, 45.0]
    return angles + list(reversed(angles)) if is_symmetric else angles


def stiffener_positions(panel_width_mm, pitch_mm):
    if pitch_mm <= 0:
        return [panel_width_mm / 2.0]

    count = max(1, int(round(panel_width_mm / pitch_mm)))
    if count == 1:
        return [panel_width_mm / 2.0]

    occupied_width = pitch_mm * (count - 1)
    edge_margin = max((panel_width_mm - occupied_width) / 2.0, 0.0)
    positions = [edge_margin + index * pitch_mm for index in range(count)]
    return [min(max(value, 0.0), panel_width_mm) for value in positions]


def resolve_analysis_spec(candidate):
    geometry = dict(candidate.get("geometry", {}))
    layup = dict(candidate.get("layup", {}))
    load_conditions = normalize_load_conditions(candidate.get("load_conditions", {}))
    boundary_conditions = normalize_boundary_conditions(candidate.get("boundary_conditions", {}))
    material_system = dict(candidate.get("material_system", {}))
    analysis = dict(candidate.get("analysis", {}))

    panel_length_mm = safe_float(geometry.get("panel_length_mm"), 700.0)
    panel_width_mm = safe_float(geometry.get("panel_width_mm"), 600.0)
    pitch_mm = max(safe_float(geometry.get("pitch_mm"), 120.0), 1.0)
    stiffener_height_mm = safe_float(geometry.get("stiffener_height_mm"), 28.0)
    flange_width_mm = safe_float(geometry.get("flange_width_mm"), 16.0)

    mesh_size_mm = max(
        12.0,
        min(
            panel_length_mm / 20.0,
            panel_width_mm / 16.0,
            pitch_mm / 5.0,
            max(stiffener_height_mm, 8.0) / 2.5,
            max(flange_width_mm, 8.0) / 1.8,
        ),
    )

    default_buckling_modes = 8
    if load_conditions.get("type") == "in_plane_shear":
        default_buckling_modes = 14
    elif load_conditions.get("type") == "compression_shear":
        default_buckling_modes = 16

    return {
        "candidate_id": str(candidate.get("candidate_id", "UNKNOWN")),
        "job_name": str(candidate.get("candidate_id", "UNKNOWN")),
        "panel_length_mm": panel_length_mm,
        "panel_width_mm": panel_width_mm,
        "skin_thickness_mm": safe_float(geometry.get("skin_thickness_mm"), 2.5),
        "pitch_mm": pitch_mm,
        "stiffener_height_mm": stiffener_height_mm,
        "web_thickness_mm": safe_float(geometry.get("web_thickness_mm"), 2.0),
        "flange_width_mm": flange_width_mm,
        "flange_thickness_mm": safe_float(geometry.get("flange_thickness_mm"), 2.0),
        "stiffener_positions_mm": stiffener_positions(panel_width_mm, pitch_mm),
        "mesh_size_mm": round(mesh_size_mm, 3),
        "buckling_modes": int(analysis.get("buckling_modes", default_buckling_modes)),
        "preload_nx_kN_per_m": safe_float(load_conditions.get("Nx_kN_per_m"), 850.0),
        "preload_nxy_kN_per_m": safe_float(load_conditions.get("Nxy_kN_per_m"), 0.0),
        "load_case_type": str(load_conditions.get("type", "axial_compression")),
        "boundary_type": str(boundary_conditions.get("type", "SSSS")),
        "simply_supported_edges": list(boundary_conditions.get("simply_supported_edges", [])),
        "clamped_edges": list(boundary_conditions.get("clamped_edges", [])),
        "layup_angles_deg": parse_layup_text(str(layup.get("skin_layup", ""))),
        "material": {
            "name": str(material_system.get("name", DEFAULT_MATERIAL["name"])),
            "density_kg_per_m3": safe_float(material_system.get("density_kg_per_m3"), DEFAULT_MATERIAL["density_kg_per_m3"]),
            "E1_GPa": safe_float(material_system.get("E1_GPa"), DEFAULT_MATERIAL["E1_GPa"]),
            "E2_GPa": safe_float(material_system.get("E2_GPa"), DEFAULT_MATERIAL["E2_GPa"]),
            "G12_GPa": safe_float(material_system.get("G12_GPa"), DEFAULT_MATERIAL["G12_GPa"]),
            "nu12": safe_float(material_system.get("nu12"), DEFAULT_MATERIAL["nu12"]),
        },
    }


def write_failure_result(candidate_id, result_json, error_type, error_log):
    run_dir = os.getcwd()
    odb_path = os.path.join(run_dir, "%s.odb" % candidate_id)
    write_json(
        result_json,
        {
            "candidate_id": candidate_id,
            "status": "failed",
            "retry_count": 0,
            "BLF_global": None,
            "BLF_local": None,
            "failure_mode": None,
            "max_displacement_mm": None,
            "weight_kg_per_m2": None,
            "verdict": None,
            "abaqus_odb": odb_path,
            "abaqus_inp": os.path.splitext(odb_path)[0] + ".inp",
            "visualization_json": None,
            "artifact_dir": run_dir,
            "error_type": error_type,
            "error_log": error_log,
        },
    )


def edge_mode(spec, edge_name):
    if edge_name in spec.get("clamped_edges", []):
        return "clamped"
    if edge_name in spec.get("simply_supported_edges", []):
        return "simply_supported"
    return "free"


def detect_error_type(message):
    lowered = (message or "").lower()
    if "too many attempts" in lowered or "mesh" in lowered:
        return "mesh_error"
    if "assembly" in lowered or "boolean merge" in lowered or "geometry" in lowered:
        return "geometry_issue"
    if "eigen" in lowered and "negative" in lowered:
        return "blf_negative"
    if "converg" in lowered:
        return "convergence_fail"
    return "process_crash"


def load_runtime_extract_module(root_dir):
    module_path = os.path.join(root_dir, "abaqus", "runtime_extract_blf.py")
    return imp.load_source("csdm_runtime_extract_blf", module_path)


def build_panel(input_json, result_json):
    candidate = read_json(input_json)
    spec = resolve_analysis_spec(candidate)
    candidate_id = spec["candidate_id"]
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    try:
        from abaqus import Mdb, mdb
        from abaqusConstants import (
            ANALYSIS,
            CARTESIAN,
            COMPUTED,
            DEFAULT,
            DEFORMABLE_BODY,
            ENGINEERING_CONSTANTS,
            FREE,
            FROM_SECTION,
            GRADIENT,
            MIDDLE_SURFACE,
            NO_IDEALIZATION,
            OFF,
            ON,
            PERCENTAGE,
            QUAD,
            S4R,
            SIMPSON,
            STANDARD,
            THREE_D,
            UNIFORM,
            UNSET,
        )
        from mesh import ElemType
        import interaction
        import regionToolset
    except Exception as exc:
        write_failure_result(candidate_id, result_json, "process_crash", "ABAQUS 模块导入失败: %s" % exc)
        return

    try:
        runtime_extract = load_runtime_extract_module(root_dir)

        Mdb()
        try:
            del mdb.models["Model-1"]
        except Exception:
            pass

        model_name = str("CSDM_%s" % candidate_id)
        model = mdb.Model(name=model_name)

        material_name = "MAT_T300_5208"
        material = model.Material(name=material_name)
        g23_gpa = spec["material"]["E2_GPa"] / (2.0 * (1.0 + 0.35))
        material.Elastic(
            type=ENGINEERING_CONSTANTS,
            table=(
                (
                    spec["material"]["E1_GPa"] * 1000.0,
                    spec["material"]["E2_GPa"] * 1000.0,
                    spec["material"]["E2_GPa"] * 1000.0,
                    spec["material"]["nu12"],
                    spec["material"]["nu12"],
                    0.35,
                    spec["material"]["G12_GPa"] * 1000.0,
                    spec["material"]["G12_GPa"] * 1000.0,
                    g23_gpa * 1000.0,
                ),
            ),
        )

        model.HomogeneousShellSection(
            name="SkinSection",
            material=material_name,
            thickness=spec["skin_thickness_mm"],
            idealization=NO_IDEALIZATION,
            poissonDefinition=DEFAULT,
            thicknessModulus=None,
            temperature=GRADIENT,
            integrationRule=SIMPSON,
            numIntPts=5,
        )
        model.HomogeneousShellSection(
            name="WebSection",
            material=material_name,
            thickness=spec["web_thickness_mm"],
            idealization=NO_IDEALIZATION,
            poissonDefinition=DEFAULT,
            thicknessModulus=None,
            temperature=GRADIENT,
            integrationRule=SIMPSON,
            numIntPts=5,
        )
        model.HomogeneousShellSection(
            name="FlangeSection",
            material=material_name,
            thickness=spec["flange_thickness_mm"],
            idealization=NO_IDEALIZATION,
            poissonDefinition=DEFAULT,
            thicknessModulus=None,
            temperature=GRADIENT,
            integrationRule=SIMPSON,
            numIntPts=5,
        )

        panel_length = spec["panel_length_mm"]
        panel_width = spec["panel_width_mm"]
        stiffener_height = spec["stiffener_height_mm"]
        flange_width = spec["flange_width_mm"]
        half_flange_width = flange_width / 2.0
        tol = max(spec["mesh_size_mm"] * 0.25, 1.0)

        skin_sketch = model.ConstrainedSketch(name="SkinSketch", sheetSize=max(panel_length, panel_width) * 2.0)
        skin_sketch.rectangle(point1=(0.0, 0.0), point2=(panel_length, panel_width))
        skin_part = model.Part(name="SkinPanel", dimensionality=THREE_D, type=DEFORMABLE_BODY)
        skin_part.BaseShell(sketch=skin_sketch)
        del skin_sketch

        partition_sketch = model.ConstrainedSketch(name="SkinPartitionSketch", sheetSize=max(panel_length, panel_width) * 2.0)
        partition_y_positions = []
        for position_mm in spec["stiffener_positions_mm"]:
            for y_value in (position_mm - half_flange_width, position_mm, position_mm + half_flange_width):
                if 0.0 <= y_value <= panel_width:
                    partition_y_positions.append(round(y_value, 6))
        for y_value in sorted(set(partition_y_positions)):
            partition_sketch.Line(point1=(0.0, y_value), point2=(panel_length, y_value))
        skin_part.PartitionFaceBySketch(faces=skin_part.faces, sketch=partition_sketch)
        del partition_sketch

        web_sketch = model.ConstrainedSketch(name="WebSketch", sheetSize=max(panel_length, stiffener_height) * 2.0)
        web_sketch.rectangle(point1=(0.0, 0.0), point2=(panel_length, stiffener_height))
        web_part = model.Part(name="WebPart", dimensionality=THREE_D, type=DEFORMABLE_BODY)
        web_part.BaseShell(sketch=web_sketch)
        del web_sketch

        flange_sketch = model.ConstrainedSketch(name="FlangeHalfSketch", sheetSize=max(panel_length, flange_width) * 2.0)
        flange_sketch.rectangle(point1=(0.0, 0.0), point2=(panel_length, half_flange_width))
        flange_part = model.Part(name="FlangeHalfPart", dimensionality=THREE_D, type=DEFORMABLE_BODY)
        flange_part.BaseShell(sketch=flange_sketch)
        del flange_sketch

        skin_part.SectionAssignment(
            region=regionToolset.Region(faces=skin_part.faces),
            sectionName="SkinSection",
            offset=0.0,
            offsetType=MIDDLE_SURFACE,
            offsetField="",
            thicknessAssignment=FROM_SECTION,
        )
        web_part.SectionAssignment(
            region=regionToolset.Region(faces=web_part.faces),
            sectionName="WebSection",
            offset=0.0,
            offsetType=MIDDLE_SURFACE,
            offsetField="",
            thicknessAssignment=FROM_SECTION,
        )
        flange_part.SectionAssignment(
            region=regionToolset.Region(faces=flange_part.faces),
            sectionName="FlangeSection",
            offset=0.0,
            offsetType=MIDDLE_SURFACE,
            offsetField="",
            thicknessAssignment=FROM_SECTION,
        )

        elem_type = ElemType(elemCode=S4R, elemLibrary=STANDARD)
        for part in (skin_part, web_part, flange_part):
            part.setElementType(regions=(part.faces,), elemTypes=(elem_type,))
            part.seedPart(size=spec["mesh_size_mm"], deviationFactor=0.1, minSizeFactor=0.1)
            part.setMeshControls(regions=part.faces, elemShape=QUAD, technique=FREE)
            part.generateMesh()

        assembly = model.rootAssembly
        assembly.DatumCsysByDefault(CARTESIAN)
        skin_instance = assembly.Instance(name="Skin-1", part=skin_part, dependent=ON)

        for index, position_mm in enumerate(spec["stiffener_positions_mm"]):
            display_index = index + 1
            web_name = "Web-%02d" % display_index
            flange_left_name = "FlangeL-%02d" % display_index
            flange_right_name = "FlangeR-%02d" % display_index

            assembly.Instance(name=web_name, part=web_part, dependent=ON)
            assembly.rotate(
                instanceList=(web_name,),
                axisPoint=(0.0, 0.0, 0.0),
                axisDirection=(1.0, 0.0, 0.0),
                angle=90.0,
            )
            assembly.translate(instanceList=(web_name,), vector=(0.0, position_mm, 0.0))

            assembly.Instance(name=flange_left_name, part=flange_part, dependent=ON)
            assembly.translate(instanceList=(flange_left_name,), vector=(0.0, position_mm - half_flange_width, 0.0))

            assembly.Instance(name=flange_right_name, part=flange_part, dependent=ON)
            assembly.translate(instanceList=(flange_right_name,), vector=(0.0, position_mm, 0.0))

            skin_edge = skin_instance.edges.getByBoundingBox(
                xMin=-tol,
                xMax=panel_length + tol,
                yMin=position_mm - tol,
                yMax=position_mm + tol,
                zMin=-tol,
                zMax=tol,
            )
            web_instance = assembly.instances[web_name]
            web_bottom_edge = web_instance.edges.getByBoundingBox(
                xMin=-tol,
                xMax=panel_length + tol,
                yMin=position_mm - tol,
                yMax=position_mm + tol,
                zMin=-tol,
                zMax=tol,
            )
            flange_left_instance = assembly.instances[flange_left_name]
            flange_right_instance = assembly.instances[flange_right_name]
            flange_left_inner = flange_left_instance.edges.getByBoundingBox(
                xMin=-tol,
                xMax=panel_length + tol,
                yMin=position_mm - tol,
                yMax=position_mm + tol,
                zMin=-tol,
                zMax=tol,
            )
            flange_right_inner = flange_right_instance.edges.getByBoundingBox(
                xMin=-tol,
                xMax=panel_length + tol,
                yMin=position_mm - tol,
                yMax=position_mm + tol,
                zMin=-tol,
                zMax=tol,
            )
            flange_inner_edges = flange_left_inner + flange_right_inner
            flange_left_face = flange_left_instance.faces.getByBoundingBox(
                xMin=-tol,
                xMax=panel_length + tol,
                yMin=position_mm - half_flange_width - tol,
                yMax=position_mm + tol,
                zMin=-tol,
                zMax=tol,
            )
            flange_right_face = flange_right_instance.faces.getByBoundingBox(
                xMin=-tol,
                xMax=panel_length + tol,
                yMin=position_mm - tol,
                yMax=position_mm + half_flange_width + tol,
                zMin=-tol,
                zMax=tol,
            )
            skin_left_face = skin_instance.faces.getByBoundingBox(
                xMin=-tol,
                xMax=panel_length + tol,
                yMin=position_mm - half_flange_width - tol,
                yMax=position_mm + tol,
                zMin=-tol,
                zMax=tol,
            )
            skin_right_face = skin_instance.faces.getByBoundingBox(
                xMin=-tol,
                xMax=panel_length + tol,
                yMin=position_mm - tol,
                yMax=position_mm + half_flange_width + tol,
                zMin=-tol,
                zMax=tol,
            )

            model.Tie(
                name="Tie_Skin_FlangeL_%02d" % display_index,
                main=assembly.Surface(name="SurfSkinL_%02d" % display_index, side1Faces=skin_left_face),
                secondary=assembly.Surface(name="SurfFlangeFaceL_%02d" % display_index, side1Faces=flange_left_face),
                positionToleranceMethod=COMPUTED,
                adjust=ON,
                tieRotations=ON,
                thickness=ON,
            )
            model.Tie(
                name="Tie_Skin_FlangeR_%02d" % display_index,
                main=assembly.Surface(name="SurfSkinR_%02d" % display_index, side1Faces=skin_right_face),
                secondary=assembly.Surface(name="SurfFlangeFaceR_%02d" % display_index, side1Faces=flange_right_face),
                positionToleranceMethod=COMPUTED,
                adjust=ON,
                tieRotations=ON,
                thickness=ON,
            )
            model.Tie(
                name="Tie_Web_Flange_%02d" % display_index,
                main=assembly.Surface(name="SurfFlangeInner_%02d" % display_index, side1Edges=flange_inner_edges),
                secondary=assembly.Surface(name="SurfWebBase_%02d" % display_index, side1Edges=web_bottom_edge),
                positionToleranceMethod=COMPUTED,
                adjust=ON,
                tieRotations=ON,
                thickness=ON,
            )

        x0_nodes = skin_instance.nodes.getByBoundingBox(
            xMin=-tol,
            xMax=tol,
            yMin=-tol,
            yMax=panel_width + tol,
            zMin=-tol,
            zMax=tol,
        )
        x1_nodes = skin_instance.nodes.getByBoundingBox(
            xMin=panel_length - tol,
            xMax=panel_length + tol,
            yMin=-tol,
            yMax=panel_width + tol,
            zMin=-tol,
            zMax=tol,
        )
        y0_nodes = skin_instance.nodes.getByBoundingBox(
            xMin=-tol,
            xMax=panel_length + tol,
            yMin=-tol,
            yMax=tol,
            zMin=-tol,
            zMax=tol,
        )
        y1_nodes = skin_instance.nodes.getByBoundingBox(
            xMin=-tol,
            xMax=panel_length + tol,
            yMin=panel_width - tol,
            yMax=panel_width + tol,
            zMin=-tol,
            zMax=tol,
        )

        x0_mode = edge_mode(spec, "X0")
        if x0_mode == "clamped":
            model.DisplacementBC(
                name="BC_X0_Clamped",
                createStepName="Initial",
                region=regionToolset.Region(nodes=x0_nodes),
                u1=0.0,
                u2=0.0,
                u3=0.0,
            )
        elif x0_mode == "simply_supported":
            model.DisplacementBC(
                name="BC_X0_Simply",
                createStepName="Initial",
                region=regionToolset.Region(nodes=x0_nodes),
                u1=0.0,
                u2=UNSET,
                u3=0.0,
            )

        x1_mode = edge_mode(spec, "X1")
        if x1_mode == "clamped":
            model.DisplacementBC(
                name="BC_X1_Clamped",
                createStepName="Initial",
                region=regionToolset.Region(nodes=x1_nodes),
                u1=UNSET,
                u2=UNSET,
                u3=0.0,
            )
        elif x1_mode == "simply_supported":
            model.DisplacementBC(
                name="BC_X1_Simply",
                createStepName="Initial",
                region=regionToolset.Region(nodes=x1_nodes),
                u1=UNSET,
                u2=UNSET,
                u3=0.0,
            )

        y0_mode = edge_mode(spec, "Y0")
        if y0_mode == "clamped":
            model.DisplacementBC(
                name="BC_Y0_Clamped",
                createStepName="Initial",
                region=regionToolset.Region(nodes=y0_nodes),
                u1=0.0,
                u2=0.0,
                u3=0.0,
            )
        elif y0_mode == "simply_supported":
            model.DisplacementBC(
                name="BC_Y0_Simply",
                createStepName="Initial",
                region=regionToolset.Region(nodes=y0_nodes),
                u1=UNSET,
                u2=0.0,
                u3=0.0,
            )

        y1_mode = edge_mode(spec, "Y1")
        if y1_mode == "clamped":
            model.DisplacementBC(
                name="BC_Y1_Clamped",
                createStepName="Initial",
                region=regionToolset.Region(nodes=y1_nodes),
                u1=0.0,
                u2=0.0,
                u3=0.0,
            )
        elif y1_mode == "simply_supported":
            model.DisplacementBC(
                name="BC_Y1_Simply",
                createStepName="Initial",
                region=regionToolset.Region(nodes=y1_nodes),
                u1=UNSET,
                u2=UNSET,
                u3=0.0,
            )

        total_force_n = spec["preload_nx_kN_per_m"] * panel_width
        total_shear_n = spec["preload_nxy_kN_per_m"] * panel_width
        load_edge_region = regionToolset.Region(nodes=x1_nodes)
        nodal_force_n = -total_force_n / max(len(x1_nodes), 1)
        nodal_shear_n = total_shear_n / max(len(x1_nodes), 1)
        model.BuckleStep(name="Buckling", previous="Initial", numEigen=max(spec["buckling_modes"], 3), vectors=max(spec["buckling_modes"] * 2, 10), maxIterations=200)
        model.ConcentratedForce(
            name="BucklingPattern",
            createStepName="Buckling",
            region=load_edge_region,
            cf1=nodal_force_n if abs(nodal_force_n) > 1e-9 else 0.0,
            cf2=nodal_shear_n if abs(nodal_shear_n) > 1e-9 else 0.0,
            distributionType=UNIFORM,
        )
        job = mdb.Job(
            name=spec["job_name"],
            model=model_name,
            type=ANALYSIS,
            memory=85,
            memoryUnits=PERCENTAGE,
            numCpus=4,
            numDomains=4,
        )
        job.submit(consistencyChecking=OFF)
        job.waitForCompletion()

        default_odb_path = os.path.join(os.getcwd(), "%s.odb" % spec["job_name"])
        runtime_extract.extract_blf(default_odb_path, result_json, input_json=input_json)
    except Exception:
        write_failure_result(candidate_id, result_json, detect_error_type(traceback.format_exc()), traceback.format_exc())
