"""在独立目录执行层序、力学或外部 API 流程验收，保存输入检查与终态证据。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
from datetime import datetime, timezone
from copy import deepcopy


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def manifest_code_hashes(manifest):
    """仅规范历史清单的路径键；指纹值和历史工件不变。"""
    normalized = {}
    for name, value in manifest.get("code_sha256", {}).items():
        key = name.replace("\\", "/")
        if key in normalized and normalized[key] != value:
            raise ValueError("源码清单的路径规范化产生指纹冲突: " + key)
        normalized[key] = value
    return normalized


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def isolate_runtime(output):
    """在业务模块导入前绑定运行目录；知识包只读，模型复制到隔离目录。"""
    from core import paths

    if output.exists():
        raise FileExistsError("验证目录已存在，拒绝覆盖: " + str(output))
    output.mkdir(parents=True)
    original_models = paths.MODELS_DIR
    for name, relative in {
        "DATA_DIR": "data", "IO_DIR": "data/io", "TASKS_DIR": "data/tasks",
        "CASES_DIR": "data/cases", "RESULTS_DIR": "data/results", "ABAQUS_RUNS_DIR": "data/abaqus_runs",
        "CHROMA_DIR": "knowledge/chroma_db", "CASE_LIBRARY_DIR": "knowledge/case_library", "MODELS_DIR": "models",
    }.items():
        target = output / relative
        setattr(paths, name, target)
        target.mkdir(parents=True, exist_ok=True)
    for kind in ("T", "BLADE", "L", "HAT"):
        if (original_models / kind).is_dir():
            shutil.copytree(original_models / kind, paths.MODELS_DIR / kind)


def inp_blocks(path):
    """解析 ASCII 求解关键字及数据，跳过可采用本地编码的注释。"""
    blocks = []
    lines = []
    for raw in Path(path).read_bytes().splitlines():
        raw = raw.strip()
        if not raw or raw.startswith(b"**"):
            continue
        lines.append(raw.decode("ascii"))
    source = iter(lines)
    for line in source:
        if line.startswith("*"):
            while line.endswith(","):
                continuation = next(source)
                if continuation.startswith("*"):
                    raise ValueError("INP 关键字续行缺失")
                line += continuation
            cells = next(csv.reader([line], skipinitialspace=True))
            options, flags = {}, set()
            for cell in cells[1:]:
                if "=" in cell:
                    key, value = cell.split("=", 1)
                    options[key.strip().upper()] = value.strip().upper()
                elif cell.strip():
                    flags.add(cell.strip().upper())
            blocks.append({"keyword": cells[0].strip().upper(), "options": options,
                           "flags": flags, "header": line, "lines": [], "data": []})
        else:
            if not blocks:
                raise ValueError("INP 数据缺少关键字")
            blocks[-1]["lines"].append(line)
            blocks[-1]["data"].append([cell.strip() for cell in next(csv.reader([line]))])
    return blocks


def number(text):
    value = float(str(text).replace("D", "E").replace("d", "e"))
    if not math.isfinite(value):
        raise ValueError("INP 数值必须有限")
    return value


def cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def transformed_point(point, rows):
    """按 *Instance 的平移、绕轴旋转顺序转换为装配坐标。"""
    result = list(point)
    if rows and len(rows[0]) == 3:
        result = [a + number(b) for a, b in zip(result, rows[0])]
        rows = rows[1:]
    if rows:
        if len(rows) != 1 or len(rows[0]) != 7:
            raise ValueError("不支持的实例变换")
        values = [number(value) for value in rows[0]]
        origin, end = values[:3], values[3:6]
        axis = [b - a for a, b in zip(origin, end)]
        length = math.sqrt(sum(value * value for value in axis))
        if length == 0:
            raise ValueError("实例旋转轴长度为零")
        axis = [value / length for value in axis]
        vector = [a - b for a, b in zip(result, origin)]
        theta = math.radians(values[6])
        cosine, sine = math.cos(theta), math.sin(theta)
        projection = sum(a * b for a, b in zip(axis, vector))
        perpendicular = cross(axis, vector)
        result = [origin[i] + cosine * vector[i] + sine * perpendicular[i]
                  + (1 - cosine) * projection * axis[i] for i in range(3)]
    return result


class MechanicsDeck:
    """读取生产 INP 的节点、集合、初始位移约束和单个屈曲步集中力。"""

    def __init__(self, path):
        self.blocks = inp_blocks(path)
        self.parts, self.instances, self.sets = {}, {}, {}
        self.part_elements, self.elsets, self.surfaces, self.ties = {}, {}, {}, []
        self.boundary_rows, self.boundary_policies, self.load_rows = {}, {}, []
        part, step = None, None
        steps, buckles = 0, 0
        unsupported = {"*INCLUDE", "*TRANSFORM", "*EQUATION", "*MPC", "*COUPLING", "*DLOAD", "*DSLOAD"}
        for block in self.blocks:
            keyword, options, rows = block["keyword"], block["options"], block["data"]
            if keyword in unsupported:
                raise ValueError("力学验收不支持该载荷/约束定义: " + keyword)
            if keyword == "*PART":
                part = options["NAME"]
                if part in self.parts:
                    raise ValueError("INP 部件重名")
                self.parts[part] = {}
                self.part_elements[part] = {}
            elif keyword == "*END PART":
                part = None
            elif keyword == "*NODE":
                if part is None or options.get("SYSTEM", "R") != "R":
                    raise ValueError("节点必须位于部件的直角坐标系")
                for row in rows:
                    label = int(row[0])
                    if label <= 0 or label in self.parts[part] or not 3 <= len(row) <= 4:
                        raise ValueError("节点编号或坐标无效")
                    self.parts[part][label] = [number(value or "0") for value in row[1:]] + [0.0] * (4 - len(row))
            elif keyword == "*ELEMENT":
                if part is None or options["TYPE"] not in ("S3", "S3R", "S4", "S4R"):
                    raise ValueError("连接验收只接受部件内的三/四节点壳单元")
                for row in rows:
                    label, *connectivity = [int(value) for value in row if value]
                    if label <= 0 or label in self.part_elements[part] or len(connectivity) not in (3, 4):
                        raise ValueError("壳单元编号或连接关系无效")
                    self.part_elements[part][label] = connectivity
                    if "ELSET" in options:
                        self.elsets.setdefault((part, options["ELSET"]), []).append((None, str(label)))
            elif keyword == "*INSTANCE":
                name = options["NAME"]
                if name in self.instances:
                    raise ValueError("INP 实例重名")
                self.instances[name] = {"part": options["PART"], "transform": rows}
            elif keyword in ("*NSET", "*ELSET"):
                store = self.sets if keyword == "*NSET" else self.elsets
                entries = store.setdefault((part, options[keyword[1:]]), [])
                for row in rows:
                    if "GENERATE" in block["flags"]:
                        start, end, increment = [int(value) for value in row]
                        if start <= 0 or increment <= 0 or end < start or (end - start) % increment:
                            raise ValueError("NSET GENERATE 范围无效")
                        row = [str(value) for value in range(start, end + 1, increment)]
                    entries.extend((options.get("INSTANCE"), value.upper()) for value in row if value)
            elif keyword == "*SURFACE":
                if options.get("TYPE", "ELEMENT") != "ELEMENT" or (part, options["NAME"]) in self.surfaces:
                    raise ValueError("Tie 表面必须是唯一的单元表面")
                self.surfaces[part, options["NAME"]] = rows
            elif keyword == "*TIE":
                self.ties.append(block)
            elif keyword == "*STEP":
                steps += 1
                step = options.get("NAME", "")
            elif keyword == "*BUCKLE":
                if step is None:
                    raise ValueError("屈曲过程缺少分析步")
                buckles += 1
            elif keyword == "*END STEP":
                step = None
            elif keyword == "*BOUNDARY":
                load_case = options.get("LOAD CASE", "1")
                policy = options.get("OP", "MOD")
                if options.get("TYPE", "DISPLACEMENT") != "DISPLACEMENT" or policy not in ("MOD", "NEW") or load_case not in ("1", "2"):
                    raise ValueError("不支持的位移约束操作")
                group = "initial" if step is None else "load_case_" + load_case
                self.boundary_rows.setdefault(group, []).extend(rows)
                self.boundary_policies.setdefault(group, set()).add(policy)
            elif keyword == "*CLOAD":
                if step is None or options.get("OP", "MOD") != "MOD" or options.get("FOLLOWER", "NO") != "NO" or "AMPLITUDE" in options:
                    raise ValueError("验收要求屈曲步内的恒定全局集中力")
                self.load_rows.extend(rows)
        if steps != 1 or buckles != 1 or step is not None:
            raise ValueError("验收要求一个完整的屈曲分析步")
        self.nodes = {(name, label): transformed_point(point, instance["transform"])
                      for name, instance in self.instances.items()
                      for label, point in self.parts[instance["part"]].items()}
        self.elements = {(name, label): [(name, node) for node in connectivity]
                         for name, instance in self.instances.items()
                         for label, connectivity in self.part_elements[instance["part"]].items()}
        if any(node not in self.nodes for connectivity in self.elements.values() for node in connectivity):
            raise ValueError("壳单元引用不存在的节点")
        skins = [name for name, instance in self.instances.items() if instance["part"] == "SKINPANEL"]
        if len(skins) != 1:
            raise ValueError("必须存在一个 SkinPanel 实例")
        self.skin_instance = skins[0]

    def resolve(self, target, instance=None, trail=(), kind="node"):
        """展开装配集合、实例集合与节点引用，保留实例身份。"""
        target = target.upper()
        if "." in target:
            prefix, suffix = target.rsplit(".", 1)
            if prefix in self.instances:
                return self.resolve(suffix, prefix, trail, kind)
        if target.isdigit():
            key = (instance, int(target))
            if key not in (self.nodes if kind == "node" else self.elements):
                raise ValueError("节点引用不存在: " + str(key))
            return {key}
        scope = self.instances[instance]["part"] if instance else None
        key = (scope, target)
        identity = (instance, target)
        sets = self.sets if kind == "node" else self.elsets
        if key not in sets or identity in trail:
            raise ValueError("节点集合缺失或循环引用: " + target)
        result = set()
        for member_instance, member in sets[key]:
            result.update(self.resolve(member, member_instance or instance, trail + (identity,), kind))
        return result

    def constraints(self, group="initial"):
        policies = self.boundary_policies.get(group, {"MOD"})
        if len(policies) != 1:
            raise ValueError("同一载荷工况的边界 OP 定义混合")
        result = {} if group == "initial" or "NEW" in policies else self.constraints()
        for row in self.boundary_rows.get(group, []):
            if not 2 <= len(row) <= 4:
                raise ValueError("位移约束行无效")
            first = int(row[1])
            last = int(row[2]) if len(row) > 2 and row[2] else first
            value = number(row[3]) if len(row) > 3 and row[3] else 0.0
            if not 1 <= first <= last <= 6:
                raise ValueError("约束自由度无效")
            for node in self.resolve(row[0]):
                for dof in range(first, last + 1):
                    key = (node, dof)
                    if key in result and result[key] != value:
                        raise ValueError("同一节点自由度的约束冲突")
                    result[key] = value
        return result

    def surface_facets(self, name):
        name, instance, scope = name.upper(), None, None
        if "." in name:
            instance, name = name.rsplit(".", 1)
            scope = self.instances[instance]["part"]
        if (scope, name) not in self.surfaces:
            raise ValueError("Tie 引用不存在的表面: " + name)
        facets = []
        for target, side in self.surfaces[scope, name]:
            for element in sorted(self.resolve(target, instance, kind="element")):
                nodes = self.elements[element]
                side = side.upper()
                if side in ("SPOS", "SNEG"):
                    facets.append(nodes)
                elif side.startswith("E") and side[1:].isdigit() and 1 <= int(side[1:]) <= len(nodes):
                    index = int(side[1:]) - 1
                    facets.append([nodes[index], nodes[(index + 1) % len(nodes)]])
                else:
                    raise ValueError("不支持的壳表面方向: " + side)
        if not facets:
            raise ValueError("Tie 表面为空: " + name)
        return facets

    def forces(self):
        components = {}
        for row in self.load_rows:
            if len(row) != 3 or int(row[1]) not in (1, 2):
                raise ValueError("仅接受全局 X/Y 方向集中力")
            for node in self.resolve(row[0]):
                key = (node, int(row[1]))
                if key in components:
                    raise ValueError("同一节点的集中力分量重复定义")
                components[key] = number(row[2])
        return {node: [components.get((node, dof), 0.0) for dof in (1, 2, 3)]
                for node, _dof in components}


def force_moment(nodes, forces):
    force, moment = [0.0] * 3, [0.0] * 3
    for node, vector in forces.items():
        torque = cross(nodes[node], vector)
        for index in range(3):
            force[index] += vector[index]
            moment[index] += torque[index]
    return force, moment


def inspect_mechanics_deck(path, candidate, *, rtol=1e-5, force_atol=1e-5, moment_atol=1e-3):
    """从实际边节点独立积分牵引，并检查合力、合矩和完整约束集合。"""
    deck = MechanicsDeck(path)
    length, width = [number(candidate["geometry"][key]) for key in ("panel_length_mm", "panel_width_mm")]
    nx, nxy = [number(candidate["load_conditions"][key]) for key in ("Nx_kN_per_m", "Nxy_kN_per_m")]
    tolerance = 1e-6 * max(length, width, 1)
    skin = {node: point for node, point in deck.nodes.items() if node[0] == deck.skin_instance}
    errors, edge_nodes, expected = [], {}, {}
    if not skin or any(not -tolerance <= p[0] <= length + tolerance or not -tolerance <= p[1] <= width + tolerance
                       or abs(p[2]) > tolerance for p in skin.values()):
        raise ValueError("蒙皮节点不在任务矩形中面内")
    boundary_type = candidate["boundary_conditions"]["type"]
    clamped = {"SSSS": set(), "SSCC": {"Y0", "Y1"}, "CCCC": {"X0", "X1", "Y0", "Y1"}}[boundary_type]
    constraints = {}
    edge_resultants = {}
    for edge, axis, location, traction in (("X0", 0, 0, (nx, -nxy)), ("X1", 0, length, (-nx, nxy)),
                                           ("Y0", 1, 0, (-nxy, 0)), ("Y1", 1, width, (nxy, 0))):
        members = {node for node, point in skin.items() if abs(point[axis] - location) <= tolerance}
        if deck.resolve("EDGE_" + edge) != members:
            errors.append(edge + " 集合与几何边界不一致")
        ordered = sorted(members, key=lambda node: skin[node][1 - axis])
        span = width if axis == 0 else length
        if len(ordered) < 2 or abs(skin[ordered[0]][1 - axis]) > tolerance or abs(skin[ordered[-1]][1 - axis] - span) > tolerance:
            raise ValueError(edge + " 缺少完整边界节点")
        edge_nodes[edge] = len(ordered)
        # 对每段恒定牵引作梯形积分，端点各承担半段载荷，角点累加两条边贡献。
        for start, end in zip(ordered, ordered[1:]):
            distance = skin[end][1 - axis] - skin[start][1 - axis]
            if distance <= 0:
                raise ValueError(edge + " 含重复边坐标")
            for node in (start, end):
                vector = expected.setdefault(node, [0.0] * 3)
                for component in (0, 1):
                    vector[component] += traction[component] * distance / 2
        edge_resultants[edge] = [value * span for value in traction]
        for node in members:
            for dof in ([3, 4, 5] if edge in clamped else [3]):
                constraints[node, dof] = 0.0
    for name, position, dofs in (("ANCHOR_ORIGIN", (0.0, 0.0), (1, 2)), ("ANCHOR_AXIS", (length, 0.0), (2,))):
        nodes = {node for node, point in skin.items() if all(abs(point[i] - position[i]) <= tolerance for i in (0, 1))}
        if len(nodes) != 1 or deck.resolve(name) != nodes:
            errors.append(name + " 必须对应指定的单个角点")
        for node in nodes:
            for dof in dofs:
                constraints[node, dof] = 0.0
    constraint_checks = {}
    for group in ("initial", "load_case_1", "load_case_2"):
        actual_constraints = deck.constraints(group)
        missing, extra = set(constraints) - set(actual_constraints), set(actual_constraints) - set(constraints)
        nonzero = [list(node) + [dof, value] for (node, dof), value in actual_constraints.items() if value != 0]
        if missing or extra or nonzero:
            errors.append(group + " 约束自由度缺失、额外固定或位移不为零")
        constraint_checks[group] = {"count": len(actual_constraints), "nonzero": nonzero,
                                   "missing": [list(node) + [dof] for node, dof in sorted(missing)],
                                   "extra": [list(node) + [dof] for node, dof in sorted(extra)]}
    actual = deck.forces()
    mismatches = []
    for node in sorted(set(expected) | set(actual)):
        wanted, observed = expected.get(node, [0.0] * 3), actual.get(node, [0.0] * 3)
        if any(abs(a - b) > force_atol + rtol * abs(b) for a, b in zip(observed, wanted)):
            mismatches.append({"instance": node[0], "node": node[1], "actual_N": observed, "expected_N": wanted})
    if mismatches:
        errors.append("节点集中力与独立边牵引积分不一致")
    total_force, total_moment = force_moment(deck.nodes, actual)
    force_scale = sum(math.sqrt(sum(value * value for value in force)) for force in expected.values())
    moment_scale = force_scale * max(length, width)
    if any(abs(value) > force_atol + rtol * force_scale for value in total_force):
        errors.append("整体载荷合力不平衡")
    if any(abs(value) > moment_atol + rtol * moment_scale for value in total_moment):
        errors.append("整体载荷合矩不平衡")
    return {"passed": not errors, "errors": errors, "sha256": digest(path), "skin_node_count": len(skin),
            "edge_node_counts": edge_nodes, "expected_edge_resultants_N": edge_resultants,
            "total_force_N": total_force, "total_moment_Nmm": total_moment, "force_scale_N": force_scale,
            "moment_scale_Nmm": moment_scale, "nodal_force_mismatches": mismatches,
            "constraints": constraint_checks,
            "tolerances": {"relative": rtol, "force_N": force_atol, "moment_Nmm": moment_atol}}


def distances_to_facets(points, facets):
    """计算节点到线段/三四节点平面的最短距离，用于 Tie 覆盖检查。"""
    import numpy as np

    points = np.asarray(points, dtype=float)
    closest = np.full(len(points), np.inf)
    for raw_facet in facets:
        facet = np.asarray(raw_facet, dtype=float)
        edges = [(facet[0], facet[1])] if len(facet) == 2 else list(zip(facet, np.roll(facet, -1, axis=0)))
        for start, end in edges:
            direction = end - start
            squared = float(direction @ direction)
            if squared <= 0:
                raise ValueError("Tie 表面存在零长度边")
            parameter = np.clip((points - start) @ direction / squared, 0, 1)
            closest = np.minimum(closest, np.linalg.norm(points - start - parameter[:, None] * direction, axis=1))
        for index in range(1, len(facet) - 1):
            a, b, c = facet[0], facet[index], facet[index + 1]
            v0, v1, delta = b - a, c - a, points - a
            d00, d01, d11 = v0 @ v0, v0 @ v1, v1 @ v1
            determinant = d00 * d11 - d01 * d01
            if determinant <= 0:
                raise ValueError("Tie 表面存在退化三角形")
            u = (d11 * (delta @ v0) - d01 * (delta @ v1)) / determinant
            v = (d00 * (delta @ v1) - d01 * (delta @ v0)) / determinant
            inside = (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9)
            normal = np.cross(v0, v1)
            distance = np.abs(delta @ normal) / np.linalg.norm(normal)
            closest = np.minimum(closest, np.where(inside, distance, np.inf))
    if not np.isfinite(closest).all():
        raise ValueError("Tie 表面距离无法计算")
    return closest.tolist()


def inspect_connections(path, candidate, *, geometric_atol=0.01):
    """检查壳中面位置、非空 Tie、显式几何容差及实例连接图。"""
    deck = MechanicsDeck(path)
    geometry, stype = candidate["geometry"], candidate["stiffener_type"]
    count = max(1, int(math.floor(geometry["panel_width_mm"] / geometry["pitch_mm"] + 0.5)))
    flange_z = 0.0 if stype == "BLADE" else (geometry["skin_thickness_mm"] + geometry["flange_thickness_mm"]) / 2
    cap_z = flange_z + geometry["stiffener_height_mm"]
    expected_counts = {"SKINPANEL": 1, "WEBPART": count * (2 if stype == "HAT" else 1),
                       "FLANGEHALFPART": count * (2 if stype in ("T", "HAT") else 1) if stype != "BLADE" else 0,
                       "CAPPART": count if stype == "HAT" else 0}
    expected_z = {"SKINPANEL": (0, 0), "WEBPART": (flange_z, cap_z),
                  "FLANGEHALFPART": (flange_z, flange_z), "CAPPART": (cap_z, cap_z)}
    counts = {key: 0 for key in expected_counts}
    bounds, errors = {}, []
    for name, instance in deck.instances.items():
        part = instance["part"]
        if part not in counts:
            raise ValueError("连接验收包含未知部件: " + part)
        counts[part] += 1
        points = [point for node, point in deck.nodes.items() if node[0] == name]
        if not points:
            raise ValueError("装配实例没有节点: " + name)
        z_bounds = [min(p[2] for p in points), max(p[2] for p in points)]
        bounds[name] = z_bounds
        if any(abs(actual - expected) > geometric_atol for actual, expected in zip(z_bounds, expected_z[part])):
            errors.append(name + " 壳中面高度不符合几何约定")
        if abs(min(p[0] for p in points)) > geometric_atol or abs(max(p[0] for p in points) - geometry["panel_length_mm"]) > geometric_atol:
            errors.append(name + " 轴向范围不符合壁板长度")
    if counts != expected_counts:
        errors.append("各部件实例数量与筋型/筋距不一致")
    expected_ties = count * {"BLADE": 1, "L": 2, "T": 3, "HAT": 6}[stype]
    if len(deck.ties) != expected_ties:
        errors.append("Tie 数量与连接拓扑不一致")
    graph = {name: set() for name in deck.instances}
    checks, names = [], set()
    allowed_pairs = [{"SKINPANEL", "WEBPART"}] if stype == "BLADE" else [
        {"SKINPANEL", "FLANGEHALFPART"}, {"WEBPART", "FLANGEHALFPART"}]
    if stype == "HAT":
        allowed_pairs.append({"WEBPART", "CAPPART"})
    for tie in deck.ties:
        options = tie["options"]
        name = options.get("NAME", "")
        if not name or name in names or len(tie["data"]) != 1 or len(tie["data"][0]) != 2:
            raise ValueError("Tie 名称或表面定义无效")
        names.add(name)
        secondary, primary = [deck.surface_facets(surface) for surface in tie["data"][0]]
        secondary_nodes = {node for facet in secondary for node in facet}
        primary_nodes = {node for facet in primary for node in facet}
        secondary_instances = {node[0] for node in secondary_nodes}
        primary_instances = {node[0] for node in primary_nodes}
        roles = {deck.instances[instance]["part"] for instance in secondary_instances | primary_instances}
        secondary_roles = {deck.instances[instance]["part"] for instance in secondary_instances}
        gap = flange_z if roles == {"SKINPANEL", "FLANGEHALFPART"} else 0.0
        if roles not in allowed_pairs or secondary_instances & primary_instances:
            errors.append(name + " 连接的部件组合错误")
        if secondary_roles == {"FLANGEHALFPART"} and roles == {"SKINPANEL", "FLANGEHALFPART"}:
            required_nodes = {node for node in deck.nodes if node[0] in secondary_instances}
        elif secondary_roles == {"WEBPART"}:
            elevation = cap_z if "CAPPART" in roles else flange_z
            required_nodes = {node for node, point in deck.nodes.items()
                              if node[0] in secondary_instances and abs(point[2] - elevation) <= geometric_atol}
        else:
            required_nodes = set()
        if not required_nodes or secondary_nodes != required_nodes:
            errors.append(name + " 从属面未完整覆盖连接面/边")
        position_tolerance = number(options["POSITION TOLERANCE"]) if "POSITION TOLERANCE" in options else None
        settings_ok = (options.get("ADJUST") == "NO" and "NO THICKNESS" in tie["flags"]
                       and "NO ROTATION" not in tie["flags"] and position_tolerance is not None
                       and gap <= position_tolerance <= gap + geometric_atol)
        if not settings_ok:
            errors.append(name + " 未采用关闭调整/厚度影响的显式几何容差")
        distances = distances_to_facets([deck.nodes[node] for node in sorted(secondary_nodes)],
                                       [[deck.nodes[node] for node in facet] for facet in primary])
        maximum = max(distances)
        if maximum > gap + geometric_atol or position_tolerance is not None and maximum > position_tolerance + 1e-5:
            errors.append(name + " 存在超出连接中面距离/显式容差的从属节点")
        for a in primary_instances:
            for b in secondary_instances:
                graph[a].add(b)
                graph[b].add(a)
        checks.append({"name": name, "secondary_node_count": len(secondary_nodes), "primary_node_count": len(primary_nodes),
                       "expected_gap_mm": gap, "max_distance_mm": maximum, "position_tolerance_mm": position_tolerance,
                       "settings_passed": settings_ok})
    reached, pending = set(), [deck.skin_instance]
    while pending:
        node = pending.pop()
        if node not in reached:
            reached.add(node)
            pending.extend(graph[node] - reached)
    if reached != set(graph):
        errors.append("存在未连接到蒙皮的部件实例")
    return {"passed": not errors, "errors": errors, "instance_counts": counts, "z_bounds_mm": bounds,
            "ties": checks, "geometric_atol_mm": geometric_atol, "disconnected_instances": sorted(set(graph) - reached)}


def inspect_input_deck(path, candidate):
    """独立解析 INP 的复合壳数据行和局部零度坐标，不以源代码调用代替输入检查。"""
    from core.laminate import section_layers, lamina_material

    blocks, orientations, section_orientations = {}, {}, {}
    current, orientation, elastic, part = None, None, False, None
    elastic_rows = []
    # 求解关键字和数据使用 ASCII；CAE 本地语言注释可使用独立编码。
    for raw_line in Path(path).read_bytes().splitlines():
        raw_line = raw_line.strip()
        if not raw_line or raw_line.startswith(b"**"):
            continue
        line = raw_line.decode("ascii")
        if line.startswith("*"):
            current = None
            orientation = None
            elastic = line.upper().startswith("*ELASTIC")
            options = {key.strip().upper(): value.strip().upper()
                       for cell in next(csv.reader([line])) if "=" in cell
                       for key, value in [cell.split("=", 1)]}
            if line.upper().startswith("*PART,"):
                part = options["NAME"]
            if line.upper().startswith("*ORIENTATION"):
                if part in orientations or options.get("SYSTEM", "RECTANGULAR") != "RECTANGULAR":
                    raise ValueError("部件材料方向重复或不是直角坐标系")
                orientation = []
                orientations[part] = {"name": options["NAME"], "rows": orientation}
            if elastic and "TYPE=LAMINA" not in line.upper().replace(" ", ""):
                raise ValueError("单层弹性材料不是 LAMINA")
            if line.upper().startswith("*SHELL SECTION"):
                if "COMPOSITE" not in line.upper() or "ORIENTATION=" not in line.upper().replace(" ", ""):
                    raise ValueError("壳截面缺少 COMPOSITE 或材料方向")
                current = []
                if part in blocks:
                    raise ValueError("部件重复定义壳截面")
                blocks[part] = current
                section_orientations[part] = options["ORIENTATION"]
            continue
        cells = next(csv.reader([line], skipinitialspace=True))
        if current is not None:
            if len(cells) < 4 or cells[2].strip().upper() != "LAMINA_MATERIAL" or int(cells[1]) != 3:
                raise ValueError("复合壳层缺少材料或角度")
            current.append((float(cells[0]), float(cells[3])))
        elif orientation is not None:
            orientation.append([float(v) for v in cells if v.strip()])
        elif elastic:
            elastic_rows.append([float(v) for v in cells[:6]])
            elastic = False
    part_names = {"skin": "SkinPanel", "web": "WebPart", "flange": "FlangeHalfPart", "cap": "CapPart"}
    expected = {part_names[name].upper(): [(layer["thickness_mm"], layer["angle_deg"]) for layer in layers]
                for name, layers in section_layers(candidate).items()}
    blocks = {name.upper(): rows for name, rows in blocks.items()}
    def equal(a, b):
        return len(a) == len(b) and all(abs(x - y) <= 1e-5 * max(1, abs(y))
                                      for p, q in zip(a, b) for x, y in zip(p, q))
    if set(blocks) != set(expected) or any(not equal(blocks[name], wanted) for name, wanted in expected.items()):
        raise ValueError("INP 层序/层厚与物理输入不一致")
    expected_axes = [[1, 0, 0, 0, 1, 0], [3, 0]]
    if set(orientations) != set(expected) or any(
            data["name"] != section_orientations[part] or len(data["rows"]) != 2
            or [len(row) for row in data["rows"]] != [6, 2] or not equal(data["rows"], expected_axes)
            for part, data in orientations.items()):
        raise ValueError("INP 局部零度方向不是部件 X 轴")
    material = lamina_material(candidate)
    expected_elastic = [material[key] * (1 if key == "nu12" else 1000) for key in
                        ("E1_GPa", "E2_GPa", "nu12", "G12_GPa", "G13_GPa", "G23_GPa")]
    if len(elastic_rows) != 1 or len(elastic_rows[0]) != 6 or any(abs(a - b) > 1e-5 * max(1, abs(b))
            for a, b in zip(elastic_rows[0], expected_elastic)):
        raise ValueError("INP 单层弹性常数与任务材料不一致")
    return {"composite_sections": blocks, "orientations": orientations, "lamina_elastic": elastic_rows[0], "sha256": digest(path)}


ODB_CHECK_SCRIPT = r'''# -*- coding: utf-8 -*-
"""只读提取 ODB 警告节点集、屈曲特征值或静力反力。"""
from __future__ import print_function
import json
import math
import re
import sys
from odbAccess import openOdb

def node_count(nodes):
    if hasattr(nodes, "label"):
        return 1
    return sum(node_count(item) for item in nodes)

odb = openOdb(path=sys.argv[1], readOnly=True)
try:
    warnings = []
    scopes = [("ASSEMBLY", odb.rootAssembly)] + list(odb.rootAssembly.instances.items())
    for scope, owner in scopes:
        for name, region in owner.nodeSets.items():
            upper = name.upper()
            if upper.startswith(("WARN", "ERR")):
                count = node_count(region.nodes)
                warnings.append({"scope": scope, "name": name, "node_count": count,
                                 "fatal": count > 0 and upper.startswith(("WARNNODEMISS", "WARNNODEUNCONNECTED", "ERRNODE"))})
    result = {"warning_node_sets": warnings, "connection_passed": not any(item["fatal"] for item in warnings)}
    if sys.argv[3] == "static":
        frame = odb.steps["MechanicsStatic"].frames[-1]
        if abs(frame.frameValue - 1.0) > 1e-8:
            raise ValueError("Static step has not reached the full reference load")
        result["frame_value"] = frame.frameValue
        for field in ("RF", "RM"):
            result[field] = [{"instance": value.instance.name, "node": value.nodeLabel,
                              "value": [float(item) for item in value.data]}
                             for value in frame.fieldOutputs[field].values]
    else:
        values = []
        for frame in odb.steps["Buckling"].frames:
            match = re.search(r"EigenValue\s*=\s*([-+0-9.eEdD]+)", frame.description, re.I)
            if match:
                value = float(match.group(1).replace("D", "E").replace("d", "e"))
                if math.isnan(value) or math.isinf(value):
                    raise ValueError("Non-finite eigenvalue")
                values.append(value)
        positive = [value for value in values if value > 0]
        if not positive:
            raise ValueError("No positive buckling eigenvalue")
        result["BLF_global"] = positive[0]
        result["mode_eigenvalues"] = values
    with open(sys.argv[2], "w") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
finally:
    odb.close()
'''


def read_odb_checks(odb_path, output_dir, *, static=False):
    """在验收输出目录运行只读后处理，原 ODB 和历史结果保持不变。"""
    from abaqus.job_utils import run_command
    from core.config_loader import load_app_config

    output_dir.mkdir(parents=True, exist_ok=True)
    script, target = output_dir / "read_odb.py", output_dir / "odb_checks.json"
    script.write_text(ODB_CHECK_SCRIPT, encoding="utf-8")
    config = load_app_config()["abaqus"]
    before = digest(odb_path)
    code, stdout, stderr = run_command([config["command"], "python", str(script), str(Path(odb_path).resolve()),
                                       str(target), "static" if static else "buckling"],
                                      workdir=output_dir, timeout=config["job_timeout_seconds"])
    save(output_dir / "odb_reader_log.json", {"exit_code": code, "stdout": stdout, "stderr": stderr})
    if code != 0 or not target.is_file():
        raise RuntimeError("ODB 检查失败，参见 " + str(output_dir / "odb_reader_log.json"))
    result = json.loads(target.read_text(encoding="utf-8"))
    if digest(odb_path) != before:
        raise RuntimeError("读取期间 ODB 内容发生变化")
    result["odb_sha256"] = before
    return result


def derive_static_deck(source, target):
    """保留模型与初始约束，独立静力步施加同一集中力，不写回屈曲输入。"""
    blocks = inp_blocks(source)
    deck = MechanicsDeck(source)
    start = next(index for index, block in enumerate(blocks) if block["keyword"] == "*STEP")
    lines = [line for block in blocks[:start] for line in [block["header"], *block["lines"]]]
    lines.extend(["*Step, name=MechanicsStatic, nlgeom=NO", "*Static", "1., 1."])
    # 生产屈曲的两个扰动载荷工况必须与 Initial 保持同一支承，静力副本继承 Initial。
    if any(deck.constraints(group) != deck.constraints() for group in ("load_case_1", "load_case_2")):
        raise ValueError("静力副本要求屈曲载荷工况与 Initial 支承一致")
    for block in blocks[start:]:
        if block["keyword"] == "*CLOAD":
            lines.extend([block["header"], *block["lines"]])
    lines.extend(["*Output, field, frequency=1", "*Node Output", "U, RF, RM", "*End Step"])
    target.write_text("\n".join(lines) + "\n", encoding="ascii")
    return {"source_inp_sha256": digest(source), "static_inp_sha256": digest(target)}


def check_static_reactions(source, audit, *, rtol, force_atol, moment_atol):
    deck = MechanicsDeck(source)
    fields = {}
    for field, dofs in (("RF", {1, 2, 3}), ("RM", {4, 5, 6})):
        values = {}
        for row in audit[field]:
            node = (row["instance"].upper(), int(row["node"]))
            vector = [number(value) for value in row["value"]]
            if node not in deck.nodes or node in values or len(vector) != 3:
                raise ValueError("静力反力节点或分量无效")
            values[node] = vector
        required = {node for node, dof in deck.constraints() if dof in dofs}
        if not values or not required.issubset(values):
            raise ValueError("静力反力字段未覆盖约束节点: " + field)
        fields[field] = values
    force, moment = force_moment(deck.nodes, deck.forces())
    reaction_force, reaction_moment = force_moment(deck.nodes, fields["RF"])
    residual_force = [a + b for a, b in zip(force, reaction_force)]
    residual_moment = [moment[i] + reaction_moment[i] + sum(value[i] for value in fields["RM"].values()) for i in range(3)]
    scale = sum(math.sqrt(sum(value * value for value in vector)) for vector in deck.forces().values())
    length = max(max(point[axis] for point in deck.nodes.values()) - min(point[axis] for point in deck.nodes.values()) for axis in range(3))
    passed = (audit["connection_passed"] and abs(number(audit["frame_value"]) - 1) <= 1e-8
              and all(abs(value) <= force_atol + rtol * scale for value in residual_force)
              and all(abs(value) <= moment_atol + rtol * scale * length for value in residual_moment))
    return {"passed": passed, "residual_force_N": residual_force, "residual_moment_Nmm": residual_moment,
            "force_scale_N": scale, "moment_scale_Nmm": scale * length, "odb_checks": audit,
            "tolerances": {"relative": rtol, "force_N": force_atol, "moment_Nmm": moment_atol}}


def run_static_check(source, output_dir, args):
    from abaqus.job_utils import run_command
    from core.config_loader import load_app_config

    output_dir.mkdir(parents=True, exist_ok=False)
    target = output_dir / "MechanicsStatic.inp"
    provenance = derive_static_deck(source, target)
    config = load_app_config()["abaqus"]
    code, stdout, stderr = run_command([config["command"], "job=MechanicsStatic", "input=MechanicsStatic.inp", "interactive"],
                                      workdir=output_dir, timeout=config["job_timeout_seconds"])
    save(output_dir / "solver_log.json", {"exit_code": code, "stdout": stdout, "stderr": stderr})
    if code != 0 or not (output_dir / "MechanicsStatic.odb").is_file():
        raise RuntimeError("派生静力求解失败，参见 " + str(output_dir / "solver_log.json"))
    audit = read_odb_checks(output_dir / "MechanicsStatic.odb", output_dir / "inspection", static=True)
    result = check_static_reactions(source, audit, rtol=args.balance_rtol,
                                    force_atol=args.force_atol_n, moment_atol=args.moment_atol_nmm)
    if digest(source) != provenance["source_inp_sha256"]:
        raise RuntimeError("生产屈曲 INP 在静力检查期间发生变化")
    return {**result, **provenance}


def mechanics_identity(candidate):
    """用实际默认网格参数匹配历史输入；载荷参考值仍由 INP 检查独立计算。"""
    from abaqus.runtime_build_panel import resolve_analysis_spec
    from core.laminate import analysis_contract, fingerprint

    contract = analysis_contract(candidate)
    spec = resolve_analysis_spec(candidate)
    physical = {key: value for key, value in contract.items() if key not in ("fingerprint", "solver_fingerprint", "analysis")}
    physical["analysis"] = {**candidate.get("analysis", {}), "mesh_size_mm": spec["mesh_size_mm"],
                            "buckling_modes": spec["buckling_modes"]}
    return fingerprint(physical)


def build_mechanics_plan(args):
    from core.doe_sampler import DOESampler
    from core.layup import normalize_layup
    from core.task_parser import TaskParser
    from core.task_contract import build_design_constraints, normalize_boundary_conditions, normalize_load_conditions

    task = TaskParser().parse_instruction("长度360mm，宽度300mm，压缩120kN/m，四边简支，生成1个候选，初筛1个")
    bases, tasks = {}, {}
    baseline_path = getattr(args, "baseline_run_dir", None)
    baseline = None
    if baseline_path is not None:
        baseline_path = Path(baseline_path) / "mechanics.json"
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        if baseline.get("status") != "passed":
            raise ValueError("力学基准必须来自 passed 的 mechanics 记录")
        prior_jobs = {job["job_id"]: job for job in baseline["jobs"]}
    for stype in args.stiffener_types:
        typed = deepcopy(task)
        typed["task"]["stiffener_type"] = stype
        typed["task"]["candidate_generation_preferences"]["stiffener_types"] = [stype]
        if baseline is not None:
            original = prior_jobs[baseline["convergence_jobs"][stype][-1]]
            if original["status"] != "passed" or original["candidate"]["stiffener_type"] != stype:
                raise ValueError("力学基准候选的筋型或状态不一致")
            bases[stype] = deepcopy(original["candidate"])
            geometry = bases[stype]["geometry"]
            typed["task"].update(
                geometry_envelope={"panel_length_mm": [geometry["panel_length_mm"]] * 2,
                                   "panel_width_mm": [geometry["panel_width_mm"]] * 2,
                                   "max_stiffener_height_mm": geometry["stiffener_height_mm"]},
                fixed_geometry=deepcopy(geometry), material_system=deepcopy(bases[stype]["material_system"]))
        else:
            candidates = DOESampler().sample_candidates(typed, 1, stiffener_type=stype)
            if len(candidates) != 1:
                raise ValueError("无法生成力学基准候选: " + stype)
            bases[stype] = candidates[0]
            bases[stype]["layup"] = normalize_layup("[0/45/-45/90]s")
        tasks[stype] = typed
    jobs, identities, convergence, matrix = [], {}, {}, []
    def schedule(stype, mesh, load_type, boundary):
        candidate, typed = deepcopy(bases[stype]), deepcopy(tasks[stype])
        loads = normalize_load_conditions({"type": load_type, "Nx_kN_per_m": 0.0 if load_type == "in_plane_shear" else args.nx_kn_per_m,
                                            "Nxy_kN_per_m": 0.0 if load_type == "axial_compression" else args.nxy_kn_per_m})
        supports = normalize_boundary_conditions(boundary)
        typed["task"].update(load_conditions=loads, boundary_conditions=supports)
        candidate.update(load_conditions=loads, boundary_conditions=supports,
                         task_constraints=build_design_constraints(typed, stype),
                         analysis={**candidate.get("analysis", {}), "mesh_size_mm": mesh})
        identity = mechanics_identity(candidate)
        if identity not in identities:
            cid = "C" + str(len(jobs) + 1)
            candidate.update(candidate_id=cid, display_name=cid)
            jobs.append({"job_id": cid, "status": "not_run", "candidate": candidate, "identity": identity})
            identities[identity] = cid
        return identities[identity]
    for stype in args.stiffener_types:
        convergence[stype] = [schedule(stype, mesh, "axial_compression", "SSSS") for mesh in args.mesh_sizes_mm]
    for stype in args.matrix_stiffener_types:
        for boundary in ("SSSS", "SSCC", "CCCC"):
            for load_type in ("axial_compression", "in_plane_shear", "compression_shear"):
                matrix.append({"stiffener_type": stype, "boundary": boundary, "load_type": load_type,
                               "job_id": schedule(stype, args.mesh_sizes_mm[-1], load_type, boundary)})
    return {"jobs": jobs, "convergence_jobs": convergence, "matrix": matrix,
            "baseline_source": {"path": str(baseline_path), "sha256": digest(baseline_path)} if baseline is not None else None,
            "mesh_sizes_mm": args.mesh_sizes_mm, "convergence_rtol": args.convergence_rtol,
            "balance_tolerances": {"relative": args.balance_rtol, "force_N": args.force_atol_n,
                                   "moment_Nmm": args.moment_atol_nmm, "connection_mm": args.connection_atol_mm}}


def convergence_check(jobs, threshold):
    values = [number(job["BLF_global"]) for job in jobs]
    if any(value <= 0 for value in values):
        raise ValueError("收敛检查要求有限正 BLF")
    changes = [abs(coarse - fine) / fine for coarse, fine in zip(values, values[1:])]
    counts = [job["mechanics"]["skin_node_count"] for job in jobs]
    refined = all(coarse < fine for coarse, fine in zip(counts, counts[1:]))
    return {"passed": refined and changes[-1] <= threshold, "BLF_global": values,
            "relative_changes": changes, "threshold": threshold, "skin_node_counts": counts,
            "acceptance_pair": "middle_to_fine", "acceptance_relative_change": changes[-1],
            "mesh_refined": refined, "mode_shape_check": "manual_review_required"}


def audit_mechanics_case(candidate, inp, odb, output, args):
    sections = inspect_input_deck(inp, candidate)
    mechanics = inspect_mechanics_deck(inp, candidate, rtol=args.balance_rtol,
                                       force_atol=args.force_atol_n, moment_atol=args.moment_atol_nmm)
    connections = inspect_connections(inp, candidate, geometric_atol=args.connection_atol_mm)
    audit = read_odb_checks(odb, output)
    return {"passed": mechanics["passed"] and connections["passed"] and audit["connection_passed"],
            "sections": sections, "mechanics": mechanics, "connections": connections,
            "odb_checks": audit, "BLF_global": audit["BLF_global"], "abaqus_inp": str(inp), "abaqus_odb": str(odb)}


def audit_reusable_runs(args, report, target):
    """只读审计已有工件；代码变更需有审查依据，复用仍要求物理输入一致。"""
    available = {}
    build_sources = ("abaqus/runtime_build_panel.py", "core/laminate.py")
    current = {name: digest(ROOT / name) for name in build_sources}
    for root in args.reuse_run_dir:
        manifest = json.loads((root / "acceptance.json").read_text(encoding="utf-8"))
        source_hashes = manifest_code_hashes(manifest)
        if any(not isinstance(source_hashes.get(name), str) or len(source_hashes[name]) != 64
               or any(c not in "0123456789abcdef" for c in source_hashes[name]) for name in build_sources):
            raise ValueError("历史运行缺少有效建模源码指纹: " + str(root))
        same_build = all(source_hashes[name] == value for name, value in current.items())
        if manifest.get("status") != "passed":
            raise ValueError("历史运行必须有 passed 终态: " + str(root))
        paths = sorted((root / "data/abaqus_runs").glob("*/candidate_input.json"))
        if not paths:
            raise ValueError("历史运行没有候选输入: " + str(root))
        for path in paths:
            candidate = json.loads(path.read_text(encoding="utf-8"))
            cid = candidate["candidate_id"]
            if path.parent.name != cid or not cid.startswith("C") or not cid[1:].isdigit():
                raise ValueError("历史作业编号与工件目录不一致")
            entry = {"origin": str(path.parent), "input_sha256": digest(path), "same_build_code": same_build, "status": "running",
                     "source_build_sha256": {name: source_hashes[name] for name in build_sources},
                     "current_build_sha256": current, "build_change_review": args.reuse_build_change_reason if not same_build else None}
            report["existing_runs"].append(entry)
            save(target, report)
            try:
                entry.update(audit_mechanics_case(candidate, path.parent / (cid + ".inp"), path.parent / (cid + ".odb"),
                                                  args.output_dir / "existing_checks" / str(len(report["existing_runs"])), args))
                entry["status"] = "passed" if entry["passed"] else "failed"
                entry["identity"] = mechanics_identity(candidate)
                if not entry["passed"]:
                    raise ValueError("历史工件未通过力学检查: " + str(path.parent))
                if same_build or args.reuse_build_change_reason:
                    available.setdefault(entry["identity"], entry)
            except BaseException as exc:
                entry.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=str(exc))
                raise
            finally:
                save(target, report)
    matches = {job["identity"] for job in report["jobs"]}
    if any(item["identity"] in matches and not item["same_build_code"] and not item["build_change_review"] for item in report["existing_runs"]):
        raise ValueError("存在同物理输入但建模源码不同的历史作业；需 --reuse-build-change-reason 记录审查原因，未自动重跑")
    return available


def run_mechanics(args):
    from agents.fem_agent import FEMAgent
    from core.laminate import ANALYSIS_VERSION, analysis_contract, solver_fingerprint

    report = {**build_mechanics_plan(args), "status": "running", "existing_runs": [], "convergence": {},
              "static_checks": [], "static_status": "pending" if args.static_check else "not_requested"}
    target = args.output_dir / "mechanics.json"
    save(target, report)
    solver_source = solver_fingerprint()
    active = None
    try:
        reusable = audit_reusable_runs(args, report, target)
        for job in report["jobs"]:
            active = job
            job["status"] = "running"
            save(target, report)
            candidate = job["candidate"]
            contract = analysis_contract(candidate)
            if (contract["version"] != ANALYSIS_VERSION or contract.get("load_definition") != "self_equilibrated_skin_edge_tractions"
                    or contract.get("connection_definition") != "fixed_midsurface_offset_ties"
                    or contract.get("boundary_definition") != {"support": "skin_edge_transverse", "in_plane_anchors": "origin_xy_axis_y"}):
                raise ValueError("力学验收要求自平衡牵引和板外支承契约")
            if solver_fingerprint() != solver_source:
                raise RuntimeError("验收期间求解源码发生变化")
            if job["identity"] in reusable:
                origin = reusable[job["identity"]]
                job.update({key: origin[key] for key in ("sections", "mechanics", "connections", "odb_checks", "BLF_global", "abaqus_inp", "abaqus_odb")})
                job.update(status="passed", execution="reused_odb", origin=origin["origin"])
            else:
                outcome = FEMAgent().run(candidate)
                job.update(execution="fem", result=outcome)
                if outcome["status"] != "success" or outcome.get("analysis_contract") != contract:
                    raise RuntimeError("求解失败或未绑定当前物理契约: " + job["job_id"])
                checks = audit_mechanics_case(candidate, outcome["abaqus_inp"], outcome["abaqus_odb"],
                                              args.output_dir / "job_checks" / job["job_id"], args)
                job.update(checks)
                if not checks["passed"]:
                    raise ValueError("INP/ODB 力学检查失败: " + job["job_id"])
                if not math.isclose(number(outcome["BLF_global"]), checks["BLF_global"], rel_tol=1e-10, abs_tol=1e-12):
                    raise ValueError("结果 BLF 与 ODB 读取值不一致")
                job["status"] = "passed"
            if solver_fingerprint() != solver_source:
                raise RuntimeError("验收期间求解源码发生变化")
            active = None
            save(target, report)
        by_id = {job["job_id"]: job for job in report["jobs"]}
        report["convergence"] = {stype: convergence_check([by_id[cid] for cid in ids], args.convergence_rtol)
                                 for stype, ids in report["convergence_jobs"].items()}
        if args.static_check:
            for cid in dict.fromkeys(item["job_id"] for item in report["matrix"]):
                active = {"job_id": cid, "status": "running"}
                report["static_checks"].append(active)
                save(target, report)
                active.update(run_static_check(by_id[cid]["abaqus_inp"], args.output_dir / "static" / cid, args))
                active["status"] = "passed" if active["passed"] else "failed"
                active = None
                save(target, report)
            report["static_status"] = "passed" if all(item["passed"] for item in report["static_checks"]) else "failed"
        report["passed"] = (all(item["passed"] for item in report["convergence"].values())
                            and report["static_status"] in ("passed", "not_requested"))
        report["status"] = "passed" if report["passed"] else "failed"
    except BaseException as exc:
        state = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        if active is not None:
            active.update(status=state, error_type=type(exc).__name__, error=str(exc))
        report.update(status=state, passed=False, error_type=type(exc).__name__, error=str(exc))
        if args.static_check and report["static_status"] == "pending":
            report["static_status"] = "not_completed"
        if isinstance(exc, KeyboardInterrupt):
            raise
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        report["new_fem_jobs"] = sum(job.get("execution") == "fem" for job in report["jobs"])
        report["reused_jobs"] = sum(job.get("execution") == "reused_odb" for job in report["jobs"])
        save(target, report)
    return report


def run_layup_comparison(stiffener_types):
    from agents.fem_agent import FEMAgent
    from core.doe_sampler import DOESampler
    from core.layup import normalize_layup
    from core.task_parser import TaskParser
    from core.io_utils import write_json
    from core import paths
    from copy import deepcopy

    task = TaskParser().parse_instruction("长度360mm，宽度300mm，压缩120kN/m，生成1个候选，初筛1个")
    pairs = []
    for stype in stiffener_types:
        typed = deepcopy(task)
        typed["task"]["stiffener_type"] = stype
        typed["task"]["candidate_generation_preferences"]["stiffener_types"] = [stype]
        base = DOESampler().sample_candidates(typed, 1, stiffener_type=stype)[0]
        results, inputs = [], []
        for text in ("[0/45/-45/90]s", "[90/45/-45/0]s"):
            candidate = deepcopy(base)
            candidate["candidate_id"] = f"C{len(pairs) * 2 + len(results) + 1}"
            candidate["display_name"] = candidate["candidate_id"]
            candidate["layup"] = normalize_layup(text)
            result = FEMAgent().run(candidate)
            write_json(paths.RESULTS_DIR / f"{candidate['candidate_id']}_result.json", result)
            if result["status"] != "success":
                raise RuntimeError(f"{candidate['candidate_id']} 求解失败，参见隔离目录中的结果与日志")
            inputs.append(inspect_input_deck(result["abaqus_inp"], candidate))
            results.append(result)
        relative_change = abs(results[0]["BLF_global"] - results[1]["BLF_global"]) / max(r["BLF_global"] for r in results)
        pair = {"stiffener_type": stype, "geometry": base["geometry"], "material": base["material_system"],
                "load_conditions": base["load_conditions"], "boundary_conditions": base["boundary_conditions"],
                "results": results, "inp_checks": inputs, "relative_BLF_change": relative_change,
                "passed": relative_change > 1e-6}
        pairs.append(pair)
        write_json(paths.RESULTS_DIR / f"{stype}_layup_comparison.json", pair)
    return {"pairs": pairs, "passed": all(pair["passed"] for pair in pairs)}


def run_api_workflow(args):
    from agents.orchestrator import OrchestratorAgent
    from core.llm_backend import LLMBackend

    api_calls = []
    original_chat = LLMBackend.chat
    def recorded_chat(backend, system_prompt, user_prompt, *positional, **kwargs):
        budget = positional[0] if positional else kwargs.get("max_tokens_override")
        record = {"model": backend.model, "system_prompt": system_prompt, "user_prompt": user_prompt, "status": "running",
                  "max_tokens": budget or backend.max_tokens, "temperature": backend.temperature,
                  "extra_body": backend.extra_body}
        api_calls.append(record)
        try:
            answer = original_chat(backend, system_prompt, user_prompt, *positional, **kwargs)
            record.update(status="completed", response=answer)
            return answer
        except Exception as exc:
            record.update(status="failed", error_type=type(exc).__name__)
            raise
        finally:
            record["response_metadata"] = backend.last_response
            save(args.output_dir / "api_calls.json", api_calls)
    LLMBackend.chat = recorded_chat
    try:
        agent = OrchestratorAgent()
        # 验证不修改代理权重；训练和独立精度验收是单独的操作。
        agent.knowledge_agent.config = {**agent.knowledge_agent.config,
                                       "pipeline": {**agent.knowledge_agent.config["pipeline"], "auto_retrain_surrogate": False}}
        if agent.report_gen.llm_backend is None:
            raise RuntimeError("外部 API 后端未初始化，不能作为 API 流程验收")
        result = agent.run(args.instruction)
    finally:
        LLMBackend.chat = original_chat
    source_counts = {}
    for candidate in result["candidates"]:
        source_counts[candidate["source"]] = source_counts.get(candidate["source"], 0) + 1
    report_paths = {name: result["report"].get(name) for name in ("markdown_path", "pdf_path", "facts_path")}
    reports_complete = all(value and Path(value).is_file() and Path(value).stat().st_size > 0
                           for value in report_paths.values())
    success = (bool(result["results"]) and len(result["results"]) == len(result["top_candidates"])
               and all(item["status"] == "success" for item in result["results"]) and reports_complete)
    inp_checks = []
    for candidate, outcome in zip(result["top_candidates"], result["results"]):
        if outcome["status"] == "success":
            inp_checks.append(inspect_input_deck(outcome["abaqus_inp"], candidate))
    requirement_used = result["task"]["task"].get("requirement_extraction", {}).get("method") == "llm_grounded"
    knowledge_context = json.loads(Path(report_paths["facts_path"]).read_text(encoding="utf-8")).get("knowledge_context", {}) if reports_complete else {}
    knowledge_used = bool(knowledge_context.get("included_chunk_ids"))
    model_explanation = result["report"]["llm_explanation_used"]
    return {"workflow": result, "source_counts": source_counts, "api_call_count": len(api_calls),
            "llm_requirement_used": requirement_used, "knowledge_used": knowledge_used,
            "llm_explanation_used": model_explanation,
            "business_complete": success, "reports_complete": reports_complete, "report_paths": report_paths,
            "inp_checks": inp_checks,
            "passed": success and bool(api_calls) and requirement_used and knowledge_used
                      and (model_explanation or not args.require_llm_explanation)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("layup", "api", "mechanics"))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--stiffener-types", nargs="+", choices=("T", "BLADE", "L", "HAT"), default=["T", "BLADE", "L", "HAT"])
    parser.add_argument("--instruction", default="请为机翼下蒙皮壁板设计T型筋方案，长度360mm，宽度300mm，压缩载荷120kN/m，生成2个候选，初筛保留1个，BLF目标1.2")
    parser.add_argument("--require-llm-explanation", action="store_true")
    parser.add_argument("--mesh-sizes-mm", nargs=3, type=float, default=[18.0, 12.0, 8.0])
    parser.add_argument("--convergence-rtol", type=float, help="mechanics 必须预先指定，例如 0.05")
    parser.add_argument("--matrix-stiffener-types", nargs="+", choices=("T", "BLADE", "L", "HAT"), default=["T"])
    parser.add_argument("--nx-kn-per-m", type=float, default=120.0)
    parser.add_argument("--nxy-kn-per-m", type=float, default=60.0)
    parser.add_argument("--balance-rtol", type=float, default=1e-5)
    parser.add_argument("--force-atol-n", type=float, default=1e-5)
    parser.add_argument("--moment-atol-nmm", type=float, default=1e-3)
    parser.add_argument("--connection-atol-mm", type=float, default=0.01)
    parser.add_argument("--static-check", action="store_true")
    parser.add_argument("--reuse-run-dir", action="append", type=Path, default=[])
    parser.add_argument("--baseline-run-dir", type=Path, help="读取已通过 mechanics 的固定几何/材料/层序，不重新采样")
    parser.add_argument("--reuse-build-change-reason", default="", help="建模源码不同但物理输入相同的复用审查依据")
    args = parser.parse_args()
    if args.mode == "mechanics":
        numbers = [*args.mesh_sizes_mm, args.convergence_rtol, args.nx_kn_per_m, args.nxy_kn_per_m,
                   args.balance_rtol, args.force_atol_n, args.moment_atol_nmm, args.connection_atol_mm]
        if any(value is None or not math.isfinite(value) or value <= 0 for value in numbers):
            parser.error("mechanics 的网格、载荷及容差必须为有限正数，--convergence-rtol 必须显式指定")
        if not 0 < args.convergence_rtol < 1 or not args.mesh_sizes_mm[0] > args.mesh_sizes_mm[1] > args.mesh_sizes_mm[2]:
            parser.error("网格尺寸须严格递减，收敛相对容差须在 (0,1) 内")
        if (len(set(args.stiffener_types)) != len(args.stiffener_types)
                or len(set(args.matrix_stiffener_types)) != len(args.matrix_stiffener_types)
                or not set(args.matrix_stiffener_types).issubset(args.stiffener_types)):
            parser.error("筋型不得重复，工况矩阵筋型必须属于网格收敛筋型集合")
        if args.reuse_run_dir and args.baseline_run_dir is None:
            parser.error("复用已有作业须用 --baseline-run-dir 指定固定物理基准，避免重新采样")
    args.output_dir = args.output_dir.resolve()
    args.reuse_run_dir = [path.resolve() for path in args.reuse_run_dir]
    if args.baseline_run_dir is not None:
        args.baseline_run_dir = args.baseline_run_dir.resolve()
    args.reuse_build_change_reason = args.reuse_build_change_reason.strip()
    isolate_runtime(args.output_dir)
    os.environ["CSDM_panel_DISABLE_LLM_AUTO"] = "0" if args.mode == "api" else "1"
    report = {"status": "running", "mode": args.mode, "started_at": datetime.now(timezone.utc).isoformat(),
              "code_sha256": {path.relative_to(ROOT).as_posix(): digest(path) for directory in ("core", "agents", "abaqus", "scripts")
                              for path in (ROOT / directory).glob("*.py")},
              "model_sha256": {path.relative_to(args.output_dir / "models").as_posix(): digest(path)
                               for path in (args.output_dir / "models").rglob("*") if path.is_file()}}
    target = args.output_dir / "acceptance.json"
    save(target, report)
    try:
        if args.mode == "layup":
            result = run_layup_comparison(args.stiffener_types)
        elif args.mode == "api":
            result = run_api_workflow(args)
        else:
            result = run_mechanics(args)
        report.update(status="passed" if result["passed"] else "failed", result=result)
    except BaseException as exc:
        report.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                      error_type=type(exc).__name__, error=str(exc))
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        save(target, report)
    print(json.dumps({"status": report["status"], "evidence": str(target)}, ensure_ascii=False))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
