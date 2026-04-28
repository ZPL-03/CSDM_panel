from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

from core.io_utils import read_json
from core.paths import CASE_LIBRARY_DIR, CASES_DIR
from core.task_contract import equivalent_in_plane_load, normalize_boundary_conditions, normalize_load_conditions


@dataclass
class RetrievedCase:
    record: Dict
    score: float


def _is_pass_verdict(value: object) -> bool:
    """仅将明确表示“通过”的结论视为通过，避免把“不通过”误判为通过。"""
    verdict = str(value or "").strip()
    if not verdict or verdict in {"None", "失败"}:
        return False
    return verdict == "通过"


class CaseRetriever:
    """基于结构化字段检索历史案例，供案例迁移路径直接复用。"""

    def __init__(self, include_archive: bool = True, include_formal: bool = True) -> None:
        self.include_archive = include_archive
        self.include_formal = include_formal

    def _candidate_paths(self) -> List:
        paths = []
        if self.include_archive:
            paths.extend(sorted(CASES_DIR.glob("CASE_*.json")))
        if self.include_formal:
            paths.extend(sorted(CASE_LIBRARY_DIR.glob("CASE_*.json")))
        deduped = []
        seen = set()
        for path in paths:
            if path.name in seen:
                continue
            seen.add(path.name)
            deduped.append(path)
        return deduped

    def _load_records(self) -> Iterable[Dict]:
        for path in self._candidate_paths():
            try:
                payload = read_json(path)
            except Exception:
                continue
            if isinstance(payload, dict):
                yield payload

    def _has_design(self, record: Dict) -> bool:
        design = record.get("design", {})
        return isinstance(design, dict) and bool(design)

    def _is_transferable(self, record: Dict) -> bool:
        results = record.get("abaqus_results", {})
        if not isinstance(results, dict):
            return False
        if results.get("status") != "success":
            return False
        if not _is_pass_verdict(results.get("verdict") or record.get("verdict")):
            return False
        return self._has_design(record)

    def _material_name(self, payload: Dict) -> str:
        material = payload.get("material_system", {})
        if not isinstance(material, dict):
            return ""
        return str(material.get("name") or "").strip().lower()

    def _geometry_center(self, task: Dict) -> Dict[str, float]:
        envelope = task.get("geometry_envelope", {})
        panel_length = envelope.get("panel_length_mm", [600.0, 800.0])
        panel_width = envelope.get("panel_width_mm", [500.0, 700.0])
        try:
            length_center = (float(panel_length[0]) + float(panel_length[1])) / 2.0
        except Exception:
            length_center = 700.0
        try:
            width_center = (float(panel_width[0]) + float(panel_width[1])) / 2.0
        except Exception:
            width_center = 600.0
        try:
            max_height = float(envelope.get("max_stiffener_height_mm", 50.0))
        except Exception:
            max_height = 50.0
        return {
            "panel_length_mm": length_center,
            "panel_width_mm": width_center,
            "max_stiffener_height_mm": max_height,
        }

    def _match_level(self, task: Dict, record: Dict) -> int:
        design = record.get("design", {})
        task_load = normalize_load_conditions(task.get("load_conditions", {}))
        task_boundary = normalize_boundary_conditions(task.get("boundary_conditions", {}))
        design_load = normalize_load_conditions(design.get("load_conditions", {}))
        design_boundary = normalize_boundary_conditions(design.get("boundary_conditions", {}))
        if str(task.get("stiffener_type") or "T") != str(design.get("stiffener_type") or "T"):
            return -1

        task_material = self._material_name(task)
        design_material = self._material_name(design)
        same_load = task_load.get("type") == design_load.get("type")
        same_boundary = task_boundary.get("type") == design_boundary.get("type")
        same_material = not task_material or not design_material or task_material == design_material

        if same_load and same_boundary and same_material:
            return 3
        if same_load and same_boundary:
            return 2
        return -1

    def _score(self, task: Dict, record: Dict) -> float:
        design = record.get("design", {})
        geometry = design.get("geometry", {})
        design_targets = task.get("design_targets", {})
        abaqus_results = record.get("abaqus_results", {})
        geometry_center = self._geometry_center(task)
        load_gap = abs(
            equivalent_in_plane_load(task.get("load_conditions", {}))
            - equivalent_in_plane_load(design.get("load_conditions", {}))
        )
        length_gap = abs(float(geometry.get("panel_length_mm", geometry_center["panel_length_mm"])) - geometry_center["panel_length_mm"])
        width_gap = abs(float(geometry.get("panel_width_mm", geometry_center["panel_width_mm"])) - geometry_center["panel_width_mm"])
        height_gap = abs(float(geometry.get("stiffener_height_mm", geometry_center["max_stiffener_height_mm"])) - geometry_center["max_stiffener_height_mm"])
        target_blf = float(design_targets.get("BLF_min", 1.2) or 1.2)
        result_blf = float(abaqus_results.get("BLF_global", target_blf) or target_blf)
        blf_gap = abs(result_blf - target_blf)
        return load_gap * 1.0 + length_gap * 0.02 + width_gap * 0.02 + height_gap * 0.05 + blf_gap * 100.0

    def retrieve_similar_cases(self, task: Dict, top_k: int = 5) -> List[Dict]:
        """返回结构上接近的历史案例，用于分析和相似样本观察。"""
        ranked: List[tuple[int, RetrievedCase]] = []
        for record in self._load_records():
            if not self._has_design(record):
                continue
            match_level = self._match_level(task, record)
            if match_level < 0:
                continue
            ranked.append((match_level, RetrievedCase(record=record, score=self._score(task, record))))
        ranked.sort(key=lambda item: (-item[0], item[1].score))
        return [item.record for _, item in ranked[:top_k]]

    def retrieve_transferable_cases(self, task: Dict, top_k: int = 5) -> List[Dict]:
        """只返回可直接迁移的通过样本，供 CASE_TRANSFER 路径使用。"""
        ranked: List[RetrievedCase] = []
        for record in self._load_records():
            if not self._is_transferable(record):
                continue
            match_level = self._match_level(task, record)
            if match_level < 3:
                continue
            ranked.append(RetrievedCase(record=record, score=self._score(task, record)))
        ranked.sort(key=lambda item: item.score)
        return [item.record for item in ranked[:top_k]]

    def transfer_candidates(self, task: Dict, top_k: int = 2) -> List[Dict]:
        transferred: List[Dict] = []
        for record in self.retrieve_transferable_cases(task, top_k=top_k):
            design = record.get("design", {})
            if isinstance(design, dict) and design:
                transferred.append(design)
        return transferred
