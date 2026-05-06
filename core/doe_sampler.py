"""DOE 候选采样器。"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from core.config_loader import load_app_config, load_material_db, load_param_ranges
from core.id_utils import format_candidate_id
from core.rule_checker import RuleChecker
from core.stiffener_profile import (
    load_param_ranges_for_type,
    required_geometry_params,
    resolve_stiffener_type,
)
from core.task_contract import task_payload_from_request


class DOESampler:
    """使用轻量 LHS 生成结构参数、铺层与材料组合（支持多筋型）。"""

    def __init__(self) -> None:
        app_config = load_app_config()
        self.random_seed = int(app_config["pipeline"]["random_seed"])
        self.param_ranges = load_param_ranges()
        self.material_db = load_material_db()
        self.rule_checker = RuleChecker()
        self.layup_templates = list(self.param_ranges.get("layup_templates", []))
        if not self.layup_templates:
            self.layup_templates = [{"name": "QI_14", "skin_layup": "[45/-45/0/90/0/-45/45]s"}]
        self.material_catalog = self._build_material_catalog()

    def _build_material_catalog(self) -> List[Dict]:
        catalog: List[Dict] = []
        for material_key, payload in self.material_db.items():
            catalog.append(
                {
                    "name": payload.get("display_name", material_key),
                    "density_kg_per_m3": float(payload.get("density_kg_per_m3", 1600.0)),
                    "E1_GPa": float(payload.get("E1_GPa", 181.0)),
                    "E2_GPa": float(payload.get("E2_GPa", 10.3)),
                    "G12_GPa": float(payload.get("G12_GPa", 7.17)),
                    "nu12": float(payload.get("nu12", 0.28)),
                    "material_key": material_key,
                }
            )
        return catalog

    def _task_payload(self, task: Dict) -> Dict:
        return task_payload_from_request(task)

    def _material_candidates(self, task: Dict) -> List[Dict]:
        material = dict(self._task_payload(task).get("material_system", {}))
        if material.get("is_user_specified", False):
            return [material]
        return [dict(item) for item in self.material_catalog] or [material]

    def _select_material(self, task: Dict, ordinal_index: int) -> Dict:
        options = self._material_candidates(task)
        if not options:
            return dict(self._task_payload(task).get("material_system", {}))
        return dict(options[(max(ordinal_index, 1) - 1) % len(options)])

    def _lhs(self, dimensions: int, samples: int, seed_offset: int = 0) -> np.ndarray:
        rng = np.random.default_rng(self.random_seed + seed_offset)
        result = np.zeros((samples, dimensions))
        for dim in range(dimensions):
            cut = np.linspace(0, 1, samples + 1)
            points = cut[:-1] + rng.random(samples) * (cut[1:] - cut[:-1])
            rng.shuffle(points)
            result[:, dim] = points
        return result

    def _parse_layup_sequence(self, layup_text: str) -> List[str]:
        text = (layup_text or "").strip()
        if not text:
            return []
        symmetric = text.endswith("s")
        if symmetric:
            text = text[:-1]
        base = [item.strip() for item in text.strip("[] ").split("/") if item.strip()]
        return base + list(reversed(base)) if symmetric else base

    def _layup_payload(self, rng: np.random.Generator) -> Dict:
        template = self.layup_templates[int(rng.integers(0, len(self.layup_templates)))]
        layup_text = str(template.get("skin_layup", "[45/-45/0/90/0/-45/45]s"))
        sequence = self._parse_layup_sequence(layup_text)
        total = max(len(sequence), 1)
        count_0 = sum(1 for angle in sequence if angle == "0")
        count_45 = sum(1 for angle in sequence if angle in {"45", "-45"})
        count_90 = sum(1 for angle in sequence if angle == "90")
        return {
            "template_name": template.get("name", "LAYUP"),
            "skin_layup": layup_text,
            "skin_f0": round(count_0 / total, 3),
            "skin_f45": round(count_45 / total, 3),
            "skin_f90": round(count_90 / total, 3),
            "ply_count": total,
        }

    def _estimate_rationale(self, material_name: str, layup_name: str) -> str:
        return f"DOE 拉丁超立方探索生成 | 材料={material_name} | 铺层={layup_name}"

    def sample_candidates(
        self,
        task: Dict,
        n_samples: int,
        start_index: int = 1,
        strict_solver_window: bool = False,
        stiffener_type: str = "T",
        id_factory=None,
    ) -> List[Dict]:
        valid_candidates: List[Dict] = []
        generation_round = 0
        candidate_id_factory = id_factory or format_candidate_id
        stype = resolve_stiffener_type(stiffener_type)
        feature_order = required_geometry_params(stype)
        type_ranges = load_param_ranges_for_type(stype)

        while len(valid_candidates) < n_samples and generation_round < 12:
            batch_multiplier = 8 if strict_solver_window else 3
            remaining = n_samples - len(valid_candidates)
            lhs_values = self._lhs(
                len(feature_order),
                max(remaining * batch_multiplier, remaining),
                seed_offset=generation_round,
            )
            rng = np.random.default_rng(self.random_seed + 1000 + generation_round)

            for row in lhs_values:
                geometry = {}
                for idx, feature in enumerate(feature_order):
                    bounds = type_ranges.get(feature)
                    if not bounds:
                        bounds = {"min": 0.0, "max": 1.0}
                    value = bounds["min"] + row[idx] * (bounds["max"] - bounds["min"])
                    geometry[feature] = round(float(value), 3)

                layup = self._layup_payload(rng)
                candidate_index = start_index + len(valid_candidates)
                material_system = self._select_material(task, candidate_index)
                task_payload = self._task_payload(task)
                candidate = {
                    "candidate_id": candidate_id_factory(candidate_index),
                    "source": "DOE",
                    "stiffener_type": stype,
                    "geometry": geometry,
                    "layup": layup,
                    "rule_check": {},
                    "surrogate_BLF": None,
                    "surrogate_weight": None,
                    "rank_score": None,
                    "rationale": self._estimate_rationale(material_system.get("name", "Unknown"), layup.get("template_name", "LAYUP")),
                    "material_system": material_system,
                    "load_conditions": task_payload["load_conditions"],
                    "boundary_conditions": task_payload["boundary_conditions"],
                    "design_targets": task_payload["design_targets"],
                }
                rule_check = self.rule_checker.run(
                    candidate, strict_solver_window=strict_solver_window, stiffener_type=stype,
                )
                candidate["rule_check"] = rule_check
                if rule_check["is_valid"]:
                    valid_candidates.append(candidate)
                if len(valid_candidates) >= n_samples:
                    break

            generation_round += 1

        return valid_candidates
