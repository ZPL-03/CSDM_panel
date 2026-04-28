"""快速筛选智能体。"""

from __future__ import annotations

from typing import Dict, List

from agents.base import BaseAgent
from core.config_loader import load_app_config
from core.surrogate_model import SurrogateModelManager
from core.task_contract import effective_screen_top_k


class ScreenerAgent(BaseAgent):
    agent_name = "SCREENER"

    def __init__(self, progress_callback=None) -> None:
        super().__init__(progress_callback=progress_callback)
        self.config = load_app_config()
        self.model_manager = SurrogateModelManager()
        score_config = dict(self.config["pipeline"].get("screening_score", {}))
        self.blf_weight = float(score_config.get("blf_weight", 1.0))
        self.weight_penalty = float(score_config.get("weight_penalty", 0.08))

    @property
    def score_formula_text(self) -> str:
        return f"score = {self.blf_weight:.2f} × BLF - {self.weight_penalty:.2f} × 面密度"

    def _estimate_weight(self, candidate: Dict) -> float:
        geometry = candidate["geometry"]
        material = candidate.get("material_system", {})
        density = float(material.get("density_kg_per_m3", 1600.0))
        panel_length = float(geometry["panel_length_mm"])
        panel_width = float(geometry["panel_width_mm"])
        skin_t = float(geometry["skin_thickness_mm"])
        height = float(geometry["stiffener_height_mm"])
        web_t = float(geometry["web_thickness_mm"])
        flange_w = float(geometry["flange_width_mm"])
        flange_t = float(geometry["flange_thickness_mm"])
        pitch = max(float(geometry["pitch_mm"]), 1.0)

        panel_area_m2 = max(panel_length * panel_width * 1e-6, 1e-9)
        skin_volume_m3 = panel_length * panel_width * skin_t * 1e-9
        stiffener_count = max(1, int(round(panel_width / pitch)))
        stiffener_volume_m3 = stiffener_count * panel_length * (web_t * height + flange_t * flange_w) * 1e-9
        total_mass_kg = density * (skin_volume_m3 + stiffener_volume_m3)
        return round(total_mass_kg / panel_area_m2, 3)

    def run(self, input_data: Dict) -> List[Dict]:
        task = input_data["task"]
        candidates = input_data["candidates"]
        predictions = self.model_manager.predict_candidates(candidates, task)
        requested_top_k = effective_screen_top_k(task, len(candidates)) or int(self.config["pipeline"]["top_k"])

        enriched: List[Dict] = []
        for candidate, predicted_blf in zip(candidates, predictions):
            updated = dict(candidate)
            updated["surrogate_BLF"] = round(float(predicted_blf), 3)
            updated["surrogate_weight"] = self._estimate_weight(candidate)
            updated["rank_score"] = round(
                float(self.blf_weight * updated["surrogate_BLF"] - self.weight_penalty * updated["surrogate_weight"]),
                4,
            )
            updated["screening_breakdown"] = {
                "formula": self.score_formula_text,
                "blf_component": round(self.blf_weight * updated["surrogate_BLF"], 4),
                "weight_component": round(self.weight_penalty * updated["surrogate_weight"], 4),
            }
            updated["screening_summary"] = (
                f"代理预测 BLF={updated['surrogate_BLF']}，"
                f"面密度={updated['surrogate_weight']} kg/m^2，"
                f"按 {self.score_formula_text} 得分 {updated['rank_score']}。"
            )
            enriched.append(updated)

        enriched.sort(key=lambda item: item["rank_score"], reverse=True)
        selected = enriched[:requested_top_k]
        for index, candidate in enumerate(selected, start=1):
            candidate["selection_reason"] = (
                f"Top-{index} 入选：{candidate['screening_summary']} "
                f"当前排序靠前，适合优先进入有限元校核。"
            )
        self.emit(
            f"已完成 {len(candidates)} 个候选的批量评分，请求保留 Top-{requested_top_k}，实际返回 {len(selected)} 个。"
        )
        return selected
