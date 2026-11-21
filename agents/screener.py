"""快速筛选智能体。"""

from __future__ import annotations

from typing import Dict, List
import math

from agents.base import BaseAgent
from core.config_loader import load_app_config
from core.display import format_number
from core.laminate import areal_density
from core.surrogate_model import SurrogateModelManager, SurrogateUnavailable
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
        return areal_density(candidate)

    def run(self, input_data: Dict) -> List[Dict]:
        task = input_data["task"]
        candidates = input_data["candidates"]
        predictions = self.model_manager.predict_candidates(candidates, task)
        if len(predictions) != len(candidates) or any(not math.isfinite(float(value)) or value <= 0 for value in predictions):
            raise SurrogateUnavailable("invalid_prediction", "预测数量或有限正 BLF 条件不满足")
        requested_top_k = effective_screen_top_k(task, len(candidates)) or int(self.config["pipeline"]["top_k"])

        enriched: List[Dict] = []
        for candidate, predicted_blf in zip(candidates, predictions):
            updated = dict(candidate)
            updated["surrogate_BLF"] = float(predicted_blf)
            updated["surrogate_weight"] = self._estimate_weight(candidate)
            updated["rank_score"] = self.blf_weight * updated["surrogate_BLF"] - self.weight_penalty * updated["surrogate_weight"]
            updated["surrogate_status"] = "available"
            updated["screening_breakdown"] = {
                "formula": self.score_formula_text,
                "blf_component": round(self.blf_weight * updated["surrogate_BLF"], 4),
                "weight_component": round(self.weight_penalty * updated["surrogate_weight"], 4),
            }
            updated["screening_summary"] = (
                f"代理预测 BLF={format_number(updated['surrogate_BLF'])}，"
                f"面密度={format_number(updated['surrogate_weight'])} kg/m^2，"
                f"按 {self.score_formula_text} 得分 {format_number(updated['rank_score'])}。"
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
