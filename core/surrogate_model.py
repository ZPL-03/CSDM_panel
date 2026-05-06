"""代理模型训练与推理。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import joblib
import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from core.io_utils import read_json
from core.paths import CASES_DIR, MODELS_DIR
from core.stiffener_profile import CANONICAL_GEOMETRY_ORDER, geometry_to_feature_vector
from core.task_contract import boundary_condition_code, load_case_code, normalize_boundary_conditions, normalize_load_conditions


FEATURE_ORDER = [
    "panel_length_mm",
    "panel_width_mm",
    "skin_thickness_mm",
    "pitch_mm",
    "stiffener_height_mm",
    "web_thickness_mm",
    "flange_width_mm",
    "flange_thickness_mm",
    "skin_f0",
    "skin_f45",
    "skin_f90",
    "ply_count",
    "Nx_kN_per_m",
    "Nxy_kN_per_m",
    "load_case_code",
    "boundary_condition_code",
    "density_kg_per_m3",
    "E1_GPa",
    "E2_GPa",
    "G12_GPa",
    "nu12",
]


def _parse_layup_sequence(layup_text: str) -> List[str]:
    text = (layup_text or "").strip()
    if not text:
        return []
    symmetric = text.endswith("s")
    if symmetric:
        text = text[:-1]
    base = [item.strip() for item in text.strip("[] ").split("/") if item.strip()]
    return base + list(reversed(base)) if symmetric else base


def candidate_to_features(candidate: Dict, task: Dict | None = None, feature_order: Sequence[str] | None = None) -> List[float]:
    geometry = candidate.get("geometry", {})
    layup = candidate.get("layup", {})
    material = candidate.get("material_system", {})
    task_load_conditions = task.get("load_conditions", {}) if task else candidate.get("load_conditions", {})
    task_boundary_conditions = task.get("boundary_conditions", {}) if task else candidate.get("boundary_conditions", {})
    load_conditions = normalize_load_conditions(task_load_conditions)
    boundary_conditions = normalize_boundary_conditions(task_boundary_conditions)

    ply_count = layup.get("ply_count")
    if ply_count is None:
        ply_count = len(_parse_layup_sequence(str(layup.get("skin_layup", ""))))

    geom_features = geometry_to_feature_vector(geometry)
    feature_map = {
        **{CANONICAL_GEOMETRY_ORDER[i]: geom_features[i] for i in range(len(CANONICAL_GEOMETRY_ORDER))},
        "skin_f0": float(layup.get("skin_f0", 0.0)),
        "skin_f45": float(layup.get("skin_f45", 0.0)),
        "skin_f90": float(layup.get("skin_f90", 0.0)),
        "ply_count": float(ply_count or 0.0),
        "Nx_kN_per_m": float(load_conditions.get("Nx_kN_per_m", 0.0)),
        "Nxy_kN_per_m": float(load_conditions.get("Nxy_kN_per_m", 0.0)),
        "load_case_code": float(load_case_code(load_conditions)),
        "boundary_condition_code": float(boundary_condition_code(boundary_conditions)),
        "density_kg_per_m3": float(material.get("density_kg_per_m3", 0.0)),
        "E1_GPa": float(material.get("E1_GPa", 0.0)),
        "E2_GPa": float(material.get("E2_GPa", 0.0)),
        "G12_GPa": float(material.get("G12_GPa", 0.0)),
        "nu12": float(material.get("nu12", 0.0)),
    }
    active_feature_order = list(feature_order or FEATURE_ORDER)
    return [float(feature_map.get(name, 0.0)) for name in active_feature_order]


def record_to_training_sample(record: Dict, feature_order: Sequence[str] | None = None) -> tuple[List[float], float] | None:
    abaqus_results = record.get("abaqus_results", {})
    if abaqus_results.get("status") != "success":
        return None
    blf = abaqus_results.get("BLF_global")
    if blf is None:
        return None
    return candidate_to_features(record.get("design", {}), record.get("task"), feature_order), float(blf)


@dataclass
class Metrics:
    mape: float
    rmse: float

    def to_dict(self) -> Dict[str, float]:
        return {"mape": float(self.mape), "rmse": float(self.rmse)}


class MLPRegressor(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 96),
            nn.ReLU(),
            nn.Linear(96, 48),
            nn.ReLU(),
            nn.Linear(48, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


class SurrogateModelManager:
    def __init__(self, models_dir: Path | None = None, case_dir: Path | None = None) -> None:
        self.models_dir = models_dir or MODELS_DIR
        self.case_dir = case_dir or CASES_DIR
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.random_forest_path = self.models_dir / "surrogate_rf.joblib"
        self.scaler_path = self.models_dir / "surrogate_scaler.joblib"
        self.mlp_path = self.models_dir / "surrogate_mlp.pt"
        self.metrics_path = self.models_dir / "surrogate_metrics.json"

    def load_training_records(self) -> List[Dict]:
        records: List[Dict] = []
        for path in sorted(self.case_dir.glob("CASE_*.json")):
            record = read_json(path)
            if record.get("abaqus_results", {}).get("status") == "success":
                records.append(record)
        return records

    def train_from_records(self, records: Sequence[Dict]) -> Dict:
        features: List[List[float]] = []
        targets: List[float] = []
        for record in records:
            sample = record_to_training_sample(record, FEATURE_ORDER)
            if sample is None:
                continue
            feature_row, target = sample
            features.append(feature_row)
            targets.append(target)
        if len(features) < 10:
            raise RuntimeError("训练代理模型至少需要 10 条成功案例")
        return self.train(features, targets)

    def train(self, feature_rows: Sequence[Sequence[float]], targets: Sequence[float]) -> Dict:
        x = np.asarray(feature_rows, dtype=float)
        y = np.asarray(targets, dtype=float)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)

        log_y_train = np.log1p(y_train)

        rf = RandomForestRegressor(n_estimators=500, random_state=42, min_samples_leaf=1)
        rf.fit(x_train, log_y_train)
        rf_pred = np.expm1(rf.predict(x_test))
        rf_pred = np.clip(rf_pred, 1e-6, None)
        rf_metrics = Metrics(
            mape=mean_absolute_percentage_error(y_test, rf_pred),
            rmse=float(np.sqrt(mean_squared_error(y_test, rf_pred))),
        )

        mlp = MLPRegressor(input_dim=x.shape[1])
        optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()
        dataset = TensorDataset(
            torch.tensor(x_train_scaled, dtype=torch.float32),
            torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32),
        )
        loader = DataLoader(dataset, batch_size=min(32, len(dataset)), shuffle=True)
        mlp.train()
        for _ in range(120):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                loss = loss_fn(mlp(batch_x), batch_y)
                loss.backward()
                optimizer.step()

        mlp.eval()
        with torch.no_grad():
            mlp_pred = mlp(torch.tensor(x_test_scaled, dtype=torch.float32)).numpy().reshape(-1)
        mlp_pred = np.clip(mlp_pred, 1e-6, None)
        mlp_metrics = Metrics(
            mape=mean_absolute_percentage_error(y_test, mlp_pred),
            rmse=float(np.sqrt(mean_squared_error(y_test, mlp_pred))),
        )

        selected = "rf" if rf_metrics.mape <= mlp_metrics.mape else "mlp"
        joblib.dump(rf, self.random_forest_path)
        joblib.dump(scaler, self.scaler_path)
        torch.save(mlp.state_dict(), self.mlp_path)
        summary = {
            "selected_model": selected,
            "rf": rf_metrics.to_dict(),
            "mlp": mlp_metrics.to_dict(),
            "feature_order": FEATURE_ORDER,
            "training_size": int(len(x)),
            "rf_target_transform": "log1p",
        }
        self.metrics_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return summary

    def _load_metrics(self) -> Dict | None:
        if not self.metrics_path.exists():
            return None
        return json.loads(self.metrics_path.read_text(encoding="utf-8"))

    def predict(self, feature_rows: Sequence[Sequence[float]]) -> np.ndarray:
        x = np.asarray(feature_rows, dtype=float)
        metrics = self._load_metrics()
        if metrics is None:
            return np.full(shape=(len(x),), fill_value=1.2, dtype=float)

        selected = metrics["selected_model"]
        if selected == "rf":
            model = joblib.load(self.random_forest_path)
            predictions = np.expm1(model.predict(x))
            return np.clip(np.asarray(predictions, dtype=float), 1e-6, None)

        scaler: StandardScaler = joblib.load(self.scaler_path)
        model = MLPRegressor(input_dim=x.shape[1])
        model.load_state_dict(torch.load(self.mlp_path, map_location="cpu"))
        model.eval()
        with torch.no_grad():
            x_scaled = scaler.transform(x)
            predictions = model(torch.tensor(x_scaled, dtype=torch.float32)).numpy().reshape(-1)
        return np.clip(predictions, 1e-6, None)

    def predict_candidates(self, candidates: Iterable[Dict], task: Dict | None = None) -> np.ndarray:
        metrics = self._load_metrics()
        feature_order = list(metrics.get("feature_order", FEATURE_ORDER)) if metrics else FEATURE_ORDER
        features = [candidate_to_features(candidate, task, feature_order) for candidate in candidates]
        return self.predict(features)
