"""逐层壁板代理的特征、训练来源、适用域和模型兼容性检查。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit

from core.io_utils import read_json
from core.laminate import ANALYSIS_VERSION, analysis_contract, fingerprint, laminate_abd, lamina_material, section_layers, solver_fingerprint
from core.layup import normalize_layup
from core.paths import CASES_DIR, MODELS_DIR
from core.stiffener_profile import CANONICAL_GEOMETRY_ORDER, STIFFENER_TYPES, required_geometry_params, geometry_to_feature_vector
from core.task_contract import boundary_condition_code, load_case_code, task_payload_from_request


FEATURE_VERSION = "panel-physical-v3-type-isolated"
ABD_COMPONENTS = [(0, 0), (1, 1), (0, 1), (0, 2), (1, 2), (2, 2)]
FEATURE_ORDER = (
    list(CANONICAL_GEOMETRY_ORDER) + ["type_" + stype for stype in STIFFENER_TYPES]
    + ["skin_f0", "skin_f45", "skin_f90", "ply_count", "Nx_kN_per_m", "Nxy_kN_per_m",
       "load_case_code", "boundary_condition_code", "density_kg_per_m3", "E1_GPa", "E2_GPa",
       "G12_GPa", "nu12", "G13_GPa", "G23_GPa"]
    + [f"{component}_{matrix}{i + 1}{j + 1}" for component in ("skin", "web", "flange", "cap")
       for matrix in ("A", "B", "D") for i, j in ABD_COMPONENTS]
)
FEATURE_SOURCES = ("core/surrogate_model.py", "core/stiffener_profile.py", "core/layup.py", "core/task_contract.py")
FEATURE_FINGERPRINT = fingerprint({"version": FEATURE_VERSION, "order": FEATURE_ORDER,
                                   "analysis_version": ANALYSIS_VERSION, "solver": solver_fingerprint(),
                                   "feature_code": {relative: hashlib.sha256((Path(__file__).resolve().parent.parent / relative)
                                                    .read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                                                    for relative in FEATURE_SOURCES},
                                   "target": "positive_BLF_global"})


class SurrogateUnavailable(RuntimeError):
    """预筛不可用原因由编排层记录并决定是否直接求解。"""

    def __init__(self, status: str, reason: str):
        self.status, self.reason = status, reason
        super().__init__(reason)

    def to_dict(self) -> Dict:
        return {"status": self.status, "reason": self.reason}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def candidate_to_features(candidate: Dict, task: Dict | None = None, feature_order: Sequence[str] | None = None) -> List[float]:
    """训练和推理使用同一任务解包与物理截面；缺少必需输入时不补零。"""
    if feature_order is not None and list(feature_order) != FEATURE_ORDER:
        raise SurrogateUnavailable("incompatible", "特征顺序不属于当前物理截面契约")
    payload = task_payload_from_request(candidate if task is None else task)
    stype = candidate["stiffener_type"]
    required = required_geometry_params(stype)
    geometry = {key: float(candidate["geometry"][key]) for key in required}
    feature_map = dict(zip(CANONICAL_GEOMETRY_ORDER, geometry_to_feature_vector(geometry)))
    feature_map.update({"type_" + kind: float(stype == kind) for kind in STIFFENER_TYPES})
    feature_map.update(normalize_layup(candidate["layup"]["skin_layup"]))
    feature_map.update(lamina_material(candidate))
    loads = payload["load_conditions"]
    if loads["Nx_kN_per_m"] + loads["Nxy_kN_per_m"] <= 0:
        raise ValueError("代理输入必须具有非零参考载荷")
    feature_map.update({"Nx_kN_per_m": loads["Nx_kN_per_m"], "Nxy_kN_per_m": loads["Nxy_kN_per_m"],
                        "load_case_code": load_case_code(loads),
                        "boundary_condition_code": boundary_condition_code(payload["boundary_conditions"])})
    sections = section_layers(candidate)
    for component in ("skin", "web", "flange", "cap"):
        matrices = laminate_abd(sections[component], feature_map) if component in sections else np.zeros((3, 3, 3))
        for label, matrix in zip(("A", "B", "D"), matrices):
            for i, j in ABD_COMPONENTS:
                feature_map[f"{component}_{label}{i + 1}{j + 1}"] = matrix[i][j]
    row = [float(feature_map[name]) for name in FEATURE_ORDER]
    if not np.isfinite(row).all():
        raise ValueError("代理特征包含非有限数值")
    return row


def record_to_training_sample(record: Dict, feature_order: Sequence[str] | None = None) -> tuple[List[float], float] | None:
    """接收版本、求解器指纹、物理输入及任务工况一致且 BLF 为有限正数的成功结果。"""
    result, design = record.get("abaqus_results", {}), record.get("design", {})
    if result.get("status") != "success" or (result.get("analysis_contract") or {}).get("version") != ANALYSIS_VERSION:
        return None
    if result["analysis_contract"].get("solver_fingerprint") != solver_fingerprint():
        return None
    if result.get("candidate_id") != design.get("candidate_id") or result["analysis_contract"] != analysis_contract(design):
        raise ValueError("训练记录的候选或物理截面来源不一致")
    target = float(result["BLF_global"])
    if not np.isfinite(target) or target <= 0:
        raise ValueError("训练 BLF 必须为有限正数")
    features = candidate_to_features(design, record.get("task"), feature_order)
    if features != candidate_to_features(design):
        raise ValueError("训练任务与求解输入工况不一致")
    return features, target


def _slice_key(candidate: Dict, task: Dict | None = None) -> str:
    payload = task_payload_from_request(candidate if task is None else task)
    return "/".join((candidate["stiffener_type"], payload["load_conditions"]["type"], payload["boundary_conditions"]["type"]))


class SurrogateModelManager:
    def __init__(self, models_dir: Path | None = None, case_dir: Path | None = None,
                 stiffener_type: str | None = None) -> None:
        if stiffener_type is not None and stiffener_type not in STIFFENER_TYPES:
            raise ValueError(f"无效代理筋型：{stiffener_type}")
        self.stiffener_type = stiffener_type
        self.models_root = models_dir or MODELS_DIR
        self.models_dir = self.models_root / stiffener_type if stiffener_type else self.models_root
        self.case_dir = case_dir or CASES_DIR
        self.random_forest_path = self.models_dir / "surrogate_rf.joblib"
        self.metrics_path = self.models_dir / "surrogate_metrics.json"

    def load_training_records(self) -> List[Dict]:
        return [record for path in sorted(self.case_dir.glob("CASE_*.json"))
                if (record := read_json(path)).get("abaqus_results", {}).get("status") == "success"
                and (self.stiffener_type is None or record.get("design", {}).get("stiffener_type") == self.stiffener_type)]

    def for_type(self, stiffener_type: str) -> "SurrogateModelManager":
        return SurrogateModelManager(self.models_root, self.case_dir, stiffener_type)

    def _require_type(self) -> None:
        if self.stiffener_type is None:
            raise ValueError("训练、验收和裸特征预测必须明确指定 stiffener_type")

    def train_from_records(self, records: Sequence[Dict]) -> Dict:
        """按物理设计分组留出，保存样本指纹；留出误差不替代独立工况验收。"""
        self._require_type()
        features, targets, provenance, groups, slices = [], [], [], [], []
        seen = set()
        excluded = 0
        for record in records:
            if record.get("design", {}).get("stiffener_type") != self.stiffener_type:
                raise ValueError("代理训练记录混入其他筋型")
            sample = record_to_training_sample(record)
            if sample is None:
                excluded += 1
                continue
            row, target = sample
            sample_id = fingerprint({"features": row, "target": target})
            if sample_id in seen:
                continue
            seen.add(sample_id)
            contract = analysis_contract(record["design"])
            features.append(row)
            targets.append(target)
            provenance.append({"case_id": record["case_id"], "record_sha256": fingerprint(record), "sample_sha256": sample_id})
            groups.append(fingerprint({key: contract[key] for key in ("geometry", "sections", "material", "stiffener_type")}))
            slices.append(_slice_key(record["design"], record.get("task")))
        if len(features) < 10 or len(set(groups)) < 5:
            raise RuntimeError(f"至少需要 10 条兼容的逐层求解记录和 5 个不同物理设计；可用 {len(features)}，排除历史/失败记录 {excluded}")
        x, y = np.asarray(features), np.asarray(targets)
        train, test = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(x, y, groups))
        model = RandomForestRegressor(n_estimators=500, random_state=42, min_samples_leaf=1)
        model.fit(x[train], np.log1p(y[train]))
        prediction = np.expm1(model.predict(x[test]))
        if not np.isfinite(prediction).all() or (prediction <= 0).any():
            raise RuntimeError("代理留出预测不是有限正数")
        def metrics(indices):
            return {"count": len(indices), "mape": float(mean_absolute_percentage_error(y[test][indices], prediction[indices])),
                    "rmse": float(np.sqrt(mean_squared_error(y[test][indices], prediction[indices])))}
        validation_slices = {key: metrics([i for i, index in enumerate(test) if slices[index] == key])
                             for key in sorted({slices[index] for index in test})}
        domain = {}
        for key in sorted({slices[index] for index in train}):
            rows = x[[index for index in train if slices[index] == key]]
            domain[key] = {"min": rows.min(axis=0).tolist(), "max": rows.max(axis=0).tolist()}
        self.models_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, self.random_forest_path)
        summary = {"selected_model": "rf", "stiffener_type": self.stiffener_type, "feature_version": FEATURE_VERSION,
                   "feature_order": FEATURE_ORDER, "feature_fingerprint": FEATURE_FINGERPRINT,
                   "analysis_version": ANALYSIS_VERSION, "training_size": len(x), "excluded_records": excluded,
                   "data_fingerprint": fingerprint(provenance), "records": provenance,
                   "split": {"train": train.tolist(), "test": test.tolist(), "groups": groups},
                   "rf": metrics(list(range(len(test)))), "validation_slices": validation_slices,
                   "domain": domain, "rf_target_transform": "log1p",
                   "artifacts": {self.random_forest_path.name: file_sha256(self.random_forest_path)},
                   "validation_status": "pending_independent_validation"}
        # 指标元数据最后落盘；中途失败的权重不能通过旧元数据的哈希校验。
        self.metrics_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        return summary

    def _load_metrics(self, *, require_validation: bool = True) -> Dict:
        self._require_type()
        if not self.metrics_path.is_file():
            raise SurrogateUnavailable("missing", "代理模型元数据不存在")
        try:
            metrics = read_json(self.metrics_path)
        except (ValueError, OSError) as exc:
            raise SurrogateUnavailable("invalid", "代理元数据无法读取") from exc
        if not isinstance(metrics, dict):
            raise SurrogateUnavailable("invalid", "代理元数据必须是对象")
        if (metrics.get("stiffener_type") != self.stiffener_type
                or metrics.get("feature_version") != FEATURE_VERSION or metrics.get("feature_order") != FEATURE_ORDER
                or metrics.get("feature_fingerprint") != FEATURE_FINGERPRINT or metrics.get("analysis_version") != ANALYSIS_VERSION):
            raise SurrogateUnavailable("incompatible", "代理特征或有限元截面版本不兼容，历史权重未参与预测")
        if metrics.get("selected_model") != "rf" or metrics.get("rf_target_transform") != "log1p":
            raise SurrogateUnavailable("incompatible", "代理类型或目标变换不兼容")
        if not self.random_forest_path.is_file():
            raise SurrogateUnavailable("missing", "代理权重不存在")
        if file_sha256(self.random_forest_path) != metrics.get("artifacts", {}).get(self.random_forest_path.name):
            raise SurrogateUnavailable("invalid", "代理权重指纹不匹配")
        if fingerprint(metrics.get("records", [])) != metrics.get("data_fingerprint"):
            raise SurrogateUnavailable("invalid", "代理数据来源指纹不匹配")
        if not require_validation:
            return metrics
        # 独立验收文件必须绑定同一权重、数据和特征，不能以留出指标冒充。
        validation_path = self.models_dir / "surrogate_validation.json"
        if not validation_path.is_file():
            raise SurrogateUnavailable("unvalidated", "缺少独立逐层 FEM 工况验收")
        try:
            validation = read_json(validation_path)
        except (ValueError, OSError) as exc:
            raise SurrogateUnavailable("invalid", "独立验收文件无法读取") from exc
        if (validation.get("status") != "passed" or not validation.get("evidence")
                or any(validation.get(key) != metrics.get(key) for key in ("stiffener_type", "artifacts", "data_fingerprint", "feature_fingerprint"))):
            raise SurrogateUnavailable("unvalidated", "独立验收未通过或未绑定当前模型")
        if fingerprint(validation["evidence"]) != validation.get("evidence_fingerprint"):
            raise SurrogateUnavailable("invalid", "独立验收证据指纹不匹配")
        metrics["validated_slices"] = validation.get("slices", [])
        return metrics

    def validate_from_records(self, records: Sequence[Dict], *, max_mape: float,
                              max_relative_error: float, min_per_slice: int) -> Dict:
        """以独立设计的真实 FEM 记录验收；阈值由调用者指定，失败报告同样保存。"""
        if min_per_slice < 2 or not all(np.isfinite(v) and v > 0 for v in (max_mape, max_relative_error)):
            raise ValueError("验收需至少两个独立设计/切片和有限正误差阈值")
        metrics = self._load_metrics(require_validation=False)
        model = joblib.load(self.random_forest_path)
        training_groups = set(metrics["split"]["groups"])
        evidence, rows_by_slice, seen = [], {}, set()
        errors = []
        for record in records:
            if record.get("design", {}).get("stiffener_type") != self.stiffener_type:
                raise ValueError("代理独立验收记录混入其他筋型")
            sample = record_to_training_sample(record)
            if sample is None:
                errors.append(f"{record.get('case_id')} 不具有兼容的成功逐层结果")
                continue
            row, target = sample
            contract = analysis_contract(record["design"])
            group = fingerprint({key: contract[key] for key in ("geometry", "sections", "material", "stiffener_type")})
            key = _slice_key(record["design"], record.get("task"))
            if group in training_groups or (key, group) in seen:
                errors.append(f"{record.get('case_id')} 不是独立物理设计")
                continue
            seen.add((key, group))
            prediction = float(np.expm1(model.predict([row]))[0])
            if not np.isfinite(prediction) or prediction <= 0:
                errors.append(f"{record.get('case_id')} 预测无效")
                continue
            evidence.append({"case_id": record["case_id"], "record_sha256": fingerprint(record),
                             "analysis_fingerprint": contract["fingerprint"], "slice": key,
                             "group": group, "target": target, "prediction": prediction})
            rows_by_slice.setdefault(key, []).append(abs(prediction - target) / target)
        slice_metrics = {}
        for key in metrics["domain"]:
            values = rows_by_slice.get(key, [])
            slice_metrics[key] = {"count": len(values), "mape": float(np.mean(values)) if values else None,
                                  "max_relative_error": max(values) if values else None,
                                  "passed": bool(len(values) >= min_per_slice and np.mean(values) <= max_mape
                                                 and max(values) <= max_relative_error)}
        passed = not errors and bool(slice_metrics) and all(item["passed"] for item in slice_metrics.values())
        report = {"status": "passed" if passed else "failed", "slices": sorted(slice_metrics),
                  "slice_metrics": slice_metrics, "evidence": evidence, "evidence_fingerprint": fingerprint(evidence),
                  "errors": errors, "thresholds": {"max_mape": max_mape, "max_relative_error": max_relative_error,
                                                    "min_per_slice": min_per_slice},
                  **{key: metrics[key] for key in ("stiffener_type", "artifacts", "data_fingerprint", "feature_fingerprint")}}
        (self.models_dir / "surrogate_validation.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        return report

    def predict(self, feature_rows: Sequence[Sequence[float]]) -> np.ndarray:
        metrics = self._load_metrics()
        x = np.asarray(feature_rows, dtype=float)
        if x.ndim != 2 or x.shape[1] != len(FEATURE_ORDER) or not np.isfinite(x).all():
            raise ValueError("代理输入维度或数值无效")
        expected_type = np.asarray([float(kind == self.stiffener_type) for kind in STIFFENER_TYPES])
        type_indices = [FEATURE_ORDER.index("type_" + kind) for kind in STIFFENER_TYPES]
        if not (x[:, type_indices] == expected_type).all():
            raise SurrogateUnavailable("incompatible", "输入筋型与代理训练筋型不一致")
        for row in x:
            if not any(self._inside_domain(row, metrics.get("domain", {}).get(key)) for key in metrics["validated_slices"]):
                raise SurrogateUnavailable("out_of_domain", "输入特征超出已验收工况的训练包络")
        try:
            predictions = np.expm1(joblib.load(self.random_forest_path).predict(x))
        except Exception as exc:
            raise SurrogateUnavailable("invalid", "代理权重加载或预测失败") from exc
        if len(predictions) != len(x) or not np.isfinite(predictions).all() or (predictions <= 0).any():
            raise SurrogateUnavailable("invalid_prediction", "代理输出不是有限正 BLF")
        return predictions

    @staticmethod
    def _inside_domain(row, domain) -> bool:
        if not isinstance(domain, dict) or set(domain) != {"min", "max"}:
            raise SurrogateUnavailable("invalid", "代理适用域定义无效")
        lower, upper = np.asarray(domain["min"]), np.asarray(domain["max"])
        if (lower.shape != (len(row),) or upper.shape != lower.shape or not np.isfinite([lower, upper]).all()
                or (lower > upper).any()):
            raise SurrogateUnavailable("invalid", "代理适用域维度或数值无效")
        tolerance = np.maximum(np.maximum(np.abs(lower), np.abs(upper)), 1) * 1e-9
        return bool(((row >= lower - tolerance) & (row <= upper + tolerance)).all())

    def predict_candidates(self, candidates: Iterable[Dict], task: Dict | None = None) -> np.ndarray:
        candidates = list(candidates)
        if not candidates:
            return np.asarray([], dtype=float)
        if self.stiffener_type is None:
            # 每个筋型独立加载权重、适用域和验收证据，输出恢复候选输入顺序。
            predictions = np.empty(len(candidates))
            for kind in dict.fromkeys(candidate["stiffener_type"] for candidate in candidates):
                indices = [i for i, candidate in enumerate(candidates) if candidate["stiffener_type"] == kind]
                predictions[indices] = self.for_type(kind).predict_candidates([candidates[i] for i in indices], task)
            return predictions
        if any(candidate["stiffener_type"] != self.stiffener_type for candidate in candidates):
            raise SurrogateUnavailable("incompatible", "候选筋型与代理训练筋型不一致")
        metrics = self._load_metrics()
        features = []
        for candidate in candidates:
            row = candidate_to_features(candidate, task)
            key = _slice_key(candidate, task)
            domain = metrics.get("domain", {}).get(key)
            if key not in metrics["validated_slices"] or not domain:
                raise SurrogateUnavailable("out_of_domain", f"筋型/工况/边界未验收: {key}")
            if not self._inside_domain(np.asarray(row), domain):
                raise SurrogateUnavailable("out_of_domain", f"{candidate['candidate_id']} 超出训练特征包络")
            features.append(row)
        return self.predict(features)
