"""代理数据来源、特征版本和不可用状态回归。"""

from copy import deepcopy

import pytest

from core.io_utils import write_json
from core.laminate import analysis_contract
from core.surrogate_model import FEATURE_ORDER, SurrogateModelManager, SurrogateUnavailable, candidate_to_features, record_to_training_sample
from tests.test_fem_agent import _real_fem_candidate


def test_flat_and_wrapped_tasks_are_identical_and_keep_load_and_boundary():
    candidate = _real_fem_candidate("C1")
    task = {"load_conditions": {"type": "compression_shear", "Nx_kN_per_m": 120.1234567, "Nxy_kN_per_m": 40},
            "boundary_conditions": {"type": "CCCC"}}
    flat = candidate_to_features(candidate, task)
    assert flat == candidate_to_features(candidate, {"task_id": "TASK_1", "task": task})
    assert flat[FEATURE_ORDER.index("Nx_kN_per_m")] == 120.1234567
    assert flat[FEATURE_ORDER.index("Nxy_kN_per_m")] == 40
    assert flat[FEATURE_ORDER.index("boundary_condition_code")] == 1
    with pytest.raises(ValueError):
        candidate_to_features(candidate, {"task": {}})


def test_features_distinguish_types_cap_and_laminate_order():
    candidates = [_real_fem_candidate("C1", stype) for stype in ("T", "L", "HAT", "BLADE")]
    assert len({tuple(candidate_to_features(c)) for c in candidates}) == 4
    hat = candidates[2]
    original = candidate_to_features(hat)
    hat["geometry"]["cap_width_mm"] += 5
    assert candidate_to_features(hat) != original
    original = candidate_to_features(hat)
    hat["geometry"]["cap_thickness_mm"] += 1
    assert candidate_to_features(hat) != original
    first, second = deepcopy(candidates[0]), deepcopy(candidates[0])
    first["layup"]["skin_layup"] = "[0/45/-45/90]s"
    second["layup"]["skin_layup"] = "[90/45/-45/0]s"
    x, y = candidate_to_features(first), candidate_to_features(second)
    assert x[FEATURE_ORDER.index("skin_f0")] == y[FEATURE_ORDER.index("skin_f0")]
    assert x[FEATURE_ORDER.index("skin_D11")] != y[FEATURE_ORDER.index("skin_D11")]


def test_missing_and_legacy_models_never_emit_predictions(tmp_path):
    manager = SurrogateModelManager(models_dir=tmp_path, stiffener_type="T")
    with pytest.raises(SurrogateUnavailable) as missing:
        manager.predict([[0] * len(FEATURE_ORDER)])
    assert missing.value.status == "missing"
    write_json(manager.metrics_path, {"feature_order": FEATURE_ORDER[:21], "selected_model": "mlp"})
    with pytest.raises(SurrogateUnavailable) as legacy:
        manager.predict_candidates([_real_fem_candidate("C1")])
    assert legacy.value.status == "incompatible"


def physical_record(index=1):
    design = _real_fem_candidate(f"C{index}")
    design["geometry"]["skin_thickness_mm"] += index * 0.01
    return {"case_id": f"CASE_{index}", "task": {"load_conditions": design["load_conditions"], "boundary_conditions": design["boundary_conditions"]},
            "design": design, "abaqus_results": {"candidate_id": design["candidate_id"], "status": "success",
            "BLF_global": 0.5 + index * 0.01, "analysis_contract": analysis_contract(design)}}


def test_training_rejects_legacy_mismatched_and_nonfinite_labels():
    record = physical_record()
    assert record_to_training_sample(record) is not None
    legacy = deepcopy(record)
    legacy["abaqus_results"].pop("analysis_contract")
    assert record_to_training_sample(legacy) is None
    broken = deepcopy(record)
    broken["design"]["geometry"]["skin_thickness_mm"] += 1
    with pytest.raises(ValueError):
        record_to_training_sample(broken)
    for value in (0, -1, float("nan"), float("inf")):
        record["abaqus_results"]["BLF_global"] = value
        with pytest.raises(ValueError):
            record_to_training_sample(record)


def test_training_fingerprints_and_validation_gate_are_real(tmp_path):
    manager = SurrogateModelManager(models_dir=tmp_path, stiffener_type="T")
    records = [physical_record(index) for index in range(1, 13)]
    summary = manager.train_from_records(records)
    assert summary["training_size"] == 12
    assert summary["feature_order"] == FEATURE_ORDER
    groups = summary["split"]["groups"]
    assert {groups[i] for i in summary["split"]["train"]}.isdisjoint({groups[i] for i in summary["split"]["test"]})
    assert summary["validation_slices"]["T/axial_compression/SSSS"]["count"] == 3
    with pytest.raises(SurrogateUnavailable) as unvalidated:
        manager.predict_candidates([records[0]["design"]])
    assert unvalidated.value.status == "unvalidated"
    with manager.random_forest_path.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(SurrogateUnavailable) as invalid:
        manager.predict_candidates([records[0]["design"]])
    assert invalid.value.status == "invalid"


def test_independent_validation_rejects_leakage_and_domain_extrapolation(tmp_path):
    manager = SurrogateModelManager(models_dir=tmp_path, stiffener_type="T")
    records = [physical_record(index) for index in range(1, 13)]
    manager.train_from_records(records)
    failed = manager.validate_from_records(records[:2], max_mape=0.1, max_relative_error=0.2, min_per_slice=2)
    assert failed["status"] == "failed" and failed["errors"]
    with pytest.raises(SurrogateUnavailable):
        manager.predict_candidates([records[2]["design"]])
    independent = []
    for index, offset in ((101, 0.045), (102, 0.075)):
        record = physical_record(index)
        record["design"]["geometry"]["skin_thickness_mm"] = 2.4 + offset
        record["abaqus_results"]["BLF_global"] = 0.5 + offset
        record["abaqus_results"]["analysis_contract"] = analysis_contract(record["design"])
        independent.append(record)
    accepted = manager.validate_from_records(independent, max_mape=0.1, max_relative_error=0.2, min_per_slice=2)
    assert accepted["status"] == "passed"
    assert len(manager.predict_candidates([independent[0]["design"]])) == 1
    outside = deepcopy(independent[0]["design"])
    outside["geometry"]["panel_length_mm"] = 700
    with pytest.raises(SurrogateUnavailable) as error:
        manager.predict_candidates([outside])
    assert error.value.status == "out_of_domain"
    with pytest.raises(SurrogateUnavailable):
        manager.predict([candidate_to_features(outside)])


def test_type_weights_and_training_provenance_are_strictly_isolated(tmp_path):
    manager = SurrogateModelManager(models_dir=tmp_path)
    typed = manager.for_type("T")
    assert typed.models_dir == tmp_path / "T"
    with pytest.raises(ValueError, match="明确指定"):
        manager.train_from_records([physical_record()])
    other = physical_record()
    other["design"]["stiffener_type"] = "L"
    with pytest.raises(ValueError, match="混入其他筋型"):
        typed.train_from_records([other])
    with pytest.raises(SurrogateUnavailable, match="筋型"):
        typed.predict_candidates([other["design"]])
    assert len(candidate_to_features(_real_fem_candidate("C1", "HAT"))) == 101
    assert len(candidate_to_features(_real_fem_candidate("C1", "BLADE"))) == 101
