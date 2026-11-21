"""维护入口的只读校验、索引更新和实际求解编排边界。"""

import sys
from copy import deepcopy
from types import SimpleNamespace

import pytest

from core import paths
from core.doe_sampler import DOESampler
from core.io_utils import read_json, write_json
from core.laminate import analysis_contract
from core.task_parser import TaskParser
from scripts import build_initial_cases as initial
from scripts import rebuild_abaqus_artifacts as artifacts
from scripts import rebuild_case_memory as memory
from scripts.case_records import read_case_record


def case_record(index=1):
    task = TaskParser().parse_instruction("长度360mm，宽度300mm，压缩120kN/m，生成1个候选，初筛1个")["task"]
    candidate = DOESampler().sample_candidates(task, 1, start_index=index)[0]
    candidate.pop("display_name")
    return {"case_id": f"CASE_{index}", "created_at": "2026-11-18T00:00:00+00:00", "source": "abaqus_auto",
            "task": task, "design": candidate, "verdict": "通过",
            "abaqus_results": {"candidate_id": candidate["candidate_id"], "status": "success", "retry_count": 0,
                               "BLF_global": 2.5, "verdict": "通过"}}


def store(record, root=None):
    path = (root or paths.CASES_DIR) / (record["case_id"] + ".json")
    write_json(path, record)
    return path


class FakeIndex:
    def __init__(self, events, *, fail=False):
        self.ids = {"CASE_99"}
        def upsert(**data):
            events.append(("upsert", data))
            if fail:
                raise RuntimeError("embedding failure")
            self.ids.update(data["ids"])
        def delete(*, ids):
            events.append(("delete", ids))
            self.ids.difference_update(ids)
        collection = SimpleNamespace(get=lambda **kw: {"ids": sorted(self.ids)}, delete=delete)
        self.engine = SimpleNamespace(upsert_documents=upsert, collection=collection, count=lambda: len(self.ids))


def test_index_rebuild_validates_sources_then_updates_and_prunes_without_json_writes(monkeypatch):
    first, second = case_record(), case_record(2)
    files = [store(first), store(second), store(first, paths.CASE_LIBRARY_DIR)]
    before = {path: path.read_bytes() for path in files}
    events = []
    monkeypatch.setattr(memory, "CaseMemoryIndex", lambda: FakeIndex(events))
    result = memory.rebuild_case_memory_index()
    assert result == {"indexed_records": 2, "archive_records": 1, "formal_records": 1, "removed_index_entries": 1}
    assert events[0][0] == "upsert" and events[1] == ("delete", ["CASE_99"])
    assert [item["case_scope"] for item in events[0][1]["metadatas"]] == ["formal", "archive"]
    assert {path: path.read_bytes() for path in files} == before
    assert "analysis_contract" not in read_json(files[0])["abaqus_results"]


@pytest.mark.parametrize("fault", ["source", "material", "identity", "nonfinite", "fraction", "load", "failed_pass"])
def test_invalid_record_blocks_index_and_solver_before_side_effects(monkeypatch, fault):
    store(case_record())
    record = case_record(2)
    if fault == "source":
        record["design"]["source"] = "unknown"
    elif fault == "material":
        del record["design"]["material_system"]["E2_GPa"]
    elif fault == "identity":
        record["abaqus_results"]["candidate_id"] = "C9"
    elif fault == "nonfinite":
        record["abaqus_results"]["BLF_global"] = float("nan")
    elif fault == "fraction":
        record["design"]["layup"]["skin_f0"] = 0.99
    elif fault == "load":
        del record["design"]["load_conditions"]["Nx_kN_per_m"]
    else:
        record["abaqus_results"]["status"] = "failed"
    path = store(record)
    before = path.read_bytes()
    monkeypatch.setattr(memory, "CaseMemoryIndex", lambda: pytest.fail("不得触碰索引"))
    monkeypatch.setattr(artifacts.FEMAgent, "run", lambda *a: pytest.fail("不得启动求解"))
    with pytest.raises(ValueError, match="CASE_2"):
        memory.rebuild_case_memory_index()
    with pytest.raises(ValueError, match="CASE_2"):
        artifacts.rebuild_case(path, force=True)
    assert path.read_bytes() == before


@pytest.mark.parametrize("fault", ["conflict", "formal_failed", "empty"])
def test_invalid_index_sources_do_not_open_index(monkeypatch, fault):
    if fault != "empty":
        record = case_record()
        store(record)
        if fault == "conflict":
            record["created_at"] = "different snapshot"
        else:
            record["verdict"] = record["abaqus_results"]["verdict"] = "不通过"
            store(record)
        store(record, paths.CASE_LIBRARY_DIR)
    monkeypatch.setattr(memory, "CaseMemoryIndex", lambda: pytest.fail("不得触碰索引"))
    with pytest.raises(ValueError):
        memory.rebuild_case_memory_index()


def test_embedding_failure_does_not_prune_index(monkeypatch):
    path = store(case_record())
    before, events = path.read_bytes(), []
    index = FakeIndex(events, fail=True)
    monkeypatch.setattr(memory, "CaseMemoryIndex", lambda: index)
    with pytest.raises(RuntimeError, match="embedding failure"):
        memory.rebuild_case_memory_index()
    assert index.ids == {"CASE_99"} and [name for name, _ in events] == ["upsert"]
    assert path.read_bytes() == before


def test_index_rebuild_uses_persistent_collection_api(monkeypatch, tmp_path):
    import numpy as np
    from core.rag_engine import RAGEngine

    engine = RAGEngine(chroma_dir=tmp_path / "index")
    engine._embedder = SimpleNamespace(encode=lambda documents, **kw: np.tile([0.0, 1.0], (len(documents), 1)))
    engine.upsert_documents(["CASE_99"], ["stale"], [{"kind": "case"}])
    store(case_record())
    monkeypatch.setattr(memory, "CaseMemoryIndex", lambda: SimpleNamespace(engine=engine))
    assert memory.rebuild_case_memory_index()["removed_index_entries"] == 1
    assert engine.collection.get(include=[])["ids"] == ["CASE_1"]


@pytest.mark.parametrize("status", ["success", "failed"])
def test_rebuild_keeps_candidate_material_and_records_actual_result(monkeypatch, status):
    record = case_record()
    record["design"]["material_system"]["E1_GPa"] = 173.0
    path = store(record)
    formal = store(record, paths.CASE_LIBRARY_DIR)
    before, captured = path.read_bytes(), []
    def solve(_self, candidate, **_kwargs):
        assert path.read_bytes() == before and formal.exists()
        captured.append(deepcopy(candidate))
        return {"candidate_id": candidate["candidate_id"], "status": status, "retry_count": 0,
                "BLF_global": 2.5 if status == "success" else None,
                "verdict": "通过" if status == "success" else None,
                "analysis_contract": analysis_contract(candidate)}
    indexed = []
    monkeypatch.setattr(artifacts.FEMAgent, "run", solve)
    monkeypatch.setattr(artifacts, "_upsert_case_memory", indexed.append)
    result = artifacts.rebuild_case(path, force=True)
    assert result["status"] == status
    assert captured[0]["material_system"] == record["design"]["material_system"]
    assert captured[0]["display_name"] == record["design"]["candidate_id"]
    updated = read_case_record(path)
    assert updated["design"] == record["design"]
    assert updated["verdict"] == ("通过" if status == "success" else "失败")
    assert formal.exists() == (status == "success")
    assert indexed == [updated]


def test_rebuild_preflights_whole_selection_before_starting_jobs(monkeypatch):
    store(case_record())
    broken = case_record(2)
    broken["design"].pop("material_system")
    store(broken)
    monkeypatch.setattr(artifacts, "rebuild_case", lambda *a, **kw: pytest.fail("不得提交部分批次"))
    monkeypatch.setattr(sys, "argv", ["rebuild_abaqus_artifacts.py", "--force"])
    with pytest.raises(ValueError, match="CASE_2"):
        artifacts.main()


def test_missing_inp_is_rebuilt_and_solver_failure_is_nonzero(monkeypatch, tmp_path):
    record = case_record()
    odb = tmp_path / "C1.odb"
    odb.write_text("fixture", encoding="ascii")
    record["abaqus_results"].update(abaqus_odb=str(odb), abaqus_inp=str(tmp_path / "missing.inp"))
    path = store(record)
    calls = []
    def rebuild(selected, **kwargs):
        calls.append(selected)
        return {"status": "failed"}
    monkeypatch.setattr(artifacts, "rebuild_case", rebuild)
    monkeypatch.setattr(sys, "argv", ["rebuild_abaqus_artifacts.py", "--case-id", "CASE_1"])
    assert artifacts.main() == 1 and calls == [path]


def task_args(**values):
    return SimpleNamespace(**{**dict(task_json=None, stiffener_types=["T", "BLADE", "L", "HAT"],
                                     load_type="compression_shear", nx=120., nxy=60., boundary="SSCC", material="T300_5208"), **values})


def test_initial_doe_covers_explicit_types_and_preserves_verified_inputs_without_training(monkeypatch):
    tasks = initial.input_tasks(task_args())
    for task in tasks:
        task["analysis"] = {"mesh_size_mm": 8., "buckling_modes": 16}
    plan = initial.build_plan(tasks, 4)
    submitted, stored = [], []
    class Knowledge:
        config = {"pipeline": {"auto_retrain_surrogate": True}}
        def run(self, payload):
            assert self.config["pipeline"]["auto_retrain_surrogate"] is False
            stored.append(payload)
            return {"case_id": "CASE_" + str(len(stored))}
    def solve(_self, candidate):
        submitted.append(deepcopy(candidate))
        return {"candidate_id": candidate["candidate_id"], "status": "success", "retry_count": 0,
                "BLF_global": 2.5, "verdict": "通过", "analysis_contract": analysis_contract(candidate)}
    monkeypatch.setattr(initial, "KnowledgeAgent", Knowledge)
    monkeypatch.setattr(initial.FEMAgent, "run", solve)
    records = initial.run_plan(plan, 1)
    assert len(records) == len(submitted) == len(stored) == 4
    assert {item["stiffener_type"] for item in submitted} == {"T", "BLADE", "L", "HAT"}
    for batch, candidate, saved in zip(plan, submitted, stored):
        assert candidate == saved["design"] == batch["candidates"][0]
        assert candidate["material_system"] == batch["task"]["task"]["material_system"]
        assert candidate["analysis"] == {"mesh_size_mm": 8., "buckling_modes": 16}
        assert candidate["load_conditions"]["Nxy_kN_per_m"] == 60
        assert candidate["boundary_conditions"]["type"] == "SSCC"
        assert read_json(paths.TASKS_DIR / (batch["task"]["task_id"] + ".json")) == batch["task"]


def test_initial_task_json_is_read_only_and_missing_physics_is_rejected(tmp_path):
    task = case_record()["task"]
    path = tmp_path / "task.json"
    write_json(path, {"task": task})
    before = path.read_bytes()
    parsed = initial.input_tasks(task_args(task_json=[path]))
    assert len(parsed) == 1 and parsed[0]["material_system"] == task["material_system"]
    assert parsed[0]["stiffener_type"] == task["stiffener_type"]
    assert path.read_bytes() == before
    del task["load_conditions"]["Nx_kN_per_m"]
    write_json(path, task)
    with pytest.raises(ValueError):
        initial.input_tasks(task_args(task_json=[path]))


def test_reference_task_generates_four_types_and_seed_controls_sampling():
    tasks = initial.input_tasks(task_args(task_json=[initial.ROOT / "reference/physical_doe_task.json"]))
    first = initial.build_plan(tasks, 4, seed=42)
    repeat = initial.build_plan(tasks, 4, seed=42)
    other = initial.build_plan(tasks, 4, seed=1042)
    assert {batch["task"]["task"]["stiffener_type"] for batch in first} == {"T", "BLADE", "L", "HAT"}
    for a, b, c in zip(first, repeat, other):
        candidate = a["candidates"][0]
        assert candidate["geometry"] == b["candidates"][0]["geometry"]
        assert candidate["geometry"] != c["candidates"][0]["geometry"]
        assert candidate["geometry"]["panel_length_mm"] == 360
        assert candidate["geometry"]["panel_width_mm"] == 300
        assert candidate["analysis"] == {"mesh_size_mm": 8.0, "buckling_modes": 16}


@pytest.mark.parametrize("arguments", [["--reset"], ["--stiffener-types", "T", "--nx", "-1"],
                                       ["--task-json", "not-read.json", "--boundary", "SSSS"]])
def test_initial_invalid_arguments_do_not_submit_jobs(monkeypatch, arguments):
    monkeypatch.setattr(initial, "run_plan", lambda *a: pytest.fail("不得启动求解"))
    monkeypatch.setattr(sys, "argv", ["build_initial_cases.py", "--count", "1", *arguments])
    with pytest.raises((SystemExit, ValueError)):
        initial.main()
