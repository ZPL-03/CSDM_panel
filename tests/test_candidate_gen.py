from __future__ import annotations

from agents.candidate_gen import CandidateGenAgent
from core.llm_backend import LLMBackend
from core.stiffener_profile import default_geometry


def _build_task() -> dict:
    return {
        "task_id": "TASK_1",
        "source": "test",
        "task": {
            "application": "复合材料加筋壁板",
            "load_conditions": {
                "type": "compression_shear",
                "label": "压剪组合",
                "Nx_kN_per_m": 1000.0,
                "Nxy_kN_per_m": 180.0,
            },
            "boundary_conditions": {
                "type": "SSCC",
                "label": "X 向简支 + Y 向固支（SSCC）",
                "description": "X0/X1 简支，Y0/Y1 固支",
                "simply_supported_edges": ["X0", "X1"],
                "clamped_edges": ["Y0", "Y1"],
            },
            "geometry_envelope": {
                "panel_length_mm": [600, 800],
                "panel_width_mm": [500, 700],
                "max_stiffener_height_mm": 50,
            },
            "candidate_generation_preferences": {
                "total_candidates": 10,
                "source_allocation_mode": "ratio",
                "source_ratio": {"llm": 2.0, "case_transfer": 1.0, "doe": 1.0},
            },
            "screening_preferences": {"top_k_candidates": 5},
            "material_system": {
                "name": "T300/5208",
                "E1_GPa": 181,
                "E2_GPa": 10.3,
                "G12_GPa": 7.17,
                "nu12": 0.28,
                "density_kg_per_m3": 1600,
            },
            "layup_constraints": {
                "allowed_angles": [0, 45, -45, 90],
                "symmetric": True,
                "balanced": True,
                "min_ratio_per_angle": 0.1,
            },
            "stiffener_type": "T",
            "design_targets": {"BLF_min": 1.2, "primary_objective": "最小重量"},
        },
    }


def _candidate_table(count: int) -> str:
    rows = [
        "| 编号 | 材料 | 壁板长度(mm) | 壁板宽度(mm) | 蒙皮厚度(mm) | 筋距(mm) | 筋高(mm) | 腹板厚度(mm) | 翼缘宽度(mm) | 翼缘厚度(mm) | 帽顶宽度(mm) | 帽顶厚度(mm) | 铺层 | f0 | f45 | f90 | 推荐理由 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for index in range(count):
        rows.append(
            f"| A{index + 1} | T300/5208 | {700 + index} | {600 + index} | 2.5 | 120 | 28 | 2.0 | 16 | 2.0 | - | - | [45/-45/0/90/0/-45/45]s | 0.286 | 0.571 | 0.143 | 结构性能与制造风险均衡 |"
        )
    return "\n".join(rows)


class _FakeNaturalBackend:
    def __init__(self, count: int = 6, sink: dict[str, str] | None = None) -> None:
        self.max_tokens = 1800
        self.count = count
        self.sink = sink if sink is not None else {}

    def chat(self, system_prompt: str, user_prompt: str, max_tokens_override: int | None = None) -> str:
        self.sink["system_prompt"] = system_prompt
        self.sink["user_prompt"] = user_prompt
        self.sink["max_tokens_override"] = str(max_tokens_override)
        return "以下为候选方案表：\n\n" + _candidate_table(self.count)


def test_candidate_gen_parses_natural_language_table_from_llm() -> None:
    captured: dict[str, str] = {}
    agent = CandidateGenAgent()
    agent.llm_backend = _FakeNaturalBackend(count=6, sink=captured)
    agent.knowledge_base = type("FakeKnowledge", (), {"format_snippets": staticmethod(lambda _task, top_k=5: [])})()

    candidates = agent.run(_build_task())

    assert len(candidates) == 10
    assert sum(1 for item in candidates if item["source"] == "LLM") == 5
    assert all(candidate["candidate_id"].startswith("TMP_") for candidate in candidates)
    assert all(candidate["display_name"] == candidate["candidate_id"] for candidate in candidates)
    assert all("persistent_candidate_id" not in candidate for candidate in candidates)
    assert candidates[0]["load_conditions"]["type"] == "compression_shear"
    assert candidates[0]["boundary_conditions"]["type"] == "SSCC"
    assert candidates[0]["layup"]["skin_layup"] == "[45/-45/0/90/0/-45/45]s"
    assert "以下为候选方案表" in candidates[0]["llm_output_excerpt"]
    assert "A1" in candidates[0]["origin_summary"]
    assert "不要输出 JSON" in captured["system_prompt"]
    assert "Markdown 表格" in captured["system_prompt"]


def test_candidate_gen_keeps_complete_llm_answer_text() -> None:
    agent = CandidateGenAgent()

    class _LongAnswerBackend(_FakeNaturalBackend):
        def chat(self, system_prompt: str, user_prompt: str, max_tokens_override: int | None = None) -> str:
            return super().chat(system_prompt, user_prompt, max_tokens_override) + "\n" + ("完整回答追踪" * 260) + "END_MARKER"

    agent.llm_backend = _LongAnswerBackend(count=6)
    agent.knowledge_base = type("FakeKnowledge", (), {"format_snippets": staticmethod(lambda _task, top_k=5: [])})()

    candidates = agent.run(_build_task())

    assert "END_MARKER" in candidates[0]["llm_output_excerpt"]
    assert len(candidates[0]["llm_output_excerpt"]) > 2000


def test_candidate_gen_respects_total_candidate_target_and_two_one_one_ratio() -> None:
    agent = CandidateGenAgent()
    agent.llm_backend = _FakeNaturalBackend(count=6)
    agent.knowledge_base = type("FakeKnowledge", (), {"format_snippets": staticmethod(lambda _task, top_k=5: [])})()

    def _fake_doe_candidates(task: dict, n_samples: int, start_index: int = 1, **_kwargs) -> list[dict]:
        candidates = []
        for offset in range(n_samples):
            raw = {
                "geometry": {
                    "panel_length_mm": 820 + offset,
                    "panel_width_mm": 720 + offset,
                    "skin_thickness_mm": 2.5,
                    "pitch_mm": 120,
                    "stiffener_height_mm": 28,
                    "web_thickness_mm": 2.0,
                    "flange_width_mm": 16,
                    "flange_thickness_mm": 2.0,
                },
                "layup": {
                    "skin_layup": "[45/-45/0/90/0/-45/45]s",
                    "skin_f0": 0.286,
                    "skin_f45": 0.571,
                    "skin_f90": 0.143,
                },
                "rationale": f"doe candidate {offset + 1}",
            }
            candidates.append(agent._normalize_candidate(task, raw, start_index + offset, "DOE"))
        return candidates

    def _fake_retrieve_transferable_cases(task: dict, top_k: int = 5) -> list[dict]:
        return [
            {
                "case_id": f"CASE_{idx}",
                "design": {
                    "stiffener_type": task.get("stiffener_type", "T"),
                    "geometry": {
                        "panel_length_mm": 760 + idx,
                        "panel_width_mm": 660 + idx,
                        "skin_thickness_mm": 2.5,
                        "pitch_mm": 120,
                        "stiffener_height_mm": 28,
                        "web_thickness_mm": 2.0,
                        "flange_width_mm": 16,
                        "flange_thickness_mm": 2.0,
                    },
                    "layup": {"skin_layup": "[45/-45/0/90/0/-45/45]s"},
                    "rationale": f"case transfer candidate {idx + 1}",
                    "load_conditions": task.get("load_conditions", {}),
                    "boundary_conditions": task.get("boundary_conditions", {}),
                    "material_system": task.get("material_system", {}),
                },
                "abaqus_results": {"status": "success", "verdict": "通过"},
            }
            for idx in range(3)
        ]

    agent.doe_sampler = type("FakeDOE", (), {"sample_candidates": staticmethod(_fake_doe_candidates)})()
    agent.case_retriever = type(
        "FakeCaseRetriever",
        (),
        {"retrieve_transferable_cases": staticmethod(_fake_retrieve_transferable_cases)},
    )()
    task = _build_task()
    task["task"]["candidate_generation_preferences"]["total_candidates"] = 12

    candidates = agent.run(task)

    assert len(candidates) == 12
    assert sum(1 for item in candidates if item["source"] == "LLM") == 6
    assert sum(1 for item in candidates if item["source"] == "CASE_TRANSFER") == 3
    assert sum(1 for item in candidates if item["source"] == "DOE") == 3


def test_candidate_gen_distributes_doe_candidates_across_requested_stiffener_types() -> None:
    agent = CandidateGenAgent()
    agent.llm_backend = None
    agent.case_retriever = type(
        "EmptyCaseRetriever",
        (),
        {"retrieve_transferable_cases": staticmethod(lambda _task, top_k=5: [])},
    )()

    def _fake_doe_candidates(
        task: dict,
        n_samples: int,
        start_index: int = 1,
        stiffener_type: str = "T",
        **_kwargs,
    ) -> list[dict]:
        candidates = []
        for offset in range(n_samples):
            geometry = default_geometry(stiffener_type)
            geometry["panel_length_mm"] += start_index + offset
            raw = {
                "geometry": geometry,
                "layup": {
                    "skin_layup": "[45/-45/0/90/0/-45/45]s",
                    "skin_f0": 0.286,
                    "skin_f45": 0.571,
                    "skin_f90": 0.143,
                },
                "rationale": f"{stiffener_type} doe candidate {offset + 1}",
            }
            candidates.append(agent._normalize_candidate(task, raw, start_index + offset, "DOE"))
        return candidates

    agent.doe_sampler = type("FakeDOE", (), {"sample_candidates": staticmethod(_fake_doe_candidates)})()
    task = _build_task()
    task["task"]["candidate_generation_preferences"] = {
        "total_candidates": 12,
        "source_allocation_mode": "ratio",
        "source_ratio": {"llm": 0.0, "case_transfer": 0.0, "doe": 1.0},
        "stiffener_types": ["T", "HAT", "BLADE"],
    }

    candidates = agent.run(task)

    assert len(candidates) == 12
    assert [candidates[index]["candidate_id"] for index in range(12)] == [f"TMP_{index}" for index in range(1, 13)]
    assert {stype: sum(1 for candidate in candidates if candidate["stiffener_type"] == stype) for stype in ["T", "HAT", "BLADE"]} == {
        "T": 4,
        "HAT": 4,
        "BLADE": 4,
    }


def test_candidate_generation_summary_reports_quota_and_effective_counts() -> None:
    messages: list[str] = []
    agent = CandidateGenAgent(progress_callback=lambda _agent, message, _event=None: messages.append(message))
    agent.llm_backend = _FakeNaturalBackend(count=6)
    agent.knowledge_base = type("FakeKnowledge", (), {"format_snippets": staticmethod(lambda _task, top_k=5: [])})()
    agent.case_retriever = type(
        "EmptyCaseRetriever",
        (),
        {"retrieve_transferable_cases": staticmethod(lambda _task, top_k=5: [])},
    )()

    candidates = agent.run(_build_task())

    assert len(candidates) == 10
    summary = messages[-1]
    assert "初始配额 LLM=5 / 案例迁移=3 / DOE=2" in summary
    assert "有效进入候选池 LLM=5，案例迁移=0，DOE补足=5" in summary
    assert "原始" not in summary
    assert "案例迁移为 0" in summary


def test_candidate_gen_uses_two_one_one_source_ratio() -> None:
    agent = CandidateGenAgent()
    task = _build_task()
    task["task"]["candidate_generation_preferences"] = {
        "total_candidates": 12,
        "source_allocation_mode": "ratio",
        "source_ratio": {"llm": 2.0, "case_transfer": 1.0, "doe": 1.0},
    }

    source_targets = agent._resolve_source_targets(task)

    assert source_targets["llm"] == 6
    assert source_targets["case_transfer"] == 3
    assert source_targets["doe"] == 3
    assert source_targets["source_ratio"] == {"llm": 2.0, "case_transfer": 1.0, "doe": 1.0}


def test_candidate_gen_build_prompt_uses_knowledge_base_guidance() -> None:
    agent = CandidateGenAgent()
    system_prompt, user_prompt = agent._build_prompt(
        _build_task(),
        ["[外部知识库 1] Composite buckling study\nabstract snippet"],
        2,
    )

    assert "外部知识库/知识图谱依据" in user_prompt
    assert "Composite buckling study" in user_prompt
    assert "参考案例" not in user_prompt
    assert "不要输出 JSON" in system_prompt
    assert "只输出合法 JSON" not in system_prompt


def test_candidate_generation_deduplicates_equivalent_designs() -> None:
    messages: list[str] = []
    agent = CandidateGenAgent(progress_callback=lambda _agent, message, _event=None: messages.append(message))

    class _DuplicateBackend(_FakeNaturalBackend):
        def chat(self, system_prompt: str, user_prompt: str, max_tokens_override: int | None = None) -> str:
            return "\n".join(
                [
                    "以下为候选方案表：",
                    "",
                    "| 编号 | 材料 | 壁板长度(mm) | 壁板宽度(mm) | 蒙皮厚度(mm) | 筋距(mm) | 筋高(mm) | 腹板厚度(mm) | 翼缘宽度(mm) | 翼缘厚度(mm) | 帽顶宽度(mm) | 帽顶厚度(mm) | 铺层 | f0 | f45 | f90 | 推荐理由 |",
                    "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | ---: | ---: | ---: | --- |",
                    "| A1 | T300/5208 | 700 | 600 | 2.5 | 120 | 28 | 2.0 | 16 | 2.0 | - | - | [45/-45/0/90/0/-45/45]s | 0.286 | 0.571 | 0.143 | 结构性能与制造风险均衡 |",
                    "| A2 | T300/5208 | 700 | 600 | 2.5 | 120 | 28 | 2.0 | 16 | 2.0 | - | - | [45/-45/0/90/0/-45/45]s | 0.286 | 0.571 | 0.143 | 结构性能与制造风险均衡 |",
                    "| A3 | T300/5208 | 700 | 600 | 2.5 | 120 | 28 | 2.0 | 16 | 2.0 | - | - | [45/-45/0/90/0/-45/45]s | 0.286 | 0.571 | 0.143 | 结构性能与制造风险均衡 |",
                ]
            )

    agent.llm_backend = _DuplicateBackend(count=3)
    agent.knowledge_base = type("FakeKnowledge", (), {"format_snippets": staticmethod(lambda _task, top_k=5: [])})()
    agent.case_retriever = type(
        "EmptyCaseRetriever",
        (),
        {"retrieve_transferable_cases": staticmethod(lambda _task, top_k=5: [])},
    )()

    task = _build_task()
    task["task"]["candidate_generation_preferences"]["total_candidates"] = 6
    candidates = agent.run(task)
    signatures = [agent._candidate_signature(candidate) for candidate in candidates]

    assert len(candidates) == 6
    assert len(signatures) == len(set(signatures))
    assert [candidate["candidate_id"] for candidate in candidates] == [f"TMP_{index}" for index in range(1, 7)]
    assert any("候选去重过滤" in message for message in messages)


def test_openai_compatible_chat_uses_plain_text_request(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            message = type("Message", (), {"content": "plain text"})()
            choice = type("Choice", (), {"message": message})()
            return type("Response", (), {"choices": [choice]})()

    class _FakeOpenAI:
        def __init__(self, **kwargs):
            captured["client_kwargs"] = kwargs
            self.chat = type("Chat", (), {"completions": _FakeCompletions()})()

    monkeypatch.setattr("openai.OpenAI", _FakeOpenAI)

    backend = LLMBackend(
        {
            "backend": {
                "provider": "openai_compatible",
                "base_url": "https://example.test/v1",
                "api_key": "test-key",
                "model": "test-model",
                "temperature": 0.2,
                "max_tokens": 1800,
                "timeout_seconds": 30,
            },
            "fallback": {"max_format_retries": 3},
        }
    )

    assert backend.chat("system", "user") == "plain text"

    assert captured["max_tokens"] == 1800
    assert captured["model"] == "test-model"
    assert "response_format" not in captured
