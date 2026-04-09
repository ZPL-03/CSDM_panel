from agents.candidate_gen import CandidateGenAgent


def _build_task() -> dict:
    return {
        "task_id": "TASK_1",
        "application": "复合材料加筋壁板",
        "load_conditions": {"type": "compression_shear", "label": "压剪组合", "Nx_kN_per_m": 1000.0, "Nxy_kN_per_m": 180.0},
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
        "candidate_generation_preferences": {"total_candidates": 10},
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
    }


def test_candidate_gen_tolerates_string_items_from_llm() -> None:
    agent = CandidateGenAgent()
    agent.llm_backend = type(
        "FakeBackend",
        (),
        {
            "generate_json": staticmethod(
                lambda _system, _user: {
                    "candidates": [
                        '{"geometry": {"panel_length_mm": 700, "panel_width_mm": 600}, "layup": {"skin_layup": "[45/-45/0/90/0/-45/45]s"}, "rationale": "string item"}',
                        {
                            "geometry": {
                                "panel_length_mm": 720,
                                "panel_width_mm": 580,
                                "skin_thickness_mm": 2.6,
                                "pitch_mm": 118,
                                "stiffener_height_mm": 27,
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
                            "rationale": "dict item",
                        },
                    ]
                }
            )
        },
    )()
    agent.rag_engine = type("FakeRAG", (), {"retrieve": staticmethod(lambda _task, top_k=5: [])})()

    candidates = agent.run(_build_task())

    assert len(candidates) >= 2
    assert all(candidate["candidate_id"].startswith("TMP_") for candidate in candidates)
    assert candidates[0]["display_name"].startswith("候选样本")
    assert candidates[0]["load_conditions"]["type"] == "compression_shear"
    assert candidates[0]["boundary_conditions"]["type"] == "SSCC"


def test_candidate_gen_tolerates_list_layup_from_llm() -> None:
    agent = CandidateGenAgent()
    agent.llm_backend = type(
        "FakeBackend",
        (),
        {
            "generate_json": staticmethod(
                lambda _system, _user: {
                    "candidates": [
                        {
                            "geometry": {
                                "panel_length_mm": 700,
                                "panel_width_mm": 600,
                                "skin_thickness_mm": 2.5,
                                "pitch_mm": 120,
                                "stiffener_height_mm": 28,
                                "web_thickness_mm": 2.0,
                                "flange_width_mm": 16,
                                "flange_thickness_mm": 2.0,
                            },
                            "layup": {
                                "skin_layup": [45, -45, 0, 90, 0, -45, 45, "s"],
                                "skin_f0": 0.286,
                                "skin_f45": 0.571,
                                "skin_f90": 0.143,
                            },
                            "rationale": "list layup",
                        }
                    ]
                }
            )
        },
    )()
    agent.rag_engine = type("FakeRAG", (), {"retrieve": staticmethod(lambda _task, top_k=5: [])})()

    candidates = agent.run(_build_task())

    assert len(candidates) >= 1
    llm_candidate = candidates[0]
    assert llm_candidate["source"] == "LLM"
    assert llm_candidate["layup"]["skin_layup"] == "[45/-45/0/90/0/-45/45]s"
    assert llm_candidate["rule_check"]["is_valid"] is True


def test_candidate_gen_respects_total_candidate_target() -> None:
    agent = CandidateGenAgent()
    agent.llm_backend = type(
        "FakeBackend",
        (),
        {
            "generate_json": staticmethod(
                lambda _system, _user: {
                    "candidates": [
                        {
                            "geometry": {
                                "panel_length_mm": 700,
                                "panel_width_mm": 600,
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
                            "rationale": "llm candidate 1",
                        },
                        {
                            "geometry": {
                                "panel_length_mm": 720,
                                "panel_width_mm": 620,
                                "skin_thickness_mm": 2.6,
                                "pitch_mm": 118,
                                "stiffener_height_mm": 29,
                                "web_thickness_mm": 2.1,
                                "flange_width_mm": 17,
                                "flange_thickness_mm": 2.1,
                            },
                            "layup": {
                                "skin_layup": "[45/-45/0/90/0/-45/45]s",
                                "skin_f0": 0.286,
                                "skin_f45": 0.571,
                                "skin_f90": 0.143,
                            },
                            "rationale": "llm candidate 2",
                        },
                    ]
                }
            )
        },
    )()
    agent.rag_engine = type("FakeRAG", (), {"retrieve": staticmethod(lambda _task, top_k=5: [])})()

    def _fake_doe_candidates(task: dict, n_samples: int, start_index: int = 1, **_kwargs) -> list[dict]:
        candidates = []
        for offset in range(n_samples):
            raw = {
                "geometry": {
                    "panel_length_mm": 700 + offset,
                    "panel_width_mm": 600 + offset,
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

    agent.doe_sampler = type("FakeDOE", (), {"sample_candidates": staticmethod(_fake_doe_candidates)})()
    task = _build_task()
    task["candidate_generation_preferences"] = {"total_candidates": 12}

    candidates = agent.run(task)

    assert len(candidates) == 12
    assert sum(1 for item in candidates if item["source"] == "LLM") == 2
    assert sum(1 for item in candidates if item["source"] == "DOE") == 10
