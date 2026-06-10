import json
from pathlib import Path

from core.domain_knowledge import DomainKnowledgeBase


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def _task() -> dict:
    return {
        "task": {
            "application": "复合材料加筋壁板",
            "load_conditions": {"type": "axial_compression", "Nx_kN_per_m": 900.0, "Nxy_kN_per_m": 0.0},
            "boundary_conditions": {"type": "SSSS"},
            "geometry_envelope": {
                "panel_length_mm": [600, 800],
                "panel_width_mm": [500, 700],
                "max_stiffener_height_mm": 50,
            },
            "material_system": {"name": "T300/5208"},
            "stiffener_type": "T",
            "design_targets": {"BLF_min": 1.2, "primary_objective": "最小重量"},
        }
    }


def test_domain_knowledge_retrieves_knowledge_base_and_graph(tmp_path: Path) -> None:
    rag_path = tmp_path / "rag" / "rag_chunks.jsonl"
    kg_dir = tmp_path / "kg"
    manifest_path = tmp_path / "manifest.json"
    _write_jsonl(
        rag_path,
        [
            {
                "chunk_id": "CHUNK_1",
                "chunk_fingerprint": "fp1",
                "record_id": "SRC_1",
                "source_id": "DOC_1",
                "retrieval_scope": "main",
                "title": "Stiffened composite panel buckling",
                "document_title": "Stiffened composite panel buckling",
                "doi": "10.1000/panel.1",
                "source_url": "https://example.com/panel",
                "year": "2026",
                "venue": "Composite Structures",
                "chunk_type": "fulltext",
                "content_plain": "stiffened composite panel axial compression buckling laminate T stiffener",
                "content_markdown": "T stiffener panel buckling guidance.",
                "task_categories": ["stiffened_panel_shell_structure", "buckling_stability"],
            },
            {
                "chunk_id": "CHUNK_2",
                "chunk_fingerprint": "fp2",
                "record_id": "SRC_1",
                "source_id": "DOC_1",
                "retrieval_scope": "main",
                "title": "Stiffened composite panel buckling",
                "document_title": "Stiffened composite panel buckling",
                "doi": "10.1000/panel.1",
                "source_url": "https://example.com/panel",
                "year": "2026",
                "venue": "Composite Structures",
                "chunk_type": "fulltext",
                "content_plain": "stiffened composite panel T stiffener compression postbuckling",
                "content_markdown": "Second panel buckling guidance from same source.",
                "task_categories": ["stiffened_panel_shell_structure"],
            }
        ],
    )
    _write_jsonl(kg_dir / "entities.jsonl", [{"type": "FailureMode", "name": "Buckling"}])
    _write_jsonl(
        kg_dir / "relations.jsonl",
        [
            {
                "source_type": "Structure",
                "source": "Stiffened Panel",
                "target_type": "FailureMode",
                "target": "Buckling",
                "relation": "EXPERIENCES",
                "record_id": "SRC_1",
                "evidence_document_title": "Stiffened composite panel buckling",
                "evidence_doi": "10.1000/panel.1",
                "evidence_source_url": "https://example.com/panel",
            },
            {
                "source_type": "Structure",
                "source": "Stiffened Panel",
                "target_type": "FailureMode",
                "target": "Buckling",
                "relation": "EXPERIENCES",
                "record_id": "SRC_1",
                "evidence_document_title": "Stiffened composite panel buckling",
                "evidence_doi": "10.1000/panel.1",
                "evidence_source_url": "https://example.com/panel",
            }
        ],
    )
    manifest_path.write_text(
        json.dumps({"rag_chunk_count": 1, "kg_entity_count": 1, "kg_relation_count": 1}, ensure_ascii=False),
        encoding="utf-8",
    )

    knowledge = DomainKnowledgeBase(
        {
            "external_knowledge": {
                "enabled": True,
                "rag_chunks_path": str(rag_path),
                "kg_dir": str(kg_dir),
                "manifest_path": str(manifest_path),
                "top_k": 2,
                "kg_top_k": 2,
                "max_snippet_chars": 200,
            }
        }
    )

    result = knowledge.retrieve(_task(), top_k=2, kg_top_k=2)
    snippets = knowledge.format_snippets(_task(), top_k=2)
    status = knowledge.status()
    snippet_text = "\n".join(snippets)

    assert {chunk["chunk_id"] for chunk in result["chunks"]} == {"CHUNK_1", "CHUNK_2"}
    assert all(chunk["source_url"] == "https://example.com/panel" for chunk in result["chunks"])
    assert result["relations"][0]["relation"] == "EXPERIENCES"
    assert "T stiffener panel buckling guidance" in snippets[0]
    assert "来源 S1" in snippet_text
    assert snippet_text.count("DOI: 10.1000/panel.1") == 1
    assert snippet_text.count("https://example.com/panel") == 1
    assert snippet_text.count("Stiffened Panel(Structure) -[EXPERIENCES]-> Buckling(FailureMode)") == 1
    assert status["rag_chunk_count"] == 1
