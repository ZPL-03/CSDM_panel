import json
import gzip
import sqlite3
from pathlib import Path

import pytest

from core.domain_knowledge import DomainKnowledgeBase, KnowledgeGraphRetriever, RagChunkRetriever, _jsonl_rows, _source_groups, tokenize


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8") as handle:
        handle.write("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n")


def _write_index(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE chunks (rowid INTEGER PRIMARY KEY, payload TEXT)")
        connection.execute("CREATE VIRTUAL TABLE chunks_fts USING fts5(title_tokens, section_tokens, content_tokens, content='')")
        for index, row in enumerate(rows, 1):
            connection.execute("INSERT INTO chunks VALUES (?, ?)", (index, json.dumps(row, ensure_ascii=False)))
            connection.execute("INSERT INTO chunks_fts(rowid, title_tokens, section_tokens, content_tokens) VALUES (?, ?, ?, ?)",
                               (index, " ".join(tokenize(row.get("title", ""))), "", " ".join(tokenize(row["content_plain"]))))


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
    rag_path = tmp_path / "rag" / "retrieval_index.sqlite"
    kg_dir = tmp_path / "kg"
    manifest_path = tmp_path / "manifest.json"
    _write_index(
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
    _write_jsonl(tmp_path / "source_metadata.compact.jsonl.gz", [{"source_id": "DOC_1", "record_id": "SRC_1"}])
    _write_jsonl(kg_dir / "entities.jsonl", [{"type": "FailureMode", "name": "Buckling"}])
    _write_jsonl(
        kg_dir / "relations.compact.jsonl.gz",
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
                "rag_index_path": str(rag_path),
                "kg_dir": str(kg_dir),
                "source_metadata_path": str(tmp_path / "source_metadata.compact.jsonl.gz"),
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


def test_source_groups_merge_doi_variants_and_incomplete_graph_metadata() -> None:
    groups = _source_groups([
        ({"record_id": "LIT_1", "source_id": "SRC_1", "doi": "https://doi.org/10.1234/ABC", "source_url": "https://publisher.test/a"}, False),
        ({"record_id": "LIT_2", "doi": "doi:10.1234/abc", "source_url": "https://repository.test/b"}, False),
        ({"record_id": "LIT_3", "evidence_source_id": "SRC_1"}, True),
        ({"record_id": "LIT_4", "document_title": "另一篇资料"}, False),
    ])
    assert groups[:3] == [0, 0, 0]
    assert groups[3] != 0


def test_bm25_deduplicates_content_and_preserves_source_diversity(tmp_path: Path) -> None:
    path = tmp_path / "index.sqlite"
    _write_index(path, [
        {"chunk_id": f"A{i}", "source_id": "SRC_A", "content_hash": f"hash{i}", "content_plain": "hat stiffener panel buckling"}
        for i in range(5)
    ] + [
        {"chunk_id": "DUP", "source_id": "SRC_A", "content_hash": "hash0", "content_plain": "hat stiffener panel buckling"},
        {"chunk_id": "B", "source_id": "SRC_B", "content_plain": "composite panel buckling"},
        {"chunk_id": "CN", "source_id": "SRC_C", "content_plain": "帽形加筋壁板屈曲及铺层设计"},
    ])
    retriever = RagChunkRetriever(path)
    hits = retriever.search("hat stiffener panel buckling", top_k=5)
    assert sum(hit["source_id"] == "SRC_A" for hit in hits) <= 2
    assert "B" in {hit["chunk_id"] for hit in hits}
    assert "CN" in {hit["chunk_id"] for hit in retriever.search("帽形加筋壁板", top_k=5)}
    assert retriever.search("panel", top_k=0) == []


def test_invalid_knowledge_record_raises(tmp_path: Path) -> None:
    path = tmp_path / "relations.jsonl"
    path.write_text('{"source":', encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        list(_jsonl_rows(path))


def test_shape_query_matches_hyphenated_stiffeners(tmp_path: Path) -> None:
    path = tmp_path / "index.sqlite"
    _write_index(path, [
        {"chunk_id": "LAMINATE", "source_id": "A", "content_plain": "L-shaped laminate composite cure optimization"},
        {"chunk_id": "PANEL", "source_id": "B", "content_plain": "L-stiffeners composite panel axial compression buckling"},
    ])
    hits = RagChunkRetriever(path).search("复合材料加筋壁板 laminate optimization 筋型：L", top_k=2)
    assert hits[0]["chunk_id"] == "PANEL"


def test_formula_is_not_truncated_mid_equation(tmp_path: Path) -> None:
    path = tmp_path / "index.sqlite"
    formula = "$$" + "a+b=" * 30 + "c$$"
    _write_index(path, [{"chunk_id": "F1", "source_id": "A", "retrieval_unit": "formula",
                         "content_plain": "composite panel buckling", "content_markdown": formula}])
    hit = RagChunkRetriever(path, max_snippet_chars=30).search("buckling", top_k=1)[0]
    assert hit["text"] == formula


def test_graph_keeps_strongest_chunk_evidence_for_each_relation(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "entities.jsonl", [{"type": "FailureMode", "name": "Buckling"}])
    relation = {"source_type": "Structure", "source": "Stiffened Panel", "target_type": "FailureMode",
                "target": "Buckling", "relation": "EXPERIENCES", "record_id": "SRC1"}
    _write_jsonl(tmp_path / "relations.compact.jsonl.gz", [
        {**relation, "evidence_chunk_id": "OTHER"},
        {**relation, "evidence_chunk_id": "MATCH"},
        {**relation, "record_id": "SRC2", "evidence_chunk_id": "DUPLICATE"},
    ])
    hits = KnowledgeGraphRetriever(tmp_path).search("Buckling", ["SRC1"], chunk_ids=["MATCH"])
    assert len(hits) == 1
    assert hits[0]["evidence_chunk_id"] == "MATCH"
