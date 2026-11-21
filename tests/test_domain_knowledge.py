import json
import gzip
import sqlite3
import csv
from copy import deepcopy
from pathlib import Path

import pytest

from core.domain_knowledge import DomainKnowledgeBase, KnowledgeGraphRetriever, RagChunkRetriever, _evidence_contract, _jsonl_rows, _source_groups, tokenize
from scripts.sync_external_knowledge import sha256_file, verify_assets


V3 = json.loads((Path(__file__).parent / "fixtures/knowledge_v3.json").read_text(encoding="utf-8"))


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8") as handle:
        handle.write("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n")


def _write_index(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE chunks (rowid INTEGER PRIMARY KEY, chunk_id TEXT, source_id TEXT, payload TEXT)")
        connection.execute("CREATE VIRTUAL TABLE chunks_fts USING fts5(title_tokens, section_tokens, content_tokens, content='')")
        for index, row in enumerate(rows, 1):
            row = _chunk(**row)
            connection.execute("INSERT INTO chunks VALUES (?, ?, ?, ?)", (index, row["chunk_id"], row["source_id"], json.dumps(row, ensure_ascii=False)))
            connection.execute("INSERT INTO chunks_fts(rowid, title_tokens, section_tokens, content_tokens) VALUES (?, ?, ?, ?)",
                               (index, " ".join(tokenize(row.get("title", ""))), "", " ".join(tokenize(row["content_plain"]))))


def _chunk(**values) -> dict:
    return {**deepcopy(V3["reference_quality"]), "chunk_fingerprint": values["chunk_id"] + "-fp", **values}


def _relation(chunk: dict) -> dict:
    return {"source_type": "Structure", "source": "Stiffened Panel", "target_type": "FailureMode",
            "target": "Buckling", "relation": "EXPERIENCES", "evidence_chunk_id": chunk["chunk_id"],
            "evidence_source_id": chunk["source_id"], "evidence_chunk_fingerprint": chunk["chunk_fingerprint"],
            **{key: chunk[key] for key in ("evidence_kind", "evidence_usage", "applicability_context")}}


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
                **_relation(_chunk(chunk_id="CHUNK_1", chunk_fingerprint="fp1", source_id="DOC_1")),
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
                **_relation(_chunk(chunk_id="CHUNK_1", chunk_fingerprint="fp1", source_id="DOC_1")),
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
                "max_context_chars": 8000,
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
    assert "T stiffener panel buckling guidance" in snippet_text
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


def test_source_groups_merge_document_copies_with_missing_publication_metadata() -> None:
    groups = _source_groups([
        ({"source_id": "SRC_A", "document_title": "Composite Panel Buckling", "doi": "10.1000/panel", "year": "2024"}, False),
        ({"source_id": "SRC_B", "document_title": "  composite  panel buckling "}, False),
        ({"evidence_source_id": "SRC_C", "evidence_document_title": "Composite Panel Buckling", "year": 2024}, True),
        ({"source_id": "SRC_D", "title": "Composite Panel Buckling"}, False),
    ])
    assert groups[:3] == [0, 0, 0]
    assert groups[3] == 3


@pytest.mark.parametrize("conflict", [{"doi": "10.1000/other"}, {"year": "2025"}])
def test_same_title_with_conflicting_publication_identity_is_not_merged(conflict: dict) -> None:
    groups = _source_groups([
        ({"source_id": "SRC_A", "document_title": "Composite Panel Buckling", "doi": "10.1000/panel", "year": "2024"}, False),
        ({"source_id": "SRC_B", "document_title": "Composite Panel Buckling", **conflict}, False),
        ({"source_id": "SRC_C", "document_title": "Composite Panel Buckling"}, False),
    ])
    assert groups == [0, 1, 2]


def test_bm25_limits_copies_of_same_document_as_one_source(tmp_path: Path) -> None:
    path = tmp_path / "copies.sqlite"
    _write_index(path, [
        {"chunk_id": f"COPY_{index}", "source_id": f"SRC_{index}",
         "document_title": "Composite Panel Buckling", "content_plain": f"composite panel buckling evidence {index}"}
        for index in range(5)
    ] + [{"chunk_id": "OTHER", "source_id": "SRC_OTHER", "document_title": "Another Document",
          "content_plain": "composite panel buckling evidence from another paper"}])
    hits = RagChunkRetriever(path).search("composite panel buckling", top_k=5)
    assert sum(hit["document_title"] == "Composite Panel Buckling" for hit in hits) == 2
    assert "OTHER" in {hit["chunk_id"] for hit in hits}


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


@pytest.mark.parametrize("kind", ["text", "table", "formula"])
def test_evidence_text_and_conditions_are_not_truncated(tmp_path: Path, kind: str) -> None:
    path = tmp_path / "index.sqlite"
    text = "\n" + "composite panel buckling\n" * 100 + "仅适用于四边简支与给定层序。\n"
    conditions = ["Nx=120 kN/m，蒙皮厚度2 mm，四边简支。"]
    table = deepcopy(V3["multilevel"]["table_evidence"]) if kind == "table" else []
    if table:
        text += V3["multilevel"]["content_markdown"]
        table[0]["condition_context"] = conditions
    _write_index(path, [{"chunk_id": "F1", "source_id": "A", "retrieval_unit": kind,
                         "evidence_kind": kind, "evidence_usage": "source_conditions_only",
                         "applicability_context": conditions, "table_evidence": table,
                         "content_plain": "composite panel buckling", "content_markdown": text}])
    hit = RagChunkRetriever(path).search("buckling", top_k=1)[0]
    assert hit["text"] == text
    assert hit["applicability_context"] == conditions and hit["table_evidence"] == table


def test_graph_keeps_strongest_chunk_evidence_for_each_relation(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "entities.jsonl", [{"type": "FailureMode", "name": "Buckling"}])
    chunk = _chunk(chunk_id="MATCH", source_id="SRC1")
    relation = _relation(chunk)
    _write_jsonl(tmp_path / "relations.compact.jsonl.gz", [
        {**relation, "evidence_chunk_id": "OTHER"},
        relation, relation,
    ])
    hits = KnowledgeGraphRetriever(tmp_path).search("Buckling", [chunk])
    assert len(hits) == 1
    assert hits[0]["evidence_chunk_id"] == "MATCH"
    assert hits[0]["evidence_contract"]["chunk_fingerprint"] == chunk["chunk_fingerprint"]


@pytest.mark.parametrize("field,value", [("evidence_chunk_fingerprint", "stale"),
                                        ("evidence_source_id", "OTHER"), ("applicability_context", ["其他工况"])])
def test_graph_rejects_stale_or_mismatched_evidence(tmp_path: Path, field: str, value) -> None:
    chunk = _chunk(chunk_id="CURRENT", source_id="SOURCE")
    _write_jsonl(tmp_path / "entities.jsonl", [])
    _write_jsonl(tmp_path / "relations.compact.jsonl.gz", [{**_relation(chunk), field: value}])
    with pytest.raises(ValueError, match="当前证据"):
        KnowledgeGraphRetriever(tmp_path).search("buckling", [chunk])


def test_retrieval_eligibility_is_explicit(tmp_path: Path) -> None:
    path = tmp_path / "index.sqlite"
    _write_index(path, [{"chunk_id": "BLOCKED", "source_id": "A", "retrieval_eligible": False, "content_plain": "panel buckling"},
                        {"chunk_id": "ALLOWED", "source_id": "B", "content_plain": "panel buckling"}])
    assert [item["chunk_id"] for item in RagChunkRetriever(path).search("panel buckling")] == ["ALLOWED"]
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE chunks SET payload=json_remove(payload, '$.retrieval_eligible') WHERE rowid=2")
    with pytest.raises(ValueError, match="retrieval_eligible"):
        RagChunkRetriever(path).search("panel buckling")


def test_prompt_budget_keeps_complete_evidence_and_bound_relations(tmp_path: Path) -> None:
    large = _chunk(chunk_id="LARGE", source_id="A", content_plain="panel buckling " * 500 + "\n" + V3["multilevel"]["content_markdown"],
                   applicability_context=["原文末尾条件"], evidence_usage="source_conditions_only",
                   table_evidence=deepcopy(V3["multilevel"]["table_evidence"]))
    small = _chunk(chunk_id="SMALL", source_id="B", content_plain="panel buckling under the stated boundary conditions.")
    _write_index(tmp_path / "index.sqlite", [large, small])
    _write_jsonl(tmp_path / "kg/entities.jsonl", [])
    _write_jsonl(tmp_path / "kg/relations.compact.jsonl.gz", [_relation(large), _relation(small)])
    _write_jsonl(tmp_path / "sources.jsonl.gz", [{"source_id": "A"}, {"source_id": "B"}])
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
    knowledge = DomainKnowledgeBase({"external_knowledge": {"rag_index_path": str(tmp_path / "index.sqlite"),
        "kg_dir": str(tmp_path / "kg"), "source_metadata_path": str(tmp_path / "sources.jsonl.gz"),
        "manifest_path": str(tmp_path / "manifest.json"), "max_context_chars": 3000}})
    context = knowledge.prepare_context(_task())
    text = "\n\n".join(context["snippets"])
    assert context["included_chunk_ids"] == ["SMALL"]
    assert context["omitted"][0]["chunk_id"] == "LARGE"
    assert context["omitted"][0]["reason"] == "context_budget"
    assert "LARGE" not in text and "原文末尾条件" not in text
    assert small["content_plain"] in text and small["chunk_fingerprint"] in text
    assert context["character_count"] == len(text) <= 3000
    knowledge.max_context_chars = 20000
    assert len(knowledge.retrieve(_task())["relations"]) == 2
    full = "\n\n".join(knowledge.prepare_context(_task())["snippets"])
    assert large["content_plain"] in full
    assert json.dumps(large["applicability_context"], ensure_ascii=False) in full
    assert json.dumps(large["table_evidence"], ensure_ascii=False) in full


def _table_rows(text: str) -> list[list[str]]:
    """逐列读取固定测试表，独立核对表头与数据行的对应关系。"""
    return [[cell.strip() for cell in next(csv.reader([line.strip()[1:-1]], delimiter="|", escapechar="\\"))]
            for line in text.splitlines() if line.strip().startswith("|")]


@pytest.mark.parametrize("start,end,expected", [
    (0, 0, ""), (None, None, ""), ("", "", ""),
    (8, 8, "第 8 页"), (8, 10, "第 8-10 页"), (8, 0, "第 8 页"),
])
def test_context_page_locator_does_not_treat_unknown_zero_as_a_page(tmp_path, start, end, expected):
    chunk = _chunk(chunk_id="PAGE", source_id="A", content_plain="panel buckling",
                   page_start=start, page_end=end)
    _write_index(tmp_path / "index.sqlite", [chunk])
    _write_jsonl(tmp_path / "kg/entities.jsonl", [])
    _write_jsonl(tmp_path / "kg/relations.compact.jsonl.gz", [_relation(chunk)])
    _write_jsonl(tmp_path / "sources.jsonl.gz", [{"source_id": "A"}])
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
    knowledge = DomainKnowledgeBase({"external_knowledge": {"rag_index_path": str(tmp_path / "index.sqlite"),
        "kg_dir": str(tmp_path / "kg"), "source_metadata_path": str(tmp_path / "sources.jsonl.gz"),
        "manifest_path": str(tmp_path / "manifest.json")}})
    context = knowledge.prepare_context(_task())
    assert len(context["evidence"]) == 1
    assert context["evidence"][0]["pages"] == expected


@pytest.mark.parametrize("name", ["multilevel", "projection_view"])
def test_v3_source_table_headers_units_and_projection_reach_prompt(tmp_path, monkeypatch, name):
    chunk = deepcopy(V3[name])
    rows = _table_rows(chunk["content_markdown"])
    table = chunk["table_evidence"][0]
    assert rows[0] == table["headers"]
    assert table["structure_issues"] == table["missing_header_columns"] == table["missing_unit_columns"] == []
    if name == "multilevel":
        assert rows[0] == ["材料", "弹性模量/GPa / 纵向 $E_f$", "弹性模量/GPa / 横向 $E_{2f}$",
                           "剪切模量/GPa", "泊松比 / $v_{12f}$", "泊松比 / $v_{23f}$"]
        assert rows[2:] == [["纤维", "233", "23.1", "8.96", "0.2", "0.4"],
                            ["基体", "", "4.62", "8.27", "0.36", "0.36"],
                            ["金属内衬", "", "71.7", "-", "0.33", "0.33"]]
        assert table["dimensionless_columns"] == [4, 5]
    else:
        source = V3["projection_source"]
        projection = table["column_projection"]
        assert projection["source_chunk_id"] == source["chunk_id"]
        assert projection["source_chunk_fingerprint"] == source["chunk_fingerprint"]
        assert rows == [[row[index] for index in projection["source_column_indices"]]
                        for row in _table_rows(source["content_markdown"])]
        assert all(column["issues"] == ["table_units_missing"] for column in projection["excluded_columns"])
        assert "| Q11 |" not in chunk["content_markdown"]

    _write_index(tmp_path / "index.sqlite", [V3["ambiguous"], V3["projection_source"], chunk])
    _write_jsonl(tmp_path / "kg/entities.jsonl", [])
    _write_jsonl(tmp_path / "kg/relations.compact.jsonl.gz", [_relation(chunk)])
    _write_jsonl(tmp_path / "sources.jsonl.gz", [{"source_id": chunk["source_id"]}])
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
    knowledge = DomainKnowledgeBase({"external_knowledge": {"rag_index_path": str(tmp_path / "index.sqlite"),
        "kg_dir": str(tmp_path / "kg"), "source_metadata_path": str(tmp_path / "sources.jsonl.gz"),
        "manifest_path": str(tmp_path / "manifest.json")}})
    monkeypatch.setattr(knowledge, "task_query_text", lambda task: "材料 模量")
    context = knowledge.prepare_context(_task())
    assert context["included_chunk_ids"] == [chunk["chunk_id"]]
    text = "\n\n".join(context["snippets"])
    assert chunk["content_markdown"] in text
    contracts = [json.loads(line.split("证据契约：", 1)[1]) for line in text.splitlines() if "证据契约：" in line]
    assert len(contracts) == 2
    assert all(contract == _evidence_contract(chunk) for contract in contracts)
    assert all(contract["table_evidence"] == chunk["table_evidence"] for contract in contracts)
    assert all(contract["content_quality_version"] == "3" for contract in contracts)
    assert V3["ambiguous"]["chunk_id"] not in text
    knowledge.max_context_chars = context["character_count"] - 1
    excluded = knowledge.prepare_context(_task())
    assert not excluded["snippets"] and excluded["omitted"][0]["reason"] == "context_budget"


@pytest.mark.parametrize("version", ["2", None, 3])
def test_retrieval_rejects_noncurrent_quality_version(tmp_path, version):
    chunk = deepcopy(V3["multilevel"])
    if version is None:
        del chunk["content_quality_version"]
    else:
        chunk["content_quality_version"] = version
    _write_index(tmp_path / "index.sqlite", [chunk])
    if version is None:
        with sqlite3.connect(tmp_path / "index.sqlite") as connection:
            connection.execute("UPDATE chunks SET payload=json_remove(payload, '$.content_quality_version')")
    with pytest.raises(ValueError, match="证据质量版本"):
        RagChunkRetriever(tmp_path / "index.sqlite").search("材料 模量")


@pytest.mark.parametrize("field,value", [("structure_issues", None),
    ("structure_issues", ["table_header_alignment_ambiguous"]), ("missing_unit_columns", [1]),
    ("numeric_columns", [100]), ("column_projection", {"source_column_indices": [0]})])
def test_v3_rejects_incomplete_or_conflicting_table_metadata(field, value):
    chunk = deepcopy(V3["projection_view"])
    if value is None:
        del chunk["table_evidence"][0][field]
    else:
        chunk["table_evidence"][0][field] = value
    with pytest.raises(ValueError, match=field):
        _evidence_contract(chunk)
    assert _evidence_contract(V3["ambiguous"]) is None


@pytest.mark.parametrize("field,value", [("source_chunk_fingerprint", ""), ("source_column_indices", [0]),
                                        ("source_column_indices", [0] * 11)])
def test_v3_projection_requires_source_binding_and_column_correspondence(field, value):
    chunk = deepcopy(V3["projection_view"])
    chunk["table_evidence"][0]["column_projection"][field] = value
    with pytest.raises(ValueError, match="column_projection"):
        _evidence_contract(chunk)


@pytest.mark.parametrize("quality_version", ["3", "2"])
def test_asset_verification_requires_current_evidence_contract(tmp_path, quality_version):
    chunk = deepcopy(V3["multilevel"])
    chunk["content_quality_version"] = quality_version
    _write_index(tmp_path / "rag/retrieval_index.sqlite", [chunk])
    _write_jsonl(tmp_path / "kg/entities.jsonl", [{"type": "Structure", "name": "Stiffened Panel"},
                                                 {"type": "FailureMode", "name": "Buckling"}])
    _write_jsonl(tmp_path / "kg/relations.compact.jsonl.gz", [_relation(chunk)])
    _write_jsonl(tmp_path / "provenance/source_metadata.compact.jsonl.gz", [{"source_id": chunk["source_id"]}])
    (tmp_path / "kg/kg_stats.json").write_text(json.dumps({"total_entities": 2, "total_relations": 1}), encoding="utf-8")
    manifest = {"updated_at": "fixture", "rag_chunk_count": 1, "kg_entity_count": 2,
                "kg_relation_count": 1, "source_metadata_count": 1,
                "files": [{"path": path.relative_to(tmp_path).as_posix(), "bytes": path.stat().st_size,
                           "sha256": sha256_file(path)} for path in tmp_path.rglob("*") if path.is_file()]}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    if quality_version == "3":
        assert verify_assets(tmp_path)["rag_chunk_count"] == 1
    else:
        with pytest.raises(ValueError, match="证据质量版本"):
            verify_assets(tmp_path)
