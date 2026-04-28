import json
from pathlib import Path

from core.literature_corpus import LiteratureCorpus
from core.literature_ingest import LiteratureIngestor
from core.paths import LITERATURE_MANIFESTS_DIR, LITERATURE_RAW_DIR, LITERATURE_RECORDS_DIR


class _FakeEngine:
    def __init__(self) -> None:
        self.calls = []
        self.upserts = []
        self.rows = [
            {
                "id": "PAPER_A::chunk::1",
                "document": "Composite buckling and surrogate modeling for stiffened panels.",
                "metadata": {
                    "paper_id": "PAPER_A",
                    "title": "Composite Buckling Study",
                    "source": "openalex",
                    "doi": "10.1000/example-a",
                    "year": 2024,
                },
            }
        ]

    def query_text(self, query_text: str, top_k: int = 5, where=None):
        self.calls.append({"query_text": query_text, "top_k": top_k, "where": where})
        return self.rows[:top_k]

    def upsert_documents(self, ids, documents, metadatas):
        self.upserts.append({"ids": list(ids), "documents": list(documents), "metadatas": list(metadatas)})


class _FakeIngestor(LiteratureIngestor):
    def __init__(self) -> None:
        super().__init__()
        self._engine = _FakeEngine()
        self.requests = []

    def _request_json(self, url: str) -> dict:
        self.requests.append(url)
        return {
            "results": [
                {
                    "id": "https://openalex.org/W123",
                    "doi": "https://doi.org/10.1000/example-a",
                    "title": "Composite Buckling Study",
                    "publication_year": 2024,
                    "cited_by_count": 12,
                    "authorships": [{"author": {"display_name": "Alice"}}],
                    "topics": [{"display_name": "Composite materials"}],
                    "concepts": [{"display_name": "Buckling"}],
                    "primary_location": {
                        "source": {"display_name": "Journal of Composite Structures"},
                        "landing_page_url": "https://example.org/paper-a",
                        "license": "cc-by",
                    },
                    "open_access": {
                        "is_oa": True,
                        "oa_status": "gold",
                        "oa_url": "https://example.org/paper-a.pdf",
                    },
                    "abstract_inverted_index": {
                        "Composite": [0],
                        "buckling": [1],
                        "study": [2],
                    },
                }
            ]
        }


def _task() -> dict:
    return {
        "task_id": "TASK_1",
        "source": "test",
        "task": {
            "application": "复合材料加筋壁板",
            "load_conditions": {"type": "compression_shear", "Nx_kN_per_m": 900.0, "Nxy_kN_per_m": 180.0},
            "boundary_conditions": {"type": "SSCC"},
            "geometry_envelope": {
                "panel_length_mm": [600, 800],
                "panel_width_mm": [500, 700],
                "max_stiffener_height_mm": 50,
            },
            "material_system": {"name": "T300/5208"},
            "layup_constraints": {"allowed_angles": [0, 45, -45, 90], "symmetric": True, "balanced": True, "min_ratio_per_angle": 0.1},
            "design_targets": {"BLF_min": 1.35, "primary_objective": "最小重量"},
            "stiffener_type": "T",
        },
    }


def test_literature_corpus_formats_snippets() -> None:
    corpus = LiteratureCorpus()
    corpus._engine = _FakeEngine()

    snippets = corpus.format_snippets(_task(), top_k=1)

    assert len(snippets) == 1
    assert "Composite Buckling Study" in snippets[0]
    assert "source=openalex" in snippets[0]
    assert "surrogate modeling" in snippets[0]
    assert corpus._engine.calls[0]["top_k"] == 1
    assert "压剪组合" in corpus._engine.calls[0]["query_text"]


def test_literature_corpus_status_prefers_manifest() -> None:
    manifest_path = LITERATURE_MANIFESTS_DIR / "latest_ingest.json"
    original = manifest_path.read_text(encoding="utf-8") if manifest_path.exists() else None
    try:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(
                {
                    "record_count": 7,
                    "chunk_count": 21,
                    "pdf_count": 2,
                    "last_ingested_at": "2026-04-27T12:00:00+00:00",
                    "queries": ["composite materials"],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        status = LiteratureCorpus().status()

        assert status["record_count"] == 7
        assert status["chunk_count"] == 21
        assert status["pdf_count"] == 2
        assert status["last_ingested_at"] == "2026-04-27T12:00:00+00:00"
    finally:
        if original is None:
            manifest_path.unlink(missing_ok=True)
        else:
            manifest_path.write_text(original, encoding="utf-8")


def test_literature_ingestor_ingests_queries_and_writes_manifest() -> None:
    ingestor = _FakeIngestor()

    existing_raw = {path.name for path in LITERATURE_RAW_DIR.glob("*.json")}
    existing_records = {path.name for path in LITERATURE_RECORDS_DIR.glob("*.json")}
    manifest_path = LITERATURE_MANIFESTS_DIR / "latest_ingest.json"
    original_manifest = manifest_path.read_text(encoding="utf-8") if manifest_path.exists() else None

    try:
        summary = ingestor.ingest_queries(["stiffened composite panel buckling"])

        assert summary["record_count"] >= 1
        assert summary["chunk_count"] >= 1
        assert summary["queries"] == ["stiffened composite panel buckling"]
        assert ingestor.requests
        assert ingestor.engine.upserts

        upsert = ingestor.engine.upserts[0]
        assert upsert["ids"][0].endswith("::chunk::1")
        assert "Composite Buckling Study" in upsert["documents"][0]
        assert upsert["metadatas"][0]["corpus"] == "literature"

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["record_count"] >= 1
        assert manifest["chunk_count"] >= 1
    finally:
        for path in LITERATURE_RAW_DIR.glob("*.json"):
            if path.name not in existing_raw:
                path.unlink(missing_ok=True)
        for path in LITERATURE_RECORDS_DIR.glob("*.json"):
            if path.name not in existing_records:
                path.unlink(missing_ok=True)
        if original_manifest is None:
            manifest_path.unlink(missing_ok=True)
        else:
            manifest_path.write_text(original_manifest, encoding="utf-8")
