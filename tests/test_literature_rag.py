import json
import shutil
from pathlib import Path

from core.literature_corpus import LiteratureCorpus
from core.literature_ingest import LiteratureIngestor
from core.paths import (
    LITERATURE_IMPORTED_MARKDOWN_DIR,
    LITERATURE_IMPORTED_PDFS_DIR,
    LITERATURE_IMPORTED_TEXTS_DIR,
    LITERATURE_MARKDOWN_DIR,
    LITERATURE_MANIFESTS_DIR,
    LITERATURE_PDFS_DIR,
    LITERATURE_RAW_DIR,
    LITERATURE_RECORDS_DIR,
    LITERATURE_TEXTS_DIR,
)


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
        self.download_calls = []
        self.parse_calls = []
        self.isolate_existing = True

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
                    "open_access": {
                        "is_oa": True,
                        "oa_status": "gold",
                        "oa_url": "https://example.org/paper-a.pdf",
                    },
                    "primary_location": {
                        "source": {"display_name": "Journal of Composite Structures"},
                        "landing_page_url": "https://example.org/paper-a",
                        "license": "cc-by",
                        "pdf_url": "https://example.org/paper-a.pdf",
                    },
                    "abstract_inverted_index": {
                        "Composite": [0],
                        "buckling": [1],
                        "study": [2],
                    },
                }
            ]
        }

    def _load_existing_records(self) -> dict:
        if self.isolate_existing:
            return {}
        return super()._load_existing_records()

    def _download_pdf(self, record: dict, force: bool = False) -> bool:
        self.download_calls.append({"paper_id": record["paper_id"], "force": force})
        target = LITERATURE_PDFS_DIR / f"{record['paper_id']}.pdf"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"%PDF-1.4\nfake")
        record["fulltext_path"] = str(target)
        record["fulltext_status"] = "pdf_downloaded"
        return True

    def _parse_pdf_for_record(self, record: dict, force: bool = False, backend=None, ocr=None) -> bool:
        self.parse_calls.append({"paper_id": record["paper_id"], "force": force, "backend": backend, "ocr": ocr})
        imported = self._is_imported_record(record)
        text_target = (LITERATURE_IMPORTED_TEXTS_DIR if imported else LITERATURE_TEXTS_DIR) / f"{record['paper_id']}.txt"
        markdown_target = (LITERATURE_IMPORTED_MARKDOWN_DIR if imported else LITERATURE_MARKDOWN_DIR) / f"{record['paper_id']}.md"
        text_target.parent.mkdir(parents=True, exist_ok=True)
        markdown_target.parent.mkdir(parents=True, exist_ok=True)
        markdown = "# Parsed Paper\n\nFull text about compression shear buckling and $\\sigma_{cr}$."
        text_target.write_text(markdown, encoding="utf-8")
        markdown_target.write_text(markdown, encoding="utf-8")
        record["fulltext_text_path"] = str(text_target)
        record["fulltext_markdown_path"] = str(markdown_target)
        record["parse_backend"] = backend or "pymupdf"
        record["fulltext_status"] = f"{record['parse_backend']}_parsed"
        return True


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


def test_literature_ingestor_downloads_parses_and_indexes_fulltext() -> None:
    ingestor = _FakeIngestor()

    existing_raw = {path.name for path in LITERATURE_RAW_DIR.glob("*.json")}
    existing_records = {path.name for path in LITERATURE_RECORDS_DIR.glob("*.json")}
    existing_pdfs = {path.name for path in LITERATURE_PDFS_DIR.glob("*.pdf")}
    existing_texts = {path.name for path in LITERATURE_TEXTS_DIR.glob("*.txt")}
    existing_markdown = {path.name for path in LITERATURE_MARKDOWN_DIR.glob("*.md")}
    manifest_path = LITERATURE_MANIFESTS_DIR / "latest_ingest.json"
    original_manifest = manifest_path.read_text(encoding="utf-8") if manifest_path.exists() else None

    try:
        summary = ingestor.ingest_queries(
            ["stiffened composite panel buckling"],
            download_pdfs=True,
            parse_pdfs=True,
            force_pdfs=True,
        )

        assert summary["pdf_count"] >= 1
        assert ingestor.download_calls
        assert ingestor.parse_calls
        upsert_documents = " ".join(ingestor.engine.upserts[0]["documents"])
        assert "Full text about compression shear buckling" in upsert_documents
    finally:
        for path in LITERATURE_RAW_DIR.glob("*.json"):
            if path.name not in existing_raw:
                path.unlink(missing_ok=True)
        for path in LITERATURE_RECORDS_DIR.glob("*.json"):
            if path.name not in existing_records:
                path.unlink(missing_ok=True)
        for path in LITERATURE_PDFS_DIR.glob("*.pdf"):
            if path.name not in existing_pdfs:
                path.unlink(missing_ok=True)
        for path in LITERATURE_TEXTS_DIR.glob("*.txt"):
            if path.name not in existing_texts:
                path.unlink(missing_ok=True)
        for path in LITERATURE_MARKDOWN_DIR.glob("*.md"):
            if path.name not in existing_markdown:
                path.unlink(missing_ok=True)
        if original_manifest is None:
            manifest_path.unlink(missing_ok=True)
        else:
            manifest_path.write_text(original_manifest, encoding="utf-8")


def test_literature_ingestor_imports_authorized_local_pdfs(tmp_path: Path) -> None:
    ingestor = _FakeIngestor()
    ingestor.isolate_existing = False
    source_dir = tmp_path / "authorized_pdfs"
    source_dir.mkdir()
    (source_dir / "composite-panel-study.pdf").write_bytes(b"%PDF-1.4\nfake")

    existing_records = {path.name for path in LITERATURE_RECORDS_DIR.glob("*.json")}
    existing_imported_pdfs = {path.name for path in LITERATURE_IMPORTED_PDFS_DIR.glob("*.pdf")}
    existing_imported_texts = {path.name for path in LITERATURE_IMPORTED_TEXTS_DIR.glob("*.txt")}
    existing_imported_markdown = {path.name for path in LITERATURE_IMPORTED_MARKDOWN_DIR.glob("*.md")}
    manifest_path = LITERATURE_MANIFESTS_DIR / "latest_ingest.json"
    original_manifest = manifest_path.read_text(encoding="utf-8") if manifest_path.exists() else None

    try:
        summary = ingestor.import_pdf_directory(source_dir, parse_pdfs=False)

        assert summary["imported"] == 1
        assert summary["record_count"] >= 1
        assert ingestor.engine.upserts
        assert any("composite-panel-study" in document for upsert in ingestor.engine.upserts for document in upsert["documents"])
        assert any(path.name.startswith("LOCAL_") for path in LITERATURE_IMPORTED_PDFS_DIR.glob("*.pdf"))
    finally:
        for path in LITERATURE_RECORDS_DIR.glob("*.json"):
            if path.name not in existing_records:
                path.unlink(missing_ok=True)
        for path in LITERATURE_IMPORTED_PDFS_DIR.glob("*.pdf"):
            if path.name not in existing_imported_pdfs:
                path.unlink(missing_ok=True)
        for path in LITERATURE_IMPORTED_TEXTS_DIR.glob("*.txt"):
            if path.name not in existing_imported_texts:
                path.unlink(missing_ok=True)
        for path in LITERATURE_IMPORTED_MARKDOWN_DIR.glob("*.md"):
            if path.name not in existing_imported_markdown:
                path.unlink(missing_ok=True)
        if original_manifest is None:
            manifest_path.unlink(missing_ok=True)
        else:
            manifest_path.write_text(original_manifest, encoding="utf-8")


def test_literature_ingestor_mineru_backend_writes_markdown_json_and_images(tmp_path: Path, monkeypatch) -> None:
    ingestor = LiteratureIngestor()
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\nfake")
    record = {
        "paper_id": "PAPER_MINERU_TEST",
        "source": "openalex",
        "title": "MinerU Parsed Paper",
        "abstract": "",
        "fulltext_path": str(pdf_path),
        "provenance": {"provider": "openalex"},
    }

    existing_records = {path.name for path in LITERATURE_RECORDS_DIR.glob("*.json")}
    existing_markdown = {path.name for path in LITERATURE_MARKDOWN_DIR.glob("*.md")}
    existing_texts = {path.name for path in LITERATURE_TEXTS_DIR.glob("*.txt")}
    manifest_path = LITERATURE_MANIFESTS_DIR / "latest_ingest.json"
    original_manifest = manifest_path.read_text(encoding="utf-8") if manifest_path.exists() else None

    def fake_run(args):
        output_dir = Path(args[args.index("-o") + 1])
        nested = output_dir / "paper"
        nested.mkdir(parents=True)
        (nested / "paper.md").write_text("Critical load $\\sigma_{cr}=\\pi^2D/b^2$.\n![mode](images/mode.png)", encoding="utf-8")
        (nested / "paper.json").write_text(json.dumps({"formulas": [{"latex": "\\sigma_{cr}"}]}), encoding="utf-8")
        image_dir = nested / "images"
        image_dir.mkdir()
        (image_dir / "mode.png").write_bytes(b"png")
        return type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    try:
        monkeypatch.setattr("core.literature_ingest.shutil.which", lambda name: "mineru.exe" if name == "mineru" else None)
        monkeypatch.setattr(ingestor, "_run_subprocess", fake_run)

        summary = ingestor.parse_pdfs([record], force=True, backend="mineru", ocr=True)

        assert summary["parsed"] == 1
        assert record["parse_backend"] == "mineru"
        assert record["parse_ocr"] is True
        assert Path(record["fulltext_markdown_path"]).exists()
        assert Path(record["fulltext_json_path"]).exists()
        assert Path(record["fulltext_image_dir"]).exists()
        assert "\\sigma_{cr}" in Path(record["fulltext_markdown_path"]).read_text(encoding="utf-8")
    finally:
        for path in LITERATURE_RECORDS_DIR.glob("*.json"):
            if path.name not in existing_records:
                path.unlink(missing_ok=True)
        for path in LITERATURE_MARKDOWN_DIR.glob("*.md"):
            if path.name not in existing_markdown:
                path.unlink(missing_ok=True)
        for path in LITERATURE_TEXTS_DIR.glob("*.txt"):
            if path.name not in existing_texts:
                path.unlink(missing_ok=True)
        if record.get("fulltext_json_path"):
            Path(record["fulltext_json_path"]).unlink(missing_ok=True)
        if record.get("fulltext_image_dir"):
            shutil.rmtree(record["fulltext_image_dir"], ignore_errors=True)
        if original_manifest is None:
            manifest_path.unlink(missing_ok=True)
        else:
            manifest_path.write_text(original_manifest, encoding="utf-8")
