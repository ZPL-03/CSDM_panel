from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List
from urllib import parse as urllib_parse
from urllib import request as urllib_request

import yaml

from core.config_loader import load_app_config
from core.io_utils import read_json, write_json
from core.paths import (
    LITERATURE_IMAGES_DIR,
    LITERATURE_IMPORTED_IMAGES_DIR,
    LITERATURE_IMPORTED_JSON_DIR,
    LITERATURE_IMPORTED_MARKDOWN_DIR,
    LITERATURE_IMPORTED_PDFS_DIR,
    LITERATURE_IMPORTED_TEXTS_DIR,
    LITERATURE_JSON_DIR,
    LITERATURE_MANIFESTS_DIR,
    LITERATURE_MARKDOWN_DIR,
    LITERATURE_PDFS_DIR,
    LITERATURE_RAW_DIR,
    LITERATURE_RECORDS_DIR,
    LITERATURE_TEXTS_DIR,
)

OPENALEX_BASE_URL = "https://api.openalex.org/works"


class LiteratureIngestor:
    """负责文献抓取、标准化、切块和向量索引构建。"""

    def __init__(self) -> None:
        app_config = load_app_config()
        literature_config = dict(app_config.get("literature", {}))
        self.collection_name = str(literature_config.get("collection_name", "csdm_literature_corpus"))
        self.max_results_per_query = int(literature_config.get("max_results_per_query", 50) or 50)
        self.chunk_chars = int(literature_config.get("chunk_chars", 1200) or 1200)
        self.chunk_overlap = int(literature_config.get("chunk_overlap", 150) or 150)
        self.download_oa_pdfs = bool(literature_config.get("download_oa_pdfs", False))
        self.parse_oa_pdfs = bool(literature_config.get("parse_oa_pdfs", False))
        self.sort_by_citations = bool(literature_config.get("sort_by_citations", False))
        self.pdf_parse_backend = str(literature_config.get("pdf_parse_backend", "pymupdf") or "pymupdf").lower()
        self.pdf_parse_ocr = bool(literature_config.get("pdf_parse_ocr", False))
        self._engine = None

    @property
    def engine(self):
        if self._engine is None:
            from core.rag_engine import RAGEngine

            self._engine = RAGEngine(collection_name=self.collection_name)
        return self._engine

    def _request_json(self, url: str) -> Dict:
        request = urllib_request.Request(url, headers={"User-Agent": "CSDM-LiteratureIngest/1.0"})
        with urllib_request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))

    def _reconstruct_abstract(self, inverted_index: Dict[str, List[int]] | None) -> str:
        if not isinstance(inverted_index, dict):
            return ""
        positions: Dict[int, str] = {}
        for word, indexes in inverted_index.items():
            if not isinstance(indexes, list):
                continue
            for index in indexes:
                try:
                    positions[int(index)] = str(word)
                except (TypeError, ValueError):
                    continue
        return " ".join(positions[index] for index in sorted(positions))

    def _paper_id(self, record: Dict) -> str:
        source_ids = record.get("source_ids", {})
        doi = str(source_ids.get("doi") or "").strip().lower()
        if doi:
            digest = hashlib.sha1(doi.encode("utf-8")).hexdigest()[:16]
            return f"DOI_{digest}"
        openalex_id = str(source_ids.get("openalex_id") or "").strip()
        if openalex_id:
            return openalex_id.rsplit("/", 1)[-1]
        title = str(record.get("title") or "untitled")
        digest = hashlib.sha1(title.encode("utf-8")).hexdigest()[:16]
        return f"PAPER_{digest}"

    def _is_imported_record(self, record: Dict) -> bool:
        provenance = record.get("provenance", {})
        provider = provenance.get("provider") if isinstance(provenance, dict) else None
        return str(record.get("source") or "") == "local_pdf" or str(provider or "") == "local_pdf"

    def _pdf_path_for(self, paper_id: str, imported: bool = False) -> Path:
        base_dir = LITERATURE_IMPORTED_PDFS_DIR if imported else LITERATURE_PDFS_DIR
        return base_dir / f"{paper_id}.pdf"

    def _text_path_for(self, paper_id: str, imported: bool = False) -> Path:
        base_dir = LITERATURE_IMPORTED_TEXTS_DIR if imported else LITERATURE_TEXTS_DIR
        return base_dir / f"{paper_id}.txt"

    def _markdown_path_for(self, paper_id: str, imported: bool = False) -> Path:
        base_dir = LITERATURE_IMPORTED_MARKDOWN_DIR if imported else LITERATURE_MARKDOWN_DIR
        return base_dir / f"{paper_id}.md"

    def _json_path_for(self, paper_id: str, imported: bool = False) -> Path:
        base_dir = LITERATURE_IMPORTED_JSON_DIR if imported else LITERATURE_JSON_DIR
        return base_dir / f"{paper_id}.json"

    def _images_dir_for(self, paper_id: str, imported: bool = False) -> Path:
        base_dir = LITERATURE_IMPORTED_IMAGES_DIR if imported else LITERATURE_IMAGES_DIR
        return base_dir / paper_id

    def _record_file_exists(self, record: Dict, field: str) -> bool:
        path_value = str(record.get(field) or "").strip()
        return bool(path_value) and Path(path_value).exists()

    def _oa_pdf_url(self, item: Dict) -> str:
        primary_location = item.get("primary_location", {}) if isinstance(item.get("primary_location"), dict) else {}
        for candidate in [
            primary_location.get("pdf_url"),
            item.get("open_access", {}).get("oa_url") if isinstance(item.get("open_access"), dict) else None,
        ]:
            url = str(candidate or "").strip()
            if url:
                return url
        return ""

    def _normalize_openalex_record(self, item: Dict, query: str) -> Dict:
        """把 OpenAlex 返回统一整理成项目内部文献记录格式。"""
        authors = []
        for authorship in item.get("authorships", []) or []:
            author = authorship.get("author", {}) if isinstance(authorship, dict) else {}
            if isinstance(author, dict) and author.get("display_name"):
                authors.append(str(author.get("display_name")))
        topics = [str(topic.get("display_name")) for topic in item.get("topics", []) if isinstance(topic, dict) and topic.get("display_name")]
        concepts = [str(concept.get("display_name")) for concept in item.get("concepts", []) if isinstance(concept, dict) and concept.get("display_name")]
        source = item.get("primary_location", {}).get("source", {}) if isinstance(item.get("primary_location"), dict) else {}
        doi_url = str(item.get("doi") or "")
        doi = doi_url.replace("https://doi.org/", "").replace("http://doi.org/", "") if doi_url else ""
        record = {
            "paper_id": "",
            "source": "openalex",
            "source_ids": {
                "openalex_id": item.get("id"),
                "doi": doi,
            },
            "title": str(item.get("title") or ""),
            "abstract": self._reconstruct_abstract(item.get("abstract_inverted_index")),
            "authors": authors,
            "year": item.get("publication_year"),
            "venue": source.get("display_name") if isinstance(source, dict) else None,
            "publisher": None,
            "keywords": [],
            "topics": topics,
            "concepts": concepts,
            "material_domain_tags": topics[:5] or concepts[:5],
            "citation_count": int(item.get("cited_by_count", 0) or 0),
            "is_open_access": bool(item.get("open_access", {}).get("is_oa", False)) if isinstance(item.get("open_access"), dict) else False,
            "oa_status": item.get("open_access", {}).get("oa_status") if isinstance(item.get("open_access"), dict) else None,
            "oa_url": item.get("open_access", {}).get("oa_url") if isinstance(item.get("open_access"), dict) else None,
            "oa_pdf_url": self._oa_pdf_url(item),
            "landing_page_url": item.get("primary_location", {}).get("landing_page_url") if isinstance(item.get("primary_location"), dict) else None,
            "license": item.get("primary_location", {}).get("license") if isinstance(item.get("primary_location"), dict) else None,
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "fulltext_status": "oa_pdf_url_found" if self._oa_pdf_url(item) else "none",
            "fulltext_path": None,
            "fulltext_text_path": None,
            "fulltext_markdown_path": None,
            "fulltext_json_path": None,
            "fulltext_image_dir": None,
            "parse_backend": None,
            "parse_ocr": False,
            "chunk_count": 0,
            "provenance": {"query": query, "provider": "openalex"},
        }
        record["paper_id"] = self._paper_id(record)
        return record

    def _download_pdf(self, record: Dict, force: bool = False) -> bool:
        url = str(record.get("oa_pdf_url") or record.get("oa_url") or "").strip()
        if not url:
            return False
        parsed = urllib_parse.urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            record["fulltext_status"] = "pdf_download_skipped_invalid_url"
            return False

        target = self._pdf_path_for(str(record["paper_id"]))
        if target.exists() and not force:
            record["fulltext_path"] = str(target)
            record["fulltext_status"] = "pdf_downloaded"
            return True

        request = urllib_request.Request(
            url,
            headers={
                "User-Agent": "CSDM-LiteratureIngest/1.0",
                "Accept": "application/pdf,*/*;q=0.8",
            },
        )
        try:
            with urllib_request.urlopen(request, timeout=90) as response:
                content_type = str(response.headers.get("Content-Type", "")).lower()
                payload = response.read()
        except Exception as exc:
            record["fulltext_status"] = f"pdf_download_failed:{type(exc).__name__}"
            return False

        if b"%PDF" not in payload[:1024] and "pdf" not in content_type:
            record["fulltext_status"] = "pdf_download_skipped_non_pdf"
            return False

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        record["fulltext_path"] = str(target)
        record["fulltext_status"] = "pdf_downloaded"
        return True

    def _extract_pdf_text(self, pdf_path: Path) -> str:
        try:
            import fitz  # type: ignore

            chunks = []
            with fitz.open(str(pdf_path)) as document:
                for page in document:
                    chunks.append(page.get_text("text"))
            return "\n".join(chunks)
        except Exception:
            pass

        try:
            from pypdf import PdfReader  # type: ignore

            reader = PdfReader(str(pdf_path))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception:
            return ""

    def _markdown_from_text(self, record: Dict, text: str) -> str:
        title = str(record.get("title") or record.get("paper_id") or "Untitled").strip()
        parts = [f"# {title}", ""]
        abstract = str(record.get("abstract") or "").strip()
        if abstract:
            parts.extend(["## Abstract", "", abstract, ""])
        parts.extend(["## Full Text", "", text.strip()])
        return "\n".join(parts).strip() + "\n"

    def _first_file(self, root: Path, suffixes: set[str]) -> Path | None:
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() in suffixes:
                return path
        return None

    def _run_subprocess(self, args: List[str]) -> subprocess.CompletedProcess:
        return subprocess.run(args, capture_output=True, text=True, timeout=1800)

    def _write_parse_outputs(
        self,
        record: Dict,
        backend: str,
        markdown: str,
        payload: Dict,
        images_source_dir: Path | None = None,
    ) -> bool:
        imported = self._is_imported_record(record)
        paper_id = str(record["paper_id"])
        markdown_path = self._markdown_path_for(paper_id, imported=imported)
        json_path = self._json_path_for(paper_id, imported=imported)
        text_path = self._text_path_for(paper_id, imported=imported)
        images_dir = self._images_dir_for(paper_id, imported=imported)

        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        text_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(markdown, encoding="utf-8")
        text_path.write_text(markdown, encoding="utf-8")
        write_json(json_path, payload)

        copied_images = 0
        if images_source_dir and images_source_dir.exists():
            images_dir.mkdir(parents=True, exist_ok=True)
            for source in images_source_dir.rglob("*"):
                if source.is_file() and source.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
                    target = images_dir / source.name
                    if target.exists():
                        target = images_dir / f"{source.stem}_{copied_images}{source.suffix}"
                    shutil.copy2(source, target)
                    copied_images += 1

        record["fulltext_text_path"] = str(text_path)
        record["fulltext_markdown_path"] = str(markdown_path)
        record["fulltext_json_path"] = str(json_path)
        record["fulltext_image_dir"] = str(images_dir) if copied_images else None
        record["parse_backend"] = backend
        record["parse_ocr"] = bool(payload.get("ocr", False))
        record["parsed_image_count"] = copied_images
        record["fulltext_status"] = f"{backend}_parsed"
        return True

    def _parse_pdf_with_pymupdf(self, record: Dict, pdf_path: Path) -> bool:
        text = self._clean_fulltext(self._extract_pdf_text(pdf_path))
        if not text:
            record["fulltext_status"] = "pymupdf_parse_failed"
            return False
        markdown = self._markdown_from_text(record, text)
        payload = {
            "paper_id": record["paper_id"],
            "backend": "pymupdf",
            "ocr": False,
            "supports_latex": False,
            "supports_images": False,
            "text": text,
        }
        return self._write_parse_outputs(record, "pymupdf", markdown, payload)

    def _parse_pdf_with_mineru(self, record: Dict, pdf_path: Path, ocr: bool = False) -> bool:
        if shutil.which("mineru") is None:
            record["fulltext_status"] = "mineru_unavailable"
            return False
        with tempfile.TemporaryDirectory(prefix="csdm_mineru_") as tmp:
            output_dir = Path(tmp) / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            result = self._run_subprocess(["mineru", "-p", str(pdf_path), "-o", str(output_dir), "-f", "true" if ocr else "false"])
            if result.returncode != 0:
                record["fulltext_status"] = "mineru_parse_failed"
                record["parse_error"] = (result.stderr or result.stdout or "").strip()[-2000:]
                return False
            markdown_path = self._first_file(output_dir, {".md"})
            if markdown_path is None:
                record["fulltext_status"] = "mineru_no_markdown"
                return False
            markdown = markdown_path.read_text(encoding="utf-8", errors="ignore")
            json_path = self._first_file(output_dir, {".json"})
            structured = None
            if json_path is not None:
                try:
                    structured = json.loads(json_path.read_text(encoding="utf-8"))
                except Exception:
                    structured = {"raw_json_path": str(json_path)}
            payload = {
                "paper_id": record["paper_id"],
                "backend": "mineru",
                "ocr": bool(ocr),
                "supports_latex": True,
                "supports_images": True,
                "source_markdown": str(markdown_path),
                "source_json": str(json_path) if json_path else None,
                "structured": structured,
            }
            return self._write_parse_outputs(record, "mineru", markdown, payload, images_source_dir=output_dir)

    def _parse_pdf_with_nougat(self, record: Dict, pdf_path: Path) -> bool:
        if shutil.which("nougat") is None:
            record["fulltext_status"] = "nougat_unavailable"
            return False
        with tempfile.TemporaryDirectory(prefix="csdm_nougat_") as tmp:
            output_dir = Path(tmp) / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            result = self._run_subprocess(["nougat", str(pdf_path), "-o", str(output_dir), "--markdown"])
            if result.returncode != 0:
                record["fulltext_status"] = "nougat_parse_failed"
                record["parse_error"] = (result.stderr or result.stdout or "").strip()[-2000:]
                return False
            markdown_path = self._first_file(output_dir, {".md", ".mmd"})
            if markdown_path is None:
                record["fulltext_status"] = "nougat_no_markdown"
                return False
            markdown = markdown_path.read_text(encoding="utf-8", errors="ignore")
            payload = {
                "paper_id": record["paper_id"],
                "backend": "nougat",
                "ocr": True,
                "supports_latex": True,
                "supports_images": False,
                "source_markdown": str(markdown_path),
                "markdown": markdown,
            }
            return self._write_parse_outputs(record, "nougat", markdown, payload)

    def _clean_fulltext(self, text: str) -> str:
        content = str(text or "")
        content = re.sub(r"\r\n?", "\n", content)
        content = re.sub(r"[ \t]+", " ", content)
        content = re.sub(r"\n{3,}", "\n\n", content)
        content = re.sub(r"\n\s*(References|Bibliography|参考文献)\s*\n.*", "", content, flags=re.IGNORECASE | re.DOTALL)
        return content.strip()

    def _parse_pdf_for_record(self, record: Dict, force: bool = False) -> bool:
        pdf_path_value = str(record.get("fulltext_path") or "").strip()
        if not pdf_path_value:
            return False
        pdf_path = Path(pdf_path_value)
        if not pdf_path.exists():
            return False

        target = self._text_path_for(str(record["paper_id"]), imported=self._is_imported_record(record))
        if target.exists() and not force:
            record["fulltext_text_path"] = str(target)
            if record.get("fulltext_status") == "pdf_downloaded":
                record["fulltext_status"] = "pdf_parsed"
            return True

        text = self._clean_fulltext(self._extract_pdf_text(pdf_path))
        if not text:
            record["fulltext_status"] = "pdf_parse_failed"
            return False

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        record["fulltext_text_path"] = str(target)
        record["fulltext_status"] = "pdf_parsed"
        return True

    def _clean_fulltext(self, text: str) -> str:
        content = str(text or "")
        content = re.sub(r"\r\n?", "\n", content)
        content = re.sub(r"[ \t]+", " ", content)
        content = re.sub(r"\n{3,}", "\n\n", content)
        content = re.sub(r"\n\s*(References|Bibliography|参考文献)\s*\n.*", "", content, flags=re.IGNORECASE | re.DOTALL)
        return content.strip()

    def _parse_pdf_for_record(self, record: Dict, force: bool = False, backend: str | None = None, ocr: bool | None = None) -> bool:
        pdf_path_value = str(record.get("fulltext_path") or "").strip()
        if not pdf_path_value:
            return False
        pdf_path = Path(pdf_path_value)
        if not pdf_path.exists():
            return False

        selected_backend = str(backend or self.pdf_parse_backend or "pymupdf").lower()
        selected_ocr = self.pdf_parse_ocr if ocr is None else bool(ocr)
        markdown_target = self._markdown_path_for(str(record["paper_id"]), imported=self._is_imported_record(record))
        if markdown_target.exists() and not force:
            record["fulltext_markdown_path"] = str(markdown_target)
            text_target = self._text_path_for(str(record["paper_id"]), imported=self._is_imported_record(record))
            if text_target.exists():
                record["fulltext_text_path"] = str(text_target)
            if record.get("fulltext_status") == "pdf_downloaded":
                record["fulltext_status"] = f"{selected_backend}_parsed"
            return True

        if selected_backend == "mineru":
            return self._parse_pdf_with_mineru(record, pdf_path, ocr=selected_ocr)
        if selected_backend == "nougat":
            return self._parse_pdf_with_nougat(record, pdf_path)
        if selected_backend in {"pymupdf", "text"}:
            return self._parse_pdf_with_pymupdf(record, pdf_path)

        record["fulltext_status"] = f"unsupported_parser:{selected_backend}"
        return False

    def _chunk_text(self, text: str) -> List[str]:
        content = str(text or "").strip()
        if not content:
            return []
        if len(content) <= self.chunk_chars:
            return [content]
        chunks: List[str] = []
        step = max(self.chunk_chars - self.chunk_overlap, 1)
        for start in range(0, len(content), step):
            chunk = content[start : start + self.chunk_chars].strip()
            if chunk:
                chunks.append(chunk)
            if start + self.chunk_chars >= len(content):
                break
        return chunks

    def _document_texts(self, record: Dict) -> List[str]:
        title = str(record.get("title") or "").strip()
        abstract = str(record.get("abstract") or "").strip()
        fulltext = ""
        markdown_path_value = str(record.get("fulltext_markdown_path") or "").strip()
        markdown_path = Path(markdown_path_value) if markdown_path_value else None
        if markdown_path is not None and markdown_path.exists() and markdown_path.is_file():
            fulltext = markdown_path.read_text(encoding="utf-8", errors="ignore").strip()
        text_path_value = str(record.get("fulltext_text_path") or "").strip()
        text_path = Path(text_path_value) if text_path_value else None
        if not fulltext and text_path is not None and text_path.exists() and text_path.is_file():
            fulltext = text_path.read_text(encoding="utf-8", errors="ignore").strip()
        if fulltext:
            base = "\n\n".join(part for part in [title, abstract, fulltext] if part)
        else:
            base = title if not abstract else f"{title}\n\n{abstract}"
        return self._chunk_text(base)

    def _load_existing_records(self) -> Dict[str, Dict]:
        existing: Dict[str, Dict] = {}
        for path in sorted(LITERATURE_RECORDS_DIR.glob("*.json")):
            try:
                record = read_json(path)
            except Exception:
                continue
            if isinstance(record, dict) and record.get("paper_id"):
                existing[str(record["paper_id"])] = record
        return existing

    def _index_records(self, records: Iterable[Dict]) -> int:
        """将文献记录切块后写入向量库，并返回累计 chunk 数。"""
        ids: List[str] = []
        docs: List[str] = []
        metadatas: List[Dict] = []
        chunk_total = 0
        for record in records:
            chunks = self._document_texts(record)
            record["chunk_count"] = len(chunks)
            for index, chunk in enumerate(chunks, start=1):
                chunk_total += 1
                ids.append(f"{record['paper_id']}::chunk::{index}")
                docs.append(chunk)
                metadatas.append(
                    {
                        "corpus": "literature",
                        "paper_id": record["paper_id"],
                        "title": str(record.get("title") or ""),
                        "year": int(record.get("year", 0) or 0),
                        "venue": str(record.get("venue") or ""),
                        "doi": str(record.get("source_ids", {}).get("doi") or ""),
                        "source": str(record.get("source") or ""),
                        "is_open_access": bool(record.get("is_open_access", False)),
                        "section_type": "title_abstract",
                    }
                )
        if ids:
            self.engine.upsert_documents(ids, docs, metadatas)
        return chunk_total

    def search_openalex(self, query: str, per_page: int | None = None) -> List[Dict]:
        params = {
            "search": query,
            "per-page": min(per_page or self.max_results_per_query, self.max_results_per_query),
        }
        if self.sort_by_citations:
            params["sort"] = "cited_by_count:desc"
        url = f"{OPENALEX_BASE_URL}?{urllib_parse.urlencode(params)}"
        payload = self._request_json(url)
        raw_name = hashlib.sha1(f"openalex::{query}".encode("utf-8")).hexdigest()[:16]
        write_json(LITERATURE_RAW_DIR / f"openalex_{raw_name}.json", payload)
        results = payload.get("results", []) if isinstance(payload, dict) else []
        normalized = [self._normalize_openalex_record(item, query) for item in results if isinstance(item, dict)]
        return normalized

    def download_open_access_pdfs(self, records: Iterable[Dict] | None = None, force: bool = False) -> Dict[str, int]:
        selected_records = list(records) if records is not None else list(self._load_existing_records().values())
        downloaded = 0
        failed = 0
        skipped = 0
        for record in selected_records:
            if self._download_pdf(record, force=force):
                downloaded += 1
            elif record.get("oa_pdf_url") or record.get("oa_url"):
                failed += 1
            else:
                skipped += 1
            write_json(LITERATURE_RECORDS_DIR / f"{record['paper_id']}.json", record)
        return {"record_count": len(selected_records), "downloaded": downloaded, "failed": failed, "skipped": skipped}

    def parse_pdfs(
        self,
        records: Iterable[Dict] | None = None,
        force: bool = False,
        backend: str | None = None,
        ocr: bool | None = None,
    ) -> Dict[str, int]:
        selected_records = list(records) if records is not None else list(self._load_existing_records().values())
        parsed = 0
        failed = 0
        skipped = 0
        for record in selected_records:
            if self._parse_pdf_for_record(record, force=force, backend=backend, ocr=ocr):
                parsed += 1
            elif record.get("fulltext_path"):
                failed += 1
            else:
                skipped += 1
            write_json(LITERATURE_RECORDS_DIR / f"{record['paper_id']}.json", record)
        return {"record_count": len(selected_records), "parsed": parsed, "failed": failed, "skipped": skipped}

    def import_pdf_directory(
        self,
        pdf_dir: Path,
        parse_pdfs: bool = True,
        force: bool = False,
        backend: str | None = None,
        ocr: bool | None = None,
    ) -> Dict[str, int]:
        source_dir = Path(pdf_dir)
        if not source_dir.exists():
            raise FileNotFoundError(f"PDF directory not found: {source_dir}")

        existing = self._load_existing_records()
        imported = 0
        parsed = 0
        for source_path in sorted(source_dir.glob("*.pdf")):
            digest = hashlib.sha1(source_path.read_bytes()).hexdigest()[:16]
            paper_id = f"LOCAL_{digest}"
            target = self._pdf_path_for(paper_id, imported=True)
            if force or not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, target)
            record = existing.get(
                paper_id,
                {
                    "paper_id": paper_id,
                    "source": "local_pdf",
                    "source_ids": {"local_path": str(source_path)},
                    "title": source_path.stem,
                    "abstract": "",
                    "authors": [],
                    "year": None,
                    "venue": None,
                    "publisher": None,
                    "keywords": [],
                    "topics": [],
                    "concepts": [],
                    "material_domain_tags": [],
                    "citation_count": 0,
                    "is_open_access": False,
                    "oa_status": None,
                    "oa_url": None,
                    "oa_pdf_url": None,
                    "landing_page_url": None,
                    "license": None,
                    "retrieved_at": datetime.now(timezone.utc).isoformat(),
                    "fulltext_status": "local_pdf_imported",
                    "fulltext_path": str(target),
                    "fulltext_text_path": None,
                    "fulltext_markdown_path": None,
                    "fulltext_json_path": None,
                    "fulltext_image_dir": None,
                    "parse_backend": None,
                    "parse_ocr": False,
                    "chunk_count": 0,
                    "provenance": {"provider": "local_pdf", "source_path": str(source_path)},
                },
            )
            record["fulltext_path"] = str(target)
            record["fulltext_status"] = "local_pdf_imported"
            if parse_pdfs and self._parse_pdf_for_record(record, force=force, backend=backend, ocr=ocr):
                parsed += 1
            write_json(LITERATURE_RECORDS_DIR / f"{paper_id}.json", record)
            imported += 1

        records = list(self._load_existing_records().values())
        chunk_count = self._index_records(records)
        manifest = {
            "record_count": len(records),
            "chunk_count": chunk_count,
            "pdf_count": sum(1 for record in records if self._record_file_exists(record, "fulltext_path")),
            "last_ingested_at": datetime.now(timezone.utc).isoformat(),
            "queries": [],
            "source": "local_pdf_import",
            "imported_pdf_count": imported,
            "parsed_pdf_count": parsed,
        }
        write_json(LITERATURE_MANIFESTS_DIR / "latest_ingest.json", manifest)
        return {"imported": imported, "parsed": parsed, "record_count": len(records), "chunk_count": chunk_count}

    def load_query_groups(self, config_path: Path) -> Dict[str, List[str]]:
        with config_path.open("r", encoding="utf-8") as file:
            payload = yaml.safe_load(file) or {}
        query_groups = payload.get("query_groups", {})
        return {
            str(group): [str(item) for item in queries or [] if str(item).strip()]
            for group, queries in query_groups.items()
            if isinstance(queries, list)
        }

    def ingest_queries(
        self,
        queries: Iterable[str],
        download_pdfs: bool | None = None,
        parse_pdfs: bool | None = None,
        force_pdfs: bool = False,
        parse_backend: str | None = None,
        parse_ocr: bool | None = None,
    ) -> Dict[str, int | str | List[str]]:
        existing = self._load_existing_records()
        updated_records: Dict[str, Dict] = dict(existing)
        processed_queries: List[str] = []
        for query in queries:
            clean_query = str(query).strip()
            if not clean_query:
                continue
            processed_queries.append(clean_query)
            for record in self.search_openalex(clean_query):
                updated_records[record["paper_id"]] = record

        records = list(updated_records.values())
        should_download = self.download_oa_pdfs if download_pdfs is None else download_pdfs
        should_parse = self.parse_oa_pdfs if parse_pdfs is None else parse_pdfs
        if should_download:
            self.download_open_access_pdfs(records, force=force_pdfs)
        if should_parse:
            self.parse_pdfs(records, force=force_pdfs, backend=parse_backend, ocr=parse_ocr)
        for record in records:
            write_json(LITERATURE_RECORDS_DIR / f"{record['paper_id']}.json", record)
        chunk_count = self._index_records(records)
        pdf_count = sum(1 for record in records if self._record_file_exists(record, "fulltext_path"))
        manifest = {
            "record_count": len(records),
            "chunk_count": chunk_count,
            "pdf_count": pdf_count,
            "last_ingested_at": datetime.now(timezone.utc).isoformat(),
            "queries": processed_queries,
            "source": "openalex",
        }
        write_json(LITERATURE_MANIFESTS_DIR / "latest_ingest.json", manifest)
        return manifest

    def reindex_from_records(self) -> Dict[str, int | str]:
        records = self._load_existing_records().values()
        chunk_count = self._index_records(records)
        existing_records = list(self._load_existing_records().values())
        manifest = {
            "record_count": len(existing_records),
            "chunk_count": chunk_count,
            "pdf_count": sum(1 for record in existing_records if self._record_file_exists(record, "fulltext_path")),
            "last_ingested_at": datetime.now(timezone.utc).isoformat(),
            "queries": [],
            "source": "reindex",
        }
        write_json(LITERATURE_MANIFESTS_DIR / "latest_ingest.json", manifest)
        return manifest
