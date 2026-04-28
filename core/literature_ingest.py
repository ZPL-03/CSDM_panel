from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List
from urllib import parse as urllib_parse
from urllib import request as urllib_request

import yaml

from core.config_loader import load_app_config
from core.io_utils import read_json, write_json
from core.paths import LITERATURE_MANIFESTS_DIR, LITERATURE_RAW_DIR, LITERATURE_RECORDS_DIR

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
            "landing_page_url": item.get("primary_location", {}).get("landing_page_url") if isinstance(item.get("primary_location"), dict) else None,
            "license": item.get("primary_location", {}).get("license") if isinstance(item.get("primary_location"), dict) else None,
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "fulltext_status": "oa_url_found" if item.get("open_access", {}).get("oa_url") else "none",
            "fulltext_path": None,
            "chunk_count": 0,
            "provenance": {"query": query, "provider": "openalex"},
        }
        record["paper_id"] = self._paper_id(record)
        return record

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
            "sort": "cited_by_count:desc",
        }
        url = f"{OPENALEX_BASE_URL}?{urllib_parse.urlencode(params)}"
        payload = self._request_json(url)
        raw_name = hashlib.sha1(f"openalex::{query}".encode("utf-8")).hexdigest()[:16]
        write_json(LITERATURE_RAW_DIR / f"openalex_{raw_name}.json", payload)
        results = payload.get("results", []) if isinstance(payload, dict) else []
        normalized = [self._normalize_openalex_record(item, query) for item in results if isinstance(item, dict)]
        return normalized

    def load_query_groups(self, config_path: Path) -> Dict[str, List[str]]:
        with config_path.open("r", encoding="utf-8") as file:
            payload = yaml.safe_load(file) or {}
        query_groups = payload.get("query_groups", {})
        return {
            str(group): [str(item) for item in queries or [] if str(item).strip()]
            for group, queries in query_groups.items()
            if isinstance(queries, list)
        }

    def ingest_queries(self, queries: Iterable[str]) -> Dict[str, int | str | List[str]]:
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
        for record in records:
            write_json(LITERATURE_RECORDS_DIR / f"{record['paper_id']}.json", record)
        chunk_count = self._index_records(records)
        pdf_count = sum(1 for record in records if record.get("fulltext_path"))
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
            "pdf_count": sum(1 for record in existing_records if record.get("fulltext_path")),
            "last_ingested_at": datetime.now(timezone.utc).isoformat(),
            "queries": [],
            "source": "reindex",
        }
        write_json(LITERATURE_MANIFESTS_DIR / "latest_ingest.json", manifest)
        return manifest
