"""外部知识库/知识图谱检索。"""

from __future__ import annotations

import json
import gzip
import re
import sqlite3
import unicodedata
from collections import Counter
from contextlib import closing
from pathlib import Path
from typing import Any, Dict, Iterable, List
from urllib.parse import unquote, urlsplit, urlunsplit

from core.config_loader import load_app_config
from core.paths import ROOT_DIR
from core.task_contract import describe_boundary_conditions, describe_load_conditions, task_payload_from_request


DOMAIN_TERMS = [
    "stiffened composite panel",
    "stiffened panel",
    "stiffened shell",
    "skin stringer",
    "skin-stringer",
    "stringer",
    "hat stiffener",
    "blade stiffener",
    "t stiffener",
    "l stiffener",
    "composite panel",
    "laminate",
    "laminated composite",
    "buckling",
    "postbuckling",
    "linear buckling",
    "eigenvalue buckling",
    "axial compression",
    "in plane shear",
    "compression shear",
    "finite element",
    "abaqus",
    "surrogate model",
    "carbon fiber",
    "cfrp",
    "加筋壁板",
    "加筋壳",
    "筋条",
    "长桁",
    "蒙皮",
    "帽型筋",
    "板式筋",
    "T型筋",
    "L型筋",
    "层合板",
    "铺层",
    "屈曲",
    "后屈曲",
    "线性屈曲",
    "轴压",
    "压缩",
    "剪切",
    "压剪",
    "有限元",
    "代理模型",
    "碳纤维",
    "复合材料",
]

QUERY_SYNONYMS = {
    "加筋壁板": ["stiffened composite panel", "skin stringer panel", "stiffened panel"],
    "加筋壳": ["stiffened shell", "stiffened composite shell"],
    "长桁": ["stringer", "skin-stringer"],
    "筋条": ["stiffener", "stringer"],
    "帽型": ["hat stiffener", "hat stringer"],
    "板式": ["blade stiffener"],
    "T型": ["t stiffener"],
    "L型": ["l stiffener"],
    "屈曲": ["buckling", "linear buckling", "postbuckling"],
    "压缩": ["axial compression", "compression"],
    "轴压": ["axial compression"],
    "剪切": ["in plane shear", "shear"],
    "压剪": ["compression shear"],
    "有限元": ["finite element", "fea", "abaqus"],
    "铺层": ["layup", "laminate", "stacking sequence"],
    "碳纤维": ["carbon fiber", "cfrp"],
    "复合材料": ["composite", "composite material"],
    "stiffened panel": ["stiffened composite panel", "skin stringer"],
    "buckling": ["linear buckling", "postbuckling", "critical load"],
    "compression": ["axial compression", "compressive load"],
    "shear": ["in plane shear", "shear load"],
}

CN_TO_ENTITY = {
    "碳纤维": "Carbon Fiber",
    "玻璃纤维": "Glass Fiber",
    "环氧树脂": "Epoxy Resin",
    "复合材料": "Composite Material",
    "CFRP": "CFRP",
    "层合板": "Laminate",
    "铺层": "Layup",
    "屈曲": "Buckling",
    "后屈曲": "Buckling",
    "疲劳": "Fatigue",
    "分层": "Delamination",
    "基体开裂": "Matrix Cracking",
    "纤维断裂": "Fiber Breakage",
    "有限元": "Finite Element Verification",
    "Abaqus": "Abaqus Verification",
    "代理模型": "Surrogate Model",
    "拉伸强度": "Tensile Strength",
    "压缩强度": "Compressive Strength",
    "剪切强度": "Shear Strength",
    "弹性模量": "Elastic Modulus",
    "剪切模量": "Shear Modulus",
    "密度": "Density",
    "加筋壁板": "Stiffened Panel",
    "加筋壳": "Stiffened Shell",
    "长桁": "Stringer",
    "筋条": "Stiffener",
}

PREFERRED_CATEGORIES = {
    "stiffened_panel_shell_structure": 3.0,
    "general_composite_structure": 2.0,
    "buckling_stability": 2.0,
    "laminate_layup_optimization": 1.5,
    "strength_stiffness": 1.5,
    "simulation_modeling": 1.2,
    "material_properties": 1.0,
    "failure_damage": 1.0,
    "data_ai_methods": 0.8,
}

STIFFENER_SEARCH_TERMS = {
    "HAT": ("帽型", "帽形", "hat stiffener", "hat stringer", "hat-stiffener", "hat-stringer",
            "hat-shaped stiffened", "hat-shaped stiffener", "hat-stiffened"),
    "BLADE": ("板式筋", "刀型筋", "blade stiffener", "blade stringer", "blade-stiffener", "blade-stiffened"),
    "T": ("T型筋", "T形筋", "T型加筋", "T形加筋", "T stiffener", "T stringer", "T-stiffener", "T-stringer",
          "T-shaped stiffened", "T-shaped stiffener"),
    "L": ("L型筋", "L形筋", "L型加筋", "L形加筋", "L stiffener", "L stringer", "L-stiffener", "L-stringer",
          "L-shaped stiffened", "L-shaped stiffener"),
}


def _as_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return ROOT_DIR / path


def _jsonl_rows(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"知识记录必须是对象：{path}")
            yield payload


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"知识清单必须是对象：{path}")
    return payload


def tokenize(text: Any) -> list[str]:
    """按 CS_LLM 索引规则提取英文词及中文二至四字片段。"""
    normalized = unicodedata.normalize("NFKC", str(text or "")).lower()
    tokens = []
    for match in re.finditer(r"[a-z0-9]+(?:[-_/][a-z0-9]+)*|[\u3400-\u4dbf\u4e00-\u9fff]+", normalized):
        value = match.group(0)
        if re.fullmatch(r"[\u3400-\u4dbf\u4e00-\u9fff]+", value):
            if len(value) == 1:
                tokens.append(value)
            for size in range(2, min(4, len(value)) + 1):
                tokens.extend(value[index:index + size] for index in range(len(value) - size + 1))
        else:
            tokens.append(value)
    return tokens


def expand_query_terms(query: str) -> list[str]:
    """按复合材料加筋壁板语境扩展检索词。"""
    query_lower = query.lower()
    terms = set(tokenize(query))
    for term in DOMAIN_TERMS:
        term_lower = term.lower()
        if term_lower in query_lower:
            terms.add(term_lower)
    for key, values in QUERY_SYNONYMS.items():
        if key.lower() in query_lower:
            terms.update(value.lower() for value in values)
    return sorted(terms, key=lambda item: (-len(item), item))


def trim_text(text: Any, limit: int) -> str:
    """按字符数裁剪检索片段。"""
    cleaned = str(text or "").strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit].rstrip() + "..."


def _list_values(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [item.strip() for item in value.split(";") if item.strip()]
    return []


def _normalize_source_key(*values: Any) -> str:
    for value in values:
        text = re.sub(r"\s+", " ", str(value or "")).strip().lower()
        if text:
            return text
    return ""


def _source_aliases(item: dict[str, Any], evidence: bool = False) -> set[str]:
    prefix = "evidence_" if evidence else ""
    aliases = set()
    doi = str(item.get(prefix + "doi") or "").strip().lower()
    doi = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", doi)
    if doi:
        aliases.add("doi:" + unquote(doi).rstrip("/"))
    for field in (prefix + "source_id", "record_id"):
        value = _normalize_source_key(item.get(field))
        if value:
            aliases.add("id:" + value)
    url = str(item.get(prefix + "source_url") or "").strip()
    if url:
        parts = urlsplit(url)
        aliases.add("url:" + urlunsplit(("", parts.netloc.lower(), parts.path.rstrip("/"), parts.query, "")))
    if not aliases:
        title = item.get("evidence_document_title") if evidence else item.get("document_title") or item.get("title")
        if title:
            aliases.add("title:" + _normalize_source_key(title))
    return aliases


def _source_groups(entries: list[tuple[dict[str, Any], bool]]) -> list[int]:
    """合并具有共同 DOI、来源编号或 URL 的证据，保留首个出现位置。"""
    parents = list(range(len(entries)))
    aliases_to_index: dict[str, int] = {}

    def root(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for index, (item, evidence) in enumerate(entries):
        for alias in _source_aliases(item, evidence):
            if alias in aliases_to_index:
                first, second = root(index), root(aliases_to_index[alias])
                parents[max(first, second)] = min(first, second)
            aliases_to_index[alias] = index
    return [root(index) for index in range(len(entries))]


def _source_line(label: str, source: dict[str, Any]) -> str:
    title = str(source.get("title") or source.get("record_id") or label).strip()
    parts = [f"[{label}] {title}"]
    pub = " ".join(str(value).strip() for value in [source.get("year"), source.get("venue")] if str(value or "").strip())
    if pub:
        parts.append(pub)
    if source.get("doi"):
        parts.append(f"DOI: {source.get('doi')}")
    if source.get("source_url"):
        parts.append(str(source.get("source_url")))
    return "；".join(parts)


class RagChunkRetriever:
    """读取 CS_LLM 完整文本索引，以 FTS5 BM25 召回证据。"""

    def __init__(self, index_path: Path, max_snippet_chars: int = 1200) -> None:
        self.index_path = index_path
        self.max_snippet_chars = max_snippet_chars

    @property
    def is_ready(self) -> bool:
        return self.index_path.is_file()

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        terms = sorted({token for term in expand_query_terms(query) for token in tokenize(term) if len(token) >= 2})
        if not terms or top_k <= 0:
            return []
        expression = " OR ".join('"' + term.replace('"', '""') + '"' for term in terms)
        expressions = [(expression, 1.0)]
        for stype, names in STIFFENER_SEARCH_TERMS.items():
            if re.search(r"筋型：" + stype + r"\b", query):
                focused = ['"' + " ".join(tokenize(name)) + '"*' for name in names]
                material = '"composite" OR "composites" OR "cfrp" OR "复合"'
                expressions.append(("(" + " OR ".join(focused) + ") AND (" + material + ")", 2.0))
                break
        candidates: dict[str, dict[str, Any]] = {}
        with closing(sqlite3.connect(self.index_path.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
            for expression, weight in expressions:
                rows = connection.execute(
                    "SELECT chunks.payload, bm25(chunks_fts, 3.0, 2.0, 1.0) AS rank "
                    "FROM chunks_fts JOIN chunks ON chunks.rowid = chunks_fts.rowid "
                    "WHERE chunks_fts MATCH ? ORDER BY rank, chunks.rowid LIMIT ?",
                    (expression, max(100, top_k * 20)),
                ).fetchall()
                for position, (payload, _) in enumerate(rows, 1):
                    chunk = json.loads(payload)
                    if str(chunk.get("retrieval_scope") or "main") not in {"main", "mixed"}:
                        continue
                    item = candidates.setdefault(chunk["chunk_id"], {**chunk, "score": 0.0})
                    item["score"] += weight / (60 + position)
        results = []
        for chunk in candidates.values():
            body = "\n".join(line for line in str(chunk.get("content_plain") or "").splitlines() if not line.startswith("#"))
            if chunk.get("retrieval_unit") == "image" and len(tokenize(body)) < 12:
                continue
            categories = _list_values(chunk.get("task_categories")) + [str(chunk.get("primary_category") or "")]
            bonus = min(sum(PREFERRED_CATEGORIES.get(category, 0.0) for category in categories), 5.0)
            chunk["score"] *= 1.0 + bonus * 0.05
            if chunk.get("retrieval_unit") == "image":
                chunk["score"] *= 0.5
            content = chunk.get("content_markdown") or chunk.get("content_plain")
            chunk["text"] = str(content or "").strip() if chunk.get("retrieval_unit") == "formula" else trim_text(content, self.max_snippet_chars)
            chunk["source"] = "KNOWLEDGE_BASE"
            results.append(chunk)
        results.sort(key=lambda item: item["score"], reverse=True)
        seen_content: set[str] = set()
        source_counts: Counter[int] = Counter()
        selected = []
        for item, source in zip(results, _source_groups([(item, False) for item in results])):
            fingerprint = str(item.get("content_hash") or item.get("chunk_fingerprint") or item["text"])
            if fingerprint in seen_content or source_counts[source] >= 2:
                continue
            seen_content.add(fingerprint)
            source_counts[source] += 1
            selected.append(item)
            if len(selected) >= top_k:
                break
        return selected


class KnowledgeGraphRetriever:
    """基于知识图谱 JSONL 的文件型关系检索器。"""

    def __init__(self, kg_dir: Path) -> None:
        self.kg_dir = kg_dir
        self.entities: list[dict[str, Any]] = []
        self.relations: list[dict[str, Any]] = []
        self.entity_names: list[str] = []
        self._loaded = False

    @property
    def is_ready(self) -> bool:
        return (self.kg_dir / "entities.jsonl").is_file() and (self.kg_dir / "relations.compact.jsonl.gz").is_file()

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self.entities = list(_jsonl_rows(self.kg_dir / "entities.jsonl"))
        self.relations = list(_jsonl_rows(self.kg_dir / "relations.compact.jsonl.gz"))
        self.entity_names = [str(item.get("name") or "") for item in self.entities if str(item.get("name") or "")]
        self._loaded = True

    def _matched_entities(self, query: str) -> set[str]:
        query_lower = query.lower()
        expanded_terms = expand_query_terms(query)
        mapped_names = {
            target
            for keyword, target in CN_TO_ENTITY.items()
            if keyword.lower() in query_lower or keyword in query
        }
        matches: set[str] = set()
        for name in self.entity_names:
            lowered = name.lower()
            if lowered in query_lower:
                matches.add(name)
                continue
            if name in mapped_names:
                matches.add(name)
                continue
            if any(term and (term in lowered or lowered in term) for term in expanded_terms):
                matches.add(name)
        return matches

    def search(self, query: str, record_ids: Iterable[str], top_k: int = 8, chunk_ids: Iterable[str] = ()) -> list[dict[str, Any]]:
        if top_k <= 0:
            return []
        self._ensure_loaded()
        if not self.relations:
            return []

        matched_entities = self._matched_entities(query)
        record_set = {str(record_id) for record_id in record_ids if str(record_id).strip()}
        chunk_set = set(chunk_ids)
        expanded_terms = [term for term in expand_query_terms(query) if len(term) >= 2]
        best_relations: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}

        for relation in self.relations:
            source = str(relation.get("source") or "")
            target = str(relation.get("target") or "")
            record_id = str(relation.get("record_id") or "")
            relation_type = str(relation.get("relation") or "")
            score = 0.0
            if source in matched_entities:
                score += 2.0
            if target in matched_entities:
                score += 2.0
            if record_id in record_set:
                score += 2.0
            if relation.get("evidence_chunk_id") in chunk_set:
                score += 6.0
            relation_text = " ".join(
                [
                    source,
                    target,
                    relation_type,
                    str(relation.get("source_type") or ""),
                    str(relation.get("target_type") or ""),
                    str(relation.get("evidence_document_title") or ""),
                ]
            ).lower()
            for term in expanded_terms:
                if term in relation_text:
                    score += 1.0
            if score <= 0:
                continue

            key = (str(relation.get("source_type") or ""), source, relation_type,
                   str(relation.get("target_type") or ""), target)
            item = dict(relation)
            item["score"] = round(score, 4)
            item["retrieval_source"] = "KNOWLEDGE_GRAPH"
            if key not in best_relations or item["score"] > best_relations[key]["score"]:
                best_relations[key] = item

        results = list(best_relations.values())
        results.sort(key=lambda item: (-float(item.get("score") or 0.0), str(item.get("record_id") or "")))
        return results[:top_k]


class DomainKnowledgeBase:
    """面向 CSDM_panel 的外部知识库/知识图谱运行时入口。"""

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        app_config = config or load_app_config()
        knowledge_config = dict(app_config.get("external_knowledge", {}))
        self.enabled = bool(knowledge_config.get("enabled", True))
        self.top_k = int(knowledge_config.get("top_k", 5) or 5)
        self.kg_top_k = int(knowledge_config.get("kg_top_k", 8) or 8)
        self.manifest_path = _as_path(str(knowledge_config.get("manifest_path", "knowledge/external/manifest.json")))
        rag_path = _as_path(str(knowledge_config.get("rag_index_path", "knowledge/external/rag/retrieval_index.sqlite")))
        kg_dir = _as_path(str(knowledge_config.get("kg_dir", "knowledge/external/kg")))
        max_snippet_chars = int(knowledge_config.get("max_snippet_chars", 1200) or 1200)
        self.rag = RagChunkRetriever(rag_path, max_snippet_chars=max_snippet_chars)
        self.kg = KnowledgeGraphRetriever(kg_dir)
        self.source_metadata_path = _as_path(str(knowledge_config.get("source_metadata_path", "knowledge/external/provenance/source_metadata.compact.jsonl.gz")))
        self._source_metadata: dict[str, dict[str, Any]] | None = None

    @property
    def is_ready(self) -> bool:
        return self.enabled and self.rag.is_ready and self.kg.is_ready and self.manifest_path.is_file() and self.source_metadata_path.is_file()

    def task_query_text(self, task: Dict[str, Any]) -> str:
        task_payload = task_payload_from_request(task)
        target = task_payload.get("design_targets", {})
        material = task_payload.get("material_system", {})
        envelope = task_payload.get("geometry_envelope", {})
        return "\n".join(
            [
                "复合材料加筋壁板 stiffened composite panel buckling",
                f"应用：{task_payload.get('application', '复合材料加筋壁板')}",
                f"工况：{describe_load_conditions(task_payload.get('load_conditions', {}))}",
                f"边界：{describe_boundary_conditions(task_payload.get('boundary_conditions', {}))}",
                f"筋型：{task_payload.get('stiffener_type', 'T')} {dict(T='T stiffener', HAT='hat stiffener', BLADE='blade stiffener', L='L stiffener').get(task_payload.get('stiffener_type', 'T'), '')}",
                f"材料：{material.get('name', '')}",
                f"几何：L={envelope.get('panel_length_mm')}, W={envelope.get('panel_width_mm')}, h_max={envelope.get('max_stiffener_height_mm')}",
                f"目标：BLF >= {target.get('BLF_min', 1.2)}，{target.get('primary_objective', '最小重量')}",
            ]
        )

    def retrieve(self, task: Dict[str, Any], top_k: int | None = None, kg_top_k: int | None = None) -> dict[str, Any]:
        if not self.enabled:
            return {"query": "", "chunks": [], "relations": []}
        if not self.is_ready:
            raise FileNotFoundError("外部知识资产不完整，请运行 scripts/sync_external_knowledge.py 校验并同步")
        query = self.task_query_text(task)
        chunks = self.rag.search(query, top_k=self.top_k if top_k is None else top_k)
        record_ids = [str(item.get("record_id") or "") for item in chunks]
        relations = self.kg.search(query, record_ids, top_k=self.kg_top_k if kg_top_k is None else kg_top_k,
                                   chunk_ids=[item["chunk_id"] for item in chunks])
        if self._source_metadata is None:
            source_metadata = {}
            for source in _jsonl_rows(self.source_metadata_path):
                for field in ("source_id", "record_id"):
                    if source.get(field):
                        source_metadata[str(source[field])] = source
            self._source_metadata = source_metadata
        for item, evidence in [(chunk, False) for chunk in chunks] + [(relation, True) for relation in relations]:
            prefix = "evidence_" if evidence else ""
            source = self._source_metadata.get(str(item.get(prefix + "source_id") or "")) or self._source_metadata.get(str(item.get("record_id") or "")) or {}
            for field in ("doi", "source_url", "document_title", "year", "venue"):
                target_field = prefix + field if evidence and field not in {"year", "venue"} else field
                if not item.get(target_field) and source.get(field):
                    item[target_field] = source[field]
        return {"query": query, "chunks": chunks, "relations": relations}

    def format_snippets(self, task: Dict[str, Any], top_k: int | None = None) -> list[str]:
        payload = self.retrieve(task, top_k=self.top_k if top_k is None else top_k)
        formatted: list[str] = []
        entries = [(item, False) for item in payload["chunks"]] + [(item, True) for item in payload["relations"]]
        groups = _source_groups(entries)
        labels_by_group: dict[int, str] = {}
        labels_by_item: dict[int, str] = {}
        source_items: dict[str, dict[str, Any]] = {}
        for (item, evidence), group in zip(entries, groups):
            title = item.get("evidence_document_title") if evidence else item.get("document_title") or item.get("title")
            source_url = item.get("evidence_source_url") if evidence else item.get("source_url")
            doi = item.get("evidence_doi") if evidence else item.get("doi")
            label = labels_by_group.setdefault(group, f"S{len(labels_by_group) + 1}")
            labels_by_item[id(item)] = label
            source = source_items.setdefault(label, {})
            values = {
                "title": title,
                "year": item.get("year"),
                "venue": item.get("venue"),
                "doi": doi,
                "source_url": source_url,
                "record_id": item.get("record_id"),
            }
            for field, value in values.items():
                if value and not source.get(field):
                    source[field] = value

        for index, item in enumerate(payload.get("chunks", []), start=1):
            title = item.get("title") or item.get("record_id") or f"资料片段{index}"
            source_label = labels_by_item[id(item)]
            meta_lines = [f"[外部知识库 {index}{f' | 来源 {source_label}' if source_label else ''}] {title}"]
            meta_lines.append(f"证据块：{item['chunk_id']}；类型：{item.get('retrieval_unit', 'text')}")
            if item.get("section_path"):
                meta_lines.append("章节：" + " > ".join(item["section_path"]))
            if item.get("page_start") not in (None, ""):
                page_end = item.get("page_end")
                page_text = str(item.get("page_start")) if page_end in (None, "", item.get("page_start")) else f"{item.get('page_start')}-{page_end}"
                meta_lines.append(f"页码：{page_text}")
            formatted.append("\n".join([*meta_lines, str(item.get("text") or "")]))

        relations = payload.get("relations", [])
        if relations:
            relation_lines = []
            for relation in relations:
                evidence_label = labels_by_item[id(relation)]
                evidence_text = f"（依据 {evidence_label}）" if evidence_label else ""
                if relation.get("evidence_chunk_id"):
                    evidence_text += f"；证据块：{relation['evidence_chunk_id']}"
                relation_lines.append(
                    "图谱 {idx}: {source}({source_type}) -[{rel}]-> {target}({target_type}){evidence}".format(
                        idx=len(relation_lines) + 1,
                        source=relation.get("source"),
                        source_type=relation.get("source_type"),
                        rel=relation.get("relation"),
                        target=relation.get("target"),
                        target_type=relation.get("target_type"),
                        evidence=evidence_text,
                    )
                )
            formatted.append("知识图谱关系：\n" + "\n".join(relation_lines))
        if source_items:
            formatted.append("来源清单：\n" + "\n".join(_source_line(label, source_items[label]) for label in sorted(source_items, key=lambda item: int(item[1:]))))
        return formatted

    def status(self) -> dict[str, Any]:
        manifest = _load_json(self.manifest_path)
        return {
            "enabled": self.enabled,
            "ready": self.is_ready,
            "manifest_path": str(self.manifest_path),
            "rag_index_path": str(self.rag.index_path),
            "kg_dir": str(self.kg.kg_dir),
            "rag_chunk_count": int(manifest.get("rag_chunk_count", 0) or 0),
            "kg_entity_count": int(manifest.get("kg_entity_count", 0) or 0),
            "kg_relation_count": int(manifest.get("kg_relation_count", 0) or 0),
            "provenance_dir": str(self.source_metadata_path.parent),
            "source_metadata_path": str(self.source_metadata_path),
            "source_metadata_count": int(manifest.get("source_metadata_count", 0) or 0),
            "updated_at": manifest.get("updated_at"),
            "retrieval_engine": "SQLite FTS5 BM25",
        }
