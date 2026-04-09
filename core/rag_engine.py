"""基于 ChromaDB 的本地 RAG 检索。"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List

import chromadb
from chromadb.errors import InvalidArgumentError, NotFoundError
from sentence_transformers import SentenceTransformer

from core.config_loader import load_app_config
from core.paths import CHROMA_DIR


class RAGEngine:
    """负责知识写入与相似案例检索。"""

    def __init__(self, chroma_dir: Path | None = None, embedding_model: str | None = None) -> None:
        app_config = load_app_config()
        rag_config = dict(app_config.get("rag", {}))

        self.chroma_dir = chroma_dir or CHROMA_DIR
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.chroma_dir))
        self.collection_name = "csdm_case_memory"
        self.collection = self.client.get_or_create_collection(name=self.collection_name)

        self.embedding_model_name = embedding_model or str(rag_config.get("embedding_model", "BAAI/bge-m3"))
        self.embedding_cache_dir = Path(str(rag_config.get("embedding_cache_dir", "models/embedding_cache")))
        self.embedding_cache_dir.mkdir(parents=True, exist_ok=True)
        self.local_files_only = bool(rag_config.get("local_files_only", True))
        self.use_hash_embedding_only = os.getenv("CSDM_USE_HASH_EMBEDDING", "0") == "1" or bool(
            rag_config.get("use_hash_embedding_only", False)
        )
        self.allow_hash_fallback = bool(rag_config.get("allow_hash_fallback", True))

        self._embedder: SentenceTransformer | None = None
        self._embedder_failed = False

    @property
    def embedder(self) -> SentenceTransformer:
        if self.use_hash_embedding_only:
            self._embedder_failed = True
            raise RuntimeError("当前配置显式启用哈希嵌入模式")

        if self._embedder is None and not self._embedder_failed:
            try:
                self._embedder = SentenceTransformer(
                    self.embedding_model_name,
                    cache_folder=str(self.embedding_cache_dir),
                    local_files_only=self.local_files_only,
                )
            except Exception:
                self._embedder_failed = True
                raise
        return self._embedder

    def _embedding(self, text: str) -> List[float]:
        if self._embedder_failed:
            return self._hash_embedding(text)

        try:
            vector = self.embedder.encode(text)
            return [float(value) for value in vector]
        except Exception:
            self._embedder_failed = self.allow_hash_fallback
            if not self.allow_hash_fallback:
                raise
            return self._hash_embedding(text)

    def _hash_embedding(self, text: str, dims: int = 32) -> List[float]:
        """在离线测试或显式降级时提供稳定的轻量嵌入。"""
        chunks = []
        seed = text.encode("utf-8", errors="replace")
        for index in range(dims):
            digest = hashlib.sha256(seed + f":{index}".encode("ascii")).digest()
            value = int.from_bytes(digest[:4], byteorder="big", signed=False) / 2**32
            chunks.append(float(value))
        return chunks

    def _to_text(self, record: Dict) -> str:
        return json.dumps(record, ensure_ascii=False, sort_keys=True)

    def _reset_collection(self) -> None:
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(name=self.collection_name)

    def upsert_records(self, records: Iterable[Dict], id_key: str) -> None:
        records = list(records)
        if not records:
            return
        ids = [str(record[id_key]) for record in records]
        docs = [self._to_text(record) for record in records]
        embeddings = [self._embedding(doc) for doc in docs]
        try:
            self.collection.upsert(
                ids=ids,
                documents=docs,
                embeddings=embeddings,
                metadatas=[{"kind": id_key} for _ in records],
            )
        except (InvalidArgumentError, NotFoundError):
            self._reset_collection()
            self.collection.upsert(
                ids=ids,
                documents=docs,
                embeddings=embeddings,
                metadatas=[{"kind": id_key} for _ in records],
            )

    def retrieve(self, query: Dict, top_k: int = 5) -> List[Dict]:
        query_text = self._to_text(query)
        try:
            results = self.collection.query(query_embeddings=[self._embedding(query_text)], n_results=top_k)
        except (InvalidArgumentError, NotFoundError):
            self._reset_collection()
            return []
        documents = results.get("documents", [[]])[0]
        output: List[Dict] = []
        for document in documents:
            try:
                output.append(json.loads(document))
            except json.JSONDecodeError:
                output.append({"raw": document})
        return output
