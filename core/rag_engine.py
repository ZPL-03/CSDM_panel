"""基于 ChromaDB 和 BGE 的案例文本向量检索。"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List

from core.config_loader import load_app_config
from core.paths import CHROMA_DIR


@lru_cache(maxsize=1)
def _load_embedder(model_name: str, local_files_only: bool):
    from huggingface_hub import snapshot_download
    from sentence_transformers import SentenceTransformer

    model_path = Path(model_name)
    if not model_path.is_dir():
        model_path = Path(snapshot_download(model_name, local_files_only=local_files_only))
    return SentenceTransformer(str(model_path))


class RAGEngine:
    """持久化案例摘要，以实际文本嵌入辅助相似案例排序。"""

    def __init__(
        self,
        chroma_dir: Path | None = None,
        embedding_model: str | None = None,
        collection_name: str = "csdm_case_memory",
    ) -> None:
        import chromadb

        rag_config = dict(load_app_config().get("rag", {}))
        self.chroma_dir = chroma_dir or CHROMA_DIR
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=str(self.chroma_dir))
        self.collection = self.client.get_or_create_collection(name=self.collection_name)
        self.embedding_model_name = embedding_model or str(rag_config.get("embedding_model", "BAAI/bge-m3"))
        self.local_files_only = bool(rag_config.get("local_files_only", True))
        self._embedder = None

    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = _load_embedder(self.embedding_model_name, self.local_files_only)
        return self._embedder

    def _embedding(self, text: str) -> List[float]:
        vector = self.embedder.encode(text, normalize_embeddings=True)
        return [float(value) for value in vector]

    def _to_text(self, record: Dict) -> str:
        return json.dumps(record, ensure_ascii=False, sort_keys=True)

    def reset_collection(self) -> None:
        """仅在显式重建时重置当前集合，不删除其他集合。"""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(name=self.collection_name)

    def count(self) -> int:
        return int(self.collection.count())

    def upsert_documents(self, ids: Iterable[str], documents: Iterable[str], metadatas: Iterable[Dict] | None = None) -> None:
        doc_ids = [str(item) for item in ids]
        docs = [str(item) for item in documents]
        if len(doc_ids) != len(docs):
            raise ValueError("ids 和 documents 数量不一致")
        if not doc_ids:
            return
        metadata_list = list(metadatas) if metadatas is not None else None
        if metadata_list is not None and len(metadata_list) != len(docs):
            raise ValueError("metadatas 和 documents 数量不一致")
        embeddings = self.embedder.encode(docs, normalize_embeddings=True).tolist()
        self.collection.upsert(ids=doc_ids, documents=docs, embeddings=embeddings, metadatas=metadata_list)

    def upsert_records(self, records: Iterable[Dict], id_key: str) -> None:
        records = list(records)
        if not records:
            return
        self.upsert_documents(
            ids=[str(record[id_key]) for record in records],
            documents=[self._to_text(record) for record in records],
            metadatas=[{"kind": id_key} for _ in records],
        )

    def query_text(self, query_text: str, top_k: int = 5, where: Dict | None = None) -> List[Dict[str, object]]:
        count = self.count()
        if count == 0 or top_k <= 0:
            return []
        results = self.collection.query(
            query_embeddings=[self._embedding(query_text)],
            n_results=min(top_k, count),
            where=where,
        )
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]
        return [
            {"id": ids[index], "document": document, "metadata": metadatas[index], "distance": distances[index]}
            for index, document in enumerate(documents)
        ]

    def retrieve(self, query: Dict, top_k: int = 5) -> List[Dict]:
        results = self.query_text(self._to_text(query), top_k=top_k)
        return [json.loads(str(item["document"])) for item in results]
