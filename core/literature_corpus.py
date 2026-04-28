from __future__ import annotations

from typing import Dict, List

from core.config_loader import load_app_config
from core.paths import LITERATURE_MANIFESTS_DIR, LITERATURE_RECORDS_DIR
from core.task_contract import describe_boundary_conditions, describe_load_conditions, task_payload_from_request


class LiteratureCorpus:
    """运行时文献检索包装层，只服务于 LLM 候选生成路径。"""

    def __init__(self) -> None:
        config = load_app_config()
        literature_config = dict(config.get("literature", {}))
        collection_name = str(literature_config.get("collection_name", "csdm_literature_corpus"))
        self.top_k = int(literature_config.get("top_k", 5) or 5)
        self.collection_name = collection_name
        self._engine = None
        self._engine_error = False

    @property
    def engine(self):
        if self._engine is not None:
            return self._engine
        if self._engine_error:
            return None
        try:
            from core.rag_engine import RAGEngine

            self._engine = RAGEngine(collection_name=self.collection_name)
        except Exception:
            self._engine_error = True
            return None
        return self._engine

    def _task_query_text(self, task: Dict) -> str:
        """把结构化任务转成文献检索文本，突出工况、边界、筋型和目标。"""
        task_payload = task_payload_from_request(task)
        target = task_payload.get("design_targets", {})
        material = task_payload.get("material_system", {})
        parts = [
            str(task_payload.get("application") or "复合材料加筋壁板"),
            f"工况：{describe_load_conditions(task_payload.get('load_conditions', {}))}",
            f"边界：{describe_boundary_conditions(task_payload.get('boundary_conditions', {}))}",
            f"筋型：{task_payload.get('stiffener_type', 'T')}",
            f"目标：BLF >= {target.get('BLF_min', 1.2)}，{target.get('primary_objective', '最小重量')}",
        ]
        material_name = str(material.get("name") or "").strip()
        if material_name:
            parts.append(f"材料体系：{material_name}")
        return "\n".join(parts)

    def retrieve_snippets(self, task: Dict, top_k: int | None = None) -> List[Dict[str, str]]:
        engine = self.engine
        if engine is None:
            return []
        query_text = self._task_query_text(task)
        matches = engine.query_text(query_text, top_k=top_k or self.top_k)
        snippets: List[Dict[str, str]] = []
        for item in matches:
            metadata = item.get("metadata") or {}
            if not isinstance(metadata, dict):
                metadata = {}
            snippets.append(
                {
                    "paper_id": str(metadata.get("paper_id") or item.get("id") or ""),
                    "title": str(metadata.get("title") or ""),
                    "source": str(metadata.get("source") or ""),
                    "doi": str(metadata.get("doi") or ""),
                    "snippet": str(item.get("document") or ""),
                }
            )
        return snippets

    def format_snippets(self, task: Dict, top_k: int | None = None) -> List[str]:
        formatted: List[str] = []
        for index, item in enumerate(self.retrieve_snippets(task, top_k=top_k), start=1):
            title = item.get("title") or item.get("paper_id") or f"文献{index}"
            source = item.get("source") or "unknown"
            snippet = item.get("snippet") or ""
            formatted.append(f"[{index}] {title} | source={source}\n{snippet}")
        return formatted

    def status(self) -> Dict[str, int | str | None | list[str]]:
        """返回知识库页需要展示的文献库状态摘要。"""
        manifest_path = LITERATURE_MANIFESTS_DIR / "latest_ingest.json"
        if manifest_path.exists():
            import json

            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            queries = payload.get("queries") if isinstance(payload.get("queries"), list) else []
            return {
                "record_count": int(payload.get("record_count", 0) or 0),
                "chunk_count": int(payload.get("chunk_count", 0) or 0),
                "pdf_count": int(payload.get("pdf_count", 0) or 0),
                "last_ingested_at": payload.get("last_ingested_at"),
                "queries": [str(item) for item in queries if str(item).strip()],
                "source": payload.get("source") or "-",
            }
        return {
            "record_count": len(list(LITERATURE_RECORDS_DIR.glob("*.json"))),
            "chunk_count": 0,
            "pdf_count": 0,
            "last_ingested_at": None,
            "queries": [],
            "source": "-",
        }
