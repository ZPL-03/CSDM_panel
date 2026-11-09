import numpy as np
import pytest

from core.rag_engine import RAGEngine


def test_embedding_failure_is_not_replaced_by_hash_vector(tmp_path) -> None:
    engine = RAGEngine(chroma_dir=tmp_path)

    class BrokenEmbedder:
        def encode(self, text, **kwargs):
            raise RuntimeError("嵌入服务错误")

    engine._embedder = BrokenEmbedder()
    with pytest.raises(RuntimeError, match="嵌入服务错误"):
        engine._embedding("复合材料")
    assert engine.count() == 0


def test_query_error_preserves_existing_collection(tmp_path) -> None:
    engine = RAGEngine(chroma_dir=tmp_path)

    class Collection:
        def count(self):
            return 1

        def query(self, **kwargs):
            raise RuntimeError("索引维度不匹配")

    class Embedder:
        def encode(self, text, **kwargs):
            return np.array([0.0, 1.0])

    engine.collection = Collection()
    engine._embedder = Embedder()
    with pytest.raises(RuntimeError, match="索引维度不匹配"):
        engine.query_text("复合材料")
    assert isinstance(engine.collection, Collection)


def test_real_bge_embedding_and_persistent_retrieval(tmp_path) -> None:
    engine = RAGEngine(chroma_dir=tmp_path)
    vector = engine._embedding("复合材料加筋壁板的压缩屈曲")
    assert len(vector) == 1024
    assert np.linalg.norm(vector) == pytest.approx(1.0, abs=1e-5)
    engine.upsert_documents(
        ["panel", "food"],
        ["复合材料加筋壁板的压缩屈曲与有限元计算", "面包烘焙与餐饮配方"],
        [{"kind": "case"}, {"kind": "case"}],
    )
    assert engine.query_text("加筋壁板压缩屈曲", top_k=1)[0]["id"] == "panel"
    assert RAGEngine(chroma_dir=tmp_path).count() == 2
    engine.upsert_documents(["extra"], ["帽形加筋壁板"])
    assert engine.count() == 3


def test_document_count_mismatch_is_explicit(tmp_path) -> None:
    engine = RAGEngine(chroma_dir=tmp_path)
    with pytest.raises(ValueError, match="数量不一致"):
        engine.upsert_documents(["case"], [])
