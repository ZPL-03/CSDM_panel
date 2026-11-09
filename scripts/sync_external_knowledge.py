"""同步并校验 CS_LLM 的完整 RAG/KG 运行资产。"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import sqlite3
import sys
import tempfile
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parent.parent
DESTINATION = ROOT / "knowledge" / "external"
ASSETS = {
    "runtime/retrieval_index.sqlite": "rag/retrieval_index.sqlite",
    "runtime/relations.compact.jsonl.gz": "kg/relations.compact.jsonl.gz",
    "kg/entities.jsonl": "kg/entities.jsonl",
    "kg/kg_stats.json": "kg/kg_stats.json",
    "provenance/source_metadata.compact.jsonl.gz": "provenance/source_metadata.compact.jsonl.gz",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"记录必须是对象：{path}")
                yield row


def verify_assets(base: Path) -> dict:
    """核对全部文件、记录数、图谱节点和 RAG 证据引用。"""
    manifest = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
    expected_files = {item["path"] for item in manifest["files"]} | {"manifest.json"}
    actual_files = {path.relative_to(base).as_posix() for path in base.rglob("*") if path.is_file()}
    if actual_files != expected_files:
        raise ValueError("知识目录的文件与清单不一致")
    for item in manifest["files"]:
        path = base / item["path"]
        if path.stat().st_size != item["bytes"] or sha256_file(path) != item["sha256"]:
            raise ValueError(f"文件校验不通过：{path}")

    with closing(sqlite3.connect((base / "rag/retrieval_index.sqlite").as_uri() + "?mode=ro", uri=True)) as connection:
        if connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise ValueError("RAG SQLite 索引完整性检查失败")
        chunks = dict(connection.execute("SELECT chunk_id, source_id FROM chunks"))
        count = connection.execute("SELECT count(*) FROM chunks_fts").fetchone()[0]
        if count != len(chunks) or len(chunks) != manifest["rag_chunk_count"]:
            raise ValueError("RAG 文本和全文索引记录数不一致")
        missing_fts = connection.execute(
            "SELECT count(*) FROM chunks LEFT JOIN chunks_fts ON chunks.rowid = chunks_fts.rowid WHERE chunks_fts.rowid IS NULL"
        ).fetchone()[0]
        if missing_fts:
            raise ValueError("RAG 存在未索引的文本块")
    entities = list(read_rows(base / "kg/entities.jsonl"))
    nodes = {(row["type"], row["name"]) for row in entities}
    sources = list(read_rows(base / "provenance/source_metadata.compact.jsonl.gz"))
    source_ids = {row["source_id"] for row in sources}
    if len(entities) != manifest["kg_entity_count"] or len(sources) != manifest["source_metadata_count"]:
        raise ValueError("实体或来源记录数不一致")
    if set(chunks.values()) - source_ids:
        raise ValueError("RAG 存在未登记的来源")
    relation_count = 0
    for row in read_rows(base / "kg/relations.compact.jsonl.gz"):
        relation_count += 1
        chunk_id = row.get("evidence_chunk_id")
        if chunk_id not in chunks or chunks[chunk_id] != row.get("evidence_source_id"):
            raise ValueError(f"图谱证据引用无效：{chunk_id}")
        if (row["source_type"], row["source"]) not in nodes or (row["target_type"], row["target"]) not in nodes:
            raise ValueError(f"图谱关系引用未定义实体：{chunk_id}")
    if relation_count != manifest["kg_relation_count"]:
        raise ValueError("知识图谱关系记录数不一致")
    stats = json.loads((base / "kg/kg_stats.json").read_text(encoding="utf-8"))
    if stats["total_entities"] != len(entities) or stats["total_relations"] != relation_count:
        raise ValueError("知识图谱统计不一致")
    return {key: manifest[key] for key in ("updated_at", "rag_chunk_count", "kg_entity_count", "kg_relation_count", "source_metadata_count")}


def synchronize(processed_dir: Path, archive_dir: Path) -> dict:
    if archive_dir.is_relative_to(ROOT):
        raise ValueError("知识资产归档目录必须位于项目外")
    package = processed_dir / "csgpt_runtime" / "csllm"
    upstream = json.loads((package / "provenance/manifest.json").read_text(encoding="utf-8"))
    file_index = {item["path"]: item for item in upstream["files"]}
    counts = upstream["record_counts"]
    with tempfile.TemporaryDirectory(prefix="csdm_panel_knowledge_", dir=ROOT.parent) as temporary:
        stage = Path(temporary) / "external"
        stage.mkdir()
        files = []
        for source_name, target_name in ASSETS.items():
            source = package / source_name
            expected = file_index[source_name]
            if source.stat().st_size != expected["bytes"] or sha256_file(source) != expected["sha256"]:
                raise ValueError(f"CS_LLM 导出文件与清单不一致：{source_name}")
            target = stage / target_name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            files.append({**expected, "path": target_name, "source_path": source_name})

        # 全文索引含完整块内容，原始块文件和导出索引的指纹必须对应。
        with closing(sqlite3.connect((stage / "rag/retrieval_index.sqlite").as_uri() + "?mode=ro", uri=True)) as connection:
            metadata = dict(connection.execute("SELECT key, value FROM metadata"))
        if metadata.get("source_sha256") != sha256_file(processed_dir / "rag/rag_chunks.jsonl"):
            raise ValueError("CS_LLM 全文索引未对应最新 RAG 文本，请先重新导出")
        for source_name, raw_name in (("runtime/retrieval_index.sqlite", "rag/retrieval_index.sqlite"),
                                      ("kg/entities.jsonl", "kg/entities.jsonl"),
                                      ("kg/kg_stats.json", "kg/kg_stats.json")):
            if sha256_file(processed_dir / raw_name) != file_index[source_name]["sha256"]:
                raise ValueError(f"CS_LLM 运行包落后于解析数据：{raw_name}")
        manifest = {
            "schema_version": 1,
            "asset_source": "CS_LLM 完整 RAG/KG 运行资产",
            "updated_at": upstream["generated_at"],
            "installed_at": datetime.now(timezone.utc).isoformat(),
            "retrieval_engine": "SQLite FTS5 BM25",
            "rag_chunk_count": counts["rag_chunks"],
            "kg_entity_count": counts["entities"],
            "kg_relation_count": counts["relations"],
            "source_metadata_count": counts["sources"],
            "files": files,
        }
        (stage / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        verified = verify_assets(stage)
        backup = archive_dir / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8])
        if DESTINATION.exists():
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(DESTINATION), str(backup))
        try:
            shutil.move(str(stage), str(DESTINATION))
        except Exception:
            if backup.exists() and not DESTINATION.exists():
                shutil.move(str(backup), str(DESTINATION))
            raise
    return {**verified, "archive_path": str(backup)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="CS_LLM 的 dataset/processed 目录")
    parser.add_argument("--verify-only", action="store_true", help="仅校验已部署的资产")
    parser.add_argument("--archive-dir", type=Path, default=ROOT.parent.parent / "CSDM_panel_knowledge_archive")
    args = parser.parse_args()
    if args.verify_only:
        result = verify_assets(DESTINATION)
    elif args.source:
        result = synchronize(args.source.resolve(), args.archive_dir.resolve())
    else:
        parser.error("同步需要 --source，已部署资产校验使用 --verify-only")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
