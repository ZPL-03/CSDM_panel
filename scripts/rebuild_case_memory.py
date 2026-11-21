"""严格读取案例，校验全部输入后更新案例记忆索引，不改写业务 JSON。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.case_memory import CaseMemoryIndex, case_record_metadata, case_record_to_search_text, is_transferable_case  # noqa: E402
from core.paths import CASES_DIR, CASE_LIBRARY_DIR  # noqa: E402
from scripts.case_records import read_case_record  # noqa: E402


def rebuild_case_memory_index() -> dict:
    """先验证来源冲突与正式案例状态，完成嵌入和写入后才删除陈旧索引项。"""
    records, scopes = {}, {}
    for root, scope in ((CASES_DIR, "archive"), (CASE_LIBRARY_DIR, "formal")):
        for path in sorted(root.glob("CASE_*.json")):
            record = read_case_record(path)
            cid = record["case_id"]
            if cid in records and record != records[cid]:
                raise ValueError(f"{path}: 正式案例与评估档案内容冲突")
            if scope == "formal" and not is_transferable_case(record):
                raise ValueError(f"{path}: 正式案例必须具有成功且通过的求解结果")
            records[cid], scopes[cid] = record, scope
    if not records:
        raise ValueError("没有可索引的案例记录，未更新索引")
    ids = sorted(records)
    documents = [case_record_to_search_text(records[cid]) for cid in ids]
    metadata = [case_record_metadata(records[cid], scope=scopes[cid]) for cid in ids]
    index = CaseMemoryIndex()
    # upsert_documents 在调用 Chroma 前计算整批嵌入；嵌入失败不清空现有集合。
    index.engine.upsert_documents(ids=ids, documents=documents, metadatas=metadata)
    stale = sorted(set(index.engine.collection.get(include=[])["ids"]) - set(ids))
    if stale:
        index.engine.collection.delete(ids=stale)
    if index.engine.count() != len(ids):
        raise RuntimeError("案例索引数量与已验证记录不一致")
    return {"indexed_records": len(ids), "archive_records": sum(scope == "archive" for scope in scopes.values()),
            "formal_records": sum(scope == "formal" for scope in scopes.values()), "removed_index_entries": len(stale)}


def main() -> int:
    print(json.dumps(rebuild_case_memory_index(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
