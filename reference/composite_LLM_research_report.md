# CSDM_panel 知识资产参考说明

本文件记录当前项目采用的知识资产边界，作为 `README.md`、`docs/接口约定.md` 和全流程开发指南的补充参考。

## 当前知识资产

CSDM_panel 当前保留两类知识能力：

- 外部知识库/知识图谱：位于 `knowledge/external/`，服务于 LLM 候选生成。
- 案例库与案例记忆：位于 `data/cases/`、`knowledge/case_library/` 和 `knowledge/chroma_db/`，服务于案例归档、案例迁移和相似案例排序。

## 外部知识库/知识图谱

外部知识库/知识图谱资产包括：

- `knowledge/external/rag/rag_chunks.jsonl`
- `knowledge/external/rag/rag_chunks_index.csv`
- `knowledge/external/kg/entities.jsonl`
- `knowledge/external/kg/relations.jsonl`
- `knowledge/external/kg/kg_stats.json`
- `knowledge/external/provenance/source_registry/`
- `knowledge/external/provenance/source_registry/source_metadata.jsonl`
- `knowledge/external/provenance/structured_text/documents.jsonl`
- `knowledge/external/provenance/structured_text/blocks.jsonl`
- `knowledge/external/provenance/structured_text/table_records.jsonl`
- `knowledge/external/provenance/structured_text/figure_records.jsonl`
- `knowledge/external/provenance/structured_text/formula_records.jsonl`
- `knowledge/external/provenance/structured_text/markdown_documents/`
- `knowledge/external/manifest.json`

运行时入口为 `core/domain_knowledge.py`。候选生成智能体只在 LLM 路径中注入这些检索结果，CASE_TRANSFER 与 DOE 路径不读取该资产。

## 案例库与案例记忆

案例侧资产包括：

- `data/cases/`：全部已校核样本评估档案。
- `knowledge/case_library/`：满足入库条件的正式案例。
- `knowledge/chroma_db/`：案例记忆向量索引，可由 `scripts/migrate_contracts.py --case-memory-only` 重建。

案例迁移以结构化硬约束为主，案例记忆向量索引只参与召回和排序，不替代工程约束。

## 回流边界

`KNOWLEDGE_AGENT` 的回流范围为：

- 写入 `data/cases/`
- 按结论写入 `knowledge/case_library/`
- 更新 `csdm_case_memory`
- 在样本数量满足阈值时触发代理模型重训

`KNOWLEDGE_AGENT` 不更新 `knowledge/external/` 下的外部知识库/知识图谱资产。

## 当前维护规则

- 运行时只读取项目内资产，不直接访问参考项目路径。
- `.env` 中只配置单一 OpenAI 兼容 LLM 后端。
- 外部知识库、知识图谱和案例记忆索引是当前主线知识来源。
