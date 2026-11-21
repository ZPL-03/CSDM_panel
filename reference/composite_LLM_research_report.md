# CSDM_panel 知识资产参考说明

本文件记录当前项目采用的知识资产边界，作为 `README.md`、`docs/接口约定.md` 和全流程开发指南的补充参考。

## 当前知识资产

CSDM_panel 当前保留两类知识能力：

- 外部知识库/知识图谱：位于 `knowledge/external/`，服务于报告解释与可追溯证据。
- 案例库与案例记忆：位于 `data/cases/`、`knowledge/case_library/` 和 `knowledge/chroma_db/`，服务于案例归档、案例迁移和相似案例排序。

## 外部知识库/知识图谱

外部知识库/知识图谱资产包括：

- `knowledge/external/rag/retrieval_index.sqlite`
- `knowledge/external/kg/entities.jsonl`
- `knowledge/external/kg/relations.compact.jsonl.gz`
- `knowledge/external/kg/kg_stats.json`
- `knowledge/external/provenance/source_metadata.compact.jsonl.gz`
- `knowledge/external/manifest.json`

运行时入口为 `core/domain_knowledge.py`。报告智能体将完整检索证据与适用条件提供给 LLM，并保存到事实 JSON；CASE_TRANSFER 与 DOE 不使用外部文献回填设计性能。

## 案例库与案例记忆

案例侧资产包括：

- `data/cases/`：全部已校核样本评估档案。
- `knowledge/case_library/`：满足入库条件的正式案例。
- `knowledge/chroma_db/`：案例记忆向量索引，可由 `conda run -n AGENT python -B scripts/rebuild_case_memory.py` 重建；源记录在索引更新前完成校验，不回写业务 JSON。

案例迁移以结构化硬约束为主，案例记忆向量索引只参与召回和排序，不替代工程约束。

## 回流边界

`KNOWLEDGE_AGENT` 的回流范围为：

- 写入 `data/cases/`
- 按结论写入 `knowledge/case_library/`
- 更新 `csdm_case_memory`
- 自动代理训练由配置控制，默认关闭；正式训练与独立 FEM 切片验收由 `scripts/train_screener.py --stiffener-type T` 执行

`KNOWLEDGE_AGENT` 不更新 `knowledge/external/` 下的外部知识库/知识图谱资产。

## 当前维护规则

- 运行时只读取项目内资产，不直接访问参考项目路径。
- `.env` 中只配置单一 OpenAI 兼容 LLM 后端。
- 外部知识库、知识图谱和案例记忆索引是当前主线知识来源。
