﻿# CSDM_panel 项目全流程梳理与开发指南

本文档描述当前仓库的实际运行方式、模块边界和继续开发入口。

## 1. 项目概览

CSDM_panel 是面向复合材料加筋壁板的多智能体设计原型系统，主链路为：

```text
自然语言需求
  -> 结构化任务
  -> 候选设计生成
  -> 代理模型初筛
  -> ABAQUS 高保真校核
  -> 结果归档与知识回流
  -> 报告输出
```

当前落地 6 个智能体：

- `ORCHESTRATOR`：主控调度
- `CANDIDATE_GEN`：候选生成
- `SCREENER`：代理模型筛选
- `FEM_AGENT`：ABAQUS 建模、求解与自动重试
- `KNOWLEDGE_AGENT`：知识回流与代理模型重训
- `REPORT_GEN`：报告生成

## 2. 当前实现状态

截至 2026-05-17，仓库内可直接确认：

- `data/cases/` 中当前有 350 个评估档案
- `knowledge/case_library/` 中当前有 41 个正式案例
- `knowledge/chroma_db/` 中当前 `csdm_case_memory` 案例记忆集合有 350 条索引记录
- `data/abaqus_runs/` 中当前有 350 个样本工件目录
- `models/surrogate_metrics.json` 当前选中模型为 `rf`
- 当前代理模型训练样本数为 317
- `knowledge/external/` 中的外部知识库/知识图谱包含知识库文本块 43651 条、知识图谱实体 1763 个、知识图谱关系 347609 条

主链路已经贯通，LLM 路径使用外部知识库/知识图谱做检索增强；案例迁移路径使用案例库与案例记忆索引；DOE 路径提供参数空间采样。

## 3. 统一运行环境

项目约定环境为 Conda `GPT`，推荐统一执行器为：

```text
D:/anaconda3/envs/GPT/python.exe
```

推荐先运行环境自检：

```powershell
D:/anaconda3/envs/GPT/python.exe scripts/check_env.py
```

当前 `environment.yml` 约定：

- Python 3.9
- PyQt6
- jinja2
- PyYAML
- python-dotenv
- jsonschema
- openai
- chromadb
- sentence-transformers
- scikit-learn
- matplotlib
- pyvista / pyvistaqt
- reportlab
- pytest

如果终端出现中文乱码，可先执行：

```powershell
chcp 65001
```

## 4. 目录结构

```text
CSDM_panel/
├─ main.py                    # GUI 启动入口
├─ agents/                    # 六大智能体
├─ abaqus/                    # ABAQUS 建模模板、运行脚本、后处理脚本
├─ config/                    # YAML 配置
├─ core/                      # 路径、契约、知识检索、LLM、DOE、代理模型等公共能力
├─ data/                      # IO、案例、Abaqus 工件、报告输出
├─ docs/                      # 开发指南与接口约定
├─ gui/                       # PyQt6 图形界面与可视化
├─ knowledge/                 # 正式案例库、案例向量库、外部知识库/知识图谱
├─ models/                    # 代理模型文件与指标
├─ schemas/                   # JSON Schema
├─ scripts/                   # 自检、训练、迁移、清理脚本
└─ tests/                     # 自动化测试
```

### 4.1 `core/`

`core/` 提供所有智能体共享的底层能力，当前关键模块包括：

- `paths.py`：统一路径常量和目录初始化
- `config_loader.py`：YAML 配置与 `.env` 加载
- `io_utils.py`：JSON / 文本读写
- `schema_validator.py`：Schema 校验
- `id_utils.py`：身份字段与编号工具
- `task_contract.py`：任务契约规范化、任务实例与任务语义工具
- `stiffener_profile.py`：筋型、几何参数和可视化网格定义
- `rule_checker.py`：候选几何与铺层规则检查
- `doe_sampler.py`：DOE 采样
- `llm_backend.py`：单一 OpenAI 兼容 LLM 后端
- `rag_engine.py`：通用文本向量引擎，当前用于案例记忆集合
- `domain_knowledge.py`：外部知识库/知识图谱运行时检索入口
- `case_retriever.py`：结构化历史案例检索与案例记忆排序
- `case_memory.py`：案例记忆文本、元数据和向量索引入口
- `surrogate_model.py`：代理模型训练与预测
- `conversation_flow.py`：UI 无关的对话流程控制器

### 4.2 `agents/`

- `orchestrator.py`：串联任务解析、候选生成、初筛、校核、回流和报告
- `candidate_gen.py`：调度 LLM、CASE_TRANSFER、DOE 三条候选路径
- `screener.py`：使用代理模型排序并标记入选理由
- `fem_agent.py`：驱动 ABAQUS 运行与失败重试
- `knowledge_agent.py`：归档案例、写入正式案例库、写入案例记忆索引、按阈值触发模型重训
- `report_gen.py`：输出 Markdown / PDF 报告

### 4.3 `knowledge/`

当前知识资产分为两类：

- 案例知识资产
  - `knowledge/case_library/`：通过样本形成的正式案例库
  - `knowledge/chroma_db/`：案例记忆集合所在的 Chroma 目录
- 外部知识库/知识图谱资产
  - `knowledge/external/rag/rag_chunks.jsonl`
  - `knowledge/external/kg/entities.jsonl`
  - `knowledge/external/kg/relations.jsonl`
  - `knowledge/external/kg/kg_stats.json`
  - `knowledge/external/manifest.json`

## 5. 主流程

### 5.1 用户视角

当前主交互是“对话驱动 + 三个关键确认节点”：

1. 用户输入自然语言设计需求
2. 系统解析任务并生成初始候选
3. 系统询问是否进行 DNN 初筛
4. 系统询问是否进行 FEM 校核
5. 系统询问是否导出报告

### 5.2 代码视角

```text
用户输入
  -> ORCHESTRATOR.parse_instruction()
  -> ORCHESTRATOR.generate_candidates()
  -> CANDIDATE_GEN.run()
       -> LLM 路径（外部知识库/知识图谱增强）
       -> CASE_TRANSFER 路径
       -> DOE 路径
  -> ORCHESTRATOR.screen_candidates()
  -> SCREENER.run()
  -> ORCHESTRATOR.evaluate_candidate()
  -> FEM_AGENT.run()
  -> KNOWLEDGE_AGENT.run()
  -> ORCHESTRATOR.generate_report()
  -> REPORT_GEN.run()
```

### 5.3 数据流视角

```text
自然语言
  -> 会话内 task / TASK_x
  -> 会话内 TMP_x 候选
  -> 正式样本 Cx
  -> data/io/input_Cx.json
  -> data/abaqus_runs/Cx/*
  -> data/io/result_Cx.json
  -> data/cases/CASE_x.json
  -> knowledge/case_library/CASE_x.json（仅通过样本）
  -> knowledge/chroma_db/*
  -> data/results/latest_report.md / latest_report.pdf
```

身份字段分工：

- `task_id`：会话任务编号，用于当前任务摘要与报告展示
- `candidate_id`：候选方案与正式样本标识
- `case_id`：归档案例标识

## 6. 候选生成边界

### 6.1 LLM 路径

- `DomainKnowledgeBase` 将任务转换为检索文本
- 从 `knowledge/external/` 检索知识库片段与知识图谱关系
- 外部知识库/知识图谱未就绪时，不注入额外知识片段
- Prompt 中包含任务、工况说明、边界说明、材料选项、检索依据和 JSON 输出约束
- 只输出候选设计字段，不输出历史案例字段

### 6.2 CASE_TRANSFER 路径

- 使用 `CaseRetriever` 在 `data/cases/` 和 `knowledge/case_library/` 中查找相似案例
- 结构化匹配条件包含筋型、工况、边界、材料兼容性和通过结论
- 使用 `CaseMemoryIndex` 查询 `csdm_case_memory`，对结构化候选做相似度排序辅助
- 只迁移“可直接复用”的通过案例
- 不把历史案例原文注入 LLM Prompt

### 6.3 DOE 路径

- `DOESampler` 按参数范围采样
- 作为兜底与探索来源
- 所有候选仍需通过 `RuleChecker`

## 7. 知识回流

`agents/knowledge_agent.py` 负责：

- 把所有校核样本写入 `data/cases/`
- 把通过样本写入 `knowledge/case_library/`
- 把案例摘要和元数据写入案例记忆向量索引
- 满足样本规模阈值时触发代理模型训练

知识回流只写入案例侧资产，不反写 `knowledge/external/` 的外部知识库/知识图谱。

## 8. 知识检索与案例检索边界

### 8.1 `core/domain_knowledge.py`

外部知识库/知识图谱入口读取：

- 知识库文本块：`knowledge/external/rag/rag_chunks.jsonl`
- 知识图谱实体：`knowledge/external/kg/entities.jsonl`
- 知识图谱关系：`knowledge/external/kg/relations.jsonl`
- 溯源资料：`knowledge/external/provenance/`

LLM 候选生成只读取知识库文本块和知识图谱关系。溯源资料不进入运行检索链路，用于核查命中片段对应的源登记、结构化文档清单和 Markdown 全文上下文。

### 8.2 `core/case_retriever.py`

结构化案例检索器直接读取案例 JSON，并可使用案例记忆向量索引做排序辅助，当前提供：

- `retrieve_similar_cases(task, top_k=5)`
- `retrieve_transferable_cases(task, top_k=5)`
- `transfer_candidates(task, top_k=2)`

它的职责是在历史案例中找可迁移设计。结构化硬约束决定能否迁移，向量索引只影响相似案例排序或非迁移场景下的观察召回。

### 8.3 `core/rag_engine.py`

`RAGEngine` 是通用文本向量引擎：

- 默认案例记忆集合：`csdm_case_memory`
- 支持 `upsert_records()` 和 `retrieve()` 兼容案例向量写法
- 支持 `upsert_documents()` 和 `query_text()` 作为通用文本接口
- 支持 `reset_collection()` 只重置当前集合

## 9. ABAQUS 层

### 9.1 双环境分离

主程序走 `GPT` 环境，Abaqus 脚本走 Abaqus 自带 Python，两边只通过 JSON 文件通信，不直接跨环境导入业务对象。

### 9.2 真实运行脚本

真实运行链路由以下文件组成：

- `abaqus/templates/t_stiffener_buckle.py.j2`
- `abaqus/runtime_build_panel.py`
- `abaqus/runtime_extract_blf.py`

### 9.3 真实有限元链路

`FEM_AGENT` 只执行真实 Abaqus 求解。求解成功后产出 `.inp`、`.odb`、模态数据 JSON 与结果 JSON；未找到 Abaqus 命令或真实求解失败时写入结构化失败结果，不生成替代性假结果。

## 10. 数据契约重点

### 10.1 任务记录

任务台账记录包含：

- `task_id`
- `created_at`
- `source`
- `task`

其中 `task` 仅保留任务语义字段。

### 10.2 候选 `candidate`

当前候选包含：

- `candidate_id`
- `task_id`
- `source`
- `stiffener_type`
- `geometry`
- `layup`
- `material_system`
- `load_conditions`
- `boundary_conditions`
- `design_targets`
- `rule_check`
- `surrogate_BLF`
- `surrogate_weight`
- `rank_score`
- `rationale`
- `origin_summary`
- `screening_summary`
- `selection_reason`
- `display_name`
- `persistent_candidate_id`

### 10.3 Abaqus 结果 `abaqus_result`

关键字段包括：

- `status`
- `candidate_id`
- `retry_count`
- `BLF_global`
- `BLF_local`
- `failure_mode`
- `weight_kg_per_m2`
- `verdict`
- `abaqus_odb`
- `abaqus_inp`
- `visualization_json`
- `mode_eigenvalues`
- `effective_mode_eigenvalues`
- `analysis_flags`
- `load_summary`
- `boundary_summary`
- `diagnosis_summary`

### 10.4 案例记录 `case_record`

一个 `CASE_x.json` 包含：

- `case_id`
- `task_id`
- `candidate_id`
- `created_at`
- `source`
- `task`
- `design`
- `abaqus_results`
- `verdict`
- `surrogate_BLF_error_pct`
- `fem_agent_retry_count`

## 11. GUI 知识库页

`gui/knowledge_widget.py` 展示：

- 评估档案数
- 正式案例数
- 案例记忆向量块数
- 已归档 ODB 数
- 模态可视化数据数
- 代理模型指标
- 外部知识库/知识图谱文本块、实体、关系数量
- 源登记、结构化文档和 Markdown 全文数量
- 知识资产清单路径、溯源资料目录和更新时间

## 12. 常用脚本

### 12.1 环境自检

```powershell
D:/anaconda3/envs/GPT/python.exe scripts/check_env.py
```

### 12.2 启动 GUI

```powershell
D:/anaconda3/envs/GPT/python.exe main.py
```

### 12.3 运行测试

```powershell
D:/anaconda3/envs/GPT/python.exe -m pytest tests -q
```

### 12.4 构建初始案例

```powershell
D:/anaconda3/envs/GPT/python.exe scripts/build_initial_cases.py --count 20 --task-count 4 --workers 2 --reset
```

### 12.5 重训代理模型

```powershell
D:/anaconda3/envs/GPT/python.exe scripts/train_screener.py
```

### 12.6 案例记忆索引

```powershell
D:/anaconda3/envs/GPT/python.exe scripts/migrate_contracts.py --case-memory-only
```

### 12.7 契约迁移与重建

```powershell
D:/anaconda3/envs/GPT/python.exe scripts/migrate_contracts.py --retrain-surrogate
```

## 13. 推荐阅读顺序

1. `README.md`
2. `docs/接口约定.md`
3. `core/paths.py`
4. `core/task_contract.py`
5. `core/conversation_flow.py`
6. `agents/orchestrator.py`
7. `agents/candidate_gen.py`
8. `core/domain_knowledge.py`
9. `core/case_retriever.py`
10. `core/case_memory.py`
11. `agents/screener.py`
12. `agents/fem_agent.py`
13. `agents/knowledge_agent.py`
14. `agents/report_gen.py`
15. `abaqus/runtime_build_panel.py`
16. `abaqus/runtime_extract_blf.py`

## 14. 当前容易踩的坑

### 坑 1：没有使用 `GPT` 环境执行

当前项目依赖集中在 `GPT` 环境里，建议直接使用：

```text
D:/anaconda3/envs/GPT/python.exe
```

### 坑 2：把案例迁移当成 LLM 检索增强

当前实现已经明确拆分：

- 外部知识库/知识图谱只服务于 LLM 路径
- 案例迁移走结构化硬过滤和案例记忆向量排序
- 直接迁移不得把历史案例原文拼进 LLM Prompt

### 坑 3：把参考项目路径当成运行时依赖

当前项目已经复制需要的知识资产，运行时只读取 `knowledge/external/`。参考项目不是 CSDM_panel 的运行时依赖。

### 坑 4：以为所有成功样本都会进入正式案例库

只有满足正式入库条件的案例才进入 `knowledge/case_library/`；全部校核结果仍会保存在 `data/cases/`。

## 15. 下一步推荐方向

- 验证外部知识库/知识图谱在加筋壁板候选生成中的命中质量
- 持续积累真实 Abaqus 校核样本，提升代理模型精度
- 扩展更多筋型、边界条件和多目标优化逻辑
- 增强报告的对比深度与工程解释能力
- 继续完善 GUI 的交互反馈和异常处理
