# CSDM 项目全流程梳理与开发指南

本文档基于当前仓库代码、配置、测试、脚本和数据目录整理，描述项目的实际运行方式、模块边界和继续开发时的推荐入口。

## 1. 项目概览

CSDM 是一个面向复合材料 T 形加筋壁板的多智能体设计原型系统，目标是把以下工程链路自动化：

```text
自然语言需求
  -> 结构化任务
  -> 候选设计生成
  -> 代理模型初筛
  -> ABAQUS 高保真校核
  -> 结果归档与知识回流
  -> 报告输出
```

当前仓库中已经落地 6 个智能体：

- `ORCHESTRATOR`：主控调度
- `CANDIDATE_GEN`：候选生成
- `SCREENER`：代理模型筛选
- `FEM_AGENT`：ABAQUS 建模、求解与自动重试
- `KNOWLEDGE_AGENT`：知识回流与代理模型重训
- `REPORT_GEN`：报告生成

## 2. 当前实现状态

截至 2026-04-28，仓库内可直接确认：

- `data/cases/` 中当前有 333 个评估档案
- `knowledge/case_library/` 中当前有 41 个正式案例
- `data/abaqus_runs/` 中当前有 333 个样本工件目录
- `models/surrogate_metrics.json` 当前选中模型为 `rf`
- 当前代理模型训练样本数为 317
- 文献知识库目录、ingestion 代码和 runtime 检索包装已经就位
- `knowledge/literature/records/` 当前为空，说明仓库内还没有保留已抓取的文献记录

这意味着项目主链路已经贯通，但文献知识库仍处于“代码可用、内容待灌注”的状态。

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

## 4. 目录脑图

```text
CSDM/
├─ main.py                    # GUI 启动入口
├─ agents/                    # 六大智能体
├─ abaqus/                    # ABAQUS 建模模板、运行脚本、后处理脚本
├─ config/                    # YAML 配置
├─ core/                      # 路径、契约、RAG、LLM、DOE、代理模型等公共能力
├─ data/                      # IO、案例、Abaqus 工件、报告输出
├─ docs/                      # 开发指南与接口约定
├─ gui/                       # PyQt6 图形界面与可视化
├─ knowledge/                 # 正式案例库、案例向量库、文献知识库
├─ models/                    # 代理模型文件与指标
├─ schemas/                   # JSON Schema
├─ scripts/                   # 自检、训练、迁移、清理、文献导入脚本
└─ tests/                     # 自动化测试
```

### 4.1 `core/`

`core/` 提供所有智能体共享的底层能力，当前关键模块包括：

- `paths.py`：统一路径常量和目录初始化
- `config_loader.py`：YAML 配置加载
- `io_utils.py`：JSON / 文本读写
- `schema_validator.py`：Schema 校验
- `id_utils.py`：身份字段与编号工具
- `task_contract.py`：任务契约规范化、任务实例与任务语义工具
- `rule_checker.py`：候选几何与铺层规则检查
- `doe_sampler.py`：DOE 采样
- `llm_backend.py`：本地 Ollama 与兼容 OpenAI 接口适配层
- `rag_engine.py`：通用文本向量引擎
- `case_retriever.py`：结构化历史案例检索
- `literature_corpus.py`：运行时文献检索包装
- `literature_ingest.py`：文献导入与索引构建
- `surrogate_model.py`：代理模型训练与预测
- `conversation_flow.py`：UI 无关的对话流程控制器

### 4.2 `agents/`

`agents/` 承载业务角色层，当前主要职责如下：

- `orchestrator.py`：串联任务解析、候选生成、初筛、校核、回流和报告
- `candidate_gen.py`：调度 LLM、CASE_TRANSFER、DOE 三条候选路径
- `screener.py`：使用代理模型排序并标记入选理由
- `fem_agent.py`：驱动 ABAQUS 运行与失败重试
- `knowledge_agent.py`：归档案例、写入正式案例库、更新案例向量库、按阈值触发模型重训
- `report_gen.py`：输出 Markdown / PDF 报告

### 4.3 `knowledge/`

`knowledge/` 当前包含两类知识资产：

- 案例知识资产
  - `knowledge/case_library/`：通过样本形成的正式案例库
  - `knowledge/chroma_db/`：案例向量库
- 文献知识资产
  - `knowledge/literature/raw/`
  - `knowledge/literature/records/`
  - `knowledge/literature/pdfs/`
  - `knowledge/literature/texts/`
  - `knowledge/literature/markdown/`
  - `knowledge/literature/json/`
  - `knowledge/literature/images/`
  - `knowledge/literature/imports/pdfs/`
  - `knowledge/literature/imports/texts/`
  - `knowledge/literature/imports/markdown/`
  - `knowledge/literature/imports/json/`
  - `knowledge/literature/imports/images/`
  - `knowledge/literature/manifests/`

## 5. 主流程如何运行

### 5.1 用户视角

当前主交互已经是“对话驱动 + 三个关键确认节点”：

1. 用户输入一句自然语言设计需求
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
       -> LLM 路径
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

身份字段分工如下：

- `task_id`：会话任务编号，用于当前任务摘要与报告展示
- `candidate_id`：候选方案与正式样本标识
- `case_id`：归档案例标识

方案识别看 `candidate_id`，案例归档看 `case_id`，任务语义快照保留在会话对象与 case 的 `task` 字段中。

同一个候选在进入 FEM 前后会从 `TMP_<n>` 切换到正式样本编号 `C<n>`。

`task` 字段只描述任务本身；`data/io/`、`data/abaqus_runs/`、`data/cases/` 与 `knowledge/case_library/` 不依赖 `task_id`、`request_id`、`task_fingerprint` 组织主数据。

当前主线统一采用“会话任务编号”“候选方案”“归档案例”三类表述。

顶层 case `source` 表示归档链路来源，当前主线为 `abaqus_auto`；`design.source` 表示候选生成来源，当前主线只使用 `LLM`、`CASE_TRANSFER`、`DOE`。

结构化案例迁移继续复用结构化任务字段与案例相似度，不依赖顺序之外的额外身份字段。

同一任务从自然语言解析到报告与知识回流结束，全链路围绕任务语义、正式样本编号和正式案例编号组织。

## 6. GUI 与对话流程

### 6.1 `main.py`

`main.py` 负责：

1. 调用 `ensure_project_dirs()` 确保目录存在
2. 创建 `QApplication`
3. 打开 `MainWindow`

### 6.2 `gui/main_window.py`

GUI 主控制器负责：

- 管理当前对话会话状态
- 创建后台线程执行 `PipelineWorker`
- 把流程事件同步到聊天页、日志页、候选页、结果页和知识库页
- 处理 DNN、FEM、报告三个确认节点

### 6.3 `core/conversation_flow.py`

当前主流程由 `ConversationFlowController` 组织为状态机，关键事件包括：

- `conversation_started`
- `task_summary`
- `candidate_summary`
- `confirmation_requested`
- `assistant_commentary`
- `screening_summary`
- `fem_summary`
- `report_summary`
- `conversation_paused`

这使得主流程逻辑不依赖具体 GUI 组件，便于后续继续扩展其他交互入口。

## 7. 六个智能体的实际职责

### 7.1 `ORCHESTRATOR`

`agents/orchestrator.py` 是主控智能体，负责：

- 把用户输入转成任务 JSON
- 调度候选生成、筛选、校核、知识回流和报告生成
- 管理临时候选编号与正式样本编号的切换

当前实现里，候选阶段使用 `TMP_<n>`，只有进入 FEM 校核时才分配正式编号 `C<n>`。

### 7.2 `CANDIDATE_GEN`

`agents/candidate_gen.py` 当前已经是“编排器”，不再把三条候选路径混写在一起。

#### LLM 路径

- `LiteratureCorpus` 将任务转换为检索文本
- 从 `csdm_literature_corpus` 检索文献片段
- Prompt 中包含任务、工况说明、边界说明、材料选项、文献依据和 JSON 输出约束
- 只输出候选设计字段，不输出历史案例字段

#### CASE_TRANSFER 路径

- 使用 `CaseRetriever` 在 `data/cases/` 和 `knowledge/case_library/` 中查找相似案例
- 结构化匹配条件包含筋型、工况、边界和材料兼容性
- 只迁移“可直接复用”的通过案例
- 不再把历史案例原文注入 LLM Prompt

#### DOE 路径

- `DOESampler` 按参数范围采样
- 作为兜底与探索来源
- 所有候选仍需通过 `RuleChecker`

### 7.3 `SCREENER`

`agents/screener.py` 负责：

- 调用 `SurrogateModelManager.predict_candidates()` 预测 `surrogate_BLF`
- 估算候选面密度
- 使用线性打分公式排序
- 为候选写入 `screening_summary` 和 `selection_reason`

当前配置中的权重在 `config/app_config.yaml -> pipeline.screening_score` 中维护。

### 7.4 `FEM_AGENT`

`agents/fem_agent.py` 负责：

- 准备 ABAQUS 输入 JSON
- 调度建模、求解、结果提取
- 在失败时进行有限次数重试
- 输出结构化 BLF 结果与诊断摘要

### 7.5 `KNOWLEDGE_AGENT`

`agents/knowledge_agent.py` 负责：

- 把成功样本写入 `data/cases/`
- 把通过样本写入 `knowledge/case_library/`
- 把正式案例写入向量库
- 满足样本规模阈值时触发代理模型训练

### 7.6 `REPORT_GEN`

`agents/report_gen.py` 负责：

- 汇总任务、候选与 FEM 结果
- 生成 Markdown 报告
- 在 Pandoc 可用时导出 PDF

## 8. RAG、案例检索与文献库的边界

### 8.1 `core/rag_engine.py`

`RAGEngine` 现在是通用文本向量引擎：

- 默认案例集合：`csdm_case_memory`
- 支持 `upsert_records()` 和 `retrieve()` 兼容案例向量写法
- 支持 `upsert_documents()` 和 `query_text()` 供文献库使用

### 8.2 `core/case_retriever.py`

结构化案例检索器直接读取案例 JSON，当前提供：

- `retrieve_similar_cases(task, top_k=5)`
- `retrieve_transferable_cases(task, top_k=5)`
- `transfer_candidates(task, top_k=2)`

它的职责是“在历史案例中找可迁移设计”，而不是“把案例文本送给 LLM 参考”。

### 8.3 `core/literature_ingest.py`

文献导入器当前基于 `OpenAlex`：

- 拉取 works 搜索结果
- 重建 abstract
- 规范化记录字段
- 可选下载开放获取 PDF
- 可导入已授权下载到本地的 PDF
- 可选用 PyMuPDF 解析 PDF 文字层
- 可选用 MinerU 解析 Markdown、JSON 和图片
- 可选用 Nougat 解析公式密集型论文 Markdown
- 切分 title + abstract 或全文片段
- 写入 `knowledge/literature/records/`
- 构建 `csdm_literature_corpus`
- 写 `knowledge/literature/manifests/latest_ingest.json`

学校账号和机构订阅不在代码中自动登录。推荐工作流是：先用学校权限在浏览器或图书馆工具中下载授权 PDF，再通过 `scripts/ingest_literature.py --import-pdf-dir ... --parse-pdfs --parse-backend mineru --ocr` 导入。导入结果写入 `knowledge/literature/imports/`，与自动开放获取下载目录分开。

### 8.4 `core/literature_corpus.py`

运行时文献检索包装层负责：

- 根据任务生成检索文本
- 查询文献向量库
- 格式化成适合 Prompt 注入的 snippets
- 汇总文献库状态供 GUI 展示

## 9. ABAQUS 层的实际落地方式

### 9.1 双环境分离

主程序走 `GPT` 环境，Abaqus 脚本走 Abaqus 自带 Python，两边只通过 JSON 文件通信，不直接跨环境导入业务对象。

### 9.2 真实运行脚本

真实运行链路由以下文件组成：

- `abaqus/templates/t_stiffener_buckle.py.j2`
- `abaqus/runtime_build_panel.py`
- `abaqus/runtime_extract_blf.py`

### 9.3 mock 与真实模式

`FEM_AGENT` 支持两条运行路径：

- mock 路径：便于本地联调和自动化测试
- 真实 Abaqus 路径：产出 `.inp`、`.odb`、模态数据 JSON 等正式工件

## 10. 数据契约重点

### 10.1 任务 request record

任务台账记录包含：

- `task_id`
- `created_at`
- `source`
- `task`

其中 `task` 仅保留任务语义字段，不携带顺序型任务主键。

### 10.2 候选 `candidate`

当前候选包含：

- `candidate_id`
- `task_id`
- `source`
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

## 11. 知识库页当前展示什么

`gui/knowledge_widget.py` 现在同时展示两类资产：

### 11.1 案例知识资产

- 评估档案数
- 正式案例数
- 已归档 ODB 数
- 模态可视化数据数

### 11.2 文献知识资产

- 文献记录数
- 文献向量块数
- 开放获取 PDF 数
- 最近同步时间
- 最近来源
- 最近查询

页面上已经明确说明：

- LLM 候选生成会优先引用文献片段
- 历史案例迁移和 DOE 采样不依赖文献库

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
D:/anaconda3/envs/GPT/python.exe -m pytest
```

### 12.4 构建初始案例

```powershell
D:/anaconda3/envs/GPT/python.exe scripts/build_initial_cases.py --count 20 --task-count 4 --workers 2 --mock --reset
```

### 12.5 重训代理模型

```powershell
D:/anaconda3/envs/GPT/python.exe scripts/train_screener.py
```

### 12.6 契约迁移与重建

```powershell
D:/anaconda3/envs/GPT/python.exe scripts/migrate_contracts.py --retrain-surrogate
```

### 12.7 文献导入

```powershell
D:/anaconda3/envs/GPT/python.exe scripts/ingest_literature.py --seed --query-group composite_basics --max-results 20
```

下载 OpenAlex 标注的开放获取 PDF 并解析：

```powershell
D:/anaconda3/envs/GPT/python.exe scripts/ingest_literature.py --seed --query-group buckling_and_panels --max-results 20 --download-oa-pdfs --parse-pdfs
```

使用 MinerU 重新解析已有 PDF，生成 Markdown / JSON / 图片：

```powershell
D:/anaconda3/envs/GPT/python.exe scripts/ingest_literature.py --parse-existing-pdfs --parse-backend mineru --ocr --force-pdfs
```

使用 Nougat 重新解析公式密集或扫描型论文：

```powershell
D:/anaconda3/envs/GPT/python.exe scripts/ingest_literature.py --parse-existing-pdfs --parse-backend nougat --force-pdfs
```

导入学校账号已授权下载到本地的 PDF：

```powershell
D:/anaconda3/envs/GPT/python.exe scripts/ingest_literature.py --import-pdf-dir D:/Project/VS_Code/CSDM/reference/authorized_pdfs --parse-pdfs --parse-backend mineru --ocr
```

### 12.8 文献索引重建

```powershell
D:/anaconda3/envs/GPT/python.exe scripts/ingest_literature.py --reindex
```

## 13. 当前项目最值得优先理解的部分

如果你是第一次接触这个仓库，推荐按下面顺序阅读：

1. `README.md`
2. `docs/接口约定.md`
3. `core/paths.py`
4. `core/task_contract.py`
5. `core/conversation_flow.py`
6. `agents/orchestrator.py`
7. `agents/candidate_gen.py`
8. `core/case_retriever.py`
9. `core/literature_corpus.py`
10. `agents/screener.py`
11. `agents/fem_agent.py`
12. `agents/knowledge_agent.py`
13. `agents/report_gen.py`
14. `abaqus/runtime_build_panel.py`
15. `abaqus/runtime_extract_blf.py`

## 14. 当前最容易踩的坑

### 坑 1：没有使用 `GPT` 环境执行

当前项目依赖集中在 `GPT` 环境里，建议直接使用：

```text
D:/anaconda3/envs/GPT/python.exe
```

### 坑 2：把案例迁移当成文献 RAG

当前实现已经明确拆分：

- 文献 RAG 只服务于 LLM 路径
- 案例迁移走结构化检索

### 坑 3：以为没有文献库就不能生成候选

不是。

- LLM 路径在没有文献片段时会退回“仅依据任务约束生成”
- CASE_TRANSFER 和 DOE 路径都可以独立工作

### 坑 4：把 `build_panel.py` 当作真实求解入口

真实运行主链路仍然是模板 + `runtime_build_panel.py` + `runtime_extract_blf.py`。

### 坑 5：以为所有成功样本都会进入正式知识库

不是。

只有满足正式入库条件的案例才进入 `knowledge/case_library/`；全部校核结果仍会保存在 `data/cases/`。

## 15. 下一步推荐方向

当前最值得继续推进的方向包括：

- 为文献知识库导入真实复合材料文献记录并验证检索质量
- 持续积累真实 Abaqus 校核样本，提升代理模型精度
- 扩展更多筋型、边界条件和多目标优化逻辑
- 增强报告的对比深度与工程解释能力
- 继续完善 GUI 的交互反馈和异常处理
