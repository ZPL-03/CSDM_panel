# CSDM

复合材料加筋壁板智能设计系统（Composite Stiffened-panel Design Manager, CSDM）是一个面向复合材料 T 形加筋壁板的多智能体设计原型系统，目标是把“自然语言需求 -> 结构化任务 -> 候选生成 -> 代理模型初筛 -> ABAQUS 校核 -> 知识回流 -> 报告输出”这条工程链路自动化。

当前仓库已经具备可运行的 Windows 原型：前端为 PyQt6 对话式 GUI，后端由多智能体协同完成候选生成、有限元校核、知识回流和报告导出；候选生成链路已经拆分为三条职责清晰的来源：LLM + 文献 RAG、历史案例迁移、DOE 参数采样。

## 1. 项目目标

本项目围绕复合材料加筋壁板智能设计场景，自动完成以下流程：

- 自然语言需求解析与任务归一化
- 候选方案生成
  - LLM 生成：使用文献知识库做 RAG 增强
  - 历史案例迁移：在归档案例和正式案例中做结构化相似检索
  - DOE 采样：在参数空间生成兜底与探索方案
- 代理模型快速初筛
- ABAQUS 建模、求解、后处理与失败重试
- 案例归档、正式知识库写入与代理模型重训
- Markdown / PDF 工程报告生成

## 2. 当前状态

截至 2026-04-28，仓库内可直接确认的状态如下：

- 已落地 6 个核心智能体：`ORCHESTRATOR`、`CANDIDATE_GEN`、`SCREENER`、`FEM_AGENT`、`KNOWLEDGE_AGENT`、`REPORT_GEN`
- 支持 3 类工况：`axial_compression`、`in_plane_shear`、`compression_shear`
- 支持 3 类边界：`SSSS`、`CCCC`、`SSCC`
- 候选生成已经按三条路径解耦：`LLM`、`CASE_TRANSFER`、`DOE`
- `data/cases/` 中当前有 333 个评估档案
- `knowledge/case_library/` 中当前有 41 个正式案例
- `data/abaqus_runs/` 中当前有 333 个样本工件目录
- `models/surrogate_metrics.json` 当前选中模型为 `rf`，训练样本数 317，RF `MAPE = 0.1216`
- `knowledge/literature/` 目录结构和 ingestion / RAG 代码已经就位，当前仓库内尚未沉淀实际文献记录

## 3. 系统架构

### 3.1 多智能体分工

| 智能体 | 标识符 | 主要职责 |
| --- | --- | --- |
| 主控智能体 | `ORCHESTRATOR` | 解析用户需求、维护状态机、串联全流程 |
| 候选生成智能体 | `CANDIDATE_GEN` | 协调 LLM 文献增强生成、案例迁移和 DOE 采样 |
| 快速筛选智能体 | `SCREENER` | 使用代理模型预测 BLF / 重量并完成 Top-K 排序 |
| 求解智能体 | `FEM_AGENT` | 驱动 ABAQUS 建模、求解、结果提取与自动重试 |
| 知识回流智能体 | `KNOWLEDGE_AGENT` | 写回案例档案、正式案例库、案例向量库，并触发代理模型重训 |
| 报告生成智能体 | `REPORT_GEN` | 汇总校核结果并导出 Markdown / PDF 报告 |

### 3.2 主流程

```text
用户自然语言需求
        ↓
ORCHESTRATOR：任务解析 / 对话流程控制
        ↓
CANDIDATE_GEN：候选生成
  ├─ LLM + 文献 RAG
  ├─ CASE_TRANSFER 结构化案例迁移
  └─ DOE 参数采样
        ↓
SCREENER：代理模型初筛
        ↓
FEM_AGENT：ABAQUS 校核与自动重试
        ↓
KNOWLEDGE_AGENT：结果归档 / 正式案例入库 / 案例向量库更新 / 模型重训
        ↓
REPORT_GEN：工程报告输出
```

### 3.3 三条候选生成路径的边界

- `LLM` 路径只使用文献知识库：`core/literature_corpus.py` 将任务转换为检索文本，从 `csdm_literature_corpus` 取回文献片段并注入 Prompt。
- `CASE_TRANSFER` 路径不走文献 RAG：`core/case_retriever.py` 直接在 `data/cases/` 与 `knowledge/case_library/` 中做结构化相似检索，只迁移结构、工况、边界和材料体系匹配的历史设计。
- `DOE` 路径独立存在：`core/doe_sampler.py` 在参数范围内采样，提供兜底与探索候选。

### 3.4 对话式 GUI

- 左侧为聊天主区，支持自然语言输入和流程事件回显
- 右侧包含候选方案、ABAQUS 结果、知识库、日志等标签页
- 主流程内置 3 个确认节点：DNN 初筛前、FEM 校核前、报告导出前
- `core/conversation_flow.py` 提供 UI 无关的对话流程控制器，GUI 只是其中一种承载方式

## 4. 仓库结构

```text
CSDM/
├─ agents/              # 六大智能体
├─ abaqus/              # ABAQUS 模板、建模脚本、结果提取与运行工具
├─ config/              # YAML 配置
├─ core/                # 路径、配置、契约、RAG、LLM、DOE、代理模型等公共能力
├─ data/                # IO、案例、Abaqus 工件、报告输出
├─ docs/                # 开发指南与接口约定
├─ gui/                 # PyQt6 对话式界面与可视化组件
├─ knowledge/           # 正式案例库、案例向量库、文献知识库
├─ models/              # 代理模型文件与指标
├─ schemas/             # 任务 / 候选 / 结果 / 案例 JSON Schema
├─ scripts/             # 自检、训练、迁移、清理、文献导入脚本
├─ tests/               # 自动化测试
├─ environment.yml      # GPT Conda 环境定义
└─ main.py              # GUI 启动入口
```

## 5. 运行环境

### 5.1 约定环境

- Windows 11
- Anaconda
- Python 3.9
- Conda 环境名：`GPT`
- 推荐统一执行器：`D:/anaconda3/envs/GPT/python.exe`
- ABAQUS 2023

### 5.2 创建环境

```powershell
conda env create -f environment.yml
```

### 5.3 LLM 配置

`config/llm_config.yaml` 当前默认启用本地 Ollama：

- `active_provider: local_ollama`
- 本地默认模型：`qwen2.5:7b`
- 云端备选：`ollama_cloud`

如需切换到云端兼容接口，可在 `.env` 中配置密钥：

```text
OLLAMA_API_KEY=your_ollama_api_key
```

## 6. 快速开始

### 6.1 环境自检

```powershell
D:/anaconda3/envs/GPT/python.exe scripts/check_env.py
```

### 6.2 启动图形界面

```powershell
D:/anaconda3/envs/GPT/python.exe main.py
```

### 6.3 运行测试

```powershell
D:/anaconda3/envs/GPT/python.exe -m pytest tests -q
```

### 6.4 强制 mock 模式回归

```powershell
$env:CSDM_USE_MOCK_ABAQUS = "1"
D:/anaconda3/envs/GPT/python.exe -m pytest tests -q
```

### 6.5 文献库导入

```powershell
D:/anaconda3/envs/GPT/python.exe scripts/ingest_literature.py --seed --query-group composite_basics --max-results 20
```

### 6.6 文献记录重建向量索引

```powershell
D:/anaconda3/envs/GPT/python.exe scripts/ingest_literature.py --reindex
```

## 7. 关键配置文件

| 文件 | 作用 |
| --- | --- |
| `config/app_config.yaml` | 路径、ABAQUS、pipeline、RAG、literature、报告输出等全局配置 |
| `config/llm_config.yaml` | 本地 / 云端 LLM 提供方与模型配置 |
| `config/material_db.yaml` | 材料数据库 |
| `config/param_ranges.yaml` | 设计变量范围、铺层模板、规则检查阈值 |
| `config/literature_queries.yaml` | 文献检索主题分组 |

## 8. 常用脚本

| 命令 | 说明 |
| --- | --- |
| `D:/anaconda3/envs/GPT/python.exe scripts/check_env.py` | 检查 Python、ABAQUS、Ollama 与核心依赖是否可用 |
| `D:/anaconda3/envs/GPT/python.exe scripts/train_screener.py` | 依据案例库重训 `SCREENER` 代理模型 |
| `D:/anaconda3/envs/GPT/python.exe scripts/migrate_contracts.py --retrain-surrogate` | 迁移案例、IO 与工件契约并可选重训模型 |
| `D:/anaconda3/envs/GPT/python.exe scripts/rebuild_abaqus_artifacts.py --limit 10 --workers 1` | 重建缺失的 ABAQUS 工件 |
| `D:/anaconda3/envs/GPT/python.exe scripts/build_initial_cases.py` | 批量生成/补齐初始案例集 |
| `D:/anaconda3/envs/GPT/python.exe scripts/clean_debug_artifacts.py` | 清理 `__pycache__`、`.pytest_cache` 与 ABAQUS 临时文件 |
| `D:/anaconda3/envs/GPT/python.exe scripts/ingest_literature.py --seed ...` | 导入文献记录并写入文献向量库 |
| `D:/anaconda3/envs/GPT/python.exe scripts/ingest_literature.py --seed ...` | 导入文献记录并写入文献向量库 |

## 9. 标识与关联

- `task_id`：会话任务编号，格式为 `TASK_<n>`，用于当前交互过程中的任务摘要与报告展示
- `candidate_id`：设计候选标识，候选阶段使用 `TMP_<n>`，进入 FEM 后切换为正式样本编号 `C<n>`
- `case_id`：归档案例标识，对应 `data/cases/` 与 `knowledge/case_library/` 中的案例记录

`data/cases/`、`data/io/`、`data/abaqus_runs/` 与 `knowledge/case_library/` 构成主数据层，围绕 `candidate_id` 与 `case_id` 组织。会话任务编号只存在于运行时任务对象、界面摘要与报告展示中，不参与主数据编号分配与关联。

## 10. 数据与知识资产

### 10.1 评估档案与正式案例

- `data/cases/`：所有已校核样本的评估档案
- `knowledge/case_library/`：仅保留当前规则下可进入正式知识库的案例
- `knowledge/chroma_db/`：案例向量库目录

### 10.2 文献知识库

- `knowledge/literature/raw/`：原始 API 返回
- `knowledge/literature/records/`：标准化文献记录
- `knowledge/literature/pdfs/`：开放获取 PDF 预留目录
- `knowledge/literature/manifests/`：最近一次 ingestion 摘要

当前文献链路基于 `OpenAlex` 实现 metadata + abstract ingestion，支持后续重建索引与任务时检索。

## 11. 回归验证

当前推荐使用如下命令进行统一验证：

```powershell
D:/anaconda3/envs/GPT/python.exe scripts/check_env.py
D:/anaconda3/envs/GPT/python.exe -m pytest tests -q
```

## 12. 已知边界

- 当前筋型仍以 `T` 形筋为主
- 真实 ABAQUS 求解依赖本机许可证与命令行环境
- 文献知识库链路已落地，但仓库内当前尚未保留实际抓取数据
- 代理模型精度仍处于原型阶段，后续仍需持续积累真实案例

## 13. 相关文档

- [项目全流程梳理与开发指南](docs/CSDM_项目全流程梳理与开发指南.md)
- [接口约定](docs/接口约定.md)

## 14. 下一步计划

- 持续积累真实校核案例，提升代理模型质量
- 为 `README` 补充 GUI 截图和示例对话
- 将关键设计案例整理为公开 benchmark 数据集
