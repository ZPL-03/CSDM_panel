# CSDM

复合材料加筋壁板智能设计系统（Composite Stiffened-panel Design Manager, CSDM）是一个面向复合材料加筋壁板的多智能体设计原型系统，目标是把"自然语言需求 -> 结构化任务 -> 候选生成 -> 代理模型初筛 -> ABAQUS 校核 -> 知识回流 -> 报告输出"这条工程链路自动化。

当前仓库已经具备可运行的 Windows 原型：前端为 PyQt6 对话式 GUI，后端由多智能体协同完成候选生成、有限元校核、知识回流和报告导出；候选生成链路已经拆分为三条职责清晰的来源：LLM + 文献 RAG、历史案例迁移、DOE 参数采样。

## 1. 项目目标

本项目围绕复合材料加筋壁板智能设计场景，自动完成以下流程：

- 自然语言需求解析与任务归一化
- 候选方案生成
  - LLM 生成：使用文献知识库做 RAG 增强
  - 历史案例迁移：在归档案例和正式案例中做结构化相似检索，并用案例记忆向量库辅助排序
  - DOE 采样：在参数空间生成兜底与探索方案
- 代理模型快速初筛
- ABAQUS 建模、求解、后处理与失败重试
- 案例归档、正式知识库写入与代理模型重训
- Markdown / PDF 工程报告生成

## 2. 支持的筋条类型

| 类型标识 | 中文名称 | 几何参数 |
| --- | --- | --- |
| `BLADE` | 板式筋 | panel_length_mm, panel_width_mm, skin_thickness_mm, pitch_mm, stiffener_height_mm, web_thickness_mm |
| `T` | T 型筋 | 上述 + flange_width_mm, flange_thickness_mm |
| `HAT` | 帽型筋 | T 型参数 + cap_width_mm, cap_thickness_mm |
| `L` | L 型角材 | 同 T 型（单侧翼缘装配） |

类型定义、参数管理和几何验证集中于 `core/stiffener_profile.py`，作为全仓库关于筋型的单点真理模块。

## 3. 当前状态

- 已落地 6 个核心智能体：`ORCHESTRATOR`、`CANDIDATE_GEN`、`SCREENER`、`FEM_AGENT`、`KNOWLEDGE_AGENT`、`REPORT_GEN`
- 支持 3 类工况：`axial_compression`、`in_plane_shear`、`compression_shear`
- 支持 3 类边界：`SSSS`、`CCCC`、`SSCC`
- 支持 4 种筋条类型：`BLADE`、`T`、`HAT`、`L`，均可走通完整设计流程
- 候选生成已经按三条路径解耦：`LLM`、`CASE_TRANSFER`、`DOE`
- 历史案例迁移采用"结构化硬过滤 + Case Memory 向量排序"混合检索；向量库不替代工程约束，也不把历史案例原文直接注入 LLM Prompt
- `data/cases/` 中当前有 350 个评估档案（T 型 344 个，BLADE/HAT/L 各 2 个）
- `knowledge/case_library/` 中当前有 41 个正式案例
- `knowledge/chroma_db/` 中当前 `csdm_case_memory` 案例记忆集合有 350 条索引记录
- `data/abaqus_runs/` 中当前有 350 个样本工件目录
- `models/surrogate_metrics.json` 当前选中模型为 `rf`，训练样本数 317，RF `MAPE = 0.1216`
- `knowledge/literature/` 目录结构和 ingestion / RAG 代码已经就位

## 4. 系统架构

### 4.1 多智能体分工

| 智能体 | 标识符 | 主要职责 |
| --- | --- | --- |
| 主控智能体 | `ORCHESTRATOR` | 解析用户需求、维护状态机、串联全流程 |
| 候选生成智能体 | `CANDIDATE_GEN` | 协调 LLM 文献增强生成、案例迁移和 DOE 采样 |
| 快速筛选智能体 | `SCREENER` | 使用代理模型预测 BLF / 重量并完成 Top-K 排序 |
| 求解智能体 | `FEM_AGENT` | 驱动 ABAQUS 建模、求解、结果提取与自动重试 |
| 知识回流智能体 | `KNOWLEDGE_AGENT` | 写回案例档案、正式案例库和案例记忆索引，并触发代理模型重训 |
| 报告生成智能体 | `REPORT_GEN` | 汇总校核结果并导出 Markdown / PDF 报告 |

### 4.2 主流程

```text
用户自然语言需求
        ↓
ORCHESTRATOR：任务解析 / 对话流程控制
        ↓
CANDIDATE_GEN：候选生成
  ├─ LLM + 文献 RAG
  ├─ CASE_TRANSFER 结构化案例迁移 + 案例记忆向量排序
  └─ DOE 参数采样
        ↓
SCREENER：代理模型初筛
        ↓
FEM_AGENT：ABAQUS 校核与自动重试
        ↓
KNOWLEDGE_AGENT：结果归档 / 正式案例入库 / 案例记忆索引写入 / 模型重训
        ↓
REPORT_GEN：工程报告输出
```

### 4.3 三条候选生成路径的边界

- `LLM` 路径只使用文献知识库：`core/literature_corpus.py` 将任务转换为检索文本，从 `csdm_literature_corpus` 取回文献片段并注入 Prompt。
- `CASE_TRANSFER` 路径不走文献 RAG：`core/case_retriever.py` 先按筋型、工况、边界、材料和通过结论做结构化硬过滤，再用 `core/case_memory.py` 中的 Case Memory 向量索引辅助相似案例排序；只迁移满足工程约束的历史设计。
- `DOE` 路径独立存在：`core/doe_sampler.py` 在参数范围内采样，提供兜底与探索候选。

### 4.4 筋型感知架构

- `core/stiffener_profile.py`：单点真理模块，定义四种筋型的参数集、默认值、范围、部件规格和 3D 可视化网格
- 任务解析阶段：从用户自然语言中提取筋型关键词（如"帽型"→`HAT`、"板式"→`BLADE`），并锁定到任务契约中
- 候选生成阶段：DOE 采样按筋型选择参数维度（BLADE 为 6 维、T/L 为 8 维、HAT 为 10 维），规则检查按筋型验证必需参数
- ABAQUS 建模阶段：按筋型调用对应的装配函数（`_assemble_blade` / `_assemble_t` / `_assemble_hat` / `_assemble_l`），HAT 型自动计算斜腹板角度与长度
- 3D 可视化阶段：按筋型生成对应的渲染网格（T/L 用 Box 基元、BLADE 无翼缘、HAT 用 PolyData 斜腹板面片）

### 4.5 对话式 GUI

- 左侧为聊天主区，支持自然语言输入和流程事件回显
- 右侧包含候选方案、ABAQUS 结果、知识库、日志等标签页
- 主流程内置 3 个确认节点：DNN 初筛前、FEM 校核前、报告导出前
- `core/conversation_flow.py` 提供 UI 无关的对话流程控制器，GUI 只是其中一种承载方式

## 5. 仓库结构

```text
CSDM/
├─ agents/              # 六大智能体
├─ abaqus/              # ABAQUS 模板、建模脚本、结果提取与运行工具
├─ config/              # YAML 配置
├─ core/                # 路径、配置、契约、筋型定义、RAG、LLM、DOE、代理模型等公共能力
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

## 6. 运行环境

### 6.1 约定环境

- Windows 11
- Anaconda
- Python 3.9
- Conda 环境名：`GPT`
- 推荐统一执行器：`D:/anaconda3/envs/GPT/python.exe`
- ABAQUS 2023

### 6.2 创建环境

```powershell
conda env create -f environment.yml
```

### 6.3 LLM 配置

`config/llm_config.yaml` 当前默认启用的提供方为 `remote_vllm`：

- 远程模型：`gpt-5.4`
- 本地备选：`local_ollama`（`qwen2.5:7b`）
- 云端备选：`ollama_cloud`

如需切换到云端兼容接口，可在 `.env` 中配置密钥：

```text
OLLAMA_API_KEY=your_ollama_api_key
```

## 7. 快速开始

### 7.1 环境自检

```powershell
D:/anaconda3/envs/GPT/python.exe scripts/check_env.py
```

### 7.2 启动图形界面

```powershell
D:/anaconda3/envs/GPT/python.exe main.py
```

### 7.3 运行测试

```powershell
D:/anaconda3/envs/GPT/python.exe -m pytest tests -q
```

### 7.4 强制 mock 模式回归

```powershell
$env:CSDM_USE_MOCK_ABAQUS = "1"
D:/anaconda3/envs/GPT/python.exe -m pytest tests -q
```

### 7.5 文献库导入

```powershell
D:/anaconda3/envs/GPT/python.exe scripts/ingest_literature.py --seed --query-group composite_basics --max-results 20
```

也可以直接指定检索词：

```powershell
D:/anaconda3/envs/GPT/python.exe scripts/ingest_literature.py --query "stiffened composite panel buckling compression" --max-results 20
```

如需同步下载 OpenAlex 标注的开放获取 PDF，并在本地解析全文：

```powershell
D:/anaconda3/envs/GPT/python.exe scripts/ingest_literature.py --seed --query-group buckling_and_panels --max-results 20 --download-oa-pdfs --parse-pdfs
```

默认解析后端是 `pymupdf`，只能抽取文字层。需要公式 LaTeX、表格结构和图片文件时，安装 MinerU 后使用：

```powershell
D:/anaconda3/envs/GPT/python.exe scripts/ingest_literature.py --parse-existing-pdfs --parse-backend mineru --ocr --force-pdfs
```

对扫描版或公式密集型论文，可安装 Nougat 后使用：

```powershell
D:/anaconda3/envs/GPT/python.exe scripts/ingest_literature.py --parse-existing-pdfs --parse-backend nougat --force-pdfs
```

如需导入已通过学校账号合法下载到本地的 PDF，请先把文件集中放入一个目录，再执行：

```powershell
D:/anaconda3/envs/GPT/python.exe scripts/ingest_literature.py --import-pdf-dir D:/Project/VS_Code/CSDM/reference/authorized_pdfs --parse-pdfs --parse-backend mineru --ocr
```

脚本不保存学校账号，也不模拟出版社登录；授权 PDF 通过本地目录导入后进入 `knowledge/literature/imports/pdfs/`、`knowledge/literature/imports/texts/` 和文献向量库，与自动下载的开放获取 PDF 分开存放。

### 7.6 文献记录重建向量索引

```powershell
D:/anaconda3/envs/GPT/python.exe scripts/ingest_literature.py --reindex
```

## 8. 关键配置文件

| 文件 | 作用 |
| --- | --- |
| `config/app_config.yaml` | 路径、ABAQUS、pipeline、RAG、case_memory、literature、报告输出等全局配置 |
| `config/llm_config.yaml` | 本地 / 远程 / 云端 LLM 提供方与模型配置 |
| `config/material_db.yaml` | 材料数据库 |
| `config/param_ranges.yaml` | 设计变量范围、铺层模板、规则检查阈值（按筋型分段） |
| `config/literature_queries.yaml` | 文献检索主题分组 |

## 9. 常用脚本

| 命令 | 说明 |
| --- | --- |
| `D:/anaconda3/envs/GPT/python.exe scripts/check_env.py` | 检查 Python、ABAQUS、LLM 连接与核心依赖是否可用 |
| `D:/anaconda3/envs/GPT/python.exe scripts/train_screener.py` | 依据案例库重训 `SCREENER` 代理模型 |
| `D:/anaconda3/envs/GPT/python.exe scripts/migrate_contracts.py --retrain-surrogate` | 迁移案例、IO 与工件契约，重建案例记忆索引并可选重训模型 |
| `D:/anaconda3/envs/GPT/python.exe scripts/migrate_contracts.py --case-memory-only` | 仅重建 `csdm_case_memory` 案例记忆集合 |
| `D:/anaconda3/envs/GPT/python.exe scripts/rebuild_abaqus_artifacts.py --limit 10 --workers 1` | 重建缺失的 ABAQUS 工件 |
| `D:/anaconda3/envs/GPT/python.exe scripts/build_initial_cases.py` | 批量生成/补齐初始案例集 |
| `D:/anaconda3/envs/GPT/python.exe scripts/clean_debug_artifacts.py` | 清理 `__pycache__`、`.pytest_cache` 与 ABAQUS 临时文件 |
| `D:/anaconda3/envs/GPT/python.exe scripts/ingest_literature.py --seed ...` | 导入文献记录并写入文献向量库 |
| `D:/anaconda3/envs/GPT/python.exe scripts/test_all_types_abaqus.py` | 对四种筋型各运行一次 ABAQUS 屈曲分析，验证参数化建模 |

## 10. 标识与关联

- `task_id`：会话任务编号，格式为 `TASK_<n>`，用于当前交互过程中的任务摘要与报告展示
- `candidate_id`：设计候选标识，候选阶段使用 `TMP_<n>`，进入 FEM 后切换为正式样本编号 `C<n>`
- `case_id`：归档案例标识，对应 `data/cases/` 与 `knowledge/case_library/` 中的案例记录

`data/cases/`、`data/io/`、`data/abaqus_runs/` 与 `knowledge/case_library/` 构成主数据层，围绕 `candidate_id` 与 `case_id` 组织。会话任务编号只存在于运行时任务对象、界面摘要与报告展示中，不参与主数据编号分配与关联。

## 11. 数据与知识资产

### 11.1 评估档案与正式案例

- `data/cases/`：所有已校核样本的评估档案
- `knowledge/case_library/`：仅保留当前规则下可进入正式知识库的案例
- `knowledge/chroma_db/`：Chroma 向量库目录，包含案例记忆集合与文献集合；案例 JSON 仍以 `data/cases/` 和 `knowledge/case_library/` 为事实源

### 11.2 文献知识库

- `knowledge/literature/raw/`：原始 API 返回
- `knowledge/literature/records/`：标准化文献记录
- `knowledge/literature/pdfs/`：自动下载的开放获取 PDF
- `knowledge/literature/texts/`：自动下载 PDF 解析后的纯文本
- `knowledge/literature/markdown/`：自动下载 PDF 解析后的 Markdown，MinerU / Nougat 可保留 LaTeX
- `knowledge/literature/json/`：自动下载 PDF 的结构化解析结果
- `knowledge/literature/images/`：MinerU 提取的图片
- `knowledge/literature/imports/pdfs/`：外部导入的授权 PDF
- `knowledge/literature/imports/texts/`：外部导入 PDF 解析后的纯文本
- `knowledge/literature/imports/markdown/`：外部导入 PDF 的 Markdown 解析结果
- `knowledge/literature/imports/json/`：外部导入 PDF 的结构化解析结果
- `knowledge/literature/imports/images/`：外部导入 PDF 中提取的图片
- `knowledge/literature/manifests/`：最近一次 ingestion 摘要

当前文献链路基于 `OpenAlex` 实现 metadata + abstract ingestion，并支持开放获取 PDF 下载、授权 PDF 本地导入、PyMuPDF 文字层解析、MinerU Markdown/JSON/图片解析、Nougat 公式 OCR Markdown 解析、全文分块索引与任务时检索。

## 12. 回归验证

当前推荐使用如下命令进行统一验证：

```powershell
D:/anaconda3/envs/GPT/python.exe scripts/check_env.py
D:/anaconda3/envs/GPT/python.exe -m pytest tests -q
```

## 13. 已知边界

- 真实 ABAQUS 求解依赖本机许可证与命令行环境
- 文献知识库链路已落地，但仓库内当前尚未保留实际抓取数据
- 代理模型精度仍处于原型阶段，后续仍需持续积累真实案例
- 代理模型特征向量为固定 22 维（含 8 个几何参数），HAT 型的 cap_width/cap_thickness 暂未纳入特征向量

## 14. 相关文档

- [项目全流程梳理与开发指南](docs/CSDM_项目全流程梳理与开发指南.md)
- [接口约定](docs/接口约定.md)
