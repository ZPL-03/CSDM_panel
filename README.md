# CSDM

复合材料加筋壁板智能设计系统（Composite Stiffened-panel Design Manager, CSDM）是一个面向复合材料 T 形加筋壁板的多智能体设计原型系统，目标是把“自然语言需求 -> 候选方案生成 -> 代理模型筛选 -> ABAQUS 高保真校核 -> 知识回流 -> 报告输出”这条工程链路尽量自动化。

当前仓库已经不是概念草图，而是一个可运行的 Windows 原型工程：前端为 PyQt6 对话式 GUI，后端由 6 个专职智能体协同执行，支持本地/云端 LLM、RAG 检索、DOE 候选补充、随机森林/MLP 代理模型筛选，以及 ABAQUS 实算或 mock 求解双路径。

## 1. 项目目标

本项目面向复合材料加筋壁板智能设计场景，核心任务是将用户自然语言需求转化为结构化设计任务，并自动完成：

- 任务解析与约束归一化
- 候选方案生成（LLM + RAG 案例迁移 + DOE）
- 代理模型快速初筛
- ABAQUS 建模、求解、后处理与失败重试
- 结果入库与知识库更新
- Markdown / PDF 工程报告生成

## 2. 当前完成度

截至 2026-04-10，本地整理与验证结果如下：

- 已落地 6 个核心智能体：`ORCHESTRATOR`、`CANDIDATE_GEN`、`SCREENER`、`FEM_AGENT`、`KNOWLEDGE_AGENT`、`REPORT_GEN`
- 支持 3 类工况：`axial_compression`、`in_plane_shear`、`compression_shear`
- 支持 3 类边界：`SSSS`、`CCCC`、`SSCC`
- 已内置代理模型权重：`rf + mlp`
- 当前案例库规模：`data/cases/` 中 328 个案例，`knowledge/case_library/` 中 41 个正式知识案例
- `models/surrogate_metrics.json` 显示当前选中模型为 `rf`，训练样本数 317，RF `MAPE = 0.1216`
- 在 `GPT / Python 3.9.25` 环境下执行 `python -m pytest tests -q`，结果为 `25 passed, 7 warnings`

## 3. 系统架构

### 3.1 多智能体分工

| 智能体 | 标识符 | 主要职责 |
| --- | --- | --- |
| 主控智能体 | `ORCHESTRATOR` | 解析用户需求、维护状态机、串联全流程 |
| 候选生成智能体 | `CANDIDATE_GEN` | 基于 LLM、RAG 与 DOE 生成候选设计 |
| 快速筛选智能体 | `SCREENER` | 使用代理模型预测 BLF / 重量并完成 Top-K 排序 |
| 求解智能体 | `FEM_AGENT` | 驱动 ABAQUS 建模、求解、结果提取与自动重试 |
| 知识回流智能体 | `KNOWLEDGE_AGENT` | 将结果写回案例库 / 向量库，并触发代理模型重训 |
| 报告生成智能体 | `REPORT_GEN` | 汇总校核结果并导出 Markdown / PDF 报告 |

### 3.2 主流程

```text
用户自然语言需求
        ↓
ORCHESTRATOR：任务解析 / 状态机控制
        ↓
CANDIDATE_GEN：候选生成
        ↓
SCREENER：代理模型初筛
        ↓
FEM_AGENT：ABAQUS 校核与自动重试
        ↓
KNOWLEDGE_AGENT：结果入库 / RAG 更新 / 模型重训
        ↓
REPORT_GEN：工程报告输出
```

### 3.3 GUI 交互形态

- 左侧为对话主区，支持自然语言输入与多智能体状态流
- 右侧为标签页视图，包含候选方案、ABAQUS 结果、知识库、日志
- 支持候选参数摘要、筛选结果、BLF 结果、PyVista 三维视图和报告打开入口
- 对话流程内置 3 个确认节点：DNN 初筛前、FEM 校核前、报告导出前

## 4. 仓库结构

```text
CSDM/
├─ agents/              # 六大智能体
├─ abaqus/              # ABAQUS 模板、建模脚本、结果提取与运行工具
├─ config/              # YAML 配置
├─ core/                # 路径、配置、Schema、RAG、LLM、DOE、代理模型等公共能力
├─ data/                # 案例、运行数据、报告输出
├─ docs/                # 设计说明与接口约定
├─ gui/                 # PyQt6 对话式界面与可视化组件
├─ knowledge/           # 正式知识案例与 ChromaDB
├─ models/              # 代理模型文件与指标
├─ schemas/             # 任务 / 候选 / 结果 / 案例 JSON Schema
├─ scripts/             # 自检、迁移、训练、清理、重建脚本
├─ tests/               # 自动化测试
├─ environment.yml      # 推荐 Conda 环境清单
└─ main.py              # GUI 启动入口
```

## 5. 运行环境

### 5.1 推荐环境

- Windows 11
- Anaconda
- Python 3.9（项目约定环境名：`GPT`）
- ABAQUS 2023
- CUDA 11.8（若使用 GPU 版 PyTorch）
- VSCode / PyCharm

### 5.2 一键创建环境

如果需要从零创建环境，优先使用仓库内的 `environment.yml`：

```powershell
conda env create -f environment.yml
conda activate GPT
```

如果你本机已经有同名环境，也可以直接：

```powershell
conda activate GPT
```

### 5.3 LLM 配置

当前 `config/llm_config.yaml` 默认激活的是本地 Ollama：

- `active_provider: local_ollama`
- 本地默认模型：`qwen2.5:7b`
- 云端备选：`ollama_cloud`

若切换到云端 Ollama API，请在 `.env` 中配置：

```powershell
OLLAMA_API_KEY=your_ollama_api_key
```

仓库提供了模板文件：

```text
.env.example
```

### 5.4 中文终端建议

PowerShell 若出现中文乱码，先执行：

```powershell
chcp 65001
```

## 6. 快速开始

### 6.1 环境自检

```powershell
conda activate GPT
python scripts/check_env.py
```

### 6.2 启动图形界面

```powershell
conda activate GPT
python main.py
```

### 6.3 运行测试

```powershell
conda activate GPT
python -m pytest tests -q
```

### 6.4 无 ABAQUS 时使用 mock 流程

项目已支持在找不到 ABAQUS 时回退到 mock 模式；如果需要强制 mock，可在 PowerShell 中设置：

```powershell
$env:CSDM_USE_MOCK_ABAQUS = "1"
python -m pytest tests -q
```

## 7. 关键配置文件

| 文件 | 作用 |
| --- | --- |
| `config/app_config.yaml` | 路径、ABAQUS、pipeline、RAG、报告输出等全局配置 |
| `config/llm_config.yaml` | 本地 / 云端 LLM 提供方与模型配置 |
| `config/material_db.yaml` | 材料数据库 |
| `config/param_ranges.yaml` | 设计变量范围、铺层模板、规则检查阈值 |

## 8. 常用脚本

| 命令 | 说明 |
| --- | --- |
| `python scripts/check_env.py` | 检查 Python、ABAQUS、Ollama 与核心依赖是否可用 |
| `python scripts/train_screener.py` | 依据案例库重训 `SCREENER` 代理模型 |
| `python scripts/migrate_contracts.py --retrain-surrogate` | 迁移历史任务/案例契约并可选重训模型 |
| `python scripts/rebuild_abaqus_artifacts.py --limit 10 --workers 1` | 重建缺失的 ABAQUS 工件 |
| `python scripts/build_initial_cases.py` | 批量生成/补齐初始案例集 |
| `python scripts/restore_tasks_from_cases.py` | 从案例档案回填任务台账 |
| `python scripts/clean_debug_artifacts.py` | 清理 `__pycache__`、`.pytest_cache` 与 ABAQUS 临时文件 |

## 9. 数据与版本管理策略

为方便公开发布，本仓库采用“源码与轻量数据入库，重型运行产物忽略”的策略：

### 9.1 版本库保留

- 源码、配置、Schema、测试、文档
- 已训练代理模型：`models/`
- 轻量案例档案：`data/cases/`
- 正式知识案例：`knowledge/case_library/`

### 9.2 默认忽略

- `data/abaqus_runs/` 下的 `.inp / .odb / mode JSON` 工件
- `data/io/` 运行时输入输出 JSON
- `knowledge/chroma_db/` 本地向量索引
- `data/results/latest_report.*` 等导出结果
- `__pycache__/`、`.pytest_cache/`、`*.pyc`
- ABAQUS 临时文件：`*.lck`、`*.023`、`*.com`、`*.jnl`、`*.sta`、`*.prt`、`*.sim`、`*.rec`

如果你需要重新生成大型工件，可使用 `FEM_AGENT` 正常运行，或调用 `scripts/rebuild_abaqus_artifacts.py` 重建。

## 10. 已验证状态

本次整理过程中已在本机 `GPT` 环境完成如下验证：

```powershell
D:\anaconda3\envs\GPT\python.exe scripts/check_env.py
D:\anaconda3\envs\GPT\python.exe -m pytest tests -q
```

结果：

- 环境自检全部通过
- `25 passed, 7 warnings`
- 当前警告均来自 PyQt6 `sipPyTypeDict()` 弃用提示，不影响主流程运行

## 11. 已知限制

- 当前代理模型精度仍处于原型阶段，距离高精度工程代理模型还有提升空间
- 真实 ABAQUS 求解依赖本机许可证、命令行环境与 Windows 路径配置
- 当前 README 与仓库结构已按“可公开发布的原型仓库”整理，但不等同于商业级交付系统

## 12. 相关文档

- [项目全流程梳理与开发指南](docs/CSDM_项目全流程梳理与开发指南.md)
- [接口约定](docs/接口约定.md)

## 13. 下一步计划

- 持续积累真实校核案例，提升代理模型质量
- 为 `README` 补充 GUI 截图和示例对话
- 将关键设计案例整理为公开 benchmark 数据集
