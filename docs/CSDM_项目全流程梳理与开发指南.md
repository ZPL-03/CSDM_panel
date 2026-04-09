# CSDM 项目全流程梳理与开发指南

最后梳理日期：2026-04-03

本文档不是“理想蓝图”，而是基于当前仓库实际代码、配置、脚本、测试、样例数据以及《复合材料加筋壁板智能设计系统.docx》方案书整理出来的“现状版项目说明书”。

它的目标是帮你回答 4 个问题：

1. 这个项目到底想做什么。
2. 当前代码已经做到什么程度。
3. 每个流程在代码里具体怎么实现。
4. 如果你从零开始，应该按什么顺序吃透并继续开发。

---

当前集成状态可以先记住几件事：

- 主交互已经是“对话主导 + 三个关键确认节点”
- 任务解析已经是“规则抽取 + 本地 LLM 结构化解析 + Schema 校验 + 归一化回填”
- RAG 默认走本地 `sentence-transformers`，模型配置为 `BAAI/bge-m3`
- 报告已经是“结构化摘要 + LLM 工程解释”的混合模式
- 候选池总数和 DNN 初筛保留数都可以通过自然语言分别指定
- 用户未指定材料时，候选会在材料库里自动分散取样，不再固定只给同一套材料
- FEM mock 与真实 ABAQUS 入口都已统一支持压缩、剪切、压剪组合，以及 `SSSS / CCCC / SSCC`
- 历史任务、案例、IO 工件和 Chroma 索引都已迁移到当前契约

---

## 1. 一句话先说清楚这个项目

CSDM 是一个“复合材料 T 形加筋壁板智能设计系统”。

它的目标是把下面这条工程链路尽可能自动化：

自然语言需求 -> 结构化任务 -> 候选设计生成 -> 代理模型快速筛选 -> ABAQUS 高保真校核 -> 结果入库 -> 报告输出

项目在架构上采用了多智能体分工协作的思路，代码里已经落地了 6 个智能体类：

- `ORCHESTRATOR`：主控调度
- `CANDIDATE_GEN`：候选生成
- `SCREENER`：代理模型筛选
- `FEM_AGENT`：ABAQUS 建模/求解/重试
- `KNOWLEDGE_AGENT`：知识回流与重训练
- `REPORT_GEN`：报告生成

---

## 2. 当前仓库的真实完成度

### 2.1 按方案书阶段来判断

结合《复合材料加筋壁板智能设计系统.docx》中的“四阶段实施路线”，当前仓库已经不是空工程，而是一个“可运行的原型系统”，大致处于：

- 阶段一：已完成
- 阶段二：已完成基础版，真实 ABAQUS 脚本和结果提取已经存在
- 阶段三：已完成原型贯通，GUI、候选生成、筛选、校核、入库、报告已经串通
- 阶段四：已部分落地，知识回流和代理模型重训已实现，但很多高级能力仍是简化版

### 2.2 当前仓库里的直接证据

截至 2026-04-03，本仓库内已有：

- `data/tasks/`：36 个任务
- `data/cases/`：322 个案例档案
- `knowledge/case_library/`：41 个正式知识库案例
- `data/abaqus_runs/`：322 个正式样本工件目录
- `models/surrogate_metrics.json`：已训练好的代理模型指标
- `data/results/contract_migration_summary.json`：当前契约迁移与重建摘要

当前代理模型信息：

- 当前选中的模型：`rf`
- 训练样本数：`317`
- RF 的 MAPE：`0.1216`
- MLP 的 MAPE：`0.1457`

这意味着：

- 项目已经积累了一批真实或半真实的样本数据
- 代理模型已经不是“空壳”
- 但模型精度距离方案书里“200+ 条数据后 MAPE < 5%”的目标还有差距

### 2.3 当前测试状态

在项目约定环境 `GPT` 中执行：

```bat
python -m pytest tests -q
```

结果是：

- `25 passed`
- 有 7 个 `PyQt6 sipPyTypeDict()` 相关的弃用警告
- 没有功能性失败

这说明主链路原型目前是稳定的。

---

## 3. 一个必须先记住的现实：环境要用 `GPT`，不要直接用当前 base

我在梳理仓库时发现一个非常关键的点：

- 当前默认 `python` 指向的是 `D:\anaconda3\python.exe`
- 版本是 `Python 3.13.5`
- 但项目约定环境是 `GPT`
- `GPT` 环境版本是 `Python 3.9.25`

而且两个环境的依赖状态完全不同：

- 在 `base / Python 3.13.5` 中，`openai`、`chromadb`、`torch`、`reportlab`、`pyvista` 等依赖缺失
- 在 `GPT / Python 3.9.25` 中，项目所需核心依赖全部通过

所以你后续开发时，第一原则就是：

```bat
conda activate GPT
```

建议先做一次环境自检：

```bat
conda activate GPT
python scripts/check_env.py
```

如果你在 PowerShell 里看到中文乱码，先执行：

```powershell
chcp 65001
```

---

## 4. 项目顶层目录怎么理解

建议先建立“目录脑图”，再去看细节。

```text
CSDM/
├─ main.py                    # GUI 入口
├─ AGENTS.md                  # 项目协作规则
├─ README.md                  # 简要说明
├─ config/                    # YAML 配置
├─ schemas/                   # JSON Schema
├─ core/                      # 公共底层能力
├─ agents/                    # 六大智能体
├─ abaqus/                    # ABAQUS 建模/提取脚本
├─ gui/                       # PyQt6 图形界面
├─ scripts/                   # 数据构建、清理、训练、自检
├─ data/                      # 任务、IO、案例、工件、报告
├─ knowledge/                 # ChromaDB 与正式案例库
├─ models/                    # 代理模型权重与指标
└─ tests/                     # 自动化测试
```

### 4.1 `config/`

放“系统参数”，不放逻辑。

- `app_config.yaml`：路径、ABAQUS 参数、pipeline 参数、报告输出路径
- `llm_config.yaml`：LLM 提供方、本地 Ollama / 云端配置
- `material_db.yaml`：材料数据库
- `param_ranges.yaml`：几何参数范围、铺层模板、规则检查阈值

### 4.2 `schemas/`

放系统的“数据契约”，用来保证模块之间传 JSON 时格式一致。

- `task.schema.json`
- `candidate.schema.json`
- `abaqus_result.schema.json`
- `case_record.schema.json`

### 4.3 `core/`

放所有智能体都会依赖的通用能力：

- 路径管理
- 配置加载
- JSON 读写
- Schema 校验
- 编号分配
- 规则检查
- DOE 采样
- LLM 接口
- RAG 接口
- 代理模型训练与预测

### 4.4 `agents/`

放业务逻辑的“角色层”。

它们基本遵守统一接口：

```python
run(input: dict) -> dict
```

### 4.5 `abaqus/`

这是整个系统最工程化、也最关键的一层。

里面有两类脚本：

- 轻量测试入口：`build_panel.py`、`extract_blf.py`
- 真实运行入口：`runtime_build_panel.py`、`runtime_extract_blf.py`

注意这个区别非常重要：

- `build_panel.py` 更像是单元测试/联调辅助入口
- 真正被 `FEM_AGENT` 通过 Jinja2 模板调用的是 `runtime_build_panel.py`

### 4.6 `gui/`

PyQt6 图形界面分成四块：

- 对话消息
- 候选方案展示
- ABAQUS 结果展示
- 知识库/日志展示

同时还集成了 `PyVista + pyvistaqt` 做三维交互显示。

### 4.7 `data/`

这是项目运行后的“事实层”。

- `data/tasks/`：任务台账
- `data/io/`：主程序和 ABAQUS 的 JSON 通信层
- `data/cases/`：所有校核案例
- `data/abaqus_runs/`：每个正式样本的工件目录
- `data/results/`：报告输出

### 4.8 `knowledge/`

这是“知识层”。

- `knowledge/case_library/`：正式知识库案例
- `knowledge/chroma_db/`：ChromaDB 向量库

### 4.9 `models/`

存代理模型文件：

- `surrogate_rf.joblib`
- `surrogate_scaler.joblib`
- `surrogate_mlp.pt`
- `surrogate_metrics.json`

### 4.10 `scripts/`

放一次性或批处理型工具。

这些脚本对你后续自己扩数据、重训模型、重建工件非常有用。

---

## 5. 整个系统从头到尾怎么跑

下面这张图是你最需要记住的主线。

```text
用户自然语言输入
  -> ORCHESTRATOR 解析为 task
  -> CANDIDATE_GEN 生成候选
     -> LLM 候选
     -> 历史案例迁移候选
     -> DOE 候选
  -> RuleChecker 过滤非法方案
  -> SCREENER 用代理模型预测 BLF 并排序
  -> FEM_AGENT 对选中样本做 ABAQUS 校核
     -> 生成脚本
     -> 写 input JSON
     -> 调用 Abaqus
     -> 读 result JSON
     -> 失败则诊断并重试
  -> KNOWLEDGE_AGENT 保存案例并更新知识库/模型
  -> REPORT_GEN 生成 Markdown/PDF 报告
  -> GUI 更新候选页、结果页、知识库页和日志页
```

再从文件角度看一次：

```text
自然语言
  -> data/tasks/TASK_x.json
  -> 会话内 TMP_x 候选
  -> 正式样本 Cx
  -> data/io/input_Cx.json
  -> data/abaqus_runs/Cx/*
  -> data/io/result_Cx.json
  -> data/cases/CASE_x.json
  -> knowledge/case_library/CASE_x.json（仅通过样本）
  -> knowledge/chroma_db/*
  -> data/results/latest_report.md / pdf
```

---

## 6. 入口程序和 GUI 是怎么工作的

### 6.1 `main.py`

`main.py` 很薄，只做三件事：

1. `ensure_project_dirs()` 确保目录存在
2. 创建 `QApplication`
3. 打开 `MainWindow`

所以真正的桌面逻辑都在 `gui/main_window.py`。

### 6.2 `gui/main_window.py`

这是 GUI 的主控制器。

它定义了一个 `PipelineSession`，里面记录当前会话状态：

- 当前任务 `task`
- 用户原始输入 `instruction`
- 初始候选 `candidates`
- 筛选后的候选 `screened_candidates`
- 已有校核结果 `results_by_session_id`
- 最终报告 `report`

当前界面已经分成两层：

1. 主路径：对话式自动流程
2. 调试路径：保留分阶段按钮入口

主路径的真实交互是：

1. 用户输入一句自然语言需求
2. 系统自动解析任务并生成初始候选
3. 系统询问是否进行 DNN 初筛
4. 系统询问是否进行 FEM 校核
5. 系统询问是否导出报告

也就是说，现在 GUI 的主操作方式已经不是旧的“手动点五个阶段按钮”，而是“对话驱动 + 关键节点确认”。调试按钮仍然保留，但已经从主路径退到辅助入口。

### 6.3 `PipelineWorker`

GUI 不直接在主线程跑智能体，而是把工作交给 `PipelineWorker + QThread`。

当前 `PipelineWorker` 同时支持两套动作：

- 对话主流程：
  - `conversation_start`
  - `conversation_continue`
- 调试入口：
  - `generate`
  - `screen`
  - `evaluate`
  - `report`

它通过 `progress_callback` 把智能体的进度消息回传到 GUI：

- 左侧聊天区显示
- 右侧日志页显示

所以你在界面里看到的“[ORCHESTRATOR] ... / [FEM_AGENT] ...”就是这样来的。

### 6.4 当前 GUI 的特点

它现在已经进入“对话式工程原型”阶段：

- 优点：
  - 一句话可以拉起主流程
  - 右侧候选/结果/知识库/日志会跟随事件同步刷新
  - DNN、FEM、报告三个关键节点可确认，适合工程审查
  - 可以在对话里分别指定“总候选数量”和“DNN 初筛保留数量”
- 当前仍需注意的限制：
  - 主进度仍以结构化阶段事件推进，并叠加 1 到 2 句自然助手说明，不是 token 级打字机式流式回复
- 前端仍然是 PyQt6，当前只把流程控制层做到了 UI 无关
  - 调试按钮仍存在，这是刻意保留的工程兜底入口

---

## 7. 六个智能体分别怎么工作

---

## 7.1 `ORCHESTRATOR` 主控智能体

代码位置：`agents/orchestrator.py`

### 它的职责

- 把用户输入转成任务 JSON
- 调度候选生成、筛选、校核、知识回流、报告生成
- 负责编号提升：`TMP_x -> Cx`

### 当前实现的真实情况

当前 `ORCHESTRATOR` 已经不是旧版的“单纯规则模板解析器”了。

现在 `parse_instruction()` 的真实链路是：

1. 规则抽取：
   - 提取 `Nx` / `Nxy`
   - 识别应用场景
   - 识别 `SSSS / CCCC / SSCC`
   - 识别目标和几何包线提示
   - 提取候选池总数、DNN 初筛数量和显式材料指定
2. 本地 LLM 结构化补全：
   - 默认走本地 Ollama 接口
   - 只补充它能确定的字段
3. Schema 校验
4. 归一化回填：
   - `load_conditions` 统一为结构化多工况格式
   - `boundary_conditions` 统一为结构化边界格式
   - 旧“单轴压缩 + 四边简支”写法自动兼容映射
   - 对候选数量、初筛数量和显式材料等高置信字段，规则结果优先，不允许被 LLM 回填覆盖

所以现在它已经是“规则 + LLM 混合解析器”，而不再是单纯正则适配层。

### 它的关键方法

- `parse_instruction(text)`：生成并持久化 `task`
- `generate_candidates(task)`：调用 `CANDIDATE_GEN`
- `screen_candidates(task, candidates)`：调用 `SCREENER`
- `evaluate_candidate(task, candidate)`：提升为正式编号后调用 `FEM_AGENT`
- `generate_report(task, results, candidates)`：调用 `REPORT_GEN`
- `run(user_instruction)`：命令行式地一口气跑完整流程

### 一个重要细节

候选生成阶段只用临时编号：

- `TMP_1`
- `TMP_2`

只有到了 FEM 阶段才会分配正式编号：

- `C1`
- `C2`

这个设计是对的，因为它避免“光生成候选就污染正式编号体系”。

---

## 7.2 `CANDIDATE_GEN` 候选方案生成智能体

代码位置：`agents/candidate_gen.py`

这是整个系统里“AI 感”最强的模块之一。

### 它的三路候选来源

方案书里定义的是三路融合，代码里也确实做了：

1. `LLM`
2. `CASE_TRANSFER`
3. `DOE`

### 实际执行顺序

#### 第一步：初始化底层能力

初始化时它会准备：

- `DOESampler`
- `RuleChecker`
- `RAGEngine`
- `LLMBackend`

如果 `LLMBackend` 初始化失败，不会让整个系统崩掉，而是自动退化为：

- 案例迁移
- DOE

这说明系统对 LLM 不可用的情况做了兜底。

#### 第二步：从知识库里找相似案例

`_successful_cases()` 会从 `RAGEngine.retrieve()` 返回结果里筛选：

- 只保留 `abaqus_results.status == success`
- 要有 `BLF_global`
- 必须含 `design`

注意这里还有一个更关键的事实：

- `KnowledgeAgent` 只会把“成功且通过”的案例写入正式知识库 `knowledge/case_library/`
- `RAGEngine` 的主集合也只在正式知识库入库时更新

所以当前 RAG 检索出来的案例，本质上是“通过案例”，不是所有成功案例。

#### 第三步：构造 LLM Prompt

`_build_prompt()` 会把下面几部分拼进 prompt：

- 当前任务 JSON
- 检索到的参考案例
- 输出格式要求
- 几何字段要求
- 铺层字段要求

#### 第四步：解析和清洗 LLM 输出

当前实现做了比较细的容错：

- 能接受纯 dict
- 能接受字符串形式 JSON
- 能接受 list 形式铺层
- 能从 `candidates/items/results/data/design` 多种包装结构中取出候选

这一点从 `tests/test_candidate_gen.py` 也能看出来。

#### 第五步：做案例迁移候选

`_case_transfer_candidates()` 会把历史通过案例的 `design` 取出来，再重新规范化为当前任务候选。

注意：

- 这里不是复杂的“参数扰动迁移算法”
- 更像是“历史优秀方案直接复用/轻度映射”

当前材料策略也已经补齐：

- 如果用户在自然语言里明确指定材料，案例迁移候选会固定沿用该材料
- 如果用户没有指定材料，案例迁移候选允许保留各自历史材料体系，后续再与 DOE 和 LLM 候选一起统一比较

#### 第六步：DOE 采样补充候选

不管前两路是否成功，DOE 都会补齐一批候选。

这保证系统即使没有：

- 可用的 LLM
- 足够的历史案例

仍然可以跑起来。

### 候选规范化做了什么

`_normalize_candidate()` 会把候选统一整理成标准结构，并补默认值：

- 统一几何字段
- 统一铺层字段
- 自动补 `material_system`
- 自动补 `load_conditions`
- 自动补 `boundary_conditions`
- 自动做 `rule_check`
- 自动过 `candidate.schema.json`

这里的 `material_system` 现在不是简单“全部覆盖成任务默认材料”：

- 任务固定材料时，所有候选统一使用该材料
- 任务未固定材料时，LLM、案例迁移和 DOE 候选都允许携带各自材料体系
- DOE 会在 `material_db.yaml` 里按材料库自动分散取样，不再总是只生成同一套材料

### 当前实现的优点

- 有容错
- 有三路兜底
- 不依赖单一来源
- 数据结构非常规整

### 当前实现的限制

- LLM 只参与“候选生成”，不参与任务理解、报告解释
- 案例迁移还比较浅
- Prompt 工程是固定模板，不是可配置策略

---

## 7.3 `SCREENER` 快速筛选智能体

代码位置：`agents/screener.py`

这个模块的作用是：用便宜的代理模型代替昂贵的 ABAQUS，对候选快速预估。

### 输入

- `task`
- `candidates`

### 输出

- 排序后的 Top-K 候选

### 它内部做了三件事

1. 调 `SurrogateModelManager.predict_candidates()` 预测每个候选的 `surrogate_BLF`
2. 用几何和密度估算 `surrogate_weight`
3. 用一个线性打分公式排序

打分公式是：

```text
rank_score = surrogate_BLF - 0.08 * surrogate_weight
```

这说明当前系统采用的是一个非常直接的多目标折中策略：

- 希望 BLF 高
- 希望重量低

### 当前实现的特点

- 非常简单
- 非常稳定
- 适合原型阶段

### 你以后可以改进的方向

- 把线性打分改为 Pareto 排序
- 引入更多约束指标
- 增加不确定性估计

---

## 7.4 `FEM_AGENT` ABAQUS 求解智能体

代码位置：`agents/fem_agent.py`

这是全系统最核心的模块。

如果你要真正吃透这个项目，这个智能体一定要反复读。

### 它到底负责什么

它不是单纯“调用一下 abaqus 命令”，而是完整负责：

1. 生成 ABAQUS 运行脚本
2. 写跨环境通信 JSON
3. 调用 ABAQUS
4. 等待结果 JSON
5. 诊断失败原因
6. 自动修改候选参数
7. 重试

### 关键路径

#### 1. 路径组织

它为每个正式样本 `C<n>` 创建单独运行目录：

```text
data/abaqus_runs/C<n>/
├─ candidate_input.json
├─ build_C<n>.py
├─ C<n>.inp
├─ C<n>.odb
└─ C<n>_mode1.json
```

#### 2. 生成 Jinja2 脚本

`generate_script()` 会把模板：

- `abaqus/templates/t_stiffener_buckle.py.j2`

渲染成：

- `build_C<n>.py`

这个生成脚本会进一步加载：

- `abaqus/runtime_build_panel.py`

所以真实的 Abaqus 建模逻辑不在模板里，而在 `runtime_build_panel.py`。

#### 3. mock 模式与真实模式

`FEMAgent.run()` 支持两种模式：

- mock 模式
- real Abaqus 模式

进入 mock 的条件有三种：

1. 候选本身带 `mock_mode=True`
2. 环境变量 `CSDM_USE_MOCK_ABAQUS=1`
3. 配置允许降级，且系统检测到 `abaqus` 不可用

这就是为什么这个项目即使在没有 Abaqus 的机器上，也能跑通很多测试。

#### 4. 真实模式如何调用 Abaqus

真实模式 `_run_real()` 大致执行：

```text
abaqus cae noGUI=build_Cx.py
```

然后轮询等待：

- `data/io/result_Cx.json`

如果结果文件出现，就说明 Abaqus 侧建模、求解、后处理链路已经跑完。

#### 5. 失败诊断

如果结果文件没出现，它会读取：

- `.msg`
- `.dat`

并通过 `abaqus/job_utils.py` 的 `diagnose_failure()` 归类成：

- `mesh_error`
- `geometry_issue`
- `convergence_fail`
- `blf_negative`
- `process_crash`
- `failed`

#### 6. 自动重试策略

`apply_adjustment()` 是当前智能的核心体现。

不同失败类型会触发不同参数修正：

- `mesh_error`：增大网格尺寸，稍微放宽筋距
- `geometry_issue`：缩小翼缘宽度，降低筋高
- `convergence_fail`：增加模态搜索规模，并把几何拉回更稳妥区间
- `blf_negative`：提高筋高
- 其他：默认清理后重试

这个逻辑和方案书是一致的，而且已经落在代码里。

### 一个非常值得注意的现实

当前仓库中的 Abaqus 工件有两种来源：

- 一部分是 mock 生成的轻量假工件，例如 `data/abaqus_runs/C1/`
- 一部分是真实 Abaqus 生成的较大工件，例如 `data/abaqus_runs/C308/`

这说明仓库里已经混合存在：

- 用于开发联调的样本
- 用于真实校核的样本

### 为什么说这个模块最关键

因为整个系统的可信度最终都来自这里：

- 没有真实 FEM，知识库质量不可靠
- 没有稳定 FEM，训练数据不可靠
- 没有训练数据，代理模型就站不住
- 没有代理模型，候选筛选就失去意义

所以方案书说“仿真第一”，这是对的。

---

## 7.5 `KNOWLEDGE_AGENT` 知识回流智能体

代码位置：`agents/knowledge_agent.py`

它负责把每次 FEM 结果变成系统长期记忆。

### 它做了什么

1. 清洗 task/design/result
2. 生成案例记录 `CASE_x`
3. 所有案例写入 `data/cases/`
4. 只有“成功且通过”的案例写入 `knowledge/case_library/`
5. 把正式案例 upsert 到 `ChromaDB`
6. 在满足条件时触发代理模型重训

### 一个关键设计

这里有“两层案例库”：

- `data/cases/`：全量评估档案
- `knowledge/case_library/`：正式知识库

这意味着系统区分：

- “仿真做过”
- “可作为优秀知识复用”

这是非常合理的。

### 重训触发条件

配置里写的是：

- `min_case_records_for_retrain: 50`

代码的真实逻辑是：

- 成功样本数达到 50 的倍数时重训

也就是：

- 50 条
- 100 条
- 150 条
- ...

### 当前实现的一个重要细节

代理模型训练读取的是 `data/cases/` 中 `status == success` 的样本，不要求 `verdict == 通过`。

也就是说：

- 训练数据包含“通过样本”
- 也包含“未通过但成功求解的样本”

这其实是合理的，因为代理模型学的是 `BLF_global` 回归，不是“是否通过”的二分类。

---

## 7.6 `REPORT_GEN` 报告生成智能体

代码位置：`agents/report_gen.py`

当前 `REPORT_GEN` 已经不再只是“固定模板报告器”。

### 当前实现

- 先构造结构化摘要：
  - 任务摘要
  - 工况说明
  - 边界条件说明
  - DNN 初筛原因
  - 有限元结果与工程诊断
- 再调用本地 LLM 生成工程解释段落
- 如果本地 LLM 不可用，则回退到稳定的规则化中文摘要
- 最后输出 Markdown 和 PDF

输出文件固定为：

- `data/results/latest_report.md`
- `data/results/latest_report.pdf`

### 当前实现的优点

- 结构化部分稳定，可重复
- LLM 只负责高价值解释，不会影响关键数值
- 本地 LLM 不可用时仍能稳定导出中文 PDF

### 当前实现仍然保留的边界

- 现在的 LLM 解释仍是短段落风格，不是长篇技术报告
- 报告已经能比较候选，但还没有做更复杂的多目标 Pareto 解释
- 目前仍以 Markdown/PDF 为主，没有单独的 Web 报告页

---

## 8. 底层核心模块怎么理解

---

## 8.1 `core/paths.py`

它定义了项目所有关键目录常量，并在 `ensure_project_dirs()` 中保证目录存在。

另外还做了一件小事：

- 支持把旧的 `data/artifacts/abaqus_runs` 自动迁移到新的 `data/abaqus_runs`

这说明项目已经经历过一次目录结构演化。

---

## 8.2 `core/config_loader.py`

很简单，但很重要。

它用 `yaml.safe_load` 读取 YAML，并用 `lru_cache` 做缓存。

优点是：

- 代码里不用反复读磁盘
- 配置访问写法统一

---

## 8.3 `core/io_utils.py`

统一的 JSON 和文本读写层。

好处是：

- 所有写文件都统一 `utf-8`
- 方便未来统一加日志、异常处理、原子写入

---

## 8.4 `core/schema_validator.py`

用 `jsonschema` 做数据校验。

只要 payload 不符合 schema，就直接抛 `SchemaValidationError`。

这层是整个项目“接口清晰”的重要基础。

---

## 8.5 `core/id_utils.py`

统一管理四类编号：

- `TASK_n`
- `TMP_n`
- `C<n>`
- `CASE_n`

它不是用数据库自增，而是通过扫描现有目录推断最大编号。

优点：

- 纯文件系统项目也能工作
- 不需要额外数据库

代价：

- 未来并发写入时要注意竞争条件

在当前桌面原型里，这种实现完全够用。

---

## 8.6 `core/rule_checker.py`

这是候选进入求解前的第一道“工程可行性门槛”。

### 它检查什么

- 几何参数范围
- 铺层是否对称
- 铺层是否平衡
- 各角度比例是否低于最小阈值
- 筋高/筋距比例
- 筋高/蒙皮厚度比例

如果 `strict_solver_window=True`，它还会进一步限制“更适合当前求解器稳定运行的窗口”：

- `skin_thickness_mm` 在更稳的区间
- `pitch_mm` 在更稳的区间
- `stiffener_height_mm` 在更稳的区间
- `flange_width_mm` 在更稳的区间
- `height/pitch` 在更稳的带内

这是一种很实用的工程策略：

- 不是只问“理论上合不合法”
- 还问“这个方案拿去给当前 Abaqus 自动流程跑，稳不稳”

---

## 8.7 `core/doe_sampler.py`

DOE 采样器采用的是轻量 Latin Hypercube Sampling。

### 它做了什么

- 从 `param_ranges.yaml` 读取几何参数范围
- 用 LHS 在多维参数空间均匀采样
- 从铺层模板库中随机抽一个模板
- 自动计算铺层比例
- 用 `RuleChecker` 过滤未通过校验的候选

### 为什么它重要

因为它提供了系统的“保底探索能力”：

- 没 LLM 也能跑
- 没知识库也能跑
- 没历史案例也能积累第一批数据

这正是方案书里 DOE 路线的意义。

---

## 8.8 `core/llm_backend.py`

这是 LLM 的适配层。

### 当前支持两类后端

1. `local_ollama`
2. `ollama_cloud`

### 真实实现方式

- 对本地 Ollama：直接走原生 `/api/chat`
- 对云端 OpenAI 兼容接口：走 `openai.OpenAI()`

### 它做了哪些容错

- 自动解析 JSON 文本
- 自动去掉 ```json 包裹
- 如果 JSON 坏了，会再发一次“JSON 修复”请求

### 当前实现的局限

- 还没有统一 prompt 配置系统
- 还没有多模型路由
- 还没有 token streaming 接入 GUI

---

## 8.9 `core/rag_engine.py`

这个模块非常值得你特别注意，因为它和方案书写法有一点“名义一致、实现简化”的差异。

### 当前真实实现

- 向量数据库：`ChromaDB`
- 文本嵌入：`SentenceTransformer`
- 默认模型名：`BAAI/bge-m3`
- 默认缓存目录：`models/embedding_cache`
- 默认策略：本地优先、项目内缓存优先

### 默认行为现在会优先尝试真实 embedding

当前控制项变成了：

```text
CSDM_USE_HASH_EMBEDDING
```

默认配置下会直接尝试加载 `SentenceTransformer`。

只有在下面两种情况下才会退回 `_hash_embedding()`：

- 显式设置 `CSDM_USE_HASH_EMBEDDING=1`
- 本地模型不可用且允许降级

也就是说，hash embedding 现在已经从“默认模式”变成了“测试/兜底模式”。

### 这意味着什么

当前项目的 RAG 有两种工作状态：

1. 开发/离线状态：哈希嵌入，轻量、稳定、便于测试
2. 真正语义检索状态：真实 embedding 模型

### 与方案书的差异

方案书里写的是：

- BGE-M3
- LangChain

但当前核心链路实际并没有使用：

- `BGE-M3`
- `LangChain`

当前主链路是：

- `chromadb + sentence-transformers + 自写封装`

`langchain` 和 `langchain_community` 目前只出现在环境自检中，不在核心执行路径里。

---

## 8.10 `core/surrogate_model.py`

这是代理模型层的真正实现。

### 输入特征有哪些

共有 18 个特征，包括：

- 几何参数 8 个
- 铺层比例 3 个
- ply 数 1 个
- 载荷 1 个
- 材料参数 5 个

### 支持两种模型

1. `RandomForestRegressor`
2. `MLPRegressor`（PyTorch）

### 训练逻辑

- 对样本做 train/test split
- RF 训练的是 `log1p(y)`
- MLP 训练的是原始值
- 比较两者 `MAPE`
- 选 `MAPE` 更低的模型作为当前主模型

### 预测逻辑

- 如果没有训练指标文件，就全部返回默认值 `1.2`
- 否则读取当前选中模型做推理

这个“没有模型也能退化运行”的思路很符合原型工程需要。

---

## 9. ABAQUS 层到底是怎么落地的

如果你是第一次做“主 Python + Abaqus Python 双环境协同”，这一节一定要搞清楚。

### 9.1 为什么必须双环境分离

项目明确要求：

- 主程序走 Anaconda `GPT`
- Abaqus 脚本走 Abaqus 自带 Python
- 两边不直接 import 对方环境里的第三方包
- 只通过 JSON 通信

这是正确的做法，因为 Abaqus 的 Python 环境通常比较老、比较封闭，和你主项目的 Conda 环境不适合强行混用。

### 9.2 真实建模脚本在哪里

真实建模脚本是：

- `abaqus/runtime_build_panel.py`

它会在 Abaqus 环境里做这些事情：

1. 读取候选 JSON
2. 解析几何和材料
3. 建立蒙皮、腹板、翼缘的 shell 几何
4. 做分区和网格
5. 装配多个筋实例
6. 用 `Tie` 连接筋与蒙皮
7. 施加边界条件
8. 建立屈曲步 `BuckleStep`
9. 施加压缩载荷
10. 提交 Job
11. 等待完成
12. 调 `runtime_extract_blf.py` 提取结果

当前这条真实链路已经统一支持：

- `axial_compression`
- `in_plane_shear`
- `compression_shear`
- `SSSS / CCCC / SSCC`

另外在 T 形筋连接上，当前实现已经把腹板底边与左右翼缘内侧边合并为单次 `Tie`，避免同一批从节点重复作为 secondary 而触发 overconstraint。

### 9.3 结果提取脚本做了什么

`abaqus/runtime_extract_blf.py` 负责：

- 打开 ODB
- 找 `Buckling` step
- 从 frame 描述里解析特征值
- 计算一阶模态位移
- 推断 `BLF_global` / `BLF_local`
- 计算面密度
- 导出模态可视化 JSON

这里有一个非常关键的实现细节：

- 系统会保留原始 `mode_eigenvalues`
- `BLF_global` 不再机械取第一阶模态，而是取“首个正特征值”
- 如果前面存在负特征值模态，会在 `analysis_flags.negative_modes_skipped` 和 `analysis_flags.first_positive_mode_index` 里明确记录

这可以避免压剪组合或较复杂边界下，前几阶出现负特征值时被误判成“求解失败”。

这个可视化 JSON 后面会被 GUI 的 PyVista 直接读取显示。

### 9.4 为什么同时还有 `build_panel.py` 和 `extract_blf.py`

这是给测试和非 Abaqus 环境联调用的轻量入口。

比如测试里可以这样跑：

- `build_panel(..., mock=True)`
- `extract_blf(..., mock=True)`

这样你不必每次单元测试都真正启动 Abaqus。

这是一种非常实用的工程分层：

- 真实生产逻辑在 `runtime_*`
- Python 侧可测试逻辑在外层轻量封装

---

## 10. 数据结构怎么串起来

---

## 10.1 任务 `task`

Schema：`schemas/task.schema.json`

最重要字段有：

- `task_id`
- `application`
- `load_conditions`
- `boundary_conditions`
- `geometry_envelope`
- `material_system`
- `layup_constraints`
- `stiffener_type`
- `design_targets`

当前任务的真实特征是：

- `load_conditions` 已支持单轴压缩、面内剪切、压剪组合
- `boundary_conditions` 已支持 `SSSS / CCCC / SSCC`
- `candidate_generation_preferences.total_candidates` 可单独控制初始候选池规模
- `screening_preferences.top_k_candidates` 可单独控制 DNN 初筛保留数量
- 默认候选池来自 `LLM 4 + CASE_TRANSFER 2 + DOE 4 = 10`
- `material_system` 会记录 `is_user_specified`，用于区分“用户锁定材料”和“系统可自动换材”
- 筋型只支持 `T`

---

## 10.2 候选 `candidate`

Schema：`schemas/candidate.schema.json`

关键字段：

- `candidate_id`
- `source`
- `geometry`
- `layup`
- `material_system`
- `rule_check`
- `surrogate_BLF`
- `surrogate_weight`
- `rank_score`
- `rationale`

编号允许两种形式：

- `TMP_n`
- `C<n>`

当前候选还有一个很重要的运行规则：

- 如果任务明确指定材料，候选材料固定
- 如果任务未指定材料，候选材料允许在材料库中自动切换，再统一送进 `SCREENER` 比较

---

## 10.3 Abaqus 结果 `abaqus_result`

Schema：`schemas/abaqus_result.schema.json`

关键字段：

- `status`
- `retry_count`
- `BLF_global`
- `BLF_local`
- `failure_mode`
- `max_displacement_mm`
- `weight_kg_per_m2`
- `verdict`
- `abaqus_odb`
- `abaqus_inp`
- `visualization_json`
- `error_type`
- `error_log`
- `mode_eigenvalues`
- `effective_mode_eigenvalues`
- `analysis_flags`

---

## 10.4 案例记录 `case_record`

Schema：`schemas/case_record.schema.json`

它把三部分打包到一起：

- `task`
- `design`
- `abaqus_results`

另外还补充：

- `case_id`
- `created_at`
- `verdict`
- `surrogate_BLF_error_pct`
- `fem_agent_retry_count`

所以一个 `CASE_x.json` 本质上就是一份完整的历史实验记录。

---

## 11. 当前 GUI 里的每一页到底在看什么

---

## 11.1 候选方案页

对应：`gui/candidate_widget.py`

会显示：

- 候选来源
- 代理预测 BLF
- 预测重量
- 排名分数
- 真实 BLF（如果做过校核）
- 当前状态和结论

右侧还能看到：

- 候选几何详情
- 材料信息
- 载荷和目标
- 铺层展开
- 规则检查结果
- 三维几何预览

### 一个很好的细节

这个页面同时显示：

- 会话内编号 `TMP_x`
- 正式编号 `C<n>`

所以你不会混淆“候选”和“已归档正式样本”。

---

## 11.2 ABAQUS 结果页

对应：`gui/abaqus_widget.py`

会显示：

- 候选显示名
- 正式编号
- 状态
- 全局/局部 BLF
- 重量
- 结论
- 失效模式

右侧还能看到：

- ODB/INP 路径
- 模态数据 JSON 路径
- 工件目录
- 特征值列表
- 模态云图三维可视化

---

## 11.3 知识库页

对应：`gui/knowledge_widget.py`

会汇总：

- 评估档案总数
- 正式知识库总数
- ODB 工件数
- 可视化数据数
- 代理模型指标
- 最新正式案例列表

它是你观察系统“是否在持续学习”的总览窗口。

---

## 11.4 日志页和聊天页

非常直接：

- `ChatWidget`：显示用户和智能体消息
- `LogWidget`：显示日志

当前两者的差别不大，后续可以进一步分层：

- 聊天页偏用户语义
- 日志页偏技术细节

---

## 12. 项目里的脚本应该怎么用

---

## 12.1 `scripts/check_env.py`

最先运行的脚本。

用途：

- 检查 Python 版本
- 检查 Abaqus
- 检查 PyQt6、openai、chromadb、torch、pyvista、reportlab 等依赖
- 检查 Torch CUDA 是否可用

建议命令：

```bat
conda activate GPT
python scripts/check_env.py
```

---

## 12.2 `scripts/build_initial_cases.py`

这是很重要的数据构建脚本。

用途：

- 批量生成任务
- DOE 采样候选
- 批量调用 FEM
- 入库
- 训练代理模型

它本质上是在帮你“批量造第一批训练数据”。

常见参数：

- `--count`
- `--task-count`
- `--workers`
- `--mock`
- `--full-range`
- `--loads`
- `--reset`

如果你刚开始开发，建议先用 mock 跑通：

```bat
conda activate GPT
python scripts/build_initial_cases.py --count 20 --task-count 4 --workers 2 --mock --reset
```

---

## 12.3 `scripts/train_screener.py`

用途：

- 从现有案例重训代理模型

命令：

```bat
conda activate GPT
python scripts/train_screener.py
```

---

## 12.4 `scripts/rebuild_abaqus_artifacts.py`

用途：

- 对已有案例重建缺失的 ODB/INP 等工件

适合：

- 你改了编号规则
- 你补齐历史数据
- 某些工件目录不完整

---

## 12.5 `scripts/restore_tasks_from_cases.py`

用途：

- 从 `data/cases/` 反向恢复 `data/tasks/`

适合：

- 任务台账丢了
- 但案例还在

---

## 12.6 `scripts/clean_debug_artifacts.py`

用途：

- 清理 `__pycache__`
- 清理 Abaqus 会话残留
- 按前缀联动清理错误工件

这个脚本很适合调试期使用。

---

## 13. 现在这个项目和方案书相比，哪些地方已经一致，哪些地方还没完全到位

这一节很关键，因为你要学的是“真实项目”，不是“口号”。

### 13.1 已经与方案书高度一致的部分

- 六个核心智能体已经全部建类落地
- GUI 已经存在并切到对话主流程
- 跨环境 JSON 通信已经落地
- T 形筋 Abaqus 自动建模和后处理已经存在
- 自动重试机制已经实现
- 知识回流和 ChromaDB 已经接上
- 代理模型训练/预测闭环已经实现
- Markdown/PDF 报告已经能生成
- 测试已经覆盖主流程关键节点
- 历史任务、案例和 IO 工件迁移脚本已经提供并跑通

### 13.2 还处于“简化版”或“原型版”的部分

#### 1. 任务解析已经进入“规则 + LLM 混合解析”，但还不是全自由语义理解

它现在已经能结构化识别多工况、多边界和设计目标，但仍然更偏工程约束场景，不是开放领域通用对话理解器。

#### 2. RAG 默认已经是本地真实语义检索，但仍保留 hash 降级路径

当前默认配置是：

- `sentence-transformers`
- `BAAI/bge-m3`
- 项目内模型缓存目录

只有显式降级或本地模型不可用时才会回退到 hash embedding。

#### 3. 方案书写的是 BGE-M3 / LangChain，当前代码走的是更轻量的本地实现

当前主链路实际使用：

- `chromadb`
- `sentence-transformers`
- 自写封装

这不是“缺失功能”，而是工程化简化选择。

#### 4. 报告已经接入 LLM 工程解释，但仍偏短摘要

当前已经能输出总体判断、候选对比和建议动作，但还没有扩展到更长的章节化技术报告。

#### 5. GUI 已经进入对话闭环，但还不是 token 级流式聊天

当前是：

- 用户一句话输入
- 后台自动推进主流程
- 在 DNN / FEM / 报告三个节点确认
- 用结构化事件持续反馈进度
- 叠加简短的自然语言助手说明
- 支持分别指定总候选数量和初筛保留数量

它已经比纯按钮式更接近对话系统，但还不是逐 token 输出的连续流式聊天。

#### 6. 结构类型还只支持 `T` 筋

二期如果扩展：

- 帽型筋
- 工字筋
- 多种边界条件
- 多种载荷工况

将涉及较大改造。

#### 7. 项目还没有标准依赖清单文件

当前仓库里没有：

- `requirements.txt`
- `environment.yml`
- `pyproject.toml`

这意味着环境复现主要靠：

- `AGENTS.md`
- `scripts/check_env.py`

这在个人研发阶段还行，但后期多人协作最好补上。

#### 8. 代理模型精度还没达到方案书目标

当前 RF `MAPE ≈ 12.16%`，还没到 `< 5%`。

---

## 14. 从零开始，你最推荐的学习顺序

如果你现在是“完全从零”，我建议你按下面顺序来，不要一上来就读 Abaqus 大脚本。

### 第 1 步：先跑环境

```bat
conda activate GPT
python scripts/check_env.py
```

先确认：

- 你真的在 `GPT` 环境
- Abaqus 可用
- Torch / Chroma / OpenAI / PyQt6 都正常

### 第 2 步：用 mock 模式把 GUI 跑起来

PowerShell 下：

```powershell
conda activate GPT
$env:CSDM_USE_MOCK_ABAQUS = "1"
python main.py
```

你先不要急着看代码，先自己按新的对话主线走一遍：

1. 输入一句自然语言需求
2. 观察任务摘要和候选来源拆分
3. 确认是否进行 DNN 初筛
4. 确认是否进入 FEM 校核
5. 确认是否导出报告

先建立“用户视角的全局感”。

### 第 3 步：读 README、接口约定和这份文档

先搞清楚：

- 目录职责
- 编号规则
- 数据流

### 第 4 步：读 `core/`

优先顺序建议：

1. `paths.py`
2. `config_loader.py`
3. `io_utils.py`
4. `schema_validator.py`
5. `id_utils.py`
6. `rule_checker.py`
7. `doe_sampler.py`
8. `rag_engine.py`
9. `llm_backend.py`
10. `surrogate_model.py`

这一步的目标是先把“地基层”搞明白。

### 第 5 步：读 `agents/`

顺序建议：

1. `base.py`
2. `orchestrator.py`
3. `candidate_gen.py`
4. `screener.py`
5. `fem_agent.py`
6. `knowledge_agent.py`
7. `report_gen.py`

这一步要学会回答：

- 每个智能体输入是什么
- 输出是什么
- 谁调谁

### 第 6 步：再去读 `gui/`

重点看：

- `main_window.py`
- `candidate_widget.py`
- `abaqus_widget.py`
- `interactive_view.py`
- `render_utils.py`

这一步的目标是看懂“界面怎么把智能体串起来”。

### 第 7 步：最后再啃 `abaqus/runtime_build_panel.py`

因为它最重，也最依赖你前面已经理解：

- JSON 输入长什么样
- 候选字段怎么来的
- 结果要输出什么

### 第 8 步：跑测试

```bat
conda activate GPT
python -m pytest tests -q
```

### 第 9 步：自己做一个最小改动练手

我建议从下面几个任务里选一个：

- 在 `material_db.yaml` 新增一种材料
- 在 `param_ranges.yaml` 新增一个铺层模板
- 改 `SCREENER` 的打分公式
- 给 `REPORT_GEN` 增加一段工程总结
- 让 `ORCHESTRATOR` 识别更多自然语言字段

### 第 10 步：再做一次真 Abaqus 联调

等你对流程足够熟了，再关掉 mock，用真实 Abaqus 跑。

---

## 15. 如果你准备继续开发，下一批建议优先做哪些改造

下面这些不是“当前还没做”的事项，而是接下来最值得继续推进的方向。

### 第一优先级：继续提升任务解析质量

虽然现在已经是“规则 + LLM 混合解析”，但还可以继续增强：

- 更复杂的工程语句理解
- 多约束冲突消解
- 更丰富的默认值策略

### 第二优先级：继续强化 RAG 与案例解释

当前已经切到真实 embedding，下一步更值得做的是：

- 检索结果解释为什么相似
- 检索案例和当前任务的差异摘要
- 为案例迁移生成更明确的迁移理由

### 第三优先级：继续增强报告深度

当前报告已经接入 LLM 工程解释，下一步可以继续往：

- 多候选对比表
- 关键设计折中
- 更像工程汇报的章节化结论

### 第四优先级：把对话流做成更细的交互状态机

当前已经有关键节点确认，下一步可以继续做：

- 用户中途修正约束
- 对 FEM 失败原因的对话式追问
- 对候选方案的追加筛选条件

### 第五优先级：扩展结构型式和更多载荷组合

当前已经支持：

- T 形筋
- 单轴压缩
- 面内剪切
- 压剪组合
- `SSSS / CCCC / SSCC`

下一批扩展方向包括：

- 新筋型
- 更多边界组合
- 多工况联合目标
- 强度、稳定性、制造性联合优化

---

## 16. 你后面自己开发时最容易踩的坑

### 坑 1：忘记进 `GPT` 环境

这是最常见也是最致命的问题。

你会看到：

- openai 导入失败
- chromadb 导入失败
- torch 导入失败
- pyvista 导入失败

### 坑 2：以为当前 RAG 一定会联网下载模型

不是。

当前默认是本地优先、项目内缓存优先，只有模型不可用时才会降级。

### 坑 3：把 `build_panel.py` 当成真实求解主脚本

不是。

真实运行主链路是：

- `FEM_AGENT`
- `t_stiffener_buckle.py.j2`
- `runtime_build_panel.py`
- `runtime_extract_blf.py`

### 坑 4：以为所有 `success` 案例都会进入正式知识库

不是。

只有：

- `status == success`
- 且 `verdict == 通过`

才会进入：

- `knowledge/case_library/`
- `ChromaDB`

### 坑 5：以为 GUI 已经完全自动化

当前已经是“对话主导 + 关键确认节点”的自动流程，但仍然保留调试按钮和人工确认，不是完全不停车的黑箱式自治。

### 坑 6：把方案书里的技术选型当成当前代码的真实实现

要分清：

- 方案书是目标架构
- 仓库代码是当前实现

两者高度一致，但不完全等价。

---

## 17. 一组最实用的命令速查

### 17.1 环境自检

```bat
conda activate GPT
python scripts/check_env.py
```

### 17.2 启动 GUI

```bat
conda activate GPT
python main.py
```

### 17.3 强制 mock 模式启动 GUI

PowerShell：

```powershell
conda activate GPT
$env:CSDM_USE_MOCK_ABAQUS = "1"
python main.py
```

### 17.4 运行测试

```bat
conda activate GPT
python -m pytest tests -q
```

### 17.5 批量构造初始数据并训练模型

```bat
conda activate GPT
python scripts/build_initial_cases.py --count 40 --task-count 4 --workers 2 --mock --reset
```

### 17.6 重训代理模型

```bat
conda activate GPT
python scripts/train_screener.py
```

### 17.7 恢复任务台账

```bat
conda activate GPT
python scripts/restore_tasks_from_cases.py
```

### 17.8 清理调试残留

```bat
conda activate GPT
python scripts/clean_debug_artifacts.py
```

---

## 18. 你现在应该形成的“全局理解”

如果你读到这里，最好能自己复述出下面这段话：

> CSDM 不是一个单纯的聊天机器人，也不是一个单纯的 Abaqus 脚本集合。它本质上是一个以 JSON 为中间契约、以 FEM 为可信核心、以代理模型提速、以知识库复用经验、以 GUI 承载交互的多智能体工程设计系统。当前代码已经打通主链路，但很多“智能”能力仍是原型级实现，真正的下一步工作重点不是再堆新概念，而是持续提升任务解析质量、RAG 语义检索质量、FEM 稳定性和代理模型精度。

如果你能真正理解上面这句话，这个项目你就已经吃透一大半了。

---

## 19. 我对你后续学习路线的建议

如果你想尽快达到“我自己能开发这个项目”的程度，我建议你分成三轮：

### 第一轮：只求看懂

- 跑 GUI
- 跑测试
- 看懂目录
- 看懂数据流

### 第二轮：只改轻逻辑

- 改配置
- 改评分公式
- 改 prompt
- 改报告模板

### 第三轮：动核心链路

- 改任务解析
- 改 RAG
- 改 FEM 自动重试
- 改 Abaqus 建模
- 扩新筋型/新工况

按这个节奏走，你会比“一上来就啃 Abaqus 大脚本”快得多，也稳得多。

---

## 20. 总结

这个项目最难的地方不是某一个 Python 技巧，而是“多层系统如何协同”：

- LLM 层负责启发式生成
- RAG 层负责经验复用
- 规则层负责工程约束
- 代理模型层负责快速缩小搜索空间
- FEM 层负责提供最终可信结果
- 知识回流层负责让系统越用越强

而当前仓库的价值就在于：

- 这些层并不是停留在 PPT 上
- 它们已经被一套可运行的代码和目录结构串起来了

你接下来真正要做的，就是在这个已经跑通的骨架上，逐步把“原型级智能”打磨成“工程级智能”。
