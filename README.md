# CSDM_panel

复合材料加筋壁板多智能体设计系统。Windows 桌面界面使用 PyQt6，支持 T、BLADE、L、HAT 四种筋型，连接需求解析、程序候选生成、代理检查、真实 Abaqus 线性屈曲、案例检索与报告。

## 职责

- LLM 抽取用户需求并逐项给出原文引句。程序核对字段、数值、单位和明确约束，再保存任务；缺少候选数或求解名额时拒绝启动。
- 候选生成器以 LLM 结构化提案为主要来源，同时支持案例迁移和 DOE。候选数、各筋型配额、物理唯一性、几何范围、材料、铺层和层厚由程序校验；LLM 提供设计输入，程序决定是否接受。
- 每种筋型单独训练 RF、保存权重、训练来源、适用域及独立验收证据。不能用 T 型权重预测 L、BLADE 或 HAT。
- 代理不可用时，按配置 `direct_fem` 或 `stop` 执行。`direct_fem` 只按生成顺序和任务名额选择，不产生代理 BLF 或性能排名。
- FEM 使用实际部件的逐层复合壳截面。输入与结果绑定候选编号、材料、层序、几何、工况、边界和求解代码指纹。
- 报告解释使用项目内 RAG/KG 及实际 FEM。在最终推荐中说明轻量化与屈曲余量的取舍、可比较方案和影响选型的验证；程序负责数值、单位和推荐排序。见 [工程设计报告](docs/工程设计报告.md)。

## 环境与启动

Python 使用 Conda `AGENT` 环境，版本不低于 3.10。依赖定义见 `environment.yml`。Abaqus 需要本机命令及有效许可证；案例嵌入使用用户级 Hugging Face 缓存中的 `BAAI/bge-m3`，默认 CPU、离线加载。

```powershell
conda env create -f environment.yml
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/start.ps1 -CheckEnvironment
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/start.ps1
```

项目 `.env` 提供 `URL`、`API_KEY`、`MODEL_NAME`、`ABAQUS_COMMAND`，不进入 Git。LLM 配置位于 `config/llm_config.yaml`，单一 OpenAI 兼容接口，SDK 自动重试关闭。DeepSeek 的默认请求参数为 `thinking.type=disabled`；其他服务通过 `CSDM_PANEL_LLM_EXTRA_BODY` 配置其实际协议。

界面以普通窗口居中启动，不强制最大化。初始尺寸按显示器可用区域确定，上限 1440×900；最小尺寸 1000×640。左侧输入与操作按钮固定可见，摘要和对话分别滚动；右侧表格、详情和三维视图可拖动分配空间。

## 操作流程

示例需求：

```text
请为机翼下蒙皮设计T型加筋壁板，长度360mm，宽度300mm，Nx120kN/m，四边简支，生成4个候选，初筛保留2个，BLF至少1.2
```

1. 输入需求并核对结构化任务。几何使用 mm、载荷使用 kN/m，数值和枚举保留原文证据；明确尺寸的单位不受支持、字段遗漏或数值冲突时拒绝继续。未明确的信息采用项目任务默认值，不伪装为用户事实。
2. 程序按全局来源比例和筋型配额分配候选数。默认 `candidate_source_ratio` 为 `llm: 3`、`case_transfer: 1`、`doe: 1`；LLM 每批最多生成 5 个候选，响应数量、字段、筋型、材料、几何、铺层、规则和物理唯一性全部通过后才接受。LLM 请求或校验失败时该来源直接报错，不由 DOE 代替。指定多筋型时，候选数不得少于筋型数。
3. 确认初筛和求解。代理结果只有通过版本、来源、权重、独立验收和适用域检查才可用。
4. 查看 Abaqus 返回值及模态。成功求解不等于满足目标，未求解或失败的承载结论为未知。
5. 导出一份总体设计方案报告，包含全部候选、送检依据、实际结果、推荐取舍与验证事项和去重文献引用。Markdown、PDF 和事实 JSON 按任务保存在 `data/results/<TASK_ID>/`。启用 LLM 时，API 或工程评议校验失败会中止导出。

## 目录与数据

| 目录 | 职责 |
|---|---|
| `agents/` | 生成、筛选、求解、案例回流、报告和流程编排 |
| `core/` | 任务、物理截面、筋型、知识、案例索引、代理与事实契约 |
| `abaqus/` | Abaqus 兼容运行时、模板和作业管理 |
| `gui/` | 黑灰主题桌面界面、表格、三维及模态展示 |
| `config/`、`schemas/` | 配置、材料、参数范围和数据结构 |
| `data/tasks/` | 任务及候选选择决策 |
| `data/results/<TASK_ID>/` | 总体设计方案报告与完整事实审计 |
| `data/cases/` | 全部真实校核档案 |
| `data/abaqus_runs/`、`data/io/` | 正式输入、CAE/INP/ODB、结果与模态数据 |
| `knowledge/case_library/` | 成功且通过的正式案例 |
| `knowledge/chroma_db/` | 独立案例记忆集合，BGE-M3 1024 维向量 |
| `knowledge/external/` | 外部 RAG/KG 运行资产 |
| `models/<筋型>/` | 该筋型 RF、来源元数据和独立验收 |

业务数据、外部知识、模型、缓存和求解输出不进入源码 Git。正式案例与 FEM 文件在本机保留；维护重算先将正式原件移至项目外归档，普通设计入口拒绝覆盖已有正式工件。案例编号与候选正式编号一致，未归档作业也占用编号。

## 外部知识

项目使用六个外部知识运行文件：SQLite 全文索引、压缩关系、实体、图谱统计、压缩来源和本地清单。SQLite 已含完整块正文，不包含重复解析文档、训练 SFT、适配器或向量副本。

导出时间为 `2026-11-20T16:38:22.769812+00:00`；155599 个 RAG 块、2217 个实体、234200 条关系、1931 条来源。准确字节数与 SHA256 以 `knowledge/external/manifest.json` 为准。

检索使用 SQLite FTS5 BM25 和任务/筋型召回融合。仅质量版本 3 且可检索的块可进入上下文。正文、表格、公式、适用条件和关系绑定整体保留；预算不足时整块排除。相同内容去重，同一来源最多两个块。来源按 DOI、来源编号、URL 合并；文献标题一致且已知 DOI、年份不冲突的副本也合并，章节标题不作为文献身份。文献数值不能替代当前候选求解结果。

## 代理与物理边界

`panel-physical-v3-type-isolated` 对四筋型使用相同的 101 维物理特征契约，但训练数据、权重与独立验收严格分开。特征含 10 项几何、4 项筋型编码、15 项铺层/工况/材料及 72 项部件 A/B/D 分量；不存在部件的项为零。T/L、BLADE、HAT 分别有 3、2、4 类截面部件，不能因维度相同而互换权重。

训练至少需要同筋型 10 条兼容成功记录、5 个不同物理设计；分组留出不代替独立 FEM 验收。当前没有可投入预测的独立验收权重，默认走 `direct_fem`。训练入口须明确类型，例如 `python scripts/train_screener.py --stiffener-type T`。

HAT 的两块底部连接板在腹板外侧。求解采用理想 Tie、逐层 `CompositeShellSection`、四边自平衡面内牵引和板外支承。`BLF_global` 表示首个正特征值，名称不表示已识别整体模态。线性屈曲不证明损伤、制造可行性或非线性极限承载。

## 验证

```powershell
conda run -n AGENT --no-capture-output python -X utf8 -B -m pytest tests -q --run-real-fem -p no:cacheprovider
conda run -n AGENT --no-capture-output python -X utf8 -B scripts/validate_physical_workflow.py --mode api --require-llm-explanation --output-dir <项目外未存在目录>
```

完整回归显式运行真实 FEM。API 验收要求真实需求抽取、外部知识、成功 FEM 和非空报告；`--require-llm-explanation` 要求实际采用通过契约的工程评议。失败证据不覆盖，不把失败或跳过计为通过。力学与层序检查见 [物理模型与集成验收](docs/物理模型与集成验收.md)。

## 文档

- [架构与开发](docs/CSDM_panel_项目全流程梳理与开发指南.md)
- [接口约定](docs/接口约定.md)
- [工程设计报告](docs/工程设计报告.md)
- [项目状态](docs/项目状态.md)
- [案例维护](docs/案例维护操作.md)
- [案例迁移](docs/案例迁移机制详解.md)
