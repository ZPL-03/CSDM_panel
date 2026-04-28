# 复合材料文献爬取与大模型微调完整调研报告

> **适用项目**：面向复合材料加筋壁板结构设计的多智能体AI辅助系统（CSDM）  
> **作者**：刘正鹏 | 西安交通大学 SV LAB  
> **日期**：2026年4月  
> **版本**：v1.0

---

## 目录

1. [背景与目标](#1-背景与目标)
2. [文献数据源调研](#2-文献数据源调研)
3. [爬取工具与技术方案](#3-爬取工具与技术方案)
4. [PDF全文解析工具对比](#4-pdf全文解析工具对比)
5. [完整爬取管线设计](#5-完整爬取管线设计)
6. [数据清洗与预处理](#6-数据清洗与预处理)
7. [微调策略选型](#7-微调策略选型)
8. [SFT训练数据集构建](#8-sft训练数据集构建)
9. [RAG知识库搭建](#9-rag知识库搭建)
10. [模型微调实操指南](#10-模型微调实操指南)
11. [CSDM多智能体架构集成方案](#11-csdm多智能体架构集成方案)
12. [数据量估算与资源规划](#12-数据量估算与资源规划)
13. [实施路线图](#13-实施路线图)
14. [参考资源汇总](#14-参考资源汇总)

---

## 1. 背景与目标

### 1.1 项目背景

CSDM（面向复合材料结构设计的多智能体AI辅助系统）是一个结合LLM、RAG、DNN代理模型与ABAQUS FEM自动化的多智能体系统，核心场景是复合材料加筋壁板的智能设计。构建该系统的知识底座，需要大量复合材料领域的高质量文献数据。

### 1.2 核心需求

| 需求维度 | 具体描述 |
|---|---|
| **覆盖广度** | 复合材料力学、结构设计、损伤分析、多尺度建模、FEM自动化 |
| **数据质量** | 公式完整保留（LaTeX）、表格结构化、图注对应 |
| **数据量** | RAG知识库：1000~5000篇；SFT微调：5000~50000条QA对 |
| **更新机制** | 支持定期增量爬取，保持知识库最新 |
| **合法合规** | 优先开放获取（OA）期刊，避免版权纠纷 |

### 1.3 报告范围

本报告系统梳理：
- 复合材料相关的文献数据库与获取渠道
- Python爬取工具链的选型与使用
- PDF解析与全文抽取方案
- 面向LLM微调的数据集构建全流程
- 针对CSDM架构的RAG + SFT混合策略
- 可执行的分阶段实施路线图

---

## 2. 文献数据源调研

### 2.1 开放获取数据库（推荐优先爬取）

#### 2.1.1 arXiv

- **地址**：https://arxiv.org
- **相关分类**：`cond-mat.mtrl-sci`（材料科学）、`physics.app-ph`（应用物理）、`cs.CE`（计算工程）
- **特点**：预印本，数量庞大，提供官方API，可批量下载PDF
- **爬取方式**：官方API（`http://export.arxiv.org/api/query`）或 `paperscraper` 库
- **局限**：复合材料力学的顶刊论文在arXiv上数量相对有限，主要是计算力学相关

```
复合材料相关arXiv关键词示例：
  - "composite laminates buckling"
  - "fiber reinforced polymer FEM"
  - "multiscale homogenization composite"
  - "progressive damage CFRP"
  - "stiffened panel optimization"
```

#### 2.1.2 Semantic Scholar (S2)

- **地址**：https://api.semanticscholar.org
- **规模**：200M+ 篇论文，覆盖自然科学、工程、材料学
- **核心优势**：
  - 提供高影响力引文列表，支持引文图谱扩散爬取
  - 结构化文本、NL摘要、向量嵌入
  - 免费API（有限速），申请研究用key后速率提升
- **爬取字段**：`title, abstract, year, citationCount, openAccessPdf, references, citations`

#### 2.1.3 OpenAlex

- **地址**：https://api.openalex.org
- **特点**：完全免费、无需注册、速率慷慨（mailto参数进入polite pool后更快）
- **支持**：关键词搜索、DOI查询、引文网络、OA全文链接
- **推荐理由**：对中国大陆网络友好，Unpaywall数据整合，OA PDF直链获取

#### 2.1.4 PubMed / PubMed Central (PMC)

- **地址**：https://pubmed.ncbi.nlm.nih.gov / https://www.ncbi.nlm.nih.gov/pmc
- **相关方向**：复合材料生物医疗应用、天然纤维复合材料、骨科植入物
- **获取方式**：E-utilities API，PMC Open Access子集可直接下载XML全文

#### 2.1.5 MDPI Open Access

- **相关期刊**：
  - *Materials*（综合材料，Q2）
  - *Polymers*（聚合物复合材料）
  - *Aerospace*（航空复合材料结构）
  - *Journal of Composites Science*
- **获取方式**：全部OA，可直接爬取HTML/PDF，也可通过OAI-PMH协议批量获取元数据

#### 2.1.6 DOAJ（Directory of Open Access Journals）

- **地址**：https://doaj.org
- **功能**：OA期刊目录，API支持，可批量检索特定领域的OA期刊全库

---

### 2.2 付费/机构访问数据库

> 以下数据库需校园网VPN访问，建议在西交大内网环境下批量下载。

#### 2.2.1 Elsevier ScienceDirect

**复合材料核心期刊**（必须覆盖）：

| 期刊名 | 影响因子（2024参考）| 复合材料方向 |
|---|---|---|
| *Composite Structures* | ~6.3 | 结构力学、加筋壁板、屈曲 |
| **Thin-Walled Structures** | ~5.7 | 薄壁结构、屈曲、后屈曲 |
| *Composites Part A* | ~8.1 | 制造工艺、材料性能 |
| *Composites Part B* | ~8.0 | 力学性能、疲劳损伤 |
| *Composites Science and Technology* | ~8.3 | 基础研究、微观机制 |
| *International Journal of Solids and Structures* | ~3.9 | 固体力学通论 |

**爬取建议**：利用 `requests` + Elsevier API（需申请key）：
```
https://api.elsevier.com/content/search/scopus?query=TITLE-ABS-KEY(composite+stiffened+panel)&apiKey=YOUR_KEY
```

#### 2.2.2 AIAA Journals（美国航空航天学会）

- **相关期刊**：*AIAA Journal*, *Journal of Aircraft*, *Structural Dynamics*
- **特点**：航空结构设计、复合材料机身/机翼，与你的方向高度契合

#### 2.2.3 Web of Science / Scopus

- **功能**：引文分析、批量导出 `.bib/.ris` 文件
- **建议用法**：通过WoS检索后批量导出记录（含DOI），再用DOI查找OA全文

#### 2.2.4 中文文献（知网/万方）

- **知网（CNKI）**：国内复合材料研究（尤其硕博论文）极为丰富
- **万方数据**：覆盖中文核心期刊
- **获取方式**：学校图书馆账号，可下载CAJ/PDF格式
- **注意**：CAJ格式需转换为PDF才能用通用解析工具处理

---

### 2.3 复合材料细分领域关键词矩阵

根据CSDM项目的核心场景，建议构建以下关键词矩阵进行组合检索：

**主题词（A类）**：
```
"composite laminate" / "fiber reinforced polymer" / "CFRP" / "GFRP"
"stiffened panel" / "stiffened plate" / "hat-stiffened" / "blade-stiffened"
"composite structure" / "laminated composite"
```

**力学行为词（B类）**：
```
"buckling" / "post-buckling" / "compressive stability"
"progressive damage" / "failure analysis" / "delamination"
"fatigue" / "fracture mechanics" / "impact damage"
"bending" / "torsion" / "free vibration"
```

**方法词（C类）**：
```
"finite element" / "FEM" / "ABAQUS" / "Nastran"
"machine learning" / "neural network" / "deep learning"
"multiscale" / "homogenization" / "RVE"
"optimization" / "surrogate model" / "genetic algorithm"
"CLT" / "classical lamination theory"
```

**组合示例**：
- `(A类) AND (B类)`：获取基础力学研究论文
- `(A类) AND (C类中AI关键词)`：获取AI辅助复合材料设计论文
- `(A类) AND (B类) AND (C类中FEM关键词)`：获取FEM仿真论文

---

## 3. 爬取工具与技术方案

### 3.1 工具总览

| 工具 | 适用数据源 | 语言 | 优势 | 局限 |
|---|---|---|---|---|
| **paperscraper** | arXiv, PubMed, S2 | Python | 开箱即用，jsonl输出 | 不含全文下载 |
| **requests + API** | S2, OpenAlex, Elsevier | Python | 灵活可控 | 需自行处理分页、限速 |
| **scholarly** | Google Scholar | Python | 覆盖全，含引用数 | 反爬严格，不稳定 |
| **Scrapy** | 通用网页爬取 | Python | 高并发，可扩展 | 需针对每个网站定制 |
| **selenium/playwright** | 动态网页（CNKI等） | Python | 处理JS渲染 | 速度慢 |

---

### 3.2 paperscraper 详细使用

安装：
```bash
pip install paperscraper
```

**arXiv批量爬取**：
```python
from paperscraper.arxiv import get_and_dump_arxiv_papers

# 复合材料加筋壁板专项检索
query_composite_stiffened = [
    ["composite", "stiffened panel"],
    ["CFRP", "buckling", "compressive"],
]
get_and_dump_arxiv_papers(
    query_composite_stiffened,
    output_filepath='data/arxiv_composite_stiffened.jsonl'
)

# 复合材料AI/ML检索
query_composite_ml = [
    ["composite laminate", "machine learning"],
    ["fiber reinforced", "neural network", "surrogate"],
    ["multiscale", "homogenization", "deep learning"],
]
get_and_dump_arxiv_papers(
    query_composite_ml,
    output_filepath='data/arxiv_composite_ml.jsonl'
)
```

**本地arXiv全量转储（最彻底的方案）**：
```python
from paperscraper.get_dumps import arxiv

# 下载2020年至今的所有arXiv元数据（约几十GB）
arxiv(start_date='2020-01-01', end_date=None)

# 然后本地关键词检索，无API限速
from paperscraper.arxiv import get_and_dump_arxiv_papers
get_and_dump_arxiv_papers(query, backend='local')
```

---

### 3.3 Semantic Scholar API 详细使用

```python
import requests
import time
import json
from pathlib import Path

class SemanticScholarCrawler:
    def __init__(self, api_key=None):
        self.base_url = "https://api.semanticscholar.org/graph/v1"
        self.headers = {"x-api-key": api_key} if api_key else {}
        self.fields = "title,abstract,year,citationCount,openAccessPdf,authors,journal,externalIds"

    def search(self, query: str, limit: int = 100, offset: int = 0) -> dict:
        """关键词检索"""
        url = f"{self.base_url}/paper/search"
        params = {
            "query": query,
            "fields": self.fields,
            "limit": min(limit, 100),
            "offset": offset
        }
        resp = requests.get(url, params=params, headers=self.headers)
        time.sleep(1)  # 限速：无key时1req/s，有key时10req/s
        return resp.json()

    def get_paper_detail(self, paper_id: str) -> dict:
        """获取单篇论文详情"""
        url = f"{self.base_url}/paper/{paper_id}"
        params = {"fields": self.fields + ",references,citations,tldr"}
        resp = requests.get(url, params=params, headers=self.headers)
        time.sleep(0.5)
        return resp.json()

    def get_citations(self, paper_id: str, depth: int = 1) -> list:
        """引文图谱扩散爬取（BFS）"""
        visited = set()
        queue = [(paper_id, 0)]
        results = []

        while queue:
            pid, d = queue.pop(0)
            if pid in visited or d > depth:
                continue
            visited.add(pid)

            url = f"{self.base_url}/paper/{pid}/citations"
            params = {"fields": self.fields, "limit": 50}
            resp = requests.get(url, params=params, headers=self.headers).json()
            time.sleep(0.5)

            for item in resp.get("data", []):
                paper = item.get("citingPaper", {})
                results.append(paper)
                if d < depth and paper.get("paperId"):
                    queue.append((paper["paperId"], d + 1))

        return results

    def bulk_crawl(self, queries: list, output_dir: str = "data/s2"):
        """批量检索多个查询词"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        all_papers = {}

        for query in queries:
            print(f"Searching: {query}")
            offset = 0
            while True:
                result = self.search(query, limit=100, offset=offset)
                papers = result.get("data", [])
                if not papers:
                    break
                for p in papers:
                    pid = p.get("paperId")
                    if pid:
                        all_papers[pid] = p
                offset += 100
                if offset >= result.get("total", 0):
                    break

        # 保存结果
        output_path = Path(output_dir) / "s2_papers.jsonl"
        with open(output_path, "w", encoding="utf-8") as f:
            for paper in all_papers.values():
                f.write(json.dumps(paper, ensure_ascii=False) + "\n")

        print(f"Saved {len(all_papers)} papers to {output_path}")
        return all_papers


# 使用示例
crawler = SemanticScholarCrawler(api_key="YOUR_S2_API_KEY")
queries = [
    "composite stiffened panel buckling",
    "CFRP laminate failure analysis",
    "fiber reinforced polymer machine learning structural design",
    "multiscale homogenization composite FEM",
    "ABAQUS composite progressive damage",
]
crawler.bulk_crawl(queries)
```

---

### 3.4 OpenAlex API 详细使用

```python
import requests
import json
from pathlib import Path

class OpenAlexCrawler:
    def __init__(self, email: str):
        self.base_url = "https://api.openalex.org"
        self.email = email  # polite pool，速率更高
        self.headers = {"User-Agent": f"mailto:{email}"}

    def search_works(self, query: str, filters: dict = None, per_page: int = 200) -> list:
        """
        filters示例：
          {"open_access.is_oa": "true", "from_publication_date": "2015-01-01"}
        """
        url = f"{self.base_url}/works"
        params = {
            "search": query,
            "per-page": per_page,
            "mailto": self.email,
            "select": "id,doi,title,abstract_inverted_index,publication_year,cited_by_count,open_access,primary_location"
        }
        if filters:
            filter_str = ",".join(f"{k}:{v}" for k, v in filters.items())
            params["filter"] = filter_str

        results = []
        cursor = "*"
        while cursor:
            params["cursor"] = cursor
            resp = requests.get(url, params=params, headers=self.headers).json()
            results.extend(resp.get("results", []))
            cursor = resp.get("meta", {}).get("next_cursor")
            if len(results) % 1000 == 0:
                print(f"  Fetched {len(results)} works...")

        return results

    def reconstruct_abstract(self, inverted_index: dict) -> str:
        """OpenAlex摘要以倒排索引存储，需重建"""
        if not inverted_index:
            return ""
        word_positions = []
        for word, positions in inverted_index.items():
            for pos in positions:
                word_positions.append((pos, word))
        word_positions.sort(key=lambda x: x[0])
        return " ".join(w for _, w in word_positions)

    def get_oa_pdf_url(self, work: dict) -> str | None:
        """获取OA全文PDF链接"""
        oa = work.get("open_access", {})
        if oa.get("is_oa") and oa.get("oa_url"):
            return oa["oa_url"]
        loc = work.get("primary_location", {})
        if loc.get("pdf_url"):
            return loc["pdf_url"]
        return None

    def crawl_composite_materials(self, output_dir: str = "data/openalex"):
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        queries_and_filters = [
            ("composite stiffened panel buckling", {"from_publication_date": "2010-01-01"}),
            ("CFRP laminate finite element analysis", {"from_publication_date": "2015-01-01"}),
            ("composite structure machine learning optimization", {"from_publication_date": "2018-01-01"}),
            ("multiscale composite homogenization RVE", {"from_publication_date": "2010-01-01"}),
        ]

        all_works = []
        for query, filters in queries_and_filters:
            print(f"\nQuery: {query}")
            works = self.search_works(query, filters=filters)
            for w in works:
                w["_abstract"] = self.reconstruct_abstract(w.pop("abstract_inverted_index", {}))
                w["_pdf_url"] = self.get_oa_pdf_url(w)
            all_works.extend(works)
            print(f"  Got {len(works)} works")

        # 去重
        seen = set()
        unique_works = []
        for w in all_works:
            doi = w.get("doi")
            if doi and doi not in seen:
                seen.add(doi)
                unique_works.append(w)

        output_path = Path(output_dir) / "openalex_papers.jsonl"
        with open(output_path, "w", encoding="utf-8") as f:
            for w in unique_works:
                f.write(json.dumps(w, ensure_ascii=False) + "\n")

        print(f"\nSaved {len(unique_works)} unique works")

        # 单独保存有PDF的记录
        with_pdf = [w for w in unique_works if w.get("_pdf_url")]
        pdf_list_path = Path(output_dir) / "papers_with_pdf.jsonl"
        with open(pdf_list_path, "w", encoding="utf-8") as f:
            for w in with_pdf:
                f.write(json.dumps({"doi": w.get("doi"), "title": w.get("title"), "pdf_url": w["_pdf_url"]}, ensure_ascii=False) + "\n")
        print(f"Papers with OA PDF: {len(with_pdf)}")


# 使用
crawler = OpenAlexCrawler(email="lzp03@stu.xjtu.edu.cn")
crawler.crawl_composite_materials()
```

---

### 3.5 PDF批量下载

获得PDF链接后批量下载：

```python
import requests
import hashlib
import time
from pathlib import Path
from urllib.parse import urlparse

def download_pdfs(pdf_records: list, output_dir: str = "data/pdfs", delay: float = 1.5):
    """
    pdf_records: [{"doi": "...", "title": "...", "pdf_url": "..."}]
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    success, failed = 0, 0

    for record in pdf_records:
        url = record.get("pdf_url")
        doi = record.get("doi", "unknown")
        if not url:
            continue

        # 文件名：用DOI的hash，避免特殊字符
        filename = hashlib.md5(doi.encode()).hexdigest() + ".pdf"
        filepath = Path(output_dir) / filename

        if filepath.exists():
            continue  # 跳过已下载

        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (research bot; mailto:lzp03@stu.xjtu.edu.cn)"
            }
            resp = requests.get(url, headers=headers, timeout=30, stream=True)
            if resp.status_code == 200 and "pdf" in resp.headers.get("content-type", "").lower():
                with open(filepath, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)
                # 记录DOI-文件名映射
                with open(Path(output_dir) / "manifest.jsonl", "a") as mf:
                    mf.write(json.dumps({"doi": doi, "title": record.get("title"), "file": filename}) + "\n")
                success += 1
            else:
                failed += 1
        except Exception as e:
            print(f"Failed: {doi} — {e}")
            failed += 1

        time.sleep(delay)

    print(f"Downloaded: {success}, Failed: {failed}")
```

---

### 3.6 中文文献爬取（知网）

知网无开放API，需借助 `cnki-spider` 或 `pycnki` 等工具，或通过学校VPN手动批量下载：

```python
# 方案1：使用 cnki_spider（需校园网环境）
# pip install cnki-spider

# 方案2：通过知网导出功能，批量导出题录（.txt格式）
# 然后用以下代码解析
def parse_cnki_export(filepath: str) -> list:
    """解析知网批量导出的NoteExpress格式"""
    papers = []
    current = {}
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("{Reference Type}"):
                if current:
                    papers.append(current)
                current = {}
            elif ": " in line:
                key, val = line.split(": ", 1)
                current[key.strip()] = val.strip()
    if current:
        papers.append(current)
    return papers
```

---

## 4. PDF全文解析工具对比

### 4.1 工具详细对比

复合材料论文的解析挑战：
- **公式密集**：本构关系、刚度矩阵、屈曲方程等大量LaTeX公式
- **表格复杂**：材料性能参数表、有限元结果对比表
- **图文混排**：应力云图、载荷-位移曲线、几何示意图

| 工具 | 公式(LaTeX) | 表格 | 图片 | 阅读顺序 | 速度 | 推荐指数 |
|---|---|---|---|---|---|---|
| **MinerU** | ✅ 优秀 | ✅ 优秀 | ✅ | ✅ | 快 | ⭐⭐⭐⭐⭐ |
| **Marker** | ✅ 良好 | ✅ 良好 | ⚠️ | ✅ | 中等 | ⭐⭐⭐⭐ |
| **Nougat** (Meta) | ✅ 优秀 | ⚠️ 中等 | ❌ | ✅ | 慢（GPU） | ⭐⭐⭐⭐ |
| **Docling** (IBM) | ⚠️ 中等 | ✅ 优秀 | ✅ base64 | ✅ | 中等 | ⭐⭐⭐ |
| **GROBID** | ❌ | ✅ | ❌ | ✅ | 快 | ⭐⭐⭐（元数据专用） |
| **LlamaParse** | ✅ | ✅ | ✅ | ✅ | 中等 | ⭐⭐⭐（需付费） |
| **PyMuPDF** | ❌ | ⚠️ | ❌ | ⚠️ | 极快 | ⭐⭐（简单文本场景） |
| **pdfminer** | ❌ | ❌ | ❌ | ⚠️ | 快 | ⭐（已过时） |

---

### 4.2 MinerU（首选方案）

MinerU 是上海人工智能实验室 OpenDataLab 开发的一站式开源PDF解析工具，对中英文科技文档支持极佳。

```bash
pip install mineru[full]
# 或使用 magic-pdf
pip install magic-pdf[full]
```

**Python API使用**：
```python
from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
from magic_pdf.data.dataset import PymuPdfDataset
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
from magic_pdf.config.make_content_config import DropMode, MakeMode
import json

def parse_pdf_with_mineru(pdf_path: str, output_dir: str) -> dict:
    """使用MinerU解析PDF，输出Markdown + 结构化JSON"""
    import os
    os.makedirs(output_dir, exist_ok=True)

    reader = FileBasedDataReader("")
    pdf_bytes = reader.read(pdf_path)

    ds = PymuPdfDataset(pdf_bytes)
    model_json = doc_analyze(ds, ocr=False)  # 对数字PDF，ocr=False更快

    writer = FileBasedDataWriter(output_dir)
    pipe = ds.apply(
        model_json,
        image_writer=writer,
    )

    pipe.pipe_classify()
    pipe.pipe_analyze()
    pipe.pipe_parse()

    # 获取Markdown
    md_content = pipe.get_markdown(output_dir)
    # 获取结构化内容（含公式、表格元数据）
    content_list = pipe.get_content_list(output_dir)

    return {
        "markdown": md_content,
        "content_list": content_list,  # 分块内容，含类型标注
    }

# 批量处理
import glob

def batch_parse(pdf_dir: str, output_base: str):
    pdf_files = glob.glob(f"{pdf_dir}/*.pdf")
    results = {}
    for pdf_path in pdf_files:
        stem = Path(pdf_path).stem
        output_dir = f"{output_base}/{stem}"
        try:
            result = parse_pdf_with_mineru(pdf_path, output_dir)
            results[stem] = result
            print(f"✓ {stem}")
        except Exception as e:
            print(f"✗ {stem}: {e}")
    return results
```

---

### 4.3 Nougat（公式密集型论文专用）

Nougat 由 Meta AI 开发，专为学术PDF设计，直接从页面像素预测 Markdown + LaTeX 公式。

```bash
pip install nougat-ocr
# 需要GPU，推荐在服务器上批量运行
```

```python
from nougat import NougatModel
from nougat.utils.dataset import LazyDataset
import torch

model = NougatModel.from_pretrained("facebook/nougat-base")
model = model.to("cuda" if torch.cuda.is_available() else "cpu")
model.eval()

def parse_with_nougat(pdf_path: str) -> str:
    from nougat.postprocessing import markdown_compatible
    from nougat.utils.checkpoint import get_checkpoint
    import pypdf

    # 将PDF转为图片后逐页处理
    dataset = LazyDataset(pdf_path, partial(model.encoder.prepare_input, random_padding=False))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)

    predictions = []
    for sample, _ in dataloader:
        with torch.no_grad():
            output = model.inference(image_tensors=sample)
        predictions.extend(output["predictions"])

    return "\n\n".join(predictions)
```

---

### 4.4 推荐组合策略

```
PDF类型判断
   ├── 数字PDF（有文字层）→ MinerU（首选，速度快，公式好）
   ├── 扫描PDF（图片）   → Nougat（OCR + 公式识别）
   └── 元数据提取        → GROBID（作者、参考文献、章节结构）

后处理
   ├── 图片/图表 → Docling（提取图片base64，传给VLM描述）
   └── 验证      → 抽样人工检查公式LaTeX是否正确
```

---

## 5. 完整爬取管线设计

### 5.1 管线架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                        数据采集层                                    │
│                                                                       │
│  arXiv API      S2 API       OpenAlex      ScienceDirect    CNKI     │
│      └──────────────┬──────────────┘            │            │       │
│                     ▼                            ▼            ▼       │
│              paperscraper              requests爬虫    selenium爬虫   │
└─────────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        元数据层                                       │
│                                                                       │
│   去重（DOI/标题哈希）→ 过滤（年份/期刊/引用数）→ 获取OA PDF链接     │
│                                                                       │
│   输出：papers_metadata.jsonl                                        │
└─────────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        全文获取层                                     │
│                                                                       │
│   OA PDF下载（Unpaywall/OpenAlex直链）→ 存储为 {doi_hash}.pdf        │
│                                                                       │
│   manifest.jsonl：记录DOI↔文件名映射                                 │
└─────────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        解析层                                         │
│                                                                       │
│   MinerU（主力）                                                      │
│      ├── 输出 Markdown（含LaTeX公式）                                │
│      ├── 输出 content_list（分块，含类型标注）                       │
│      └── 输出 figures/（图片文件）                                   │
│                                                                       │
│   GROBID（辅助）→ 提取参考文献、作者、章节结构                      │
└─────────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        存储层                                         │
│                                                                       │
│   MongoDB / Elasticsearch：结构化检索                                │
│   原始 JSONL 文件：离线处理                                          │
│   向量数据库（Chroma/Milvus）：RAG检索                               │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.2 统一数据Schema

每篇论文处理后存储为以下格式：

```json
{
  "id": "md5_of_doi",
  "doi": "10.1016/j.tws.2025.112340",
  "title": "Buckling analysis of CFRP stiffened panels under combined compression and shear",
  "authors": ["Wang, X.", "Li, Y.", "Zhang, Z."],
  "journal": "Thin-Walled Structures",
  "year": 2025,
  "citation_count": 12,
  "keywords": ["CFRP", "stiffened panel", "buckling", "combined loading"],
  "abstract": "This paper investigates the buckling behavior...",
  "full_text_markdown": "## 1. Introduction\n\nComposite stiffened panels...\n\n$$K_c = \\frac{\\pi^2 D_{11}}{b^2 t}$$\n\n",
  "formulas": [
    {"latex": "K_c = \\frac{\\pi^2 D_{11}}{b^2 t}", "context": "critical buckling coefficient"},
    {"latex": "\\sigma_{cr} = K_c \\frac{\\pi^2 E}{12(1-\\nu^2)}\\left(\\frac{t}{b}\\right)^2", "context": "critical stress"}
  ],
  "tables": [
    {
      "caption": "Material properties of T300/5208 CFRP",
      "headers": ["E1 (GPa)", "E2 (GPa)", "G12 (GPa)", "v12"],
      "data": [["181", "10.3", "7.17", "0.28"]]
    }
  ],
  "sections": {
    "introduction": "...",
    "methodology": "...",
    "results": "...",
    "conclusion": "..."
  },
  "source": "sciencedirect",
  "pdf_path": "data/pdfs/a1b2c3d4.pdf",
  "parse_status": "success",
  "parse_date": "2026-04-27"
}
```

---

### 5.3 管线编排脚本

```python
import json
import hashlib
from pathlib import Path
from datetime import datetime

class CompositeLiteraturePipeline:
    def __init__(self, output_dir: str = "data/composite_corpus"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "pdfs").mkdir(exist_ok=True)
        (self.output_dir / "parsed").mkdir(exist_ok=True)

    def run_step1_metadata(self):
        """Step 1: 从多个数据源爬取元数据"""
        print("=== Step 1: Crawling metadata ===")
        # ... 调用前文各爬虫 ...

    def run_step2_dedup_filter(self):
        """Step 2: 去重+过滤"""
        print("=== Step 2: Deduplication & filtering ===")
        seen_dois = set()
        filtered = []
        with open(self.output_dir / "raw_metadata.jsonl") as f:
            for line in f:
                paper = json.loads(line)
                doi = paper.get("doi")
                if doi and doi not in seen_dois:
                    # 过滤条件：年份>=2005，有摘要，引用数>0（或近2年新论文）
                    year = int(paper.get("year", 0))
                    has_abstract = bool(paper.get("abstract", "").strip())
                    if year >= 2005 and has_abstract:
                        seen_dois.add(doi)
                        filtered.append(paper)
        print(f"After dedup+filter: {len(filtered)} papers")
        return filtered

    def run_step3_download_pdfs(self, papers: list):
        """Step 3: 下载OA PDF"""
        print("=== Step 3: Downloading PDFs ===")
        # ... 调用download_pdfs函数 ...

    def run_step4_parse(self):
        """Step 4: 解析PDF全文"""
        print("=== Step 4: Parsing PDFs ===")
        # ... 调用batch_parse函数 ...

    def run_step5_finalize(self):
        """Step 5: 合并元数据与全文，输出最终语料库"""
        print("=== Step 5: Finalizing corpus ===")
        # 合并逻辑 ...

    def run_all(self):
        self.run_step1_metadata()
        papers = self.run_step2_dedup_filter()
        self.run_step3_download_pdfs(papers)
        self.run_step4_parse()
        self.run_step5_finalize()
        print("Pipeline complete!")


if __name__ == "__main__":
    pipeline = CompositeLiteraturePipeline()
    pipeline.run_all()
```

---

## 6. 数据清洗与预处理

### 6.1 文本质量过滤

```python
import re
from typing import Optional

def filter_paper_quality(paper: dict) -> tuple[bool, str]:
    """
    返回 (is_valid, reason)
    """
    text = paper.get("full_text_markdown", "")
    abstract = paper.get("abstract", "")

    # 1. 文本长度过滤
    if len(text) < 2000:
        return False, "text_too_short"

    # 2. 解析质量检查（公式被破坏的症状）
    if text.count("?") / max(len(text), 1) > 0.05:
        return False, "too_many_question_marks"  # 乱码迹象

    # 3. 领域相关性过滤（确保在复合材料领域）
    keywords_required = [
        "composite", "fiber", "laminate", "CFRP", "GFRP", "buckling",
        "stiffened", "delamination", "homogenization", "RVE",
        "复合材料", "纤维增强", "层合板", "屈曲"
    ]
    text_lower = (text + abstract).lower()
    if not any(kw.lower() in text_lower for kw in keywords_required):
        return False, "off_topic"

    # 4. 重复内容检测（简单n-gram去重）
    # （在批量处理时用MinHash/SimHash更高效）

    return True, "ok"


def clean_paper_text(text: str) -> str:
    """清洗Markdown文本，保留公式"""
    # 去除参考文献章节
    text = re.sub(r'\n#{1,3}\s*(References|Bibliography|参考文献).*', '', text, flags=re.DOTALL)
    # 去除页眉页脚残留（通常是短行+期刊名）
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        # 跳过纯图号行（"Fig. 3."或"Figure 3"单独成行）
        if re.match(r'^(Fig\.|Figure|Table|图|表)\s*\d+[\.\s]*$', line.strip()):
            continue
        cleaned_lines.append(line)
    text = '\n'.join(cleaned_lines)
    # 合并多个连续空行
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()
```

### 6.2 数据分块（Chunking）

对于RAG和SFT，需要对全文进行合理分块：

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

def create_composite_splitter():
    """
    复合材料论文专用分块器
    关键：不在公式中间切割，优先在段落边界切割
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=512,        # token数
        chunk_overlap=64,      # 上下文重叠
        length_function=len,
        separators=[
            "\n## ",           # 一级标题边界（最优先）
            "\n### ",          # 二级标题边界
            "\n\n",            # 段落边界
            "\n",              # 行边界
            "$$\n",            # 公式块边界（不在公式内切割）
            "。",              # 中文句号
            ". ",              # 英文句号
            " ",               # 词边界
        ]
    )

def chunk_paper(paper: dict) -> list[dict]:
    """将一篇论文分块，保留元数据"""
    splitter = create_composite_splitter()
    text = paper.get("full_text_markdown", "")
    chunks = splitter.split_text(text)

    return [
        {
            "chunk_id": f"{paper['id']}_chunk_{i:04d}",
            "paper_id": paper["id"],
            "doi": paper.get("doi"),
            "title": paper.get("title"),
            "year": paper.get("year"),
            "journal": paper.get("journal"),
            "chunk_text": chunk,
            "chunk_index": i,
        }
        for i, chunk in enumerate(chunks)
    ]
```

---

## 7. 微调策略选型

### 7.1 策略对比分析

| 策略 | 核心作用 | 算力要求 | 数据量 | 知识更新 | 推荐场景 |
|---|---|---|---|---|---|
| **RAG** | 检索增强生成，无需修改模型权重 | 低（推理时） | 无上限 | 实时 | 知识库问答，引文溯源 |
| **Prompt Engineering** | 通过精心设计提示词适配任务 | 极低 | 0 | 即时 | 快速验证，临时方案 |
| **SFT（LoRA）** | 指令跟随、输出格式、专业风格 | 中（1-2 GPU） | 5K~50K条 | 需重训 | ABAQUS脚本生成，格式化输出 |
| **CPT（续训预训练）** | 注入领域词汇、概念、符号体系 | 高（8+ GPU） | 1B+ tokens | 需重训 | 术语理解，长文本处理 |
| **CPT + SFT** | 领域知识注入 + 指令跟随，最全面 | 高 | 大量 | 需重训 | 研究级别的领域专用模型 |
| **DPO/ORPO** | 偏好对齐，提升回答质量 | 中 | 数千对比对 | 需重训 | 专业性与流畅度的平衡 |

### 7.2 CSDM项目推荐策略

考虑到你的实际约束（个人/实验室级别GPU资源有限，项目以工程落地为目标），建议：

```
Phase 1（立即可执行）：RAG为主
  → KNOWLEDGE_AGENT基于向量数据库+复合材料语料库
  → 使用Claude/Qwen-Max等通用大模型API

Phase 2（资源充足时）：SFT补充
  → 针对FEM_AGENT的ABAQUS脚本生成任务微调Qwen2.5-7B
  → LoRA参数高效微调，可在单A100上完成

Phase 3（长期规划）：模型合并/蒸馏
  → 将多个LoRA适配器合并，获得复合材料全能模型
```

### 7.3 基座模型选择

| 模型 | 参数量 | 中英文 | 理工科能力 | 推荐理由 |
|---|---|---|---|---|
| **Qwen2.5-7B-Instruct** | 7B | ✅ 强 | ✅ | 国内首选，理工科语料充分，中英双语 |
| **Qwen2.5-14B-Instruct** | 14B | ✅ 强 | ✅ | 更强推理，算力允许时推荐 |
| **DeepSeek-R1-7B** | 7B | ✅ | ✅ 强 | 推理能力突出，适合复杂力学推理 |
| **LLaMA-3.1-8B-Instruct** | 8B | ⚠️ 中英 | ✅ | 英文文献处理更好 |
| **InternLM2.5-7B** | 7B | ✅ 强 | ✅ | 上海AI Lab出品，理工科强 |

---

## 8. SFT训练数据集构建

### 8.1 任务类型设计

针对CSDM架构，设计以下SFT任务类型：

| 任务类型 | 说明 | 数据来源 | 优先级 |
|---|---|---|---|
| **材料力学QA** | 复合材料本构、失效准则、CLT理论 | 论文+教材+自动生成 | 高 |
| **ABAQUS脚本生成** | 给定结构描述→生成Python脚本 | 手工构造+LLM生成 | 高 |
| **FEM结果分析** | 给定仿真结果→分析失效模式/建议 | 论文结果章节 | 高 |
| **设计参数推理** | 给定设计要求→推荐铺层方案 | 设计手册+论文 | 高 |
| **公式推导解释** | 解释某公式的物理含义/推导过程 | 教材+论文 | 中 |
| **文献摘要总结** | 给定论文全文→输出结构化摘要 | 论文自动处理 | 中 |
| **跨模态关联** | 描述图表→解释力学含义 | 论文图表+人工标注 | 低 |

### 8.2 自动QA生成管线

```python
import anthropic
import json
import random
from pathlib import Path

client = anthropic.Anthropic()

# 不同任务类型的prompt模板
TASK_PROMPTS = {
    "mechanics_qa": """你是复合材料力学专家。基于以下论文片段，生成{n}个高质量的问答对。
要求：
- 问题类型包括：概念理解、公式推导含义、参数影响分析、工程应用
- 答案要专业精准，必要时包含LaTeX公式（用$$包围独立公式，$包围行内公式）
- 难度要有梯度（基础题/中级题/高级题）
- 输出纯JSON，格式：{{"pairs": [{{"question": "...", "answer": "...", "difficulty": "basic/intermediate/advanced", "type": "..."}}]}}

论文片段：
{chunk}
""",

    "abaqus_script": """你是ABAQUS有限元仿真专家。基于以下论文中的建模描述，生成{n}个ABAQUS Python脚本生成的问答对。
要求：
- 问题描述具体的建模需求（材料、几何、边界条件、载荷）
- 答案是完整可运行的ABAQUS Python脚本
- 包含注释说明关键步骤
- 输出纯JSON，格式：{{"pairs": [{{"question": "...", "answer": "..."}}]}}

论文片段（关注建模方法部分）：
{chunk}
""",

    "design_reasoning": """你是复合材料结构设计工程师。基于以下设计案例，生成{n}个结构设计推理问答对。
要求：
- 问题涉及铺层设计、加筋布置、优化目标权衡
- 答案要有工程判断依据，说明选择理由
- 输出纯JSON，格式：{{"pairs": [{{"question": "...", "answer": "..."}}]}}

参考材料：
{chunk}
""",
}

def generate_qa_batch(chunks: list[dict], task_type: str, n_per_chunk: int = 3) -> list[dict]:
    """批量生成QA对"""
    all_pairs = []
    prompt_template = TASK_PROMPTS[task_type]

    for i, chunk in enumerate(chunks):
        chunk_text = chunk.get("chunk_text", "")
        if len(chunk_text) < 200:
            continue

        prompt = prompt_template.format(
            n=n_per_chunk,
            chunk=chunk_text[:2500]  # 限制长度避免超出上下文
        )

        try:
            resp = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )
            raw = resp.content[0].text

            # 清理JSON
            raw = raw.strip()
            if raw.startswith("```"):
                raw = re.sub(r'^```json?\n?', '', raw)
                raw = re.sub(r'\n?```$', '', raw)

            data = json.loads(raw)
            pairs = data.get("pairs", [])

            for pair in pairs:
                pair["source_chunk_id"] = chunk.get("chunk_id")
                pair["source_doi"] = chunk.get("doi")
                pair["task_type"] = task_type
                all_pairs.append(pair)

            if (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{len(chunks)} chunks, {len(all_pairs)} pairs generated")

        except Exception as e:
            print(f"  Error on chunk {i}: {e}")
            continue

    return all_pairs


def quality_filter(pairs: list[dict]) -> list[dict]:
    """QA对质量过滤"""
    filtered = []
    for pair in pairs:
        q = pair.get("question", "")
        a = pair.get("answer", "")

        # 基本长度过滤
        if len(q) < 10 or len(a) < 30:
            continue
        # 答案不能太短（专业问答需要足够详细）
        if len(a) < 100 and pair.get("task_type") != "simple_fact":
            continue
        # 去除明显不相关的
        if any(bad in q.lower() for bad in ["please", "thank you", "could you"]):
            continue

        filtered.append(pair)

    # 简单去重（基于问题文本前50字符）
    seen = set()
    deduped = []
    for pair in filtered:
        key = pair["question"][:50]
        if key not in seen:
            seen.add(key)
            deduped.append(pair)

    return deduped
```

### 8.3 数据集格式化

```python
def to_sharegpt_format(pairs: list[dict], system_prompt: str) -> list[dict]:
    """ShareGPT格式（兼容LLaMA Factory, ms-swift等主流框架）"""
    return [
        {
            "conversations": [
                {"from": "system", "value": system_prompt},
                {"from": "human", "value": pair["question"]},
                {"from": "gpt", "value": pair["answer"]}
            ],
            "source": pair.get("source_doi", ""),
            "task_type": pair.get("task_type", ""),
        }
        for pair in pairs
    ]

# 系统提示词（为CSDM定制）
SYSTEM_PROMPT = """你是一个专注于复合材料结构设计的AI专家助手，具备以下核心能力：
1. 复合材料力学：CLT层合板理论、屈曲分析、失效准则（Hashin, Tsai-Wu, LaRC等）
2. 有限元仿真：ABAQUS建模、周期边界条件、渐进损伤分析
3. 结构设计：加筋壁板铺层优化、质量-强度权衡、工程规范
4. 多尺度分析：RVE均质化、代理模型、数据驱动方法

回答时请：
- 使用专业术语，公式用LaTeX格式（独立公式用$$...$$，行内用$...$）
- 给出工程实践中的具体建议和依据
- 必要时引用失效准则或设计标准"""

# 生成并保存
composite_sft_data = to_sharegpt_format(all_filtered_pairs, SYSTEM_PROMPT)

with open("data/composite_sft_dataset.json", "w", encoding="utf-8") as f:
    json.dump(composite_sft_data, f, ensure_ascii=False, indent=2)

print(f"SFT dataset: {len(composite_sft_data)} samples")
```

### 8.4 数据集质量统计

在提交训练前，运行以下统计检查：

```python
import statistics

def analyze_dataset(data: list[dict]):
    questions = [d["conversations"][1]["value"] for d in data]
    answers = [d["conversations"][2]["value"] for d in data]
    task_types = [d.get("task_type", "unknown") for d in data]

    print(f"Total samples: {len(data)}")
    print(f"Avg question length: {statistics.mean(len(q) for q in questions):.0f} chars")
    print(f"Avg answer length: {statistics.mean(len(a) for a in answers):.0f} chars")
    print(f"Samples with LaTeX formulas: {sum(1 for a in answers if '$$' in a or '$' in a)}")
    print("\nTask type distribution:")
    from collections import Counter
    for task, count in Counter(task_types).most_common():
        print(f"  {task}: {count} ({100*count/len(data):.1f}%)")
```

---

## 9. RAG知识库搭建

### 9.1 向量化方案

```python
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
import json

# 嵌入模型选择（本地部署，无需API费用）
# 推荐：BAAI/bge-m3（中英双语，效果最好）
# 备选：BAAI/bge-large-zh-v1.5（中文优化）

def build_vectorstore(chunks_jsonl: str, persist_dir: str = "data/vectorstore"):
    """构建复合材料专属向量数据库"""

    # 加载嵌入模型
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True}  # cosine相似度需要归一化
    )

    # 加载分块数据
    documents = []
    with open(chunks_jsonl, encoding="utf-8") as f:
        for line in f:
            chunk = json.loads(line)
            doc = Document(
                page_content=chunk["chunk_text"],
                metadata={
                    "chunk_id": chunk["chunk_id"],
                    "doi": chunk.get("doi", ""),
                    "title": chunk.get("title", ""),
                    "year": chunk.get("year", 0),
                    "journal": chunk.get("journal", ""),
                    "chunk_index": chunk.get("chunk_index", 0),
                }
            )
            documents.append(doc)

    print(f"Building vectorstore from {len(documents)} chunks...")

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name="composite_materials"
    )
    vectorstore.persist()
    print(f"Vectorstore built and saved to {persist_dir}")
    return vectorstore
```

### 9.2 检索策略

```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

def build_hybrid_retriever(vectorstore, documents: list):
    """
    混合检索：向量检索（语义）+ BM25（关键词）
    适合复合材料场景：既要语义理解，也要精确匹配专业术语
    """
    # 向量检索器
    vector_retriever = vectorstore.as_retriever(
        search_type="mmr",  # Maximum Marginal Relevance，避免重复
        search_kwargs={"k": 5, "fetch_k": 20}
    )

    # BM25关键词检索器
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 5

    # 混合（0.6权重给向量，0.4给BM25）
    hybrid_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.6, 0.4]
    )
    return hybrid_retriever


def retrieve_for_knowledge_agent(query: str, retriever, top_k: int = 5) -> list[dict]:
    """KNOWLEDGE_AGENT的检索接口"""
    docs = retriever.get_relevant_documents(query)
    results = []
    for doc in docs[:top_k]:
        results.append({
            "content": doc.page_content,
            "doi": doc.metadata.get("doi"),
            "title": doc.metadata.get("title"),
            "year": doc.metadata.get("year"),
            "journal": doc.metadata.get("journal"),
        })
    return results
```

### 9.3 增量更新机制

```python
def incremental_update(new_papers_jsonl: str, vectorstore_dir: str):
    """增量更新向量数据库（新论文爬取后调用）"""
    # 加载现有vectorstore
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    vectorstore = Chroma(
        persist_directory=vectorstore_dir,
        embedding_function=embeddings,
        collection_name="composite_materials"
    )

    # 获取已有DOI集合
    existing_ids = set(
        m["doi"] for m in vectorstore.get()["metadatas"] if m.get("doi")
    )

    # 加载新论文，只处理未收录的
    new_docs = []
    with open(new_papers_jsonl, encoding="utf-8") as f:
        for line in f:
            paper = json.loads(line)
            if paper.get("doi") not in existing_ids:
                chunks = chunk_paper(paper)
                for chunk in chunks:
                    new_docs.append(Document(
                        page_content=chunk["chunk_text"],
                        metadata={k: v for k, v in chunk.items() if k != "chunk_text"}
                    ))

    if new_docs:
        vectorstore.add_documents(new_docs)
        vectorstore.persist()
        print(f"Added {len(new_docs)} new chunks from {new_papers_jsonl}")
    else:
        print("No new documents to add")
```

---

## 10. 模型微调实操指南

### 10.1 环境配置

```bash
# 推荐使用 LLaMA Factory（支持Qwen, LLaMA, DeepSeek等）
pip install llamafactory

# 或使用 ms-swift（魔搭社区，国内更友好）
pip install ms-swift

# LoRA微调基础依赖
pip install transformers>=4.40 peft>=0.10 datasets accelerate bitsandbytes
```

### 10.2 LLaMA Factory 配置文件

`composite_sft_config.yaml`：
```yaml
### 模型配置
model_name_or_path: Qwen/Qwen2.5-7B-Instruct

### 训练方式
stage: sft
do_train: true
finetuning_type: lora

### LoRA参数
lora_rank: 16           # 秩，16是性能/参数平衡点
lora_alpha: 32          # 通常设为rank的2倍
lora_dropout: 0.05
lora_target: q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj

### 数据集
dataset: composite_materials_sft
dataset_dir: data/
template: qwen           # 对应Qwen2模型的聊天模板

### 训练参数
output_dir: models/composite_qwen25_7b_lora
num_train_epochs: 3
per_device_train_batch_size: 2
gradient_accumulation_steps: 8   # 等效batch_size = 2*8 = 16
learning_rate: 2.0e-4
lr_scheduler_type: cosine
warmup_ratio: 0.05
max_grad_norm: 1.0
fp16: true                # 或 bf16: true（A100等支持BF16的GPU）

### 序列长度
cutoff_len: 2048           # 复合材料文献可能较长，可适当增大

### 评估
val_size: 0.01
eval_strategy: steps
eval_steps: 100
save_steps: 500

### 日志
logging_steps: 10
report_to: tensorboard
```

运行训练：
```bash
llamafactory-cli train composite_sft_config.yaml
```

### 10.3 QLoRA（量化微调，降低显存要求）

当GPU显存不足（<24GB）时，使用4-bit量化：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
import torch

# 4-bit量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    quantization_config=bnb_config,
    device_map="auto"
)

# LoRA配置
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    bias="none"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# 输出示例: trainable params: 20,971,520 || all params: 7,721,734,144 || trainable%: 0.27
```

### 10.4 训练后验证

```python
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

# 加载基座模型 + LoRA适配器
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
model = PeftModel.from_pretrained(base_model, "models/composite_qwen25_7b_lora")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

# 测试用例
test_cases = [
    "解释T300/5208碳纤维复合材料加筋壁板的屈曲失效机制",
    "写一个ABAQUS Python脚本，建立一个[0/90/±45]_s铺层的复合材料板，施加单轴压缩载荷并进行屈曲分析",
    "对于设计临界载荷为200kN/m的CFRP加筋壁板，如何选择加筋形式和铺层比例？",
]

for question in test_cases:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=500, temperature=0.7)
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)

    print(f"\nQ: {question}")
    print(f"A: {response[:500]}...")
    print("-" * 80)
```

---

## 11. CSDM多智能体架构集成方案

### 11.1 各Agent知识需求分析

| Agent | 核心知识需求 | 推荐方案 |
|---|---|---|
| **ORCHESTRATOR** | 任务分解逻辑、Agent路由 | 通用大模型API（Claude/Qwen-Max） |
| **CANDIDATE_GEN** | 铺层方案生成、设计空间探索 | RAG（设计手册+历史方案） + SFT |
| **SCREENER** | 工程可行性评估、约束检查 | RAG（规范/标准文献） |
| **FEM_AGENT** | ABAQUS脚本生成、结果解析 | SFT微调（ABAQUS专项） |
| **KNOWLEDGE_AGENT** | 文献检索、知识问答 | **RAG核心，全量文献语料库** |
| **REPORT_GEN** | 报告写作、结果可视化 | 通用大模型API |

### 11.2 KNOWLEDGE_AGENT集成代码

```python
from anthropic import Anthropic

client = Anthropic()

class KnowledgeAgent:
    def __init__(self, retriever):
        self.retriever = retriever
        self.system_prompt = """你是CSDM系统中的知识检索专家。
你的职责是：
1. 从复合材料文献数据库中检索相关知识
2. 基于检索结果回答技术问题
3. 为其他Agent提供权威的文献支撑

回答时必须注明知识来源（论文标题/DOI）。"""

    def query(self, question: str, conversation_history: list = None) -> str:
        # 检索相关文献
        retrieved_docs = self.retriever.retrieve_for_knowledge_agent(question, top_k=5)

        # 构建上下文
        context = "\n\n".join([
            f"[来源：{d['title']} ({d['year']}, {d['journal']})]\n{d['content']}"
            for d in retrieved_docs
        ])

        # 构建消息
        messages = conversation_history or []
        messages.append({
            "role": "user",
            "content": f"""请基于以下文献内容回答问题。

问题：{question}

检索到的相关文献：
{context}

请给出专业、准确的回答，并标注信息来源。"""
        })

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1500,
            system=self.system_prompt,
            messages=messages
        )
        return response.content[0].text


class FEMAgent:
    """FEM_AGENT：基于微调模型的ABAQUS脚本生成"""

    def __init__(self, use_finetuned: bool = True):
        self.use_finetuned = use_finetuned
        if use_finetuned:
            # 加载微调后的本地模型
            self._load_finetuned_model()
        else:
            # 降级到API模式
            self.client = Anthropic()

    def generate_abaqus_script(self, design_spec: dict) -> str:
        """
        design_spec示例：
        {
            "geometry": "stiffened panel, 500mm x 500mm, blade stiffeners at 100mm spacing",
            "layup": "[0/90/±45]_2s",
            "material": "T300/5208 CFRP",
            "loading": "uniaxial compression, 300kN total",
            "analysis_type": "linear buckling"
        }
        """
        prompt = f"""为以下复合材料结构生成完整的ABAQUS Python建模脚本：

几何：{design_spec['geometry']}
铺层：{design_spec['layup']}
材料：{design_spec['material']}
载荷：{design_spec['loading']}
分析类型：{design_spec['analysis_type']}

要求：
1. 完整的脚本，包含材料定义、几何建模、网格划分、边界条件、载荷施加和提交步骤
2. 添加详细注释
3. 使用SI单位（Pa, N, m）"""

        if self.use_finetuned:
            return self._generate_local(prompt)
        else:
            resp = self.client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=3000,
                messages=[{"role": "user", "content": prompt}]
            )
            return resp.content[0].text
```

---

## 12. 数据量估算与资源规划

### 12.1 文献数据量目标

| 分类 | 数量目标 | 来源 |
|---|---|---|
| 复合材料结构力学 | ~1000篇 | ScienceDirect, S2 |
| 屈曲/后屈曲分析 | ~500篇 | Composite Structures, TWS |
| 多尺度/均质化 | ~300篇 | Composites Science and Technology |
| FEM建模方法 | ~300篇 | Int. J. Solids and Structures |
| AI辅助设计 | ~200篇 | 近5年跨领域论文 |
| ABAQUS应用案例 | ~200篇 | 各期刊 |
| 合计目标 | **~2500篇** | |

### 12.2 SFT数据集规模

| 任务类型 | 目标数量 | 生成方式 |
|---|---|---|
| 材料力学QA | 10,000条 | 论文自动生成（每篇5条） |
| ABAQUS脚本生成 | 3,000条 | 半人工构造 |
| FEM结果分析 | 3,000条 | 论文结果章节提取 |
| 设计推理 | 2,000条 | 设计手册+案例 |
| 公式解释 | 2,000条 | 教材+论文 |
| **合计** | **~20,000条** | |

### 12.3 计算资源估算

| 任务 | 配置需求 | 时间估计 |
|---|---|---|
| 爬取2500篇元数据 | 普通CPU | ~4小时 |
| 下载2500篇PDF（OA部分约1000篇） | 普通网络 | ~8小时 |
| MinerU解析1000篇PDF | 1 GPU (V100/A100) | ~10小时 |
| 生成20,000条QA（Claude API） | API调用费用约$80-150 | ~20小时 |
| LoRA微调Qwen2.5-7B（20K条，3 epochs） | 1 A100 40G | ~8小时 |
| 向量化2500篇文档（bge-m3） | 1 GPU | ~2小时 |

---

## 13. 实施路线图

### Phase 1：数据采集（第1-2周）

```
Week 1:
  □ 配置Python爬取环境（paperscraper, requests, OpenAlex）
  □ 爬取arXiv + S2 + OpenAlex 元数据（目标2500篇）
  □ 获取OA PDF链接，启动批量下载（预计1000篇有OA全文）
  □ 校园网VPN下载ScienceDirect重要论文（~200篇核心）

Week 2:
  □ MinerU批量解析已下载PDF
  □ 运行数据清洗脚本，过滤低质量解析结果
  □ 人工抽查20篇，验证公式/表格解析质量
  □ 输出：papers_corpus.jsonl（含元数据+全文Markdown）
```

### Phase 2：RAG系统搭建（第3周）

```
Week 3:
  □ 实现论文分块逻辑（公式感知分块）
  □ 部署BAAI/bge-m3嵌入模型
  □ 构建Chroma向量数据库（~50万个chunk）
  □ 实现BM25+向量混合检索
  □ 接入KNOWLEDGE_AGENT，进行10个测试用例验证
  □ 输出：可查询的复合材料RAG系统
```

### Phase 3：SFT数据集构建（第4-5周）

```
Week 4:
  □ 实现自动QA生成管线（基于Claude API）
  □ 针对3种任务类型（力学QA、ABAQUS脚本、设计推理）分批生成
  □ 运行质量过滤脚本

Week 5:
  □ 人工抽查200条QA，标注质量
  □ 格式化为ShareGPT格式
  □ 统计分析数据集覆盖度和质量
  □ 输出：composite_sft_20k.json
```

### Phase 4：模型微调（第6-7周）

```
Week 6:
  □ 配置LLaMA Factory微调环境
  □ Qwen2.5-7B-Instruct QLoRA微调（力学QA + 设计推理任务）
  □ 监控训练loss，调整超参数

Week 7:
  □ FEM_AGENT专项：ABAQUS脚本生成LoRA微调
  □ 评估：10个标准测试用例对比微调前后
  □ 模型合并（可选）：将多个LoRA适配器合并
  □ 输出：composite_qwen25_7b_v1.0（部署就绪模型）
```

### Phase 5：集成与迭代（第8周+）

```
□ 将微调模型集成到CSDM架构（替换FEM_AGENT的API调用）
□ 建立增量更新机制（每月爬取新论文，更新RAG知识库）
□ A/B测试：RAG vs 微调模型在各任务上的表现
□ 根据实际使用反馈迭代数据集和微调策略
```

---

## 14. 参考资源汇总

### 14.1 核心工具链

| 工具 | GitHub/地址 | 用途 |
|---|---|---|
| paperscraper | https://github.com/jannisborn/paperscraper | 论文元数据爬取 |
| MinerU | https://github.com/opendatalab/MinerU | PDF解析首选 |
| Nougat | https://github.com/facebookresearch/nougat | 公式密集型PDF |
| Docling | https://github.com/DS4SD/docling | IBM开源，图表提取 |
| GROBID | https://github.com/kermitt2/grobid | 参考文献/元数据提取 |
| LLaMA Factory | https://github.com/hiyouga/LLaMA-Factory | 微调框架（推荐） |
| ms-swift | https://github.com/modelscope/ms-swift | 微调框架（国内备选） |
| LangChain | https://github.com/langchain-ai/langchain | RAG管线 |
| Chroma | https://github.com/chroma-core/chroma | 向量数据库 |
| BAAI/bge-m3 | https://huggingface.co/BAAI/bge-m3 | 中英双语嵌入模型 |

### 14.2 API 接入方式

| API | 文档地址 | 免费额度 |
|---|---|---|
| Semantic Scholar | https://api.semanticscholar.org/api-docs | 1 req/s（无key），10 req/s（有key） |
| OpenAlex | https://docs.openalex.org | 完全免费，推荐mailto参数 |
| Unpaywall | https://unpaywall.org/products/api | 完全免费 |
| Elsevier | https://dev.elsevier.com | 机构订阅下可申请 |

### 14.3 参考文献

1. Kinney, D. et al. (2023). The Semantic Scholar Open Data Platform. *arXiv:2301.10140*.
2. Blecher, L. et al. (2024). Nougat: Neural Optical Understanding for Academic Documents. *ICLR 2024*.
3. Hu, B. et al. (2025). SciLitLLM: How to Adapt LLMs for Scientific Literature Understanding. *ICLR 2025*.
4. Auer, C. et al. (2024). Docling Technical Report. *arXiv:2408.09869*.
5. Sun, G. et al. (2025). Fine-tuning large language models for domain adaptation. *npj Computational Materials*, 11, 84.
6. Hu, E. et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. *ICLR 2022*.

---

> **最后更新**：2026年4月27日  
> **维护**：刘正鹏 | lzp03@stu.xjtu.edu.cn | @ZPL-03  
> **项目**：CSDM - 面向复合材料结构设计的多智能体AI辅助系统
