# 跨境电商合规智能问答系统 (Cross-Border Compliance Agent)

<p align="center">
  <a href="https://github.com/chenqianlei51525/cross-border-compliance-agent">
    <img src="https://img.shields.io/badge/license-MIT-green" alt="license" />
  </a>
  <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="python" />
  <img src="https://img.shields.io/badge/RAG-Agentic-orange" alt="agentic rag" />
  <img src="https://img.shields.io/badge/Milvus-2.x-0095d3" alt="milvus" />
  <img src="https://img.shields.io/badge/Neo4j-5.x-018bff" alt="neo4j" />
</p>

> 应小合规 · 跨境电商卖家的 CE / FCC / RoHS / UN38.3 / PSE / KCC 全合规闭环助手。

---

## 这是什么

把 **RAG 当成 Agent 的工具组件**——LLM 自己决定何时调混合检索、何时调知识图谱多跳推理、何时让反思检索、何时拉一手法规网页。

业务目标：
- 跨境卖家在欧盟/美国/日韩/东南亚卖货时，**几秒钟**拿到产品需要的合规清单（CE-RED / RoHS / UN38.3 / FCC / PSE 等）。
- 法规经常翻新，避免"凭印象答 / 凭旧文本答"。
- 审计/平台合规：每条回答都能追溯到底层法规 chunk 或图谱三元组。

简历话术对应实现：

| 简历描述 | 代码位置 |
| --- | --- |
| 知识图谱驱动的跨文档法规推理 | `kg/` (Neo4j client + 抽取 + 多跳) |
| Parent-Child 分块 + 混合召回 | `rag_qa/core/` (BGE-M3 + BM25 + RRF) |
| Reflective RAG 反思检索 | `rag_qa/core/reflective.py` (Self-RAG + Corrective RAG) |
| Query 增强与精排 | `agent/tools.py` (query_rewrite / sub_question / hyde / rerank) |
| 两级问答链路 + RAGAS 评测 | MySQL+Redis 缓存 + `evaluate/pipeline.py` (RAGAS + Agent metrics) |
| **RAG 作为 Agent 组件** | `agent/agent.py` (ReAct + Plan-and-Execute) |

---

## 架构

```
                         ┌──────────────────────────────────┐
                         │   ComplianceAgent (ReAct/Plan)  │
                         │   LLM 自主决策: Thought→Action   │
                         └──────────────┬───────────────────┘
                                        │
        ┌─────────┬─────────┬───────────┼────────────┬────────────┐
        ▼         ▼         ▼           ▼            ▼            ▼
   hybrid_search  kg_query  query_rewrite sub_question self_critique   web_fetch
   (BM25+Dense+RRF)  (Neo4j hops)  ─────────┬────────────────────────────┘
                                          ▼
                          ┌──────────────────────────────┐
                          │  RAG (Reflective + Rerank)   │
                          │  Milvus + BGE-M3 + BGE-reranker v2 │
                          └──────────────────────────────┘
                                          │
                                          ▼
                          ┌──────────────────────────────┐
                          │   MinerU 风格 PDF→结构化      │
                          │   → Parent-Child 分块          │
                          └──────────────────────────────┘
```

---

## 快速开始

### 1. 准备基础设施
- Python 3.10+
- Docker（推荐）：`docker compose -f docker/docker-compose.yml up -d`
  - MySQL 8（合规问答标准答案库）
  - Redis 7（热点缓存）
  - Milvus 2.x（向量库）
  - Neo4j 5.x（法规图谱，可选，未连接时自动降级）
- LLM：阿里云 DashScope（默认）/ OpenAI / 通义千问 OpenAI 兼容 API
- Embedding：`BAAI/bge-m3`（已下载到 `rag_qa/models/bge-m3`）

### 2. 安装 & 启动
```bash
git clone https://github.com/chenqianlei51525/cross-border-compliance-agent.git
cd cross-border-compliance-agent
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-windows.txt   # 或 requirements-mac.txt
cp config.example.ini config.ini          # 填入 DASHSCOPE_API_KEY 等
python app.py                             # http://127.0.0.1:8003
```

### 3. 访问
- 前端演示：`http://127.0.0.1:8003/`
- API 文档：`http://127.0.0.1:8003/docs`
- 工具清单：`GET /api/agent/tools`
- 同步问答：`POST /api/agent/chat`
- 流式推理：`POST /api/agent/stream`

---

## 核心能力

### 1. Agent 编排（最核心）
- **ReAct 模式**：单步思考→工具调用→反馈，迭代直到 FinalAnswer
- **Plan-and-Execute 模式**：复杂问题先列计划再分步执行
- **对话记忆**：每会话保留事实 + 最近 8 轮

```python
from compliance.system import get_system
sys_ = get_system()
result = sys_.ask("蓝牙耳机出口德国需要哪些认证？", use_agent=True)
print(result["answer"], result["trace"])
```

### 2. 知识图谱（Neo4j）
- 实体类型：ProductCategory / CertificationStandard / MarketRegion / TestItem
- 关系：REQUIRES / APPLIES_TO / CONTAINS / CITES
- 多跳推理：自动抽取入口节点，BFS/Neo4j Cypher 双路实现
- Neo4j 未连接时 in-memory 降级，仍然能演示

```bash
curl -X POST http://127.0.0.1:8003/api/kg/reason \
     -G --data-urlencode "question=蓝牙耳机出口德国需要哪些认证" --data-urlencode "hops=3"
```

### 3. Reflective RAG（Self-RAG + Corrective RAG）
- 每条候选由 LLM 打 relevant / partial / irrelevant
- 比例不足 → 触发 Corrective：query rewrite → 再检索
- 最多 3 轮，自动剔除低质文档

### 4. MinerU 风格文档解析
- PDF (PyMuPDF / pdfplumber 兜底 / 官方 MinerU API)
- HTML / Markdown / DOCX / TXT
- 输出 MinerU 兼容 JSON：blocks[] / page_idx / bbox

### 5. 两级问答链路
- MySQL 标准答案库（人工校准的高频问答）
- Redis 热点缓存，BM25 阈值 0.85 高置信时直接返回
- 失败回退到 RAG/Agent

### 6. 评测流水线
```bash
python -m evaluate.run --dataset evaluate/sample_set.json
# 输出 evaluate/report.md + evaluate/results.csv
```
指标：faithfulness / answer_relevancy / context_precision / context_recall + Agent 轨迹指标。

---

## 仓库目录

```
├── agent/                 # Agent 编排 (ReAct + Plan-Execute)
│   ├── agent.py
│   ├── tools.py           # 工具集合（hybrid_search / kg_query / rewrite / ...）
│   ├── prompts.py
│   ├── registry.py
│   └── memory.py
├── compliance/            # 业务系统集成
│   └── system.py          # CrossBorderComplianceSystem
├── kg/                    # Neo4j 法规知识图谱
├── rag_qa/
│   ├── core/              # 切分/检索/RAGSystem/RAGAS
│   │   ├── reflective.py  # Reflective RAG 反思循环
│   │   └── ...
│   ├── mineru/            # MinerU 风格文档解析
│   └── models/            # bge-m3 / bge-reranker-large
├── evaluate/              # 评测流水线 + 样本集
├── knowledge_storage/
│   └── seeds/             # 法规种子数据 + FAQ 样例
├── static/                # 前端
├── new_main.py / app.py   # FastAPI 入口
└── config.ini             # 配置
```

---

## Roadmap

- [x] Agentic RAG（ReAct + Plan-and-Execute）
- [x] Neo4j 法规知识图谱 + 多跳推理（in-memory 降级）
- [x] Reflective RAG 反思检索
- [x] MinerU 适配（PyMuPDF / pdfplumber / 官方 API 三档）
- [x] FastAPI SSE 流式推理 + 轨迹可视化前端
- [x] RAGAS + Agent 轨迹评测
- [ ] Web 抓取工具实装（突破 SaaS 防火墙 / Cloudflare）
- [ ] 多语言（中文 / 英文 / 日文 法规）
- [ ] 法规增量更新 push 通知

## 贡献

欢迎提 Issue / PR：
- 新增法规类别（KC、UKCA、CCC 等）只改 `compliance/__init__.py`
- 新增工具只改 `agent/tools.py`
- 新增评测样本改 `evaluate/sample_set.py`

## License

MIT — see [LICENSE](LICENSE).
