# Changelog

All notable changes to **Cross-Border Compliance Agent** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased] – 2026-08-26

### Added (本期改造)
- **Agentic RAG 层**：把 RAG 变成 Agent 的工具组件；LLM 通过 ReAct 与 Plan-and-Execute 两套模式自主决策 `hybrid_search` / `knowledge_graph_query` / `query_rewrite` / `sub_question_decompose` / `hyde_search` / `self_critique` / `mysql_qa_search` / `web_fetch` / `doc_lookup` 等 9 个工具。
- **Reflective RAG 反思检索**：`rag_qa/core/reflective.py`，Self-RAG 风格的 relevance 打分 + Corrective RAG 风格的 query rewrite 二次检索。
- **Neo4j 法规知识图谱**：`kg/`，从法规文本抽 (实体-关系-实体)，多跳推理；Neo4j 不可用时自动 in-memory 降级。
- **MinerU 风格文档解析适配层**：`rag_qa/mineru/`，输出 MinerU 兼容 JSON；本地 PyMuPDF / pdfplumber / 官方 MinerU API 三档兜底。
- **FastAPI Agent 端点**：`/api/agent/chat` `/api/agent/stream`(SSE) `/api/agent/tools` `/api/kg/reason` `/api/kg/build` `/api/parser/parse`。
- **评测流水线**：`evaluate/pipeline.py` 同时输出 RAGAS 与 Agent 特有指标，生成 Markdown 报告 + CSV。
- **前端可视化**：`static/` 重写 Agent 轨迹 + 工具清单 + 引用片段三栏布局。
- **业务场景切换**：从黑马 IT 教育领域改为跨境电商合规（CE/FCC/RoHS/UN38.3/PSE/KCC），`config.ini` 与 `compliance/__init__.py` 集中管理。
- **GitHub 提交准备**：`README.md`（架构图 + 模块对照表 + 快速开始）、`LICENSE`（MIT）、`.gitignore`、CI 工作流 `.github/workflows/ci.yml`。

### Changed
- `base/config.py` 默认 `VALID_SOURCES` 从 IT 学科切换为合规业务分类。
- `app.py` 标题与问候模板统一为『应小合规』品牌。

### Notes
- 项目原名 `Itcast_qa_system` (黑马 IT 教育场景)，本次改造统一为跨境电商合规场景。
