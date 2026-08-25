"""内置工具集合。

把现有的 RAG / BM25 / Neo4j / MinerU / Web 能力包装为 Agent 工具。

工具集：
- hybrid_search:  混合召回（BM25 + BGE-M3 稠密，RRF 融合）
- knowledge_graph_query:  Neo4j 多跳推理（产品→认证→检测项）
- query_rewrite:   Query 标准化（口语化→结构化）
- sub_question_decompose: 复杂问题拆解
- hyde_search:    HyDE 假设答案检索
- self_critique:  自批判检索结果，决定是否二次检索
- web_fetch:      网页抓取最新法规
- mysql_qa_search: 高频问题精确匹配（Redis 缓存）
- doc_lookup:     通过 MinerU 解析的 chunk ID 直接拉原文

每个工具的 schema 都设计为 LLM 友好（字段少，注释清晰）。
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Optional

from agent.registry import ToolRegistry, ToolSpec, tool
from base import logger, Config


def build_default_registry(
    vector_store=None,
    rag_system=None,
    kg_client=None,
    bm25_search=None,
    mineru_loader=None,
    web_fetcher=None,
    llm=None,
) -> ToolRegistry:
    """构造默认工具注册中心。

    所有依赖都是可选的——没注入就返回提示，由 Agent 自行决定回退方案。
    """
    reg = ToolRegistry()

    # ---- 1. hybrid_search ----
    def hybrid_search(q: str, top_k: int = 6,
                      source_filter: Optional[str] = None) -> Dict[str, Any]:
        """混合语义+关键词检索。"""
        if vector_store is None:
            return _unavailable("hybrid_search")
        try:
            docs = vector_store.hybrid_search_with_rerank(q, k=top_k)
        except Exception as e:
            logger.warning("hybrid_search failed: %s", e)
            return {"text": f"（hybrid_search 调用失败：{e}）",
                    "citations": []}
        return _docs_to_payload(docs, source_filter=source_filter, prefix="hybrid")

    reg.register(ToolSpec(
        name="hybrid_search",
        description="混合检索（BM25 + BGE-M3 稠密向量 + RRF 融合 + BGE reranker v2），语义匹配首选。",
        schema={
            "type": "object",
            "properties": {
                "q": {"type": "string", "description": "用户问题或子问题原文"},
                "top_k": {"type": "integer", "default": 6},
                "source_filter": {"type": "string",
                                  "description": "可选：ce / fcc / rohs / un38.3 等法规类别"},
            },
            "required": ["q"],
        },
        fn=hybrid_search,
        examples=[{"q": "CE-RED 认证覆盖哪些产品", "top_k": 6}],
    ))

    # ---- 2. knowledge_graph_query ----
    def knowledge_graph_query(question: str, hops: int = 2) -> Dict[str, Any]:
        """基于 Neo4j 法规图谱多跳推理，适合『某产品出口某国需要哪些认证』这类跨文档问题。"""
        if kg_client is None or not getattr(kg_client, "available", False):
            return _unavailable("knowledge_graph_query",
                                hint="Neo4j 未连接，将回退到 hybrid_search")
        try:
            triples = kg_client.multi_hop_reasoning(question, hops=hops)
            text_lines = ["图谱多跳推理结果："]
            citations = []
            for i, t in enumerate(triples, 1):
                text_lines.append(
                    f"{i}. {t.get('from_label','')} -[{t.get('rel','')}]-> "
                    f"{t.get('to_label','')}（来源：{t.get('evidence','')[:80]}）"
                )
                citations.append({"id": f"kg_{i}", "title": t.get("evidence", "图谱片段")})
            return {"text": "\n".join(text_lines), "citations": citations}
        except Exception as e:
            return {"text": f"（知识图谱调用失败：{e}）", "citations": []}

    reg.register(ToolSpec(
        name="knowledge_graph_query",
        description="Neo4j 法规图谱多跳推理，适用于跨文档法规关系问题（产品→认证→标准）。",
        schema={
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "hops": {"type": "integer", "default": 2},
            },
            "required": ["question"],
        },
        fn=knowledge_graph_query,
        examples=[{"question": "蓝牙耳机出口德国需要哪些认证",
                  "hops": 3}],
    ))

    # ---- 3. query_rewrite ----
    def query_rewrite(q: str) -> Dict[str, Any]:
        """把口语化问题标准化为检索词，输出更精准的检索串。"""
        prompt = (
            "你是跨境电商合规检索专家。把用户口语化问题改写为 1~3 条更精确的检索 query，"
            "保留法规/标准/产品/国家关键词，补全隐含上下文，去掉寒暄词。\n"
            "只返回 JSON：{\"rewrites\":[\"...\",\"...\"],\"standard_query\":\"...\"}\n"
            f"用户问题：{q}"
        )
        out = _call_llm(llm, prompt)
        try:
            m = re.search(r"\{.*\}", out, re.DOTALL)
            data = json.loads(m.group(0)) if m else {}
        except Exception:
            data = {"rewrites": [q], "standard_query": q}
        lines = ["改写检索串："]
        for r in data.get("rewrites", []):
            lines.append(f"- {r}")
        lines.append(f"标准化串：{data.get('standard_query', q)}")
        return {
            "text": "\n".join(lines),
            "citations": [],
            "_rewritten": data.get("standard_query", q),
        }

    reg.register(ToolSpec(
        name="query_rewrite",
        description="改写用户问题为更精准的检索串（含关键词扩写、隐含信息补全），"
                    "复杂问题或冷门术语先调此工具再检索。",
        schema={
            "type": "object",
            "properties": {"q": {"type": "string"}},
            "required": ["q"],
        },
        fn=query_rewrite,
    ))

    # ---- 4. sub_question_decompose ----
    def sub_question_decompose(q: str) -> Dict[str, Any]:
        """把复杂问题拆成 2~4 个子问题，便于并行检索。"""
        prompt = (
            "把下面跨境合规问题拆成 2~4 个子问题，便于并行检索。"
            "输出 JSON：{\"sub_questions\":[\"...\",\"...\"]}\n"
            f"原问题：{q}"
        )
        out = _call_llm(llm, prompt)
        try:
            m = re.search(r"\{.*\}", out, re.DOTALL)
            data = json.loads(m.group(0)) if m else {"sub_questions": [q]}
        except Exception:
            data = {"sub_questions": [q]}
        subs = data.get("sub_questions") or [q]
        lines = ["子问题拆解："]
        for i, s in enumerate(subs, 1):
            lines.append(f"{i}. {s}")
        return {
            "text": "\n".join(lines),
            "citations": [],
            "_sub_questions": subs,
        }

    reg.register(ToolSpec(
        name="sub_question_decompose",
        description="把复杂合规问题拆成多个子问题，配合多次 hybrid_search 并行使用。"
                    "问题超过 1 个国家/产品/认证时优先调此工具。",
        schema={
            "type": "object",
            "properties": {"q": {"type": "string"}},
            "required": ["q"],
        },
        fn=sub_question_decompose,
    ))

    # ---- 5. hyde_search ----
    def hyde_search(q: str, top_k: int = 4) -> Dict[str, Any]:
        """HyDE：用 LLM 生成假设答案再 embedding 检索，扩展短问题的语义匹配。"""
        if rag_system is None:
            return _unavailable("hyde_search", hint="RAGSystem 未加载")
        try:
            docs = rag_system._retrieve_with_hyde(q)  # noqa: SLF001（内部复用）
            return _docs_to_payload(docs[:top_k], prefix="hyde")
        except Exception as e:
            return {"text": f"（hyde_search 失败：{e}）", "citations": []}

    reg.register(ToolSpec(
        name="hyde_search",
        description="HyDE 假设文档检索——先用 LLM 生成假设答案，再用答案 embedding 检索。",
        schema={
            "type": "object",
            "properties": {
                "q": {"type": "string"},
                "top_k": {"type": "integer", "default": 4},
            },
            "required": ["q"],
        },
        fn=hyde_search,
    ))

    # ---- 6. self_critique ----
    def self_critique(question: str, current_evidence: str) -> Dict[str, Any]:
        """自我批判当前证据是否足够，输出 JSON：sufficient / missing[] / next_action。"""
        prompt = (
            "你是合规答案审查员。基于下面【当前证据】判断能否回答【问题】。\n"
            "输出严格 JSON："
            "{\"sufficient\": bool, \"missing\": [\"...\"], \"next_action\": \"...\"}\n"
            f"问题：{question}\n"
            f"当前证据（前 1500 字）：{current_evidence[:1500]}"
        )
        out = _call_llm(llm, prompt)
        try:
            m = re.search(r"\{.*\}", out, re.DOTALL)
            data = json.loads(m.group(0)) if m else {"sufficient": True}
        except Exception:
            data = {"sufficient": True}
        return {
            "text": json.dumps(data, ensure_ascii=False, indent=2),
            "citations": [],
            "_critique": data,
        }

    reg.register(ToolSpec(
        name="self_critique",
        description="自批判当前证据是否足够，不足时建议下一步动作（query_rewrite / sub_question / web_fetch）。"
                    "Reflective RAG 核心工具，检索一轮后建议调用。",
        schema={
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "current_evidence": {"type": "string"},
            },
            "required": ["question", "current_evidence"],
        },
        fn=self_critique,
    ))

    # ---- 7. mysql_qa_search ----
    def mysql_qa_search(q: str) -> Dict[str, Any]:
        """高频合规问题的标准答案（MySQL + Redis 缓存）。"""
        if bm25_search is None:
            return _unavailable("mysql_qa_search")
        try:
            answer, need_rag = bm25_search.search(q, threshold=0.85)
            if answer:
                return {"text": f"MySQL 标准答案：{answer}", "citations": [{
                    "id": "qa_db", "title": "高频合规问答库"
                }]}
            return {"text": "（标准问答库无匹配，need_rag=True）", "citations": []}
        except Exception as e:
            return {"text": f"（mysql_qa_search 失败：{e}）", "citations": []}

    reg.register(ToolSpec(
        name="mysql_qa_search",
        description="高频合规问题的精确匹配（MySQL+Redis），置信度高时直接返回标准答案。",
        schema={
            "type": "object",
            "properties": {"q": {"type": "string"}},
            "required": ["q"],
        },
        fn=mysql_qa_search,
    ))

    # ---- 8. doc_lookup ----
    def doc_lookup(chunk_id: str) -> Dict[str, Any]:
        """通过 MinerU 解析的 chunk_id 拉原文片段。"""
        if mineru_loader is None:
            return _unavailable("doc_lookup",
                                hint="MinerU 适配层未启用")
        try:
            text = mineru_loader.fetch_chunk(chunk_id)
            return {"text": text, "citations": [{"id": chunk_id, "title": chunk_id}]}
        except Exception as e:
            return {"text": f"（doc_lookup 失败：{e}）", "citations": []}

    reg.register(ToolSpec(
        name="doc_lookup",
        description="通过 chunk_id 直接取原文片段，用于回答中需要引用原文细节时。",
        schema={
            "type": "object",
            "properties": {"chunk_id": {"type": "string"}},
            "required": ["chunk_id"],
        },
        fn=doc_lookup,
    ))

    # ---- 9. web_fetch ----
    def web_fetch(url: str, max_chars: int = 4000) -> Dict[str, Any]:
        """网页抓取——查最新法规或非本地资料。"""
        if web_fetcher is None:
            return _unavailable("web_fetch", hint="未启用网络工具，可能被沙箱策略阻止")
        try:
            text = web_fetcher.fetch(url, max_chars=max_chars)
            return {"text": text[:max_chars], "citations": [
                {"id": "web", "title": url}
            ]}
        except Exception as e:
            return {"text": f"（web_fetch 失败：{e}）", "citations": []}

    reg.register(ToolSpec(
        name="web_fetch",
        description="抓取公开网页，用于最新法规、官方公告查询。本地资料不足时使用。",
        schema={
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "max_chars": {"type": "integer", "default": 4000},
            },
            "required": ["url"],
        },
        fn=web_fetch,
    ))

    return reg


# ---------- helpers ----------
def _unavailable(name: str, hint: str = "") -> Dict[str, Any]:
    msg = f"（{name} 当前不可用：{hint or '依赖未连接或部署未启用'}）"
    return {"text": msg, "citations": []}


def _docs_to_payload(docs: List[Any], source_filter: Optional[str] = None,
                      prefix: str = "hit") -> Dict[str, Any]:
    """把内部 Document 转成 {'text':..., 'citations':[...]}。"""
    if not docs:
        return {"text": "（未检索到相关片段）", "citations": []}
    lines, cites = [], []
    for i, d in enumerate(docs, 1):
        meta = getattr(d, "metadata", {}) or {}
        title = meta.get("source") or meta.get("title") or f"{prefix}_{i}"
        if source_filter and meta.get("source") != source_filter:
            continue
        snippet = (d.page_content if hasattr(d, "page_content")
                   else str(d))[:600]
        lines.append(f"[{i}] {title}\n{snippet}")
        cites.append({"id": f"{prefix}_{i}", "title": title})
    if not lines:
        lines = ["（按来源过滤后无命中）"]
    return {"text": "\n\n".join(lines), "citations": cites}


def _call_llm(llm, prompt: str) -> str:
    """统一包装 llm 调用——llm 可能是 stream 或非 stream，统一收成字符串。"""
    if llm is None:
        return ""
    try:
        out = llm(prompt)
    except Exception as e:
        logger.warning("LLM call failed: %s", e)
        return ""
    if hasattr(out, "__iter__") and not isinstance(out, str):
        chunks = []
        for chunk in out:
            if isinstance(chunk, str):
                chunks.append(chunk)
            else:
                # 流式 OpenAI 风格
                try:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        chunks.append(delta)
                except Exception:
                    pass
        return "".join(chunks)
    return str(out)
