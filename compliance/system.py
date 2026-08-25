"""跨境电商合规智能问答系统——集成层。

把现有能力：
- BM25 高频问答（MySQL+Redis）
- Hybrid search 混合检索（Milvus+BGE-M3+RRF+rgrank）
- Reflective 反思检索
- Neo4j 知识图谱
- MinerU 风格文档解析
- Agent 编排（ReAct + Plan-and-Execute）
- 评估流水线

统一成一个入口，新 / 旧 FastAPI 路由都从这里取。
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Generator, List, Optional

from base import logger, Config
from mysql_qa import MySQLClient, RedisClient, BM25Search
from rag_qa import VectorStore, RAGSystem
from rag_qa.core.reflective import ReflectiveRetriever
from rag_qa.mineru import MinerUStyleParser
from kg import Neo4jKGClient, KGBuilder
from agent import ComplianceAgent, ConversationMemory
from agent.tools import build_default_registry


class CrossBorderComplianceSystem:
    """系统主入口。"""

    def __init__(self):
        self.conf = Config()
        self.logger = logger

        # ---- 基础服务 ----
        try:
            self.mysql_client = MySQLClient()
            self.redis_client = RedisClient()
            self.bm25_search = BM25Search(self.redis_client, self.mysql_client)
        except Exception as e:
            self.logger.warning("MySQL/Redis init failed, BM25 disabled: %s", e)
            self.mysql_client = None
            self.redis_client = None
            self.bm25_search = None

        # ---- LLM ----
        from openai import OpenAI
        self._llm_client = OpenAI(
            api_key=self.conf.DASHSCOPE_API_KEY,
            base_url=self.conf.DASHSCOPE_BASE_URL,
        )

        # ---- RAG 组件 ----
        self._llm_callable = self._call_dashscope
        try:
            self.vector_store = VectorStore()
            self.rag_system = RAGSystem(self.vector_store, self._llm_callable)
        except Exception as e:
            self.logger.warning("Vector store init failed: %s", e)
            self.vector_store = None
            self.rag_system = None

        # ---- 反思检索 ----
        if self.vector_store is not None:
            self.reflective = ReflectiveRetriever(
                retrieve_fn=self.vector_store.hybrid_search_with_rerank,
                llm=self._llm_callable,
                max_rounds=3,
                top_k=self.conf.RETRIEVAL_K,
            )
        else:
            self.reflective = None

        # ---- MinerU 风格文档解析 ----
        self.parser = MinerUStyleParser(
            mineru_endpoint=os.environ.get("MINERU_ENDPOINT"),
        )

        # ---- 知识图谱 ----
        neo4j_uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        neo4j_user = os.environ.get("NEO4J_USER", "neo4j")
        neo4j_pwd = os.environ.get("NEO4J_PASSWORD", "neo4j")
        self.kg_client = Neo4jKGClient(neo4j_uri, neo4j_user, neo4j_pwd)
        self.kg_builder = KGBuilder(self.kg_client, llm=self._llm_callable)

        # ---- Agent 工具注册 ----
        self.tool_registry = build_default_registry(
            vector_store=self.vector_store,
            rag_system=self.rag_system,
            kg_client=self.kg_client,
            bm25_search=self.bm25_search,
            mineru_loader=self.parser,
            web_fetcher=_SimpleWebFetcher(),
            llm=self._llm_callable,
        )

        # ---- Agent ----
        self.agent = ComplianceAgent(
            llm=self._llm_callable,
            tools=self.tool_registry,
        )

        # 会话记忆池
        self._memories: Dict[str, ConversationMemory] = {}

    # ---------- 基础 LLM 调用 ----------
    def _call_dashscope(self, prompt: str) -> str:
        """同步版 LLM 包装。"""
        try:
            resp = self._llm_client.chat.completions.create(
                model=self.conf.LLM_MODEL,
                messages=[
                    {"role": "system", "content": "你是跨境电商合规助手『应小合规』。"},
                    {"role": "user", "content": prompt},
                ],
                timeout=30,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            self.logger.error("LLM call failed: %s", e)
            return ""

    def call_dashscope_stream(self, prompt: str):
        """流式版 LLM 包装（给传统 FastAPI/SSE 用）。"""
        try:
            completion = self._llm_client.chat.completions.create(
                model=self.conf.LLM_MODEL,
                messages=[
                    {"role": "system", "content": "你是跨境电商合规助手『应小合规』。"},
                    {"role": "user", "content": prompt},
                ],
                timeout=30,
                stream=True,
            )
            for chunk in completion:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            self.logger.error("LLM stream failed: %s", e)
            yield f"[ERROR] {e}"

    # ---------- 业务接口 ----------
    def ask(self, question: str, source_filter: Optional[str] = None,
            use_agent: bool = True,
            session_id: Optional[str] = None) -> Dict[str, Any]:
        """同步版问答。

        - use_agent=True：走 ComplianceAgent（自动决策工具）
        - use_agent=False：直接走 Reflective RAG
        """
        t0 = time.time()
        if use_agent:
            mem = self._memories.setdefault(session_id or "default",
                                            ConversationMemory())
            final, trace = None, None
            for ev in self.agent.stream(question, session_id=session_id, memory=mem):
                if ev["type"] == "final":
                    final = ev["answer"]
                    trace = ev["trace"]
            return {
                "answer": final or "",
                "trace": trace or {},
                "session_id": session_id,
                "elapsed_ms": int((time.time() - t0) * 1000),
            }

        # 非 Agent 模式：直接 Reflective RAG
        if self.reflective is None:
            return {"answer": "（RAG 组件未就绪）", "trace": {}}
        res = self.reflective.retrieve(question)
        context_text = "\n".join(
            (j.doc.page_content if hasattr(j.doc, "page_content") else str(j.doc))
            for j in res.docs[: self.conf.RETRIEVAL_K]
        )
        prompt = (
            "基于以下参考资料回答合规问题，引用要写 [1][2] 等编号，"
            "末尾列出引用清单。"
            f"\n\n参考资料：{context_text}\n\n问题：{question}"
        )
        answer_chunks: List[str] = []
        for chunk in self.call_dashscope_stream(prompt):
            answer_chunks.append(chunk)
        answer = "".join(answer_chunks)
        return {
            "answer": answer,
            "trace": {"rounds": res.rounds, "judged_docs": [
                j.relevance for j in res.docs[: self.conf.RETRIEVAL_K]
            ]},
            "session_id": session_id,
            "elapsed_ms": int((time.time() - t0) * 1000),
        }

    def stream_agent(self, question: str,
                     session_id: Optional[str] = None
                     ) -> Generator[Dict[str, Any], None, None]:
        """流式 Agent 输出，便于 FastAPI SSE。"""
        mem = self._memories.setdefault(session_id or "default",
                                        ConversationMemory())
        yield from self.agent.stream(question, session_id=session_id, memory=mem)


# 一个最小 web fetcher，避免引入 requests
class _SimpleWebFetcher:
    def fetch(self, url: str, max_chars: int = 4000) -> str:
        try:
            import requests  # type: ignore
        except ImportError:
            return "（requests 未安装，web_fetch 不可用）"
        try:
            resp = requests.get(url, timeout=10,
                                headers={"User-Agent": "ComplianceAgent/1.0"})
            resp.raise_for_status()
            return resp.text[:max_chars]
        except Exception as e:
            return f"（web_fetch 失败：{e}）"


# Singleton
_singleton: Optional[CrossBorderComplianceSystem] = None


def get_system() -> CrossBorderComplianceSystem:
    global _singleton
    if _singleton is None:
        _singleton = CrossBorderComplianceSystem()
    return _singleton
