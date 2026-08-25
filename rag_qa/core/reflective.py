"""Reflective RAG 反思检索循环。

借鉴：
- Self-RAG（Asai et al., 2023）：每个检索结果都让 LLM 自评 [Relevant] / [Partially] / [Irrelevant]
- Corrective RAG（Yan et al., 2024）：检索结果不达标时触发 Query Rewrite / Web 检索

实现：
1. retrieve 一轮拿到候选文档
2. 让 LLM 对每个候选打 relevance 与 supporting 标签（Self-RAG 思路）
3. 不达标的过滤；都不达标就走 corrective 子流程：query_rewrite → 再检索
4. 最多 N 轮，超过则返回当前最优
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from base import logger


@dataclass
class JudgedDoc:
    doc: Any
    relevance: str = "unknown"     # relevant / partial / irrelevant
    is_support: bool = False       # 是否能支撑当前事实
    reason: str = ""
    score: float = 0.0


@dataclass
class ReflectionResult:
    docs: List[JudgedDoc]
    rounds: int
    transformed_query: str
    trace: List[Dict[str, Any]] = field(default_factory=list)


CRITIQUE_PROMPT = """你是检索质量评估员。判断下面【文档】是否回答了【问题】。
严格输出 JSON：
{
  "relevance": "relevant" | "partial" | "irrelevant",
  "is_support": true | false,
  "reason": "一句话理由"
}
问题：{question}
文档：{document}
"""


REWRITE_PROMPT = """你是问题改写专家。基于原问题和上轮检索失败原因（evidence_insufficient），
改写一条更可能命中标准答案的检索 query。
只输出 JSON：{{"rewritten":"...", "strategy":"broaden|add_constraints|switch_terms"}}
原问题：{question}
"""


def _judge_one(llm, question: str, doc_text: str) -> JudgedDoc:
    out = llm(CRITIQUE_PROMPT.format(question=question, document=doc_text[:1200]))
    out_str = "".join(out) if hasattr(out, "__iter__") and not isinstance(out, str) else str(out)
    import json, re as _re
    m = _re.search(r"\{.*\}", out_str, _re.DOTALL)
    try:
        data = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        data = {}
    relevance = data.get("relevance", "irrelevant")
    return JudgedDoc(
        doc=None,
        relevance=relevance,
        is_support=bool(data.get("is_support", False)),
        reason=data.get("reason", "") or "",
        score={"relevant": 1.0, "partial": 0.5}.get(relevance, 0.0),
    )


class ReflectiveRetriever:
    """反思检索器。

    用法：
        rr = ReflectiveRetriever(retrieve_fn=vs.hybrid_search_with_rerank,
                                llm=call_model, max_rounds=3)
        result = rr.retrieve("蓝牙耳机出口德国需要哪些认证？")
    """

    def __init__(self, retrieve_fn: Callable[..., List[Any]],
                 llm: Callable, max_rounds: int = 3,
                 min_relevant_ratio: float = 0.3,
                 top_k: int = 6):
        self.retrieve_fn = retrieve_fn
        self.llm = llm
        self.max_rounds = max_rounds
        self.min_relevant_ratio = min_relevant_ratio
        self.top_k = top_k

    def retrieve(self, question: str,
                 initial_query: Optional[str] = None) -> ReflectionResult:
        """跑完整反思检索循环。"""
        query = initial_query or question
        all_docs: List[Any] = []
        trace: List[Dict[str, Any]] = []
        for round_idx in range(self.max_rounds):
            docs = list(self.retrieve_fn(query, k=self.top_k)) or []
            judged = [_judge_one(self.llm, question, _doc_text(d))
                     for d in docs]
            for j, d in zip(judged, docs):
                j.doc = d
            all_docs.extend(judged)

            relevant = [j for j in judged if j.relevance == "relevant"]
            ratio = (len(relevant) / max(1, len(judged)))
            trace.append({
                "round": round_idx,
                "query": query,
                "retrieved": len(judged),
                "relevant": len(relevant),
                "ratio": ratio,
            })
            if ratio >= self.min_relevant_ratio or round_idx == self.max_rounds - 1:
                break
            # corrective：query rewrite
            new_q = self._corrective_rewrite(question, "evidence_insufficient")
            query = new_q or query

        # 去重（按 page_content）
        seen, deduped = set(), []
        for j in all_docs:
            content = _doc_text(j.doc)
            if content in seen:
                continue
            seen.add(content)
            deduped.append(j)

        # 按分数排序
        deduped.sort(key=lambda j: j.score, reverse=True)
        return ReflectionResult(
            docs=deduped[: self.top_k],
            rounds=len(trace),
            transformed_query=query,
            trace=trace,
        )

    def _corrective_rewrite(self, question: str, reason: str) -> Optional[str]:
        try:
            out = self.llm(REWRITE_PROMPT.format(question=question))
            out_str = "".join(out) if hasattr(out, "__iter__") else str(out)
            import json, re as _re
            m = _re.search(r"\{.*\}", out_str, _re.DOTALL)
            if not m:
                return None
            data = json.loads(m.group(0))
            return data.get("rewritten")
        except Exception as e:
            logger.warning("corrective rewrite failed: %s", e)
            return None


def _doc_text(doc: Any) -> str:
    if doc is None:
        return ""
    if hasattr(doc, "page_content"):
        return doc.page_content
    return str(doc)
