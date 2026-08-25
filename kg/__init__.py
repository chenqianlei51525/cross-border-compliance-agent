"""知识图谱模块：法规实体 + 关系三元组 + 多跳推理。

节点类型：
  ProductCategory  产品类别
  CertificationStandard  认证标准（CE-RED / FCC Part 15 / RoHS / UN38.3 ...）
  MarketRegion  适用市场（EU / US / JP / KR / CN ...）
  TestItem  检测项目（EMC / RF / EMF / LVD / Chemical ...）
  Regulation  法规正文（指向文件 + chunk_id）

关系：
  (ProductCategory)-[:REQUIRES]->(CertificationStandard)
  (CertificationStandard)-[:APPLIES_TO]->(MarketRegion)
  (CertificationStandard)-[:CONTAINS]->(TestItem)
  (CertificationStandard)-[:SOURCE_FROM]->(Regulation)
  (Regulation)-[:CITES]->(Regulation)

构造思路：
  1. 用 LLM 从法规文本抽 (head, relation, tail) 三元组
  2. 写入 Neo4j
  3. 多跳推理：CYPHER 模板查询 + 文本回退

Neo4j 未连接时全部以"in-memory"模式跑，便于演示。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from base import logger


# ---------- 数据模型 ----------
@dataclass
class Triple:
    head: str
    rel: str
    tail: str
    evidence: str = ""
    source: str = ""

    def to_dict(self) -> Dict[str, str]:
        return {
            "head": self.head, "rel": self.rel,
            "tail": self.tail, "evidence": self.evidence,
            "source": self.source,
        }


@dataclass
class KGSchema:
    entities: Dict[str, str] = field(default_factory=dict)  # 实体名 → 类型
    relations: List[Triple] = field(default_factory=list)


# ---------- 抽取器（LLM） ----------
EXTRACT_PROMPT = """你是跨境合规法规实体抽取专家。
从下面这段法规文本里抽取 (head, relation, tail) 三元组，关系词使用：

- REQUIRES    产品类别→认证标准
- APPLIES_TO  认证标准→适用市场
- CONTAINS    认证标准→检测项目
- CITES       法规↔法规
- VERSION_OF  标准版本关系

只输出严格 JSON：
{
  "triples": [
    {"head":"...","rel":"...","tail":"...","evidence":"原文支撑句（<=80字）"}
  ]
}

文本：
{text}
"""


def extract_triples_with_llm(text: str, llm=None) -> List[Triple]:
    """LLM 抽三元组；llm 不可用时返回空。"""
    if llm is None:
        return []
    prompt = EXTRACT_PROMPT.format(text=text[:4000])
    out = llm(prompt)
    out_str = "".join(out) if hasattr(out, "__iter__") and not isinstance(out, str) else str(out)
    m = re.search(r"\{[\s\S]*\}", out_str)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    triples: List[Triple] = []
    for t in data.get("triples", []):
        triples.append(Triple(
            head=(t.get("head") or "").strip(),
            rel=(t.get("rel") or "").strip().upper(),
            tail=(t.get("tail") or "").strip(),
            evidence=(t.get("evidence") or "").strip(),
            source=text[:200],
        ))
    return [t for t in triples if t.head and t.tail and t.rel]


# ---------- Neo4j 客户端 ----------
class Neo4jKGClient:
    """Neo4j 法规知识图谱客户端。

    - available=False 时所有读操作降级到内存模式，便于离线 / 演示。
    - available=True 时走真实 Neo4j（pip install neo4j）。
    """

    def __init__(self, uri: str = "bolt://localhost:7687",
                 user: str = "neo4j", password: str = "neo4j"):
        self.uri, self.user, self.password = uri, user, password
        self._driver = None
        self._in_mem: KGSchema = KGSchema()
        self.available: bool = False
        self._connect()

    def _connect(self) -> None:
        try:
            from neo4j import GraphDatabase  # type: ignore
            self._driver = GraphDatabase.driver(
                self.uri, auth=(self.user, self.password))
            with self._driver.session() as s:
                s.run("RETURN 1").consume()
            self.available = True
            logger.info("Connected to Neo4j %s", self.uri)
        except Exception as e:
            logger.warning("Neo4j not available, fallback in-memory: %s", e)
            self.available = False

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()

    # ---------- 写入 ----------
    def upsert_triples(self, triples: List[Triple]) -> int:
        if not triples:
            return 0
        if self.available:
            return self._upsert_neo4j(triples)
        return self._upsert_inmem(triples)

    def _upsert_neo4j(self, triples: List[Triple]) -> int:
        cypher = """
        MERGE (h:Entity {name:$head})
        MERGE (t:Entity {name:$tail})
        MERGE (h)-[r:REL {kind:$rel}]->(t)
        SET r.evidence = $evidence, r.source = $source
        RETURN h, r, t
        """
        with self._driver.session() as s:  # type: ignore[union-attr]
            n = 0
            for t in triples:
                s.run(cypher, head=t.head, tail=t.tail, rel=t.rel,
                      evidence=t.evidence, source=t.source).consume()
                n += 1
        return n

    def _upsert_inmem(self, triples: List[Triple]) -> int:
        for t in triples:
            self._in_mem.entities.setdefault(t.head, "Entity")
            self._in_mem.entities.setdefault(t.tail, "Entity")
            self._in_mem.relations.append(t)
        return len(triples)

    # ---------- 多跳推理 ----------
    def multi_hop_reasoning(self, question: str, hops: int = 2) -> List[Dict[str, str]]:
        """根据自然语言问题做多跳查询。"""
        if self.available:
            return self._multi_hop_neo4j(question, hops)
        return self._multi_hop_inmem(question, hops)

    def _multi_hop_neo4j(self, question: str, hops: int) -> List[Dict[str, str]]:
        # 简化：用关键词匹配入口节点，再 N 步外扩
        keywords = self._extract_keywords(question)
        cypher = (
            "MATCH p=(h:Entity)-[*1..%d]->(t:Entity) "
            "WHERE any(k IN $keys WHERE toLower(h.name) CONTAINS toLower(k)) "
            "RETURN h.name AS h, t.name AS t, "
            "       [r IN relationships(p) | r.kind] AS rels, "
            "       [r IN relationships(p) | r.evidence][0] AS evidence "
            "LIMIT 30"
        ) % hops
        with self._driver.session() as s:  # type: ignore[union-attr]
            res = s.run(cypher, keys=keywords)
            return [
                {
                    "from_label": r["h"],
                    "rel": "->".join(r["rels"]),
                    "to_label": r["t"],
                    "evidence": r["evidence"] or "",
                } for r in res
            ]

    def _multi_hop_inmem(self, question: str, hops: int) -> List[Dict[str, str]]:
        keywords = self._extract_keywords(question)
        if not keywords:
            return []
        # 找到入口节点
        starts = {h for h in self._in_mem.entities
                  if any(k in h.lower() for k in keywords)}
        # BFS
        visited: Dict[str, List[Triple]] = {s: [] for s in starts}
        frontier = set(starts)
        results: List[Dict[str, str]] = []
        for _ in range(hops):
            next_frontier = set()
            for h in frontier:
                for t in self._in_mem.relations:
                    if t.head == h and t.tail not in visited:
                        visited.setdefault(t.tail, visited[h] + [t])
                        next_frontier.add(t.tail)
                        results.append({
                            "from_label": t.head,
                            "rel": t.rel,
                            "to_label": t.tail,
                            "evidence": t.evidence,
                        })
            frontier = next_frontier
            if not frontier:
                break
        return results

    @staticmethod
    def _extract_keywords(question: str) -> List[str]:
        # 简单关键词提取：保留中文 2~6 字、英文 3+ 字的 token
        tokens = re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}|[\u4e00-\u9fa5]{2,6}", question)
        # 黑名单过滤（很泛的词）
        blacklist = {"哪些", "什么", "怎么", "需要", "出口", "产品",
                     "标准", "认证"}
        return [t for t in tokens if t not in blacklist][:8]


# ---------- 构建器 ----------
class KGBuilder:
    """从一堆 chunk 文本构造知识图谱。"""

    def __init__(self, client: Neo4jKGClient, llm=None):
        self.client = client
        self.llm = llm
        self.stats = {"chunks": 0, "triples": 0}

    def build_from_chunks(self, chunks: List[Dict[str, str]]) -> int:
        """chunks: [{chunk_id, text, source}]"""
        for chunk in chunks:
            triples = extract_triples_with_llm(chunk.get("text", ""), self.llm)
            for t in triples:
                t.source = chunk.get("source", "")
            n = self.client.upsert_triples(triples)
            self.stats["chunks"] += 1
            self.stats["triples"] += n
        return self.stats["triples"]
