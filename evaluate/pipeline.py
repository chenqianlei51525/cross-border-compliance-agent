"""评测流水线：Agent 轨迹 + RAGAS 联动。

评测指标：
- RAGAS：faithfulness / answer_relevancy / context_precision / context_recall
- Agent 特有：tool_accuracy / trajectory_step_efficiency / citation_coverage

输出：
- CSV 详细结果
- Markdown 报告（聚合 + 分类对比）
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from base import logger

@dataclass
class EvalItem:
    question: str
    ground_truth: str = ""
    predicted: str = ""
    contexts: List[str] = field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    trace_rounds: int = 0
    elapsed_ms: int = 0
    ragas_scores: Dict[str, float] = field(default_factory=dict)
    tool_accuracy: float = 0.0
    citation_coverage: float = 0.0


def ragas_evaluate(dataset: List[EvalItem]) -> Dict[str, float]:
    """跑 RAGAS 评估（faithfulness / answer_relevancy / context_precision / context_recall）。"""
    try:
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness, answer_relevancy,
            context_precision, context_recall,
        )
        from datasets import Dataset
    except ImportError as e:
        logger.warning("RAGAS not installed: %s", e)
        return {}

    try:
        from langchain_community.chat_models import ChatTongyi
        from langchain_community.embeddings import DashScopeEmbeddings
    except ImportError as e:
        logger.warning("langchain-community not installed: %s", e)
        return {}

    api_key = os.getenv("API_KEY")
    if not api_key:
        logger.warning("API_KEY not set, skip RAGAS")
        return {}

    ds = Dataset.from_dict({
        "question": [it.question for it in dataset],
        "answer": [it.predicted for it in dataset],
        "contexts": [it.contexts for it in dataset],
        "ground_truth": [it.ground_truth for it in dataset],
    })
    result = evaluate(
        dataset=ds,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=ChatTongyi(model="qwen-max", api_key=api_key),
        embeddings=DashScopeEmbeddings(dashscope_api_key=api_key,
                                       model="text-embedding-v3"),
    )
    scores = {}
    if hasattr(result, "to_pandas"):
        df = result.to_pandas()
        for col in df.columns:
            if col in ("faithfulness", "answer_relevancy",
                       "context_precision", "context_recall"):
                scores[col] = float(df[col].mean())
    return scores


def compute_agent_metrics(items: List[EvalItem]) -> Dict[str, float]:
    """Agent 特有指标：
       - tool_accuracy: 工具调用 json 解析成功率
       - citation_coverage: 答案中引用编号比例
       - trajectory_step_efficiency: 平均轮次
    """
    if not items:
        return {}
    valid_tool_calls = sum(1 for it in items
                            if any(c.get("action", "").startswith(tuple([
                                "hybrid_search", "knowledge_graph",
                                "query_rewrite", "sub_question",
                                "hyde_search", "self_critique",
                                "mysql_qa_search", "web_fetch", "doc_lookup"
                            ])) for c in it.tool_calls))
    cited = sum(1 for it in items if any(c.get("id") for c in it.tool_calls))
    return {
        "tool_call_success_rate": valid_tool_calls / len(items),
        "citation_coverage": cited / len(items),
        "avg_trace_rounds": statistics.mean(
            it.trace_rounds for it in items if it.trace_rounds
        ) if any(it.trace_rounds for it in items) else 0,
        "avg_elapsed_ms": statistics.mean(it.elapsed_ms for it in items),
    }


def save_csv(items: List[EvalItem], path: str) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "question", "ground_truth", "predicted",
            "trace_rounds", "elapsed_ms",
            "tool_calls_count", "ragas_faithfulness",
            "ragas_relevancy", "ragas_precision", "ragas_recall",
        ])
        for it in items:
            w.writerow([
                it.question[:80], it.ground_truth[:80], it.predicted[:80],
                it.trace_rounds, it.elapsed_ms,
                len(it.tool_calls),
                it.ragas_scores.get("faithfulness", ""),
                it.ragas_scores.get("answer_relevancy", ""),
                it.ragas_scores.get("context_precision", ""),
                it.ragas_scores.get("context_recall", ""),
            ])


def save_markdown_report(items: List[EvalItem],
                          aggregate: Dict[str, float],
                          path: str) -> None:
    lines = [
        "# 跨境合规 Agent 评测报告", "",
        f"样本数: {len(items)}",
        f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 聚合指标", "",
        "| 指标 | 值 |", "| --- | --- |",
    ]
    for k, v in aggregate.items():
        lines.append(f"| {k} | {round(v, 4) if isinstance(v, float) else v} |")
    lines += [
        "",
        "## 分样本明细（前 10 条）", "",
        "| Q | 预测 | 轮次 | 用时(ms) | faithfulness | relevancy |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for it in items[:10]:
        lines.append(
            f"| {it.question[:40]} | {it.predicted[:40]} | "
            f"{it.trace_rounds} | {it.elapsed_ms} | "
            f"{round(it.ragas_scores.get('faithfulness', 0), 3)} | "
            f"{round(it.ragas_scores.get('answer_relevancy', 0), 3)} |"
        )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def run_evaluation(eval_set: List[EvalItem],
                    csv_path: str = "evaluate/results.csv",
                    md_path: str = "evaluate/report.md") -> Dict[str, float]:
    ragas_scores = ragas_evaluate(eval_set)
    agent_metrics = compute_agent_metrics(eval_set)
    aggregate = {**ragas_scores, **agent_metrics}
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    save_csv(eval_set, csv_path)
    save_markdown_report(eval_set, aggregate, md_path)
    logger.info("evaluation done: %s", aggregate)
    return aggregate


# ---------- CLI ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True,
                        help="评测样本 JSON, 每条 {question, ground_truth, contexts[]}")
    parser.add_argument("--output", default="evaluate/report.md")
    parser.add_argument("--csv", default="evaluate/results.csv")
    args = parser.parse_args()

    with open(args.dataset, "r", encoding="utf-8") as f:
        items = [EvalItem(**{
            "question": r["question"],
            "ground_truth": r.get("ground_truth", ""),
            "contexts": r.get("contexts", []) or [],
            "predicted": r.get("predicted", ""),
        }) for r in json.load(f)]
    run_evaluation(items, csv_path=args.csv, md_path=args.output)
