"""Agentic RAG package.

把检索/知识图谱/查询改写/子问题/网页抓取都注册成 Agent 可调用的工具，
让 LLM 自主决策何时调哪个工具——RAG 退化为 Agent 的一种能力。
"""

from .agent import ComplianceAgent, AgentStep, AgentTrace
from .registry import ToolRegistry, ToolSpec
from .memory import ConversationMemory

__all__ = [
    "ComplianceAgent",
    "AgentStep",
    "AgentTrace",
    "ToolRegistry",
    "ToolSpec",
    "ConversationMemory",
]
