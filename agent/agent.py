"""核心 Agent 编排：ReAct + Plan-and-Execute 双模式。

整段 Agent 调用都是 streaming-friendly 的——每走一步把当前轨迹
（Thought / Action / Observation）通过 callback 推给上层（FastAPI SSE），便于调试。
"""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generator, List, Optional

from base import logger, Config
from .prompts import REACT_SYSTEM, PLAN_SYSTEM, VERIFIER_SYSTEM, TOOL_RESULT_FORMAT
from .registry import ToolRegistry
from .memory import ConversationMemory


@dataclass
class AgentStep:
    """单步轨迹。"""
    thought: str = ""
    action: str = ""
    action_input: Dict[str, Any] = field(default_factory=dict)
    observation: str = ""
    is_final: bool = False
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "thought": self.thought,
            "action": self.action,
            "action_input": self.action_input,
            "observation": self.observation[:400],  # 截断防爆
            "is_final": self.is_final,
            "error": self.error,
        }


@dataclass
class AgentTrace:
    """整次调用的轨迹。"""
    session_id: str
    question: str
    mode: str
    steps: List[AgentStep] = field(default_factory=list)
    final_answer: str = ""
    elapsed_ms: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "question": self.question,
            "mode": self.mode,
            "steps": [s.to_dict() for s in self.steps],
            "final_answer": self.final_answer,
            "elapsed_ms": self.elapsed_ms,
        }


class ComplianceAgent:
    """跨境电商合规 Agent。

    使用方式：
        agent = ComplianceAgent(llm=my_llm, tools=registry)
        for ev in agent.stream("某品类产品出口德国需要哪些认证"):
            ...  # 逐 token / step 消费
    """

    def __init__(
        self,
        llm: Callable[[str], str],
        tools: ToolRegistry,
        max_steps: int = 8,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps
        self.conf = Config()

    # ---------- 流式入口 ----------
    def stream(self, question: str, session_id: Optional[str] = None,
               memory: Optional[ConversationMemory] = None
               ) -> Generator[Dict[str, Any], None, AgentTrace]:
        """流式产出事件，事件类型：step / token / trace / error / final。"""
        sid = session_id or str(uuid.uuid4())
        mem = memory or ConversationMemory()
        mem.add("user", question)

        # 闲聊/简写问题免去工具调用，走直答
        if self._is_chitchat(question):
            trace = AgentTrace(session_id=sid, question=question,
                               mode="chitchat", steps=[], final_answer="",
                               elapsed_ms=0)
            reply = self._chitchat_reply(question)
            trace.final_answer = reply
            mem.add("assistant", reply)
            yield {"type": "final", "answer": reply, "trace": trace.to_dict()}
            return

        # 复杂问题走 Plan-and-Execute，简单问题走 ReAct
        mode = "plan" if self._is_complex(question) else "react"
        trace = AgentTrace(session_id=sid, question=question, mode=mode,
                           steps=[], final_answer="", elapsed_ms=0)
        t0 = time.time()

        yield {"type": "trace_start", "mode": mode, "session_id": sid}

        # 把工具描述 + 历史 + 问题拼到 system prompt
        system_prompt = self._build_system(mode)
        history_prompt = mem.to_prompt()
        full_prompt = (
            f"{system_prompt}\n\n"
            f"[历史对话]\n{history_prompt}\n\n"
            f"[当前问题]\n{question}\n"
        )

        citations: List[Dict[str, Any]] = []  # 收集所有工具产生的引用
        transcript: List[str] = [full_prompt]  # 累积对话，给 LLM 复读用

        try:
            for step_idx in range(self.max_steps):
                # 调 LLM 让它思考 + 决定下一步
                llm_out = self.llm("\n".join(transcript))
                step = self._parse_step(llm_out, mode=mode)
                step.observation = ""

                # 检测结束
                if step.is_final:
                    final = self._compose_final(llm_out, citations)
                    step.is_final = True
                    trace.steps.append(step)
                    trace.final_answer = final
                    mem.add("assistant", final)
                    yield {"type": "step", "step": step.to_dict()}
                    break

                # 执行工具
                try:
                    observation = self._execute_tool(step.action, step.action_input)
                    step.observation = observation.get("text", "")
                    citations.extend(observation.get("citations", []))
                    # 把这次执行反馈给 LLM
                    observation_block = TOOL_RESULT_FORMAT.format(
                        result=step.observation[:1500]
                    )
                except Exception as exc:  # 工具失败要回到模型重规划
                    step.error = str(exc)
                    observation_block = f"Observation: 工具执行失败 - {exc}"

                transcript.append(llm_out)
                transcript.append(observation_block)
                trace.steps.append(step)
                yield {"type": "step", "step": step.to_dict()}
            else:  # 超过 max_steps 强制收尾
                final = self._compose_final("", citations)
                trace.final_answer = final
                yield {"type": "step", "step": AgentStep(
                    thought="达到最大步数限制，强制收尾", is_final=True,
                ).to_dict()}

        except Exception as exc:
            logger.exception("agent error")
            trace.final_answer = f"抱歉，回答过程出现异常：{exc}"
            yield {"type": "error", "error": str(exc)}

        trace.elapsed_ms = int((time.time() - t0) * 1000)
        yield {"type": "trace", "trace": trace.to_dict()}
        yield {"type": "final", "answer": trace.final_answer,
               "trace": trace.to_dict()}
        return trace

    # ---------- 构造与解析 ----------
    def _build_system(self, mode: str) -> str:
        if mode == "plan":
            return PLAN_SYSTEM.format(tool_descriptions=self.tools.describe_all())
        return REACT_SYSTEM.format(tool_descriptions=self.tools.describe_all())

    def _is_chitchat(self, q: str) -> bool:
        q = q.strip()
        return len(q) <= 6 or bool(re.match(
            r"^(你好|您好|hi|hello|你是谁|你是做什么的|你叫什么|在吗|thanks|谢谢)+[!.?。?！]?$",
            q, re.IGNORECASE))

    def _chitchat_reply(self, q: str) -> str:
        q_l = q.lower()
        if "你是谁" in q or "你叫什么" in q:
            return ('我是『应小合规』，跨境电商合规智能助手，'
                    '能告诉你 CE/FCC/RoHS/UN38.3 等认证要求。')
        if "谢谢" in q_l or "thanks" in q_l:
            return "不客气！需要再查哪条标准随时叫我。"
        return "你好！我是『应小合规』，跨境卖家合规问题随时问。"

    def _is_complex(self, q: str) -> bool:
        """判断是否需要 Plan-and-Execute 模式。"""
        markers = [
            "哪些认证", "出口", "多个国家", "同时", "以及对", "还有",
            "跨文档", "比较", "对比", "多步", "推理",
        ]
        return any(m in q for m in markers) or len(q) > 60

    def _parse_step(self, llm_out: str, mode: str) -> AgentStep:
        """从 LLM 输出里抽 Thought / Action / ActionInput。"""
        step = AgentStep()
        # FinalAnswer 检测
        m_final = re.search(r"FinalAnswer\s*:\s*(.+)$", llm_out, re.DOTALL | re.IGNORECASE)
        if m_final:
            step.is_final = True
            step.thought = self._between(r"Thought\s*:", r"FinalAnswer\s*:", llm_out).strip()
            step.action = "__final__"
            step.action_input = {"answer": m_final.group(1).strip()}
            return step

        m_thought = re.search(r"Thought\s*:\s*(.+?)(?=\n\s*Action\s*:|$)", llm_out, re.DOTALL)
        m_action = re.search(r"Action\s*:\s*([a-zA-Z_][\w]*)", llm_out)
        m_input = re.search(r"ActionInput\s*:\s*(\{.*?\})", llm_out, re.DOTALL)

        step.thought = m_thought.group(1).strip() if m_thought else ""
        step.action = m_action.group(1).strip() if m_action else ""
        if m_input:
            try:
                step.action_input = json.loads(m_input.group(1))
            except json.JSONDecodeError:
                # 容错：把括号内容当字符串
                step.action_input = {"_raw": m_input.group(1)}
        return step

    @staticmethod
    def _between(start_pat: str, end_pat: str, text: str) -> str:
        s = re.search(start_pat, text, re.DOTALL)
        e = re.search(end_pat, text, re.DOTALL)
        if not s or not e:
            return ""
        return text[s.end():e.start()]

    def _execute_tool(self, action: str, action_input: Dict[str, Any]) -> Dict[str, Any]:
        """执行工具，返回 {'text': str, 'citations': [...]}。"""
        if action == "__final__":
            return {"text": action_input.get("answer", ""), "citations": []}
        spec = self.tools.get(action)
        # 用 inspect 把 kwargs 过滤掉不在签名里的
        sig = spec.fn.__annotations__
        kwargs = {k: v for k, v in action_input.items() if k in sig}
        out = spec.fn(**kwargs)
        if isinstance(out, dict) and "text" in out:
            return out
        return {"text": str(out), "citations": []}

    def _compose_final(self, llm_out: str, citations: List[Dict[str, Any]]) -> str:
        """如有 FinalAnswer 字段优先返回，否则摘 LLM 输出。"""
        m = re.search(r"FinalAnswer\s*:\s*(.+)$", llm_out, re.DOTALL | re.IGNORECASE)
        ans = m.group(1).strip() if m else ""
        if citations:
            ref_lines = ["", "## 引用"]
            seen = set()
            for c in citations:
                key = c.get("id") or c.get("chunk_id")
                if key in seen:
                    continue
                seen.add(key)
                title = c.get("title", "片段")
                ref_lines.append(f"- [{key}] {title}")
            ans = ans.rstrip() + "\n" + "\n".join(ref_lines)
        return ans or "（未能生成答案，请换个问法或补充背景信息）"
