"""对话记忆。

为了让 Agent 真正能做多轮问答，记忆模块要保存：
- 简短摘要（控制窗口长度）
- 最近几轮问答
- 用户偏好 / 国家市场（用于上下文补全）
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional


@dataclass
class Turn:
    role: str  # "user" / "assistant" / "tool"
    content: str
    meta: Dict[str, str] = field(default_factory=dict)


class ConversationMemory:
    """基于 deque 的有界对话记忆。"""

    def __init__(self, max_turns: int = 8) -> None:
        self._buf: Deque[Turn] = deque(maxlen=max_turns * 2)
        self._facts: Dict[str, str] = {}

    def add(self, role: str, content: str, **meta: str) -> None:
        self._buf.append(Turn(role=role, content=content, meta=dict(meta)))

    def set_fact(self, key: str, value: str) -> None:
        self._facts[key] = value

    def get_fact(self, key: str, default: Optional[str] = None) -> Optional[str]:
        return self._facts.get(key, default)

    def to_prompt(self) -> str:
        lines = []
        if self._facts:
            lines.append("[已知上下文]")
            for k, v in self._facts.items():
                lines.append(f"- {k}: {v}")
            lines.append("")
        lines.append("[最近对话]")
        for turn in self._buf:
            lines.append(f"{turn.role}: {turn.content}")
        return "\n".join(lines)

    def clear(self) -> None:
        self._buf.clear()
        self._facts.clear()
