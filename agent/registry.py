"""工具注册中心。

每个工具是一个可被 Agent 调用的能力单元，必须满足：
- 有清晰的 name / description（让 LLM 知道什么时候该用）
- 有 schema（输入 JSON schema，让 LLM 知道该怎么填参数）
- 有 callable（实际执行）

工具注册后，Agent 拿到工具列表就能在提示里描述它们。
"""

from __future__ import annotations

import json
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class ToolSpec:
    """单个工具的规格说明。"""

    name: str
    description: str
    schema: Dict[str, Any]  # JSON schema 风格的输入定义（简化）
    fn: Callable[..., Any]
    examples: List[Dict[str, Any]] = field(default_factory=list)

    def to_prompt_description(self) -> str:
        """生成给 LLM 阅读的工具描述。"""
        schema_str = json.dumps(self.schema, ensure_ascii=False, indent=2)
        ex_lines = []
        for i, ex in enumerate(self.examples, 1):
            ex_lines.append(
                f"  Example{i}: ActionInput={json.dumps(ex, ensure_ascii=False)}"
            )
        examples_block = "\n".join(ex_lines) if ex_lines else ""
        return (
            f"- {self.name}: {self.description}\n"
            f"  Input schema: {schema_str}"
            f"\n{examples_block}" if examples_block else
            f"- {self.name}: {self.description}\n  Input schema: {schema_str}"
        )


class ToolRegistry:
    """工具注册中心。"""

    def __init__(self) -> None:
        self._tools: Dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"Tool {spec.name} already registered")
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec:
        if name not in self._tools:
            raise KeyError(f"Tool {name} not found. Available: {list(self._tools)}")
        return self._tools[name]

    def list_names(self) -> List[str]:
        return sorted(self._tools.keys())

    def describe_all(self) -> str:
        return "\n".join(s.to_prompt_description() for s in self._tools.values())

    def __contains__(self, name: str) -> bool:
        return name in self._tools


def tool(name: str, description: str, schema: Dict[str, Any],
         examples: Optional[List[Dict[str, Any]]] = None):
    """装饰器：把一个函数注册为工具。

    用法：
        @tool("hybrid_search", "...",
              schema={"type":"object","properties":{"q":{"type":"string"}}, "required":["q"]})
        def hybrid_search(q: str, top_k: int = 6):
            ...
    """
    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        spec = ToolSpec(
            name=name,
            description=description,
            schema=schema,
            fn=fn,
            examples=examples or [],
        )
        # 把 spec 挂在函数上方便后面统一注册
        fn.__tool_spec__ = spec  # type: ignore[attr-defined]
        return fn
    return decorator
