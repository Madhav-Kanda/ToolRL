from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class ToolCall:
    tool_name: str
    args: Dict[str, Any]


@dataclass
class ToolResult:
    ok: bool
    content: str
    meta: Dict[str, Any]


@dataclass
class Step:
    call: ToolCall
    result: ToolResult


