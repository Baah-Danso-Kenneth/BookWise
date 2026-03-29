from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List
from datetime import datetime


class MCPToolStatus(Enum):
    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"


@dataclass
class MCPToolResult:
    """Standardized MCP tool result envelope"""
    tool_name: str
    status: MCPToolStatus
    content: Dict[str, Any]
    metadata: Dict[str, Any] = None
    timestamp: str = None

    def __post_init__(self):          # ← was __post_init_ (missing underscore)
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict:
        return {
            "tool_name": self.tool_name,
            "status": self.status.value,
            "content": self.content,
            "metadata": self.metadata or {},
            "timestamp": self.timestamp
        }


@dataclass
class MCPToolDescriptor:
    """MCP tool registration descriptor"""
    name: str
    description: str
    version: str
    input_schema: Dict[str, Any]


class MCPToolServer:
    """Base class for all MCP-compliant tools"""

    def __init__(self):
        self.descriptor = self._register()

    def _register(self) -> MCPToolDescriptor:
        raise NotImplementedError

    def execute(self, **kwargs) -> MCPToolResult:
        raise NotImplementedError

    def __call__(self, **kwargs) -> Dict:
        result = self.execute(**kwargs)
        return result.to_dict()