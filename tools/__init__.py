"""
Tools Module - Exposes Jarvis agent capabilities as callable tools.

This module provides:
- ToolDefinition: Schema for defining tools with metadata
- ToolRegistry: Central registry of all available tools
- ToolExecutor: Async execution engine that routes through EventBus
- ToolPermissions: Permission checking for tool execution

Design Principles:
- Tools are abstractions over agent capabilities (no duplicate implementations)
- All tool execution goes through EventBus (observable, async)
- Tool results are tracked and can be awaited
"""

from tools.registry import (
    ToolDefinition,
    ToolParameter,
    ToolRegistry,
    ToolPermission,
    ToolCategory,
    get_tool_registry,
)
from tools.executor import (
    ToolExecutor,
    ToolExecutionResult,
    ToolExecutionStatus,
    get_tool_executor,
)

__all__ = [
    # Registry
    "ToolDefinition",
    "ToolParameter",
    "ToolRegistry",
    "ToolPermission",
    "ToolCategory",
    "get_tool_registry",
    # Executor
    "ToolExecutor",
    "ToolExecutionResult",
    "ToolExecutionStatus",
    "get_tool_executor",
]
