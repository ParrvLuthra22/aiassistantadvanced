"""
Tool Registry - Central registry for all Jarvis tools.

This module defines the tool schema and provides a registry for discovering
and validating tools before execution.

Key Design:
- ToolDefinition describes a tool's interface (name, params, target agent)
- ToolRegistry is a singleton that holds all registered tools
- Tools map directly to agent capabilities - no duplicate implementations
- Permission levels control which tools can be used
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set
from uuid import UUID

from utils.logger import get_logger


logger = get_logger(__name__)


# =============================================================================
# Tool Categories and Permissions
# =============================================================================

class ToolCategory(Enum):
    """Categories of tools for organization and filtering."""
    
    SYSTEM = auto()       # System control (apps, volume, brightness)
    MEMORY = auto()       # Memory/storage operations
    VISION = auto()       # Camera/vision operations
    WEB = auto()          # Web search/browsing
    FILE = auto()         # File system operations
    CODE = auto()         # Code execution
    DATA = auto()         # Data analysis
    COMMUNICATION = auto() # Notifications, messaging
    UTILITY = auto()      # General utilities


class ToolPermission(Enum):
    """
    Permission levels for tool execution.
    
    Higher levels require more trust/confirmation.
    """
    
    # No confirmation needed - read-only, safe operations
    SAFE = "safe"
    
    # Low risk - may modify minor settings
    LOW_RISK = "low_risk"
    
    # Medium risk - modifies system state
    MEDIUM_RISK = "medium_risk"
    
    # High risk - significant system changes
    HIGH_RISK = "high_risk"
    
    # Critical - requires explicit user confirmation
    CRITICAL = "critical"
    
    # Blocked - never allowed (admin only)
    BLOCKED = "blocked"


# =============================================================================
# Tool Parameter Definition
# =============================================================================

@dataclass
class ToolParameter:
    """
    Definition of a tool parameter.
    
    Attributes:
        name: Parameter name (used as key in args dict)
        param_type: Expected type ("string", "number", "boolean", "array", "object")
        description: Human-readable description for LLM
        required: Whether the parameter is required
        default: Default value if not provided
        enum: List of allowed values (if constrained)
        examples: Example values for documentation
    """
    
    name: str
    param_type: str = "string"
    description: str = ""
    required: bool = True
    default: Any = None
    enum: Optional[List[Any]] = None
    examples: List[Any] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON schema generation."""
        result = {
            "name": self.name,
            "type": self.param_type,
            "description": self.description,
            "required": self.required,
        }
        if self.default is not None:
            result["default"] = self.default
        if self.enum:
            result["enum"] = self.enum
        if self.examples:
            result["examples"] = self.examples
        return result


# =============================================================================
# Tool Definition
# =============================================================================

@dataclass
class ToolDefinition:
    """
    Definition of a tool that can be called by orchestrator agents.
    
    Tools are abstractions over agent capabilities. They define:
    - What the tool does (name, description)
    - What parameters it accepts
    - Which agent handles it (target_agent)
    - What action to invoke (action)
    - Permission requirements
    
    Attributes:
        name: Unique tool name (e.g., "open_app", "search_web")
        description: Detailed description for LLM to understand when to use
        parameters: List of parameter definitions
        target_agent: Name of the agent that handles this tool
        action: Action name to invoke on the target agent
        category: Category for organization
        permission: Required permission level
        is_async: Whether the tool is async (most are)
        timeout_seconds: Maximum execution time
        examples: Example usage for documentation
        tags: Additional tags for filtering
    """
    
    name: str
    description: str
    parameters: List[ToolParameter] = field(default_factory=list)
    target_agent: str = ""
    action: str = ""
    category: ToolCategory = ToolCategory.UTILITY
    permission: ToolPermission = ToolPermission.LOW_RISK
    is_async: bool = True
    timeout_seconds: float = 30.0
    examples: List[Dict[str, Any]] = field(default_factory=list)
    tags: Set[str] = field(default_factory=set)
    
    def __post_init__(self):
        """Set action to name if not specified."""
        if not self.action:
            self.action = self.name
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON schema/API."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": [p.to_dict() for p in self.parameters],
            "target_agent": self.target_agent,
            "action": self.action,
            "category": self.category.name,
            "permission": self.permission.value,
            "is_async": self.is_async,
            "timeout_seconds": self.timeout_seconds,
            "examples": self.examples,
            "tags": list(self.tags),
        }
    
    def to_openai_function(self) -> Dict[str, Any]:
        """
        Convert to OpenAI function calling format.
        
        This format is commonly used by LLMs for tool/function calling.
        """
        properties = {}
        required = []
        
        for param in self.parameters:
            prop = {
                "type": param.param_type,
                "description": param.description,
            }
            if param.enum:
                prop["enum"] = param.enum
            properties[param.name] = prop
            
            if param.required:
                required.append(param.name)
        
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }
    
    def validate_args(self, args: Dict[str, Any]) -> List[str]:
        """
        Validate arguments against parameter definitions.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        for param in self.parameters:
            if param.required and param.name not in args:
                errors.append(f"Missing required parameter: {param.name}")
            elif param.name in args:
                value = args[param.name]
                # Type checking (simplified)
                if param.param_type == "string" and not isinstance(value, str):
                    errors.append(f"Parameter {param.name} must be a string")
                elif param.param_type == "number" and not isinstance(value, (int, float)):
                    errors.append(f"Parameter {param.name} must be a number")
                elif param.param_type == "boolean" and not isinstance(value, bool):
                    errors.append(f"Parameter {param.name} must be a boolean")
                elif param.param_type == "array" and not isinstance(value, list):
                    errors.append(f"Parameter {param.name} must be an array")
                
                # Enum checking
                if param.enum and value not in param.enum:
                    errors.append(
                        f"Parameter {param.name} must be one of: {param.enum}"
                    )
        
        return errors


# =============================================================================
# Tool Registry
# =============================================================================

class ToolRegistry:
    """
    Central registry for all tools available to the assistant.
    
    The registry:
    - Stores all tool definitions
    - Provides tool discovery by name, category, or tags
    - Validates tool existence before execution
    - Generates tool schemas for LLM consumption
    
    Usage:
        registry = get_tool_registry()
        tool = registry.get_tool("open_app")
        all_tools = registry.list_tools()
        system_tools = registry.list_tools(category=ToolCategory.SYSTEM)
    """
    
    _instance: Optional[ToolRegistry] = None
    
    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}
        self._initialized = False
        self._allowed_tools: Set[str] = set()  # Empty = all allowed
        self._blocked_tools: Set[str] = set()
    
    @classmethod
    def get_instance(cls) -> ToolRegistry:
        """Get the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def register(self, tool: ToolDefinition) -> None:
        """
        Register a tool definition.
        
        Args:
            tool: The tool definition to register
        
        Raises:
            ValueError: If a tool with the same name already exists
        """
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        
        self._tools[tool.name] = tool
        logger.debug(f"Registered tool: {tool.name} -> {tool.target_agent}.{tool.action}")
    
    def unregister(self, name: str) -> bool:
        """
        Unregister a tool.
        
        Args:
            name: Tool name to unregister
        
        Returns:
            True if tool was found and removed
        """
        if name in self._tools:
            del self._tools[name]
            logger.debug(f"Unregistered tool: {name}")
            return True
        return False
    
    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        """Get a tool by name."""
        return self._tools.get(name)
    
    def has_tool(self, name: str) -> bool:
        """Check if a tool exists."""
        return name in self._tools
    
    def list_tools(
        self,
        category: Optional[ToolCategory] = None,
        permission_max: Optional[ToolPermission] = None,
        tags: Optional[Set[str]] = None,
        include_blocked: bool = False,
    ) -> List[ToolDefinition]:
        """
        List tools with optional filtering.
        
        Args:
            category: Filter by category
            permission_max: Only include tools up to this permission level
            tags: Filter by tags (tool must have all specified tags)
            include_blocked: Whether to include blocked tools
        
        Returns:
            List of matching tool definitions
        """
        permission_order = [
            ToolPermission.SAFE,
            ToolPermission.LOW_RISK,
            ToolPermission.MEDIUM_RISK,
            ToolPermission.HIGH_RISK,
            ToolPermission.CRITICAL,
            ToolPermission.BLOCKED,
        ]
        
        results = []
        for tool in self._tools.values():
            # Check blocked
            if not include_blocked and tool.name in self._blocked_tools:
                continue
            if not include_blocked and tool.permission == ToolPermission.BLOCKED:
                continue
            
            # Check allowed list
            if self._allowed_tools and tool.name not in self._allowed_tools:
                continue
            
            # Check category
            if category and tool.category != category:
                continue
            
            # Check permission level
            if permission_max:
                tool_idx = permission_order.index(tool.permission)
                max_idx = permission_order.index(permission_max)
                if tool_idx > max_idx:
                    continue
            
            # Check tags
            if tags and not tags.issubset(tool.tags):
                continue
            
            results.append(tool)
        
        return results
    
    def list_tool_names(self, **kwargs) -> List[str]:
        """List tool names with optional filtering."""
        return [t.name for t in self.list_tools(**kwargs)]
    
    def set_allowed_tools(self, tools: List[str]) -> None:
        """
        Set the list of allowed tools.
        
        If empty, all tools are allowed (except blocked).
        """
        self._allowed_tools = set(tools)
        logger.info(f"Set allowed tools: {len(self._allowed_tools)} tools")
    
    def block_tool(self, name: str) -> None:
        """Block a tool from being used."""
        self._blocked_tools.add(name)
        logger.info(f"Blocked tool: {name}")
    
    def unblock_tool(self, name: str) -> None:
        """Unblock a tool."""
        self._blocked_tools.discard(name)
        logger.info(f"Unblocked tool: {name}")
    
    def is_tool_allowed(self, name: str) -> bool:
        """Check if a tool is allowed to be used."""
        if name in self._blocked_tools:
            return False
        if self._allowed_tools and name not in self._allowed_tools:
            return False
        tool = self.get_tool(name)
        if tool and tool.permission == ToolPermission.BLOCKED:
            return False
        return True
    
    def get_tools_for_agent(self, agent_name: str) -> List[ToolDefinition]:
        """Get all tools that target a specific agent."""
        return [t for t in self._tools.values() if t.target_agent == agent_name]
    
    def to_openai_functions(
        self,
        category: Optional[ToolCategory] = None,
        permission_max: Optional[ToolPermission] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate OpenAI function calling format for tools.
        
        This is useful for passing to LLMs that support function calling.
        """
        tools = self.list_tools(category=category, permission_max=permission_max)
        return [t.to_openai_function() for t in tools]
    
    def initialize_default_tools(self) -> None:
        """
        Initialize the registry with default Jarvis tools.
        
        This maps existing agent capabilities to tool definitions.
        """
        if self._initialized:
            return
        
        # Register all default tools
        for tool in _create_default_tools():
            self.register(tool)
        
        self._initialized = True
        logger.info(f"Initialized tool registry with {len(self._tools)} tools")


# =============================================================================
# Default Tool Definitions
# =============================================================================

def _create_default_tools() -> List[ToolDefinition]:
    """
    Create default tool definitions that map to Jarvis agent capabilities.
    
    Each tool maps to an existing agent action - no duplicate implementations.
    """
    tools = []
    
    # =========================================================================
    # SystemAgent Tools
    # =========================================================================
    
    # Application Control
    tools.append(ToolDefinition(
        name="open_app",
        description="Open an application by name. Launches the specified application on macOS.",
        parameters=[
            ToolParameter(
                name="app_name",
                param_type="string",
                description="Name of the application to open (e.g., 'Safari', 'Finder', 'Terminal')",
                required=True,
                examples=["Safari", "Finder", "Notes", "Calendar"],
            ),
        ],
        target_agent="SystemAgent",
        action="open_app",
        category=ToolCategory.SYSTEM,
        permission=ToolPermission.LOW_RISK,
        examples=[
            {"app_name": "Safari"},
            {"app_name": "Visual Studio Code"},
        ],
        tags={"application", "launch", "open"},
    ))
    
    tools.append(ToolDefinition(
        name="close_app",
        description="Close a running application by name.",
        parameters=[
            ToolParameter(
                name="app_name",
                param_type="string",
                description="Name of the application to close",
                required=True,
            ),
            ToolParameter(
                name="force",
                param_type="boolean",
                description="Force quit the application",
                required=False,
                default=False,
            ),
        ],
        target_agent="SystemAgent",
        action="close_app",
        category=ToolCategory.SYSTEM,
        permission=ToolPermission.MEDIUM_RISK,
        tags={"application", "close", "quit"},
    ))
    
    tools.append(ToolDefinition(
        name="focus_app",
        description="Bring an application to the foreground.",
        parameters=[
            ToolParameter(
                name="app_name",
                param_type="string",
                description="Name of the application to focus",
                required=True,
            ),
        ],
        target_agent="SystemAgent",
        action="focus_app",
        category=ToolCategory.SYSTEM,
        permission=ToolPermission.SAFE,
        tags={"application", "focus", "switch"},
    ))
    
    tools.append(ToolDefinition(
        name="list_apps",
        description="List all currently running applications.",
        parameters=[],
        target_agent="SystemAgent",
        action="list_apps",
        category=ToolCategory.SYSTEM,
        permission=ToolPermission.SAFE,
        tags={"application", "list", "running"},
    ))
    
    # Volume Control
    tools.append(ToolDefinition(
        name="set_volume",
        description="Set the system volume to a specific level.",
        parameters=[
            ToolParameter(
                name="level",
                param_type="number",
                description="Volume level from 0 to 100",
                required=True,
                examples=[50, 75, 0, 100],
            ),
        ],
        target_agent="SystemAgent",
        action="control_volume",
        category=ToolCategory.SYSTEM,
        permission=ToolPermission.SAFE,
        tags={"volume", "audio", "sound"},
    ))
    
    tools.append(ToolDefinition(
        name="get_volume",
        description="Get the current system volume level.",
        parameters=[],
        target_agent="SystemAgent",
        action="get_volume",
        category=ToolCategory.SYSTEM,
        permission=ToolPermission.SAFE,
        tags={"volume", "audio", "sound"},
    ))
    
    tools.append(ToolDefinition(
        name="mute",
        description="Mute the system audio.",
        parameters=[],
        target_agent="SystemAgent",
        action="mute",
        category=ToolCategory.SYSTEM,
        permission=ToolPermission.SAFE,
        tags={"volume", "audio", "mute"},
    ))
    
    tools.append(ToolDefinition(
        name="unmute",
        description="Unmute the system audio.",
        parameters=[],
        target_agent="SystemAgent",
        action="unmute",
        category=ToolCategory.SYSTEM,
        permission=ToolPermission.SAFE,
        tags={"volume", "audio", "mute"},
    ))
    
    # Brightness Control
    tools.append(ToolDefinition(
        name="set_brightness",
        description="Set the screen brightness to a specific level.",
        parameters=[
            ToolParameter(
                name="level",
                param_type="number",
                description="Brightness level from 0 to 100",
                required=True,
            ),
        ],
        target_agent="SystemAgent",
        action="set_brightness",
        category=ToolCategory.SYSTEM,
        permission=ToolPermission.SAFE,
        tags={"brightness", "display", "screen"},
    ))
    
    tools.append(ToolDefinition(
        name="get_brightness",
        description="Get the current screen brightness level.",
        parameters=[],
        target_agent="SystemAgent",
        action="get_brightness",
        category=ToolCategory.SYSTEM,
        permission=ToolPermission.SAFE,
        tags={"brightness", "display", "screen"},
    ))
    
    # Time/Date
    tools.append(ToolDefinition(
        name="get_time",
        description="Get the current time.",
        parameters=[],
        target_agent="SystemAgent",
        action="get_time",
        category=ToolCategory.UTILITY,
        permission=ToolPermission.SAFE,
        tags={"time", "clock"},
    ))
    
    tools.append(ToolDefinition(
        name="get_date",
        description="Get the current date.",
        parameters=[],
        target_agent="SystemAgent",
        action="get_date",
        category=ToolCategory.UTILITY,
        permission=ToolPermission.SAFE,
        tags={"date", "calendar"},
    ))
    
    # System Info
    tools.append(ToolDefinition(
        name="get_system_info",
        description="Get comprehensive system information including CPU, memory, disk, and battery.",
        parameters=[],
        target_agent="SystemAgent",
        action="system_info",
        category=ToolCategory.SYSTEM,
        permission=ToolPermission.SAFE,
        tags={"system", "info", "status"},
    ))
    
    tools.append(ToolDefinition(
        name="get_cpu_usage",
        description="Get current CPU usage percentage.",
        parameters=[],
        target_agent="SystemAgent",
        action="get_cpu",
        category=ToolCategory.SYSTEM,
        permission=ToolPermission.SAFE,
        tags={"cpu", "system", "performance"},
    ))
    
    tools.append(ToolDefinition(
        name="get_memory_usage",
        description="Get current memory (RAM) usage.",
        parameters=[],
        target_agent="SystemAgent",
        action="get_memory",
        category=ToolCategory.SYSTEM,
        permission=ToolPermission.SAFE,
        tags={"memory", "ram", "system"},
    ))
    
    tools.append(ToolDefinition(
        name="get_battery_status",
        description="Get battery level and charging status.",
        parameters=[],
        target_agent="SystemAgent",
        action="get_battery",
        category=ToolCategory.SYSTEM,
        permission=ToolPermission.SAFE,
        tags={"battery", "power", "charging"},
    ))
    
    tools.append(ToolDefinition(
        name="get_disk_usage",
        description="Get disk space usage.",
        parameters=[],
        target_agent="SystemAgent",
        action="get_disk",
        category=ToolCategory.SYSTEM,
        permission=ToolPermission.SAFE,
        tags={"disk", "storage", "space"},
    ))
    
    # Screen Control
    tools.append(ToolDefinition(
        name="lock_screen",
        description="Lock the screen immediately.",
        parameters=[],
        target_agent="SystemAgent",
        action="lock_screen",
        category=ToolCategory.SYSTEM,
        permission=ToolPermission.MEDIUM_RISK,
        tags={"screen", "lock", "security"},
    ))
    
    tools.append(ToolDefinition(
        name="sleep_display",
        description="Put the display to sleep.",
        parameters=[],
        target_agent="SystemAgent",
        action="sleep_display",
        category=ToolCategory.SYSTEM,
        permission=ToolPermission.LOW_RISK,
        tags={"display", "sleep", "screen"},
    ))
    
    tools.append(ToolDefinition(
        name="take_screenshot",
        description="Take a screenshot of the current screen.",
        parameters=[
            ToolParameter(
                name="region",
                param_type="string",
                description="Screenshot region: 'full', 'window', or 'selection'",
                required=False,
                default="full",
                enum=["full", "window", "selection"],
            ),
        ],
        target_agent="SystemAgent",
        action="screenshot",
        category=ToolCategory.SYSTEM,
        permission=ToolPermission.LOW_RISK,
        tags={"screenshot", "capture", "screen"},
    ))
    
    # Web
    tools.append(ToolDefinition(
        name="search_web",
        description="Search the web using the default browser.",
        parameters=[
            ToolParameter(
                name="query",
                param_type="string",
                description="The search query",
                required=True,
                examples=["weather today", "python documentation"],
            ),
        ],
        target_agent="SystemAgent",
        action="search_web",
        category=ToolCategory.WEB,
        permission=ToolPermission.LOW_RISK,
        tags={"web", "search", "browser"},
    ))
    
    tools.append(ToolDefinition(
        name="open_url",
        description="Open a URL in the default browser.",
        parameters=[
            ToolParameter(
                name="url",
                param_type="string",
                description="The URL to open",
                required=True,
                examples=["https://google.com", "https://github.com"],
            ),
        ],
        target_agent="SystemAgent",
        action="open_url",
        category=ToolCategory.WEB,
        permission=ToolPermission.LOW_RISK,
        tags={"web", "url", "browser"},
    ))
    
    # Notifications
    tools.append(ToolDefinition(
        name="show_notification",
        description="Show a system notification.",
        parameters=[
            ToolParameter(
                name="title",
                param_type="string",
                description="Notification title",
                required=True,
            ),
            ToolParameter(
                name="message",
                param_type="string",
                description="Notification message body",
                required=True,
            ),
        ],
        target_agent="SystemAgent",
        action="notify",
        category=ToolCategory.COMMUNICATION,
        permission=ToolPermission.SAFE,
        tags={"notification", "alert", "message"},
    ))
    
    # =========================================================================
    # MemoryAgent Tools
    # =========================================================================
    
    tools.append(ToolDefinition(
        name="store_memory",
        description="Store information in long-term memory for later retrieval.",
        parameters=[
            ToolParameter(
                name="key",
                param_type="string",
                description="Unique key to identify this memory",
                required=True,
            ),
            ToolParameter(
                name="value",
                param_type="string",
                description="The information to store",
                required=True,
            ),
            ToolParameter(
                name="category",
                param_type="string",
                description="Category for organization (e.g., 'user_preferences', 'facts')",
                required=False,
                default="general",
            ),
        ],
        target_agent="MemoryAgent",
        action="store",
        category=ToolCategory.MEMORY,
        permission=ToolPermission.LOW_RISK,
        tags={"memory", "store", "remember"},
    ))
    
    tools.append(ToolDefinition(
        name="recall_memory",
        description="Retrieve information from memory by key or query.",
        parameters=[
            ToolParameter(
                name="query",
                param_type="string",
                description="Key or search query to find memory",
                required=True,
            ),
        ],
        target_agent="MemoryAgent",
        action="query",
        category=ToolCategory.MEMORY,
        permission=ToolPermission.SAFE,
        tags={"memory", "recall", "retrieve"},
    ))
    
    tools.append(ToolDefinition(
        name="delete_memory",
        description="Delete a memory by key.",
        parameters=[
            ToolParameter(
                name="key",
                param_type="string",
                description="Key of the memory to delete",
                required=True,
            ),
        ],
        target_agent="MemoryAgent",
        action="delete",
        category=ToolCategory.MEMORY,
        permission=ToolPermission.MEDIUM_RISK,
        tags={"memory", "delete", "forget"},
    ))
    
    # =========================================================================
    # VisionAgent Tools (Future)
    # =========================================================================
    
    tools.append(ToolDefinition(
        name="start_camera",
        description="Start the camera for vision processing.",
        parameters=[],
        target_agent="VisionAgent",
        action="toggle_vision",
        category=ToolCategory.VISION,
        permission=ToolPermission.MEDIUM_RISK,
        tags={"vision", "camera", "start"},
    ))
    
    tools.append(ToolDefinition(
        name="stop_camera",
        description="Stop the camera and vision processing.",
        parameters=[],
        target_agent="VisionAgent",
        action="toggle_vision",
        category=ToolCategory.VISION,
        permission=ToolPermission.SAFE,
        tags={"vision", "camera", "stop"},
    ))
    
    tools.append(ToolDefinition(
        name="detect_gesture",
        description="Detect hand gestures from camera feed. Returns the detected gesture.",
        parameters=[
            ToolParameter(
                name="timeout_seconds",
                param_type="number",
                description="How long to wait for a gesture",
                required=False,
                default=10.0,
            ),
        ],
        target_agent="VisionAgent",
        action="detect_gesture",
        category=ToolCategory.VISION,
        permission=ToolPermission.MEDIUM_RISK,
        tags={"vision", "gesture", "hand"},
    ))
    
    tools.append(ToolDefinition(
        name="recognize_face",
        description="Attempt to recognize faces in the camera feed.",
        parameters=[],
        target_agent="VisionAgent",
        action="recognize_face",
        category=ToolCategory.VISION,
        permission=ToolPermission.MEDIUM_RISK,
        tags={"vision", "face", "recognition"},
    ))
    
    tools.append(ToolDefinition(
        name="enroll_face",
        description="Enroll a new face with a name for future recognition.",
        parameters=[
            ToolParameter(
                name="name",
                param_type="string",
                description="Name to associate with the face",
                required=True,
            ),
        ],
        target_agent="VisionAgent",
        action="enroll_face",
        category=ToolCategory.VISION,
        permission=ToolPermission.HIGH_RISK,
        tags={"vision", "face", "enroll"},
    ))
    
    return tools


# =============================================================================
# Module-level singleton accessor
# =============================================================================

_registry_instance: Optional[ToolRegistry] = None


def get_tool_registry() -> ToolRegistry:
    """Get the global tool registry instance."""
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = ToolRegistry.get_instance()
        _registry_instance.initialize_default_tools()
    return _registry_instance
