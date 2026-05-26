"""Built-in tool catalogue and dispatch (package).

Public re-export surface for the ``cici.tools`` package. The actual
implementation lives in eight focused submodules:

* :mod:`schemas` — constants, ``tool_definitions``, activation, result helpers.
* :mod:`permission` — rule loading, dangerous-command detection, ``check_permission``.
* :mod:`filesystem` — read/write/edit/list/grep handlers.
* :mod:`shell` — persistent-cwd shell execution + scaffold preflight.
* :mod:`web` — ``web_fetch`` handler.
* :mod:`todos` — ``todo_write`` handler.
* :mod:`registry` — :class:`BaseTool` subclasses + default registry.
* :mod:`runner` — :func:`execute_tool` dispatcher (string + structured).
"""

from __future__ import annotations

from .permission import (  # noqa: F401
    DANGEROUS_PATTERNS,
    check_permission,
    is_dangerous,
    load_permission_rules,
    reset_permission_cache,
)
from .registry import (  # noqa: F401
    AskUserQuestionTool,
    EditFileTool,
    GrepSearchTool,
    ListFilesTool,
    ReadFileTool,
    RunShellTool,
    ToolSearchTool,
    WebFetchTool,
    WriteFileTool,
    create_default_registry,
    get_default_registry,
)
from .runner import (  # noqa: F401
    execute_tool,
    execute_tool_structured,
    is_concurrency_safe_tool,
)
from .schemas import (  # noqa: F401
    CONCURRENCY_SAFE_TOOLS,
    EDIT_TOOLS,
    IS_WIN,
    MAX_FILE_SIZE_BYTES,
    MAX_RESULT_CHARS,
    READ_TOOLS,
    SENSITIVE_PATH_PATTERNS,
    PermissionMode,
    ToolDef,
    ToolResult,
    _derive_is_error,
    get_active_tool_definitions,
    get_deferred_tool_names,
    reset_activated_tools,
    tool_definitions,
)
from .shell import reset_shell_state  # noqa: F401

__all__ = [
    "CONCURRENCY_SAFE_TOOLS",
    "DANGEROUS_PATTERNS",
    "EDIT_TOOLS",
    "IS_WIN",
    "MAX_FILE_SIZE_BYTES",
    "MAX_RESULT_CHARS",
    "PermissionMode",
    "READ_TOOLS",
    "SENSITIVE_PATH_PATTERNS",
    "ToolDef",
    "tool_definitions",
    "get_active_tool_definitions",
    "get_deferred_tool_names",
    "reset_activated_tools",
    "check_permission",
    "is_dangerous",
    "load_permission_rules",
    "reset_permission_cache",
    "execute_tool",
    "execute_tool_structured",
    "is_concurrency_safe_tool",
    "reset_shell_state",
    "ToolResult",
    "AskUserQuestionTool",
    "EditFileTool",
    "GrepSearchTool",
    "ListFilesTool",
    "ReadFileTool",
    "RunShellTool",
    "ToolSearchTool",
    "WebFetchTool",
    "WriteFileTool",
    "create_default_registry",
    "get_default_registry",
]
