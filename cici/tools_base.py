"""Tool abstractions — BaseTool / ToolRegistry / ToolExecutionContext / ToolResult.

Individual tools subclass ``BaseTool`` and are registered with a
``ToolRegistry``, which owns the ``name -> tool`` mapping and generates
the Anthropic Messages API schema.

This module is intentionally dependency-free (only ``pydantic``) so it
can be imported from anywhere in the package without risking a circular
import with ``tools.py`` / ``agent.py``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel


@dataclass
class ToolExecutionContext:
    """Shared execution context threaded through every tool call.

    ``metadata`` carries per-invocation state that is currently passed to
    ``execute_tool`` as loose kwargs (``read_file_state``) or reached
    through module-global singletons (``_activated_tools``, shell cwd,
    plan-file path, parent-agent reference for the ``agent`` / ``skill``
    tools). Consolidating those into a single dict keeps tool signatures
    uniform and makes the registry-based dispatcher straightforward.
    """

    cwd: Path
    metadata: dict[str, Any] = field(default_factory=dict)
    # Placeholder for the hook executor wired in a later phase. Typed as
    # ``Any`` so this module stays dependency-free.
    hook_executor: Any = None


@dataclass(frozen=True)
class ToolResult:
    """Normalized tool execution result.

    Replaces the legacy "string with an ``Error:`` prefix" contract with
    an explicit ``is_error`` flag plus room for structured ``metadata``
    (e.g. spawned agent ids, exit codes, artifact paths).
    """

    output: str
    is_error: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseTool(ABC):
    """Base class for all coding-agent tools.

    Subclasses declare four class attributes and one coroutine:

    - ``name``: stable identifier used by the model and the registry.
    - ``description``: model-facing prose shipped in the API schema.
    - ``input_model``: Pydantic model the registry uses for validation
      and schema generation.
    - ``deferred`` (optional): if ``True`` the tool is hidden from the
      default tool list and only surfaced after ``tool_search``
      activates it. Replaces the legacy ``"deferred": True`` dict key.
    - ``execute``: the async entry point that runs the tool.

    ``is_read_only`` / ``is_concurrency_safe`` are methods (not static
    sets) so tools that care about their arguments — e.g. ``run_shell``
    inspecting ``command`` — can decide dynamically.
    """

    name: str
    description: str
    input_model: type[BaseModel]
    deferred: bool = False

    @abstractmethod
    async def execute(
        self,
        arguments: BaseModel,
        context: ToolExecutionContext,
    ) -> ToolResult:
        """Execute the tool and return its normalized result."""

    def is_read_only(self, arguments: BaseModel) -> bool:
        """Whether this invocation performs no side effects.

        Read-only tools bypass permission confirmation in every mode
        except ``plan`` (where mutation is blocked outright).
        """
        del arguments
        return False

    def is_concurrency_safe(self, arguments: BaseModel) -> bool:
        """Whether this invocation can run in parallel with siblings.

        Defaults to ``is_read_only``. Override when a tool is read-only
        but shares mutable process state (e.g. a shell session with a
        persistent ``cwd``).
        """
        return self.is_read_only(arguments)

    def to_api_schema(self) -> dict[str, Any]:
        """Return the tool schema expected by the Anthropic Messages API."""
        schema = self.input_model.model_json_schema()
        _strip_titles(schema)
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": schema,
        }


class ToolRegistry:
    """Map tool names to ``BaseTool`` instances."""

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Register a tool instance. Re-registering the same name overwrites."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool | None:
        """Return the registered tool or ``None`` if ``name`` is unknown."""
        return self._tools.get(name)

    def list_tools(self) -> list[BaseTool]:
        """Return every registered tool in registration order."""
        return list(self._tools.values())

    def to_api_schema(
        self,
        *,
        include_deferred: bool = False,
        activated: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Return tool schemas in Anthropic Messages API format.

        Deferred tools are hidden unless explicitly activated (matching
        the legacy ``get_active_tool_definitions`` behavior) or the
        caller passes ``include_deferred=True`` to dump every tool.
        """
        activated = activated or set()
        return [
            tool.to_api_schema()
            for tool in self._tools.values()
            if include_deferred or not tool.deferred or tool.name in activated
        ]


def _strip_titles(obj: object) -> None:
    """Strip pydantic-generated ``title`` keys in-place.

    Pydantic emits a ``title`` for every field derived from the class
    name; the Anthropic API accepts them but they add noise without
    changing semantics.
    """
    if isinstance(obj, dict):
        obj.pop("title", None)
        for value in obj.values():
            _strip_titles(value)
    elif isinstance(obj, list):
        for value in obj:
            _strip_titles(value)
