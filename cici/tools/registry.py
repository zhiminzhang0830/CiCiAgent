"""Registry-backed BaseTool subclasses + default registry singleton.

Each of the nine simple tools is wrapped in a :class:`BaseTool`
subclass so :mod:`runner` can dispatch through the registry uniformly.
The Anthropic-shaped JSON schemas in :data:`schemas.tool_definitions`
remain the source of truth for what the LLM sees; these classes are
only used at execution time to validate input and run the handler.
"""

from __future__ import annotations

import json

from ..tool_models import (
    AskUserQuestionInput,
    EditFileInput,
    GrepSearchInput,
    ListFilesInput,
    ReadFileInput,
    RunShellInput,
    ToolSearchInput,
    WebFetchInput,
    WriteFileInput,
)
from ..tools_base import BaseTool
from ..tools_base import ToolExecutionContext as _ToolExecutionContext
from ..tools_base import ToolRegistry
from ..tools_base import ToolResult as _ToolResult
from .filesystem import _edit_file, _grep_search, _list_files, _read_file, _write_file
from .schemas import _activated_tools, _derive_is_error, tool_definitions
from .shell import _run_shell
from .web import _web_fetch

# Single source of truth for tool descriptions: schemas.tool_definitions.
# Registry classes look themselves up by ``name`` so the description prose
# lives in exactly one place.
_DESCRIPTIONS_BY_NAME: dict[str, str] = {
    t["name"]: t.get("description", "") for t in tool_definitions
}


class ReadFileTool(BaseTool):
    name = "read_file"
    description = _DESCRIPTIONS_BY_NAME["read_file"]
    input_model = ReadFileInput

    def is_read_only(self, arguments: ReadFileInput) -> bool:
        return True

    async def execute(
        self, arguments: ReadFileInput, context: _ToolExecutionContext
    ) -> _ToolResult:
        del context
        raw = _read_file(arguments.model_dump(exclude_none=True))
        return _ToolResult(output=raw, is_error=_derive_is_error(raw))


class WriteFileTool(BaseTool):
    name = "write_file"
    description = _DESCRIPTIONS_BY_NAME["write_file"]
    input_model = WriteFileInput

    async def execute(
        self, arguments: WriteFileInput, context: _ToolExecutionContext
    ) -> _ToolResult:
        del context
        raw = _write_file(arguments.model_dump())
        return _ToolResult(output=raw, is_error=_derive_is_error(raw))


class EditFileTool(BaseTool):
    name = "edit_file"
    description = _DESCRIPTIONS_BY_NAME["edit_file"]
    input_model = EditFileInput

    async def execute(
        self, arguments: EditFileInput, context: _ToolExecutionContext
    ) -> _ToolResult:
        del context
        raw = _edit_file(arguments.model_dump())
        return _ToolResult(output=raw, is_error=_derive_is_error(raw))


class ListFilesTool(BaseTool):
    name = "list_files"
    description = _DESCRIPTIONS_BY_NAME["list_files"]
    input_model = ListFilesInput

    def is_read_only(self, arguments: ListFilesInput) -> bool:
        return True

    async def execute(
        self, arguments: ListFilesInput, context: _ToolExecutionContext
    ) -> _ToolResult:
        del context
        raw = _list_files(arguments.model_dump(exclude_none=True))
        return _ToolResult(output=raw, is_error=_derive_is_error(raw))


class GrepSearchTool(BaseTool):
    name = "grep_search"
    description = _DESCRIPTIONS_BY_NAME["grep_search"]
    input_model = GrepSearchInput

    def is_read_only(self, arguments: GrepSearchInput) -> bool:
        return True

    async def execute(
        self, arguments: GrepSearchInput, context: _ToolExecutionContext
    ) -> _ToolResult:
        del context
        raw = _grep_search(arguments.model_dump(exclude_none=True))
        return _ToolResult(output=raw, is_error=_derive_is_error(raw))


class RunShellTool(BaseTool):
    name = "run_shell"
    description = _DESCRIPTIONS_BY_NAME["run_shell"]
    input_model = RunShellInput

    # Shell commands mutate the process cwd across calls, so they are
    # never concurrency-safe even if the command itself is read-only.
    def is_concurrency_safe(self, arguments: RunShellInput) -> bool:
        del arguments
        return False

    async def execute(
        self, arguments: RunShellInput, context: _ToolExecutionContext
    ) -> _ToolResult:
        del context
        raw = _run_shell(arguments.model_dump(exclude_none=True))
        return _ToolResult(output=raw, is_error=_derive_is_error(raw))


class WebFetchTool(BaseTool):
    name = "web_fetch"
    description = _DESCRIPTIONS_BY_NAME["web_fetch"]
    input_model = WebFetchInput

    def is_read_only(self, arguments: WebFetchInput) -> bool:
        return True

    async def execute(
        self, arguments: WebFetchInput, context: _ToolExecutionContext
    ) -> _ToolResult:
        del context
        raw = _web_fetch(arguments.model_dump(exclude_none=True))
        return _ToolResult(output=raw, is_error=_derive_is_error(raw))


class ToolSearchTool(BaseTool):
    """Activate deferred tools matching a query and return their schemas.

    Mutates module-level ``_activated_tools``. Read-only from the
    filesystem's perspective but not concurrency-safe: parallel calls
    would race on the shared activation set and could return partial
    schema lists.
    """

    name = "tool_search"
    description = _DESCRIPTIONS_BY_NAME["tool_search"]
    input_model = ToolSearchInput

    def is_read_only(self, arguments: ToolSearchInput) -> bool:
        del arguments
        return True

    def is_concurrency_safe(self, arguments: ToolSearchInput) -> bool:
        del arguments
        return False

    async def execute(
        self, arguments: ToolSearchInput, context: _ToolExecutionContext
    ) -> _ToolResult:
        del context
        query = (arguments.query or "").lower()
        deferred = [t for t in tool_definitions if t.get("deferred")]
        matches = [
            t
            for t in deferred
            if query in t["name"].lower()
            or query in (t.get("description") or "").lower()
        ]
        if not matches:
            return _ToolResult(output="No matching deferred tools found.")
        for m in matches:
            _activated_tools.add(m["name"])
        payload = json.dumps(
            [
                {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "input_schema": t["input_schema"],
                }
                for t in matches
            ],
            indent=2,
        )
        return _ToolResult(output=payload)


_DEFAULT_REGISTRY: ToolRegistry | None = None


class AskUserQuestionTool(BaseTool):
    """Prompt the user with one or more multiple-choice questions.

    Renders each question and its options to the terminal and blocks on
    ``input()`` until the user answers. An ``Other`` escape hatch lets
    the user type a free-form reply. Returns a JSON-shaped string mapping
    each question to the user's selected labels (and free-form text when
    ``Other`` was chosen).
    """

    name = "ask_user_question"
    description = _DESCRIPTIONS_BY_NAME["ask_user_question"]
    input_model = AskUserQuestionInput

    # Interactive: no filesystem side effects, but not concurrency-safe
    # (parallel prompts would race on stdin).
    def is_read_only(self, arguments: AskUserQuestionInput) -> bool:
        del arguments
        return True

    def is_concurrency_safe(self, arguments: AskUserQuestionInput) -> bool:
        del arguments
        return False

    async def execute(
        self,
        arguments: AskUserQuestionInput,
        context: _ToolExecutionContext,
    ) -> _ToolResult:
        del context
        from ..ui import ask_questions

        # Serialize to plain dicts so backends don't need pydantic.
        questions: list[dict] = []
        for q in arguments.questions:
            questions.append(
                {
                    "question": q.question,
                    "header": q.header,
                    "multi_select": q.multi_select,
                    "options": [
                        {"label": opt.label, "description": opt.description}
                        for opt in q.options
                    ],
                }
            )
        try:
            answers = await ask_questions(questions)
        except (EOFError, KeyboardInterrupt):
            return _ToolResult(
                output="Error: user aborted while answering questions.",
                is_error=True,
            )
        return _ToolResult(output=json.dumps({"answers": answers}, indent=2))


def create_default_registry() -> ToolRegistry:
    """Build a ``ToolRegistry`` populated with every registry-backed tool.

    The remaining tools (``agent``, ``skill``, ``enter_plan_mode``,
    ``exit_plan_mode``) are tightly coupled to :class:`~ocsci.coding_agent.agent.Agent`
    instance state and intentionally stay as ``Agent`` methods rather
    than ``BaseTool`` subclasses; see ``agent.py`` for their dispatch.
    """
    registry = ToolRegistry()
    for cls in (
        ReadFileTool,
        WriteFileTool,
        EditFileTool,
        ListFilesTool,
        GrepSearchTool,
        RunShellTool,
        WebFetchTool,
        ToolSearchTool,
        AskUserQuestionTool,
    ):
        registry.register(cls())
    return registry


def get_default_registry() -> ToolRegistry:
    """Return a lazily-initialized shared registry.

    Safe to call from anywhere (including ``tests``) — the factory is
    deterministic and idempotent.
    """
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = create_default_registry()
    return _DEFAULT_REGISTRY
