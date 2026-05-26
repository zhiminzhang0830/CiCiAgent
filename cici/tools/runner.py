"""Tool dispatch: ``execute_tool`` (legacy str) and ``execute_tool_structured``.

This is the orchestration layer that wires permissions, the registry,
and the legacy handlers dict together. It is NOT the source of truth
for any tool — that's the registry — it just sequences pre-checks,
dispatches, and applies the post-call truncation/mtime bookkeeping
that's specific to the read-before-edit guard.

Dispatch order:
1. read-before-edit + mtime freshness checks (``read_file`` fast path,
   ``write_file``/``edit_file`` precondition).
2. ``ToolRegistry`` lookup for the nine registry-backed tools.
3. Legacy ``handlers`` dict for ``todo_write``.
"""

from __future__ import annotations

import os
from pathlib import Path

from ..tools_base import ToolExecutionContext as _ToolExecutionContext
from .filesystem import _read_file
from .registry import get_default_registry
from .schemas import (
    CONCURRENCY_SAFE_TOOLS,
    ToolResult,
    _derive_is_error,
    _truncate_result,
)
from .todos import _todo_write

async def execute_tool(
    name: str, inp: dict, read_file_state: dict[str, float] | None = None
) -> str:
    # ─── read-before-edit + mtime freshness checks ───────────
    if name == "read_file":
        result = _read_file(inp)
        if read_file_state is not None and not result.startswith("Error"):
            abs_path = str(Path(inp["file_path"]).resolve())
            try:
                read_file_state[abs_path] = os.path.getmtime(abs_path)
            except OSError:
                pass
        # Return raw result — _read_file already self-bounds via MAX_FILE_SIZE_BYTES
        # and the offset/limit slice. Don't truncate: truncation would lose the
        # original content, and persisting a truncated read_file output creates
        # a circular Read→file→Read loop (the model reads the saved file, which
        # is also truncated, forever).
        return result

    if name in ("write_file", "edit_file") and read_file_state is not None:
        abs_path = str(Path(inp["file_path"]).resolve())
        if os.path.exists(abs_path):
            if abs_path not in read_file_state:
                verb = "writing" if name == "write_file" else "editing"
                return f"Error: You must read this file before {verb}. Use read_file first to see its current contents."
            if os.path.getmtime(abs_path) != read_file_state[abs_path]:
                verb = "writing" if name == "write_file" else "editing"
                return f"Warning: {inp['file_path']} was modified externally since your last read. Please read_file again before {verb}."

    # tool_search: handled by ``ToolSearchTool`` via the registry
    # dispatch below. The activation of deferred tools is a side-effect
    # on the module-global ``_activated_tools`` set.

    # Registry-backed dispatch for migrated tools. The registry is the
    # source of truth for all registry-backed tools; legacy dict-based
    # handlers below remain the fallback for dead code paths that the
    # model cannot currently reach (todo_write is not in
    # ``tool_definitions``).
    result = await _dispatch_via_registry(name, inp)
    if result is not None:
        truncated = _truncate_result(result)
        if (
            name in ("write_file", "edit_file")
            and read_file_state is not None
            and not truncated.startswith("Error")
        ):
            abs_path = str(Path(inp["file_path"]).resolve())
            try:
                read_file_state[abs_path] = os.path.getmtime(abs_path)
            except OSError:
                pass
        return truncated

    handlers: dict = {
        "todo_write": _todo_write,
    }
    handler = handlers.get(name)
    if not handler:
        return f"Unknown tool: {name}"
    return _truncate_result(handler(inp))


async def _dispatch_via_registry(name: str, inp: dict) -> str | None:
    """Run a registered ``BaseTool`` and return its string output.

    Returns ``None`` when ``name`` is not in the default registry, so
    ``execute_tool`` can fall through to the legacy handlers dict.
    Validation errors are converted to the legacy ``Error: ...`` string
    contract so callers that string-sniff for errors keep working.
    """
    registry = get_default_registry()
    tool = registry.get(name)
    if tool is None:
        return None
    try:
        parsed = tool.input_model.model_validate(inp)
    except Exception as exc:
        return f"Error: invalid arguments for {name}: {exc}"
    context = _ToolExecutionContext(cwd=Path.cwd())
    result = await tool.execute(parsed, context)
    return result.output

def is_concurrency_safe_tool(name: str, inp: dict | None = None) -> bool:
    """Return whether a tool invocation can run alongside its siblings.

    Replaces the legacy ``name in CONCURRENCY_SAFE_TOOLS`` check. For
    registered tools the decision comes from
    :meth:`BaseTool.is_concurrency_safe`, which can inspect arguments —
    ``run_shell`` for instance always returns ``False`` because it
    mutates the shared persistent cwd. For unmigrated tools we fall
    back to the static set so behaviour stays identical.
    """
    tool = get_default_registry().get(name)
    if tool is not None:
        try:
            parsed = tool.input_model.model_validate(inp or {})
        except Exception:
            # Malformed input → be conservative; caller will surface the
            # validation error once it actually executes.
            return False
        return tool.is_concurrency_safe(parsed)
    return name in CONCURRENCY_SAFE_TOOLS

async def execute_tool_structured(
    name: str, inp: dict, read_file_state: dict[str, float] | None = None
) -> ToolResult:
    """Structured wrapper around ``execute_tool``.

    Returns a :class:`ToolResult` with ``is_error`` set when the underlying
    handler returned an error-prefixed string. Prefer this API for new
    callers; the legacy str-returning ``execute_tool`` is retained so
    existing code paths keep working unchanged.
    """
    raw = await execute_tool(name, inp, read_file_state)
    return ToolResult(output=raw, is_error=_derive_is_error(raw))
