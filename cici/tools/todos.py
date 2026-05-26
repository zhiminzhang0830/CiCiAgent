"""Persistent TODO checklist handler.

Stored as a markdown file next to the session memory directory so the
list survives across turns. Validates input via the
:class:`cici.tool_models.TodoWriteInput` model.
"""

from __future__ import annotations

from pathlib import Path

from ..memory import get_memory_dir

# ─── todo_write ────────────────────────────────────────────
# Persistent TODO list for the current session. Stored as a markdown
# checklist next to the session memory so the agent (and the user) can
# resume planning context across turns.


def _todo_file_path() -> Path:
    return get_memory_dir() / "TODO.md"


def _todo_write(inp: dict) -> str:
    try:
        # Validate via the Pydantic model so schema mismatches surface early.
        from ..tool_models import TodoWriteInput

        parsed = TodoWriteInput.model_validate(inp)
    except Exception as e:
        return f"Error: invalid todo_write arguments: {e}"

    in_progress = [t for t in parsed.todos if t.status == "in_progress"]
    if len(in_progress) > 1:
        return (
            f"Error: exactly one task should be in_progress at a time, "
            f"got {len(in_progress)}."
        )

    status_icon = {"pending": "[ ]", "in_progress": "[~]", "completed": "[x]"}
    lines = ["# TODO", ""]
    for item in parsed.todos:
        lines.append(f"- {status_icon[item.status]} {item.content}")

    rendered = "\n".join(lines)
    try:
        path = _todo_file_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered + "\n")
    except Exception as e:
        return f"Error writing TODO list: {e}"

    total = len(parsed.todos)
    completed = sum(1 for t in parsed.todos if t.status == "completed")
    summary = f"TODO updated ({completed}/{total} completed)"
    return f"{summary}\n\n{rendered}"

