"""Pydantic input models for coding-agent tools.

Phase 1 goal: define the input contract of every tool here so the
Anthropic schema can be auto-derived from a Pydantic model, eliminating
schema drift and giving us runtime validation for free.

Older tools (``read_file``, ``write_file``, ...) still use hand-written
JSON schemas in ``tools.py`` while phase 2 migrates them to subclass
``BaseTool``. The models below are additive: importing them has no
behavioural effect on the existing code path.

Usage (phase 2, once ``BaseTool`` is wired in)::

    class ReadFileTool(BaseTool):
        name = "read_file"
        description = "..."
        input_model = ReadFileInput

        async def execute(self, arguments, context):
            ...
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .tools_base import _strip_titles

# ─── File I/O ───────────────────────────────────────────────


class ReadFileInput(BaseModel):
    """Arguments for the read_file tool."""

    file_path: str = Field(description="The path to the file to read")
    offset: int = Field(
        default=0,
        ge=0,
        description=(
            "The line number to start reading from (0-indexed). "
            "Only provide if the file is too large to read at once."
        ),
    )
    limit: int | None = Field(
        default=None,
        ge=1,
        description=(
            "The number of lines to read. Only provide if the file "
            "is too large to read at once."
        ),
    )


class WriteFileInput(BaseModel):
    """Arguments for the write_file tool."""

    file_path: str = Field(description="The path to the file to write")
    content: str = Field(description="The content to write to the file")


class EditFileInput(BaseModel):
    """Arguments for the edit_file tool."""

    file_path: str = Field(description="The path to the file to edit")
    old_string: str = Field(description="The exact string to find and replace")
    new_string: str = Field(description="The string to replace it with")


# ─── Discovery ──────────────────────────────────────────────


class ListFilesInput(BaseModel):
    """Arguments for the list_files tool."""

    pattern: str = Field(
        description='Glob pattern to match files (e.g., "**/*.ts", "src/**/*")'
    )
    path: str | None = Field(
        default=None,
        description="Base directory to search from. Defaults to current directory.",
    )


class GrepSearchInput(BaseModel):
    """Arguments for the grep_search tool."""

    pattern: str = Field(description="The regex pattern to search for")
    path: str | None = Field(
        default=None,
        description="Directory or file to search in. Defaults to current directory.",
    )
    include: str | None = Field(
        default=None,
        description='File glob pattern to include (e.g., "*.ts", "*.py")',
    )


# ─── Shell / network ────────────────────────────────────────


class RunShellInput(BaseModel):
    """Arguments for the run_shell tool."""

    command: str = Field(description="The shell command to execute.")
    timeout: float | None = Field(
        default=None,
        gt=0,
        description=(
            "Timeout in milliseconds. Default: 30000 (30s). Max: 600000 (10min)."
        ),
    )


class WebFetchInput(BaseModel):
    """Arguments for the web_fetch tool."""

    url: str = Field(description="The URL to fetch")
    max_length: int | None = Field(
        default=None,
        gt=0,
        description="Maximum content length in characters (default 50000)",
    )


# ─── Skill / planning / sub-agents ──────────────────────────


class SkillInput(BaseModel):
    """Arguments for the skill tool."""

    skill_name: str = Field(description="The name of the skill to invoke")
    args: str | None = Field(
        default=None,
        description="Optional arguments to pass to the skill",
    )


class EmptyInput(BaseModel):
    """Shared empty-input model for tools that take no arguments.

    Used by ``enter_plan_mode`` / ``exit_plan_mode``.
    """


class AgentInput(BaseModel):
    """Arguments for the agent tool."""

    description: str = Field(
        description="Short (3-5 word) description of the sub-agent's task"
    )
    prompt: str = Field(description="Detailed task instructions for the sub-agent")
    type: str = Field(
        default="general",
        description=(
            "Agent type. Built-ins: 'explore', 'plan', 'general'. Custom "
            "agents from .claude/agents/*.md are also accepted. Default: general"
        ),
    )
    run_in_background: bool = Field(
        default=False,
        description=(
            "If true, launch the sub-agent asynchronously and return "
            "immediately with a task_id. Use agent_result to retrieve the "
            "output. Default: false (blocks until completion)."
        ),
    )
    timeout_sec: float | None = Field(
        default=None,
        gt=0,
        description="Wall-clock timeout for the sub-agent in seconds. Default: 300.",
    )


class AgentResultInput(BaseModel):
    """Arguments for the agent_result tool."""

    task_id: str = Field(
        description=(
            "The task_id returned by the agent tool when the sub-agent "
            "was launched in background mode."
        )
    )
    wait_sec: float | None = Field(
        default=None,
        ge=0,
        description=(
            "Optional: block up to this many seconds waiting for "
            "completion (capped at 120). Default: 0 (non-blocking poll)."
        ),
    )


class EnterPlanModeInput(BaseModel):
    """Arguments for the enter_plan_mode tool (no fields)."""


class ExitPlanModeInput(BaseModel):
    """Arguments for the exit_plan_mode tool (no fields)."""


class ToolSearchInput(BaseModel):
    """Arguments for the tool_search tool."""

    query: str = Field(description="Tool name or search keywords")


# ─── ask_user_question ──────────────────────────────────────


class QuestionOption(BaseModel):
    """A single choice within an ask_user_question question."""

    label: str = Field(
        description=(
            "The display text for this option that the user will see and "
            "select. Should be concise (1-5 words) and clearly describe "
            "the choice."
        )
    )
    description: str = Field(
        default="",
        description=(
            "Explanation of what this option means or what will happen if "
            "chosen. Useful for providing context about trade-offs or "
            "implications."
        ),
    )


class Question(BaseModel):
    """A single multiple-choice question."""

    question: str = Field(
        description=(
            "The complete question to ask the user. Should be clear, "
            'specific, and end with a question mark. Example: "Which '
            'library should we use for date formatting?"'
        )
    )
    header: str = Field(
        default="",
        description=(
            "Very short label shown as a chip/tag above the question "
            '(e.g. "Auth method", "Library", "Approach").'
        ),
    )
    options: list[QuestionOption] = Field(
        description=(
            "The available choices for this question. Must have 2-4 "
            "options. Each option should be a distinct, mutually "
            "exclusive choice (unless multi_select is enabled). An "
            "'Other' option is appended automatically."
        ),
        min_length=2,
        max_length=4,
    )
    multi_select: bool = Field(
        default=False,
        description=(
            "Set to true to allow the user to select multiple options "
            "instead of just one. Use when choices are not mutually "
            "exclusive."
        ),
    )


class AskUserQuestionInput(BaseModel):
    """Arguments for the ask_user_question tool."""

    questions: list[Question] = Field(
        description="Questions to ask the user (1-4 questions).",
        min_length=1,
        max_length=4,
    )


# ─── todo_write ─────────────────────────────────────────────


class TodoItem(BaseModel):
    """A single TODO checklist entry."""

    content: str = Field(description="Imperative description of the task")
    status: Literal["pending", "in_progress", "completed"] = Field(
        default="pending",
        description="Current status of this task",
    )


class TodoWriteInput(BaseModel):
    """Arguments for the todo_write tool."""

    todos: list[TodoItem] = Field(
        description=(
            "The full TODO list for the current session. Each call replaces "
            "the previous list. Exactly one task should be 'in_progress'."
        ),
        min_length=1,
    )


# ─── schema helper ──────────────────────────────────────────


def to_tool_schema(name: str, description: str, model: type[BaseModel]) -> dict:
    """Build the Anthropic tool schema dict from a Pydantic model.

    Pydantic emits a JSON schema with a top-level ``$defs`` section for
    nested models (e.g. ``TodoItem`` inside ``TodoWriteInput``). The
    Anthropic API accepts this format, but we strip ``title`` fields that
    pydantic adds — they add noise to the tool registration without
    changing semantics.
    """
    schema = model.model_json_schema()
    _strip_titles(schema)
    return {
        "name": name,
        "description": description,
        "input_schema": schema,
    }
