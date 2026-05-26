"""Tool catalogue, constants, activation, and result helpers.

This is the leaf module of the ``cici.tools`` package: it defines the
JSON schema list (:data:`tool_definitions`) sent to the LLM, the
permission-mode tags, the sensitive-path hard-block, and the tiny
result helpers (``ToolResult`` dataclass, error-prefix detection,
output truncation) shared by the registry and the dispatcher.

It deliberately depends on **nothing** else inside ``cici.tools`` so
both ``registry.py`` and ``runner.py`` can import from it without
risking a circular import.
"""

from __future__ import annotations

import fnmatch
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

# ─── Permission modes ──────────────────────────────────────

PermissionMode = (
    str  # "default" | "plan" | "acceptEdits" | "bypassPermissions" | "dontAsk"
)

READ_TOOLS = {
    "read_file",
    "list_files",
    "grep_search",
    "web_fetch",
    "ask_user_question",
}
EDIT_TOOLS = {"write_file", "edit_file"}

# Concurrency-safe tools can run in parallel (read-only, no side effects)
CONCURRENCY_SAFE_TOOLS = {
    "read_file",
    "list_files",
    "grep_search",
    "web_fetch",
}

IS_WIN = sys.platform == "win32"

# Max file size for read_file without offset/limit. Files larger than this
# require offset+limit to read specific portions. Prevents OOM and bounds
# output so it doesn't need to be truncated or persisted (which would create
# a circular Read→file→Read loop).
MAX_FILE_SIZE_BYTES = 256 * 1024  # 256 KB

# Max bytes inspected for binary detection in read_file
_BINARY_SNIFF_BYTES = 8192

# ─── Sensitive path hard-block (cannot be overridden) ───────
# Inspired by OpenHarness: even if user rules/allow lists are permissive,
# these paths must never be read or written by the agent. Guards against
# prompt injection tricking the model into exfiltrating credentials.
SENSITIVE_PATH_PATTERNS: list[str] = [
    "*/.ssh/*",
    "*/.aws/*",
    "*/.aws/credentials",
    "*/.config/gcloud/*",
    "*/.azure/*",
    "*/.gnupg/*",
    "*/.docker/config.json",
    "*/.kube/config",
    "*/.netrc",
    "*/.pypirc",
    "*/.npmrc",
    "*/.git-credentials",
]


def _is_sensitive_path(path_str: str) -> bool:
    """Check if a path matches any hard-blocked sensitive pattern.
    Resolves the path (following ~ and symlinks best-effort) to defeat
    trivial obfuscation like ``~/.ssh/../.ssh/id_rsa``."""
    try:
        resolved = str(Path(path_str).expanduser().resolve(strict=False))
    except Exception:
        resolved = path_str
    normalized = resolved.replace(os.sep, "/")
    return any(fnmatch.fnmatch(normalized, pat) for pat in SENSITIVE_PATH_PATTERNS)


# ─── Type alias ──────────────────────────────────────────────

ToolDef = dict  # Anthropic tool schema dict

# ─── Tool definitions (Pydantic SSOT) ──────────────────────
# Every entry is built from a Pydantic input model in
# :mod:`cici.tool_models` via :func:`to_tool_schema`. The description
# strings stay here because they are the user-facing prompt copy and
# reference module-level constants like :data:`MAX_FILE_SIZE_BYTES`.

from ..tool_models import (  # noqa: E402
    AgentInput,
    AgentResultInput,
    AskUserQuestionInput,
    EditFileInput,
    EnterPlanModeInput,
    ExitPlanModeInput,
    GrepSearchInput,
    ListFilesInput,
    ReadFileInput,
    RunShellInput,
    SkillInput,
    TodoWriteInput,
    ToolSearchInput,
    WebFetchInput,
    WriteFileInput,
    to_tool_schema,
)

_READ_FILE_DESCRIPTION = (
    "Read the contents of a file. Returns the file content with line numbers. "
    f"Files larger than {MAX_FILE_SIZE_BYTES // 1024} KB must be read with "
    "offset and limit to select a specific range."
)

_WRITE_FILE_DESCRIPTION = (
    "Write content to a file. Creates the file if it doesn't exist, "
    "overwrites if it does.\n\n"
    "Path rules:\n"
    "- For deliverable files that belong to the user's project, write under the working directory.\n"
    "- For temporary/intermediate artifacts (working notes, captured outputs, throwaway scripts), "
    "write under the session scratchpad directory given in the system prompt (Temporary files section). "
    "Do NOT write temp files to `/tmp` or scatter them across arbitrary paths like `/tmp/foo.md`.\n"
    "- Do not create files unless necessary; prefer edit_file on an existing file when possible."
)

_RUN_SHELL_DESCRIPTION = (
    "Execute a shell command and return its output.\n\n"
    "The working directory persists between commands (POSIX: `cd /some/dir` "
    "in one call affects the next), but environment variables do NOT persist "
    "across calls. On timeout, the entire process group is terminated.\n\n"
    "IMPORTANT: Do NOT use this tool for tasks that have a dedicated tool:\n"
    "- Read files: use read_file (NOT cat/head/tail)\n"
    "- Edit files: use edit_file (NOT sed/awk)\n"
    "- Write files: use write_file (NOT echo >, cat <<EOF)\n"
    "- Find files: use list_files (NOT find/ls)\n"
    "- Search content: use grep_search (NOT grep/rg)\n"
    "- Communicate: output text directly (NOT echo/printf)\n\n"
    "# Instructions\n"
    "- Before creating files/dirs, run `ls` to verify the parent directory exists.\n"
    '- Quote paths containing spaces: cd "path with spaces".\n'
    "- Prefer absolute paths. Avoid `cd` unless the user explicitly requests a directory change.\n"
    "- Temporary files: do NOT write to `/tmp` or scatter files across the project. "
    "Use the session scratchpad directory given in the system prompt (Temporary files section). "
    "For one-off verification, prefer inline execution (`python -c '...'`, shell heredoc) over "
    "writing a script to disk. If output is truncated, capture to a file inside the scratchpad "
    "(e.g. `... 2>&1 | tee <scratchpad>/out.txt`) and delete it when done.\n"
    "- When issuing multiple commands:\n"
    "  - Independent commands: issue multiple run_shell calls in parallel (one message, multiple tool uses).\n"
    "  - Dependent commands: chain with `&&` in a single call.\n"
    "  - Use `;` only when you don't care if earlier commands fail.\n"
    "  - Do NOT use newlines to separate commands (newlines in quoted strings are fine).\n"
    "- For git: never pass `--no-verify`, `--force`, `-i`, or `--amend` unless the user explicitly asks.\n"
    "- Avoid unnecessary `sleep`: don't poll in a loop, don't retry failing commands with sleep — diagnose instead."
)

_ASK_USER_QUESTION_DESCRIPTION = (
    "Ask the user one or more multiple-choice questions and return "
    "their answers. Use this when you need specific input from the "
    "user to proceed (e.g. clarifying ambiguous requirements, "
    "choosing between implementation approaches, picking a library). "
    "Each question has 2-4 options; an 'Other' option for free-text "
    "input is appended automatically. Prefer this over open-ended "
    "text questions when the answer space is bounded. Do not use "
    "for yes/no confirmations of dangerous actions — permission "
    "prompts already handle those."
)

tool_definitions: list[ToolDef] = [
    to_tool_schema("read_file", _READ_FILE_DESCRIPTION, ReadFileInput),
    to_tool_schema("write_file", _WRITE_FILE_DESCRIPTION, WriteFileInput),
    to_tool_schema(
        "edit_file",
        (
            "Edit a file by replacing an exact string match with new content. "
            "The old_string must match exactly (including whitespace and indentation)."
        ),
        EditFileInput,
    ),
    to_tool_schema(
        "list_files",
        "List files matching a glob pattern. Returns matching file paths.",
        ListFilesInput,
    ),
    to_tool_schema(
        "grep_search",
        (
            "Search for a pattern in files. Returns matching lines with file "
            "paths and line numbers."
        ),
        GrepSearchInput,
    ),
    to_tool_schema("run_shell", _RUN_SHELL_DESCRIPTION, RunShellInput),
    to_tool_schema(
        "skill",
        (
            "Invoke a registered skill by name. Skills are prompt templates "
            "loaded from .claude/skills/. Returns the skill's resolved "
            "prompt to follow."
        ),
        SkillInput,
    ),
    to_tool_schema(
        "web_fetch",
        (
            "Fetch a URL and return its content as text. For HTML pages, "
            "tags are stripped to return readable text. For JSON/text "
            "responses, content is returned directly."
        ),
        WebFetchInput,
    ),
    {
        **to_tool_schema(
            "enter_plan_mode",
            (
                "Enter plan mode to switch to a read-only planning phase. "
                "In plan mode, you can only read files and write to the "
                "plan file."
            ),
            EnterPlanModeInput,
        ),
        "deferred": True,
    },
    {
        **to_tool_schema(
            "exit_plan_mode",
            "Exit plan mode after you have finished writing your plan to the plan file.",
            ExitPlanModeInput,
        ),
        "deferred": True,
    },
    to_tool_schema(
        "agent",
        (
            "Launch a sub-agent to handle a task autonomously. Sub-agents "
            "have isolated context and return their result. Types: "
            "'explore' (read-only), 'plan' (read-only, structured "
            "planning), 'general' (full tools). Set "
            "run_in_background=true to launch asynchronously and "
            "retrieve the result later with the agent_result tool."
        ),
        AgentInput,
    ),
    to_tool_schema(
        "agent_result",
        (
            "Retrieve the output of a background sub-agent launched via "
            "agent(run_in_background=true, ...). Returns the final text "
            "if the sub-agent has finished, or a 'still running' status "
            "message otherwise."
        ),
        AgentResultInput,
    ),
    to_tool_schema(
        "tool_search",
        (
            "Search for available tools by name or keyword. Returns full "
            "schema definitions for matching deferred tools so you can "
            "use them."
        ),
        ToolSearchInput,
    ),
    to_tool_schema(
        "ask_user_question",
        _ASK_USER_QUESTION_DESCRIPTION,
        AskUserQuestionInput,
    ),
    to_tool_schema(
        "todo_write",
        (
            "Maintain a persistent TODO checklist for the current session. "
            "Each call replaces the full list. Use this proactively for "
            "multi-step tasks so progress is visible. Exactly one task "
            "should be in_progress at a time."
        ),
        TodoWriteInput,
    ),
]
# ─── Deferred tool activation ───────────────────────────────

_activated_tools: set[str] = set()


def reset_activated_tools() -> None:
    _activated_tools.clear()


def get_active_tool_definitions(
    all_tools: list[ToolDef] | None = None,
) -> list[ToolDef]:
    """Return tool definitions, excluding deferred tools that haven't been activated.
    Strips the 'deferred' key so it's not sent to the API."""
    tools = all_tools if all_tools is not None else tool_definitions
    return [
        {k: v for k, v in t.items() if k != "deferred"}
        for t in tools
        if not t.get("deferred") or t["name"] in _activated_tools
    ]


def get_deferred_tool_names(all_tools: list[ToolDef] | None = None) -> list[str]:
    """Return names of deferred tools that haven't been activated yet."""
    tools = all_tools if all_tools is not None else tool_definitions
    return [
        t["name"]
        for t in tools
        if t.get("deferred") and t["name"] not in _activated_tools
    ]


# ─── Result helpers (shared by registry + runner) ───────────
# Lifted out of runner.py so registry's BaseTool subclasses can derive
# ``is_error`` without forcing a circular import (runner imports the
# registry to dispatch; registry needs the helper at class-body
# evaluation time).


@dataclass(frozen=True)
class ToolResult:
    """Normalized tool execution result (OpenHarness-style)."""

    output: str
    is_error: bool = False
    metadata: dict = field(default_factory=dict)


_ERROR_PREFIXES: tuple[str, ...] = (
    "Error:",
    "Error ",
    "Warning:",
    "Command failed",
    "HTTP error:",
    "Unknown tool:",
)


def _derive_is_error(raw: str) -> bool:
    """Best-effort in-band error detection for string-returning handlers.
    Matches the legacy prefix convention so existing handlers don't need
    to be rewritten."""
    if not isinstance(raw, str):
        return False
    return any(raw.startswith(p) for p in _ERROR_PREFIXES)


# ─── Truncate long tool results ─────────────────────────────

MAX_RESULT_CHARS = 50000


def _truncate_result(result: str) -> str:
    if len(result) <= MAX_RESULT_CHARS:
        return result
    keep_each = (MAX_RESULT_CHARS - 60) // 2
    return (
        result[:keep_each]
        + f"\n\n[... truncated {len(result) - keep_each * 2} chars ...]\n\n"
        + result[-keep_each:]
    )
