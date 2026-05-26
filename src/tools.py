"""Tool definitions and execution — 10 tools with 5 permission modes.
Mirrors Claude Code's tool system: read_file, write_file, edit_file, list_files,
grep_search, run_shell, skill, enter/exit_plan_mode, agent."""

from __future__ import annotations

import fnmatch
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .memory import get_memory_dir
from .paths import project_settings_file, user_settings_file

# ─── Permission modes ──────────────────────────────────────

PermissionMode = (
    str  # "default" | "plan" | "acceptEdits" | "bypassPermissions" | "dontAsk"
)

READ_TOOLS = {
    "read_file",
    "list_files",
    "grep_search",
    "web_fetch",
    "web_search",
    "ask_user_question",
}
EDIT_TOOLS = {"write_file", "edit_file"}

# Concurrency-safe tools can run in parallel (read-only, no side effects)
CONCURRENCY_SAFE_TOOLS = {
    "read_file",
    "list_files",
    "grep_search",
    "web_fetch",
    "web_search",
}

IS_WIN = sys.platform == "win32"

# Max file size for read_file without offset/limit. Files larger than this
# require offset+limit to read specific portions. Prevents OOM and bounds
# output so it doesn't need to be truncated or persisted (which would create
# a circular Read→file→Read loop).
MAX_FILE_SIZE_BYTES = 256 * 1024  # 256 KB

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


# Max bytes inspected for binary detection in read_file
_BINARY_SNIFF_BYTES = 8192

# ─── Type alias ──────────────────────────────────────────────

ToolDef = dict  # Anthropic tool schema dict

# ─── Tool definitions ───────────────────────────────────────

tool_definitions: list[ToolDef] = [
    {
        "name": "read_file",
        "description": (
            "Read the contents of a file. Returns the file content with line numbers. "
            f"Files larger than {MAX_FILE_SIZE_BYTES // 1024} KB must be read with "
            "offset and limit to select a specific range."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "The path to the file to read",
                },
                "offset": {
                    "type": "integer",
                    "description": (
                        "The line number to start reading from (0-indexed). "
                        "Only provide if the file is too large to read at once."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "description": (
                        "The number of lines to read. Only provide if the file "
                        "is too large to read at once."
                    ),
                },
            },
            "required": ["file_path"],
        },
    },
    {
        "name": "write_file",
        "description": (
            "Write content to a file. Creates the file if it doesn't exist, "
            "overwrites if it does.\n\n"
            "Path rules:\n"
            "- For deliverable files that belong to the user's project, write under the working directory.\n"
            "- For temporary/intermediate artifacts (working notes, captured outputs, throwaway scripts), "
            "write under the session scratchpad directory given in the system prompt (Temporary files section). "
            "Do NOT write temp files to `/tmp` or scatter them across arbitrary paths like `/tmp/foo.md`.\n"
            "- Do not create files unless necessary; prefer edit_file on an existing file when possible."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "The path to the file to write",
                },
                "content": {
                    "type": "string",
                    "description": "The content to write to the file",
                },
            },
            "required": ["file_path", "content"],
        },
    },
    {
        "name": "edit_file",
        "description": "Edit a file by replacing an exact string match with new content. The old_string must match exactly (including whitespace and indentation).",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "The path to the file to edit",
                },
                "old_string": {
                    "type": "string",
                    "description": "The exact string to find and replace",
                },
                "new_string": {
                    "type": "string",
                    "description": "The string to replace it with",
                },
            },
            "required": ["file_path", "old_string", "new_string"],
        },
    },
    {
        "name": "list_files",
        "description": "List files matching a glob pattern. Returns matching file paths.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": 'Glob pattern to match files (e.g., "**/*.ts", "src/**/*")',
                },
                "path": {
                    "type": "string",
                    "description": "Base directory to search from. Defaults to current directory.",
                },
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "grep_search",
        "description": "Search for a pattern in files. Returns matching lines with file paths and line numbers.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The regex pattern to search for",
                },
                "path": {
                    "type": "string",
                    "description": "Directory or file to search in. Defaults to current directory.",
                },
                "include": {
                    "type": "string",
                    "description": 'File glob pattern to include (e.g., "*.ts", "*.py")',
                },
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "run_shell",
        "description": (
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
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute.",
                },
                "timeout": {
                    "type": "number",
                    "description": "Timeout in milliseconds. Default: 30000 (30s). Max: 600000 (10min).",
                },
            },
            "required": ["command"],
        },
    },
    {
        "name": "skill",
        "description": "Invoke a registered skill by name. Skills are prompt templates loaded from .claude/skills/. Returns the skill's resolved prompt to follow.",
        "input_schema": {
            "type": "object",
            "properties": {
                "skill_name": {
                    "type": "string",
                    "description": "The name of the skill to invoke",
                },
                "args": {
                    "type": "string",
                    "description": "Optional arguments to pass to the skill",
                },
            },
            "required": ["skill_name"],
        },
    },
    {
        "name": "web_fetch",
        "description": "Fetch a URL and return its content as text. For HTML pages, tags are stripped to return readable text. For JSON/text responses, content is returned directly.",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The URL to fetch"},
                "max_length": {
                    "type": "number",
                    "description": "Maximum content length in characters (default 50000)",
                },
            },
            "required": ["url"],
        },
    },
    {
        "name": "enter_plan_mode",
        "description": "Enter plan mode to switch to a read-only planning phase. In plan mode, you can only read files and write to the plan file.",
        "input_schema": {"type": "object", "properties": {}},
        "deferred": True,
    },
    {
        "name": "exit_plan_mode",
        "description": "Exit plan mode after you have finished writing your plan to the plan file.",
        "input_schema": {"type": "object", "properties": {}},
        "deferred": True,
    },
    {
        "name": "agent",
        "description": "Launch a sub-agent to handle a task autonomously. Sub-agents have isolated context and return their result. Types: 'explore' (read-only), 'plan' (read-only, structured planning), 'general' (full tools). Set run_in_background=true to launch asynchronously and retrieve the result later with the agent_result tool.",
        "input_schema": {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "Short (3-5 word) description of the sub-agent's task",
                },
                "prompt": {
                    "type": "string",
                    "description": "Detailed task instructions for the sub-agent",
                },
                "type": {
                    "type": "string",
                    "description": "Agent type. Built-ins: 'explore', 'plan', 'general'. Custom agents from .claude/agents/*.md are also accepted. Default: general",
                },
                "run_in_background": {
                    "type": "boolean",
                    "description": "If true, launch the sub-agent asynchronously and return immediately with a task_id. Use agent_result to retrieve the output. Default: false (blocks until completion).",
                },
                "timeout_sec": {
                    "type": "number",
                    "description": "Wall-clock timeout for the sub-agent in seconds. Default: 300.",
                },
            },
            "required": ["description", "prompt"],
        },
    },
    {
        "name": "agent_result",
        "description": "Retrieve the output of a background sub-agent launched via agent(run_in_background=true, ...). Returns the final text if the sub-agent has finished, or a 'still running' status message otherwise.",
        "input_schema": {
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "The task_id returned by the agent tool when the sub-agent was launched in background mode.",
                },
                "wait_sec": {
                    "type": "number",
                    "description": "Optional: block up to this many seconds waiting for completion (capped at 120). Default: 0 (non-blocking poll).",
                },
            },
            "required": ["task_id"],
        },
    },
    # ─── Tool search (deferred tool loader) ─────────────────────
    {
        "name": "tool_search",
        "description": "Search for available tools by name or keyword. Returns full schema definitions for matching deferred tools so you can use them.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Tool name or search keywords",
                },
            },
            "required": ["query"],
        },
    },
    # ─── Ask user question (interactive multiple-choice) ────────
    {
        "name": "ask_user_question",
        "description": (
            "Ask the user one or more multiple-choice questions and return "
            "their answers. Use this when you need specific input from the "
            "user to proceed (e.g. clarifying ambiguous requirements, "
            "choosing between implementation approaches, picking a library). "
            "Each question has 2-4 options; an 'Other' option for free-text "
            "input is appended automatically. Prefer this over open-ended "
            "text questions when the answer space is bounded. Do not use "
            "for yes/no confirmations of dangerous actions — permission "
            "prompts already handle those."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "questions": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 4,
                    "description": "Questions to ask the user (1-4 questions).",
                    "items": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "The complete question to ask. Should end with a question mark.",
                            },
                            "header": {
                                "type": "string",
                                "description": 'Short label for the question (e.g. "Library", "Approach").',
                            },
                            "options": {
                                "type": "array",
                                "minItems": 2,
                                "maxItems": 4,
                                "description": "2-4 distinct options. An 'Other' option is appended automatically.",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "label": {
                                            "type": "string",
                                            "description": "Short display text for this option (1-5 words).",
                                        },
                                        "description": {
                                            "type": "string",
                                            "description": "Optional explanation of what this option means.",
                                        },
                                    },
                                    "required": ["label"],
                                },
                            },
                            "multi_select": {
                                "type": "boolean",
                                "description": "Allow the user to select multiple options. Default: false.",
                            },
                        },
                        "required": ["question", "options"],
                    },
                },
            },
            "required": ["questions"],
        },
    },
]


# ─── New tools (Pydantic-defined schemas) ───────────────────
# Additions inspired by OpenHarness. Kept in a separate block so older
# hand-written defs above are untouched. The schemas are generated from
# Pydantic models in tool_models.py to eliminate schema drift.

# tool_definitions.extend(
#     [
#         # to_tool_schema(
#         #     name="todo_write",
#         #     description=(
#         #         "Maintain a persistent TODO checklist for the current session. "
#         #         "Each call replaces the full list. Use this proactively for "
#         #         "multi-step tasks so progress is visible. Exactly one task "
#         #         "should be in_progress at a time."
#         #     ),
#         #     model=TodoWriteInput,
#         # ),
#         to_tool_schema(
#             name="web_search",
#             description=(
#                 "Search the web via DuckDuckGo and return a list of result "
#                 "titles, URLs, and snippets. For fetching a specific known "
#                 "URL use web_fetch instead."
#             ),
#             model=WebSearchInput,
#         ),
#     ]
# )

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


# ─── Tool execution ─────────────────────────────────────────


def _read_file(inp: dict) -> str:
    try:
        path = Path(inp["file_path"])
        if _is_sensitive_path(str(path)):
            return (
                f"Error: Access denied — {inp['file_path']} matches a "
                f"sensitive-path pattern (credentials/keys). This block "
                f"cannot be overridden."
            )
        offset = inp.get("offset", 0)
        limit = inp.get("limit")

        # Pre-read size check — reject whole-file reads of oversized files.
        # When offset/limit is given, the caller has opted into a bounded
        # slice, so the file-size cap doesn't apply.
        if limit is None and offset == 0:
            try:
                file_size = path.stat().st_size
            except OSError as e:
                return f"Error reading file: {e}"
            if file_size > MAX_FILE_SIZE_BYTES:
                return (
                    f"Error: File too large ({file_size / 1024:.1f} KB > "
                    f"{MAX_FILE_SIZE_BYTES / 1024:.0f} KB). Use offset and limit "
                    f"parameters to read specific portions of the file, or use "
                    f"grep_search to locate specific content."
                )

        # Binary detection — sniff the first few KB for a NUL byte. Reading
        # a binary file as text returns garbled mojibake that wastes the
        # model's context; fail fast instead.
        try:
            with open(path, "rb") as f:
                head = f.read(_BINARY_SNIFF_BYTES)
        except OSError as e:
            return f"Error reading file: {e}"
        if b"\x00" in head:
            return (
                f"Error: {inp['file_path']} appears to be a binary file "
                f"(NUL byte in first {_BINARY_SNIFF_BYTES} bytes). "
                f"read_file only supports text."
            )

        content = path.read_text(errors="replace")
        lines = content.split("\n")
        total_lines = len(lines)

        # Apply offset/limit slicing
        start = max(0, offset)
        if limit is not None:
            selected = lines[start : start + limit]
        else:
            selected = lines[start:]

        # Number lines using absolute line numbers (1-indexed)
        numbered = "\n".join(
            f"{start + i + 1:4d} | {line}" for i, line in enumerate(selected)
        )

        # Add range header when a partial slice was requested
        if limit is not None or offset > 0:
            end = start + len(selected)
            header = (
                f"[Showing lines {start + 1}-{end} of {total_lines} total lines]\n\n"
            )
            return header + numbered
        return numbered
    except Exception as e:
        return f"Error reading file: {e}"


def _write_file(inp: dict) -> str:
    try:
        path = Path(inp["file_path"])
        if _is_sensitive_path(str(path)):
            return (
                f"Error: Write denied — {inp['file_path']} matches a "
                f"sensitive-path pattern (credentials/keys). This block "
                f"cannot be overridden."
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(inp["content"])
        _auto_update_memory_index(str(path))
        lines = inp["content"].split("\n")
        line_count = len(lines)
        preview = "\n".join(f"{i+1:4d} | {line}" for i, line in enumerate(lines[:30]))
        trunc = f"\n  ... ({line_count} lines total)" if line_count > 30 else ""
        return f"Successfully wrote to {inp['file_path']} ({line_count} lines)\n\n{preview}{trunc}"
    except Exception as e:
        return f"Error writing file: {e}"


def _auto_update_memory_index(file_path: str) -> None:
    try:
        mem_dir = str(get_memory_dir())
        if (
            file_path.startswith(mem_dir)
            and file_path.endswith(".md")
            and not file_path.endswith("MEMORY.md")
        ):
            mem_path = Path(mem_dir)
            lines = ["# Memory Index", ""]
            for f in sorted(mem_path.glob("*.md")):
                if f.name == "MEMORY.md":
                    continue
                try:
                    raw = f.read_text()
                    name_match = re.search(r"^name:\s*(.+)$", raw, re.MULTILINE)
                    type_match = re.search(r"^type:\s*(.+)$", raw, re.MULTILINE)
                    desc_match = re.search(r"^description:\s*(.+)$", raw, re.MULTILINE)
                    if name_match and type_match:
                        n = name_match.group(1).strip()
                        t = type_match.group(1).strip()
                        d = desc_match.group(1).strip() if desc_match else ""
                        lines.append(f"- **[{n}]({f.name})** ({t}) — {d}")
                except Exception:
                    pass
            (mem_path / "MEMORY.md").write_text("\n".join(lines))
    except Exception:
        pass


# ─── Edit helpers: quote normalization + diff ───────────────


def _normalize_quotes(s: str) -> str:
    s = re.sub("[\u2018\u2019\u2032]", "'", s)
    s = re.sub("[\u201c\u201d\u2033]", '"', s)
    return s


def _find_actual_string(file_content: str, search_string: str) -> str | None:
    if search_string in file_content:
        return search_string
    norm_search = _normalize_quotes(search_string)
    norm_file = _normalize_quotes(file_content)
    idx = norm_file.find(norm_search)
    if idx != -1:
        return file_content[idx : idx + len(search_string)]
    return None


def _generate_diff(old_content: str, old_string: str, new_string: str) -> str:
    before_change = old_content.split(old_string)[0]
    line_num = before_change.count("\n") + 1
    old_lines = old_string.split("\n")
    new_lines = new_string.split("\n")

    parts = [f"@@ -{line_num},{len(old_lines)} +{line_num},{len(new_lines)} @@"]
    for line in old_lines:
        parts.append(f"- {line}")
    for line in new_lines:
        parts.append(f"+ {line}")
    return "\n".join(parts)


def _edit_file(inp: dict) -> str:
    try:
        path = Path(inp["file_path"])
        if _is_sensitive_path(str(path)):
            return (
                f"Error: Edit denied — {inp['file_path']} matches a "
                f"sensitive-path pattern (credentials/keys). This block "
                f"cannot be overridden."
            )
        content = path.read_text()

        actual = _find_actual_string(content, inp["old_string"])
        if not actual:
            return f"Error: old_string '{inp['old_string']}' not found in {inp['file_path']}"

        count = content.count(actual)
        if count > 1:
            return f"Error: old_string '{inp['old_string']}' found {count} times in {inp['file_path']}. Must be unique."

        new_content = content.replace(actual, inp["new_string"], 1)
        path.write_text(new_content)

        diff = _generate_diff(content, actual, inp["new_string"])
        quote_note = (
            " (matched via quote normalization)" if actual != inp["old_string"] else ""
        )
        return f"Successfully edited {inp['file_path']}{quote_note}\n\n{diff}"
    except Exception as e:
        return f"Error editing file: {e}"


def _list_files(inp: dict) -> str:
    try:
        base = Path(inp.get("path") or ".")
        pattern = inp["pattern"]
        files = []
        for p in base.glob(pattern):
            if p.is_file():
                rel = str(p.relative_to(base) if base != Path(".") else p)
                # Skip node_modules and .git
                if "node_modules" in rel or ".git" in rel.split(os.sep):
                    continue
                files.append(rel)
                if len(files) >= 200:
                    break
        if not files:
            return "No files found matching the pattern."
        result = "\n".join(files[:200])
        if len(files) > 200:
            result += f"\n... and {len(files) - 200} more"
        return result
    except Exception as e:
        return f"Error listing files: {e}"


def _grep_search(inp: dict) -> str:
    pattern = inp["pattern"]
    path = inp.get("path") or "."
    include = inp.get("include")

    # Try system grep first (Linux/macOS)
    if not IS_WIN:
        try:
            args = ["grep", "--line-number", "--color=never", "-r"]
            if include:
                args.append(f"--include={include}")
            args.extend(["--", pattern, path])
            result = subprocess.run(args, capture_output=True, text=True, timeout=10)
            if result.returncode == 1:
                return "No matches found."
            if result.returncode == 0:
                lines = [line for line in result.stdout.split("\n") if line]
                output = "\n".join(lines[:100])
                if len(lines) > 100:
                    output += f"\n... and {len(lines) - 100} more matches"
                return output
            # Non-zero exit (not 1) — fall through to Python fallback
        except Exception:
            pass  # Fall through to Python fallback

    # Pure Python fallback (Windows, or system grep unavailable)
    return _grep_python(pattern, path, include)


def _grep_python(pattern: str, directory: str, include: str | None) -> str:
    regex = re.compile(pattern)
    include_pattern = include
    matches: list[str] = []

    def walk(d: str) -> None:
        if len(matches) >= 200:
            return
        try:
            entries = os.listdir(d)
        except Exception:
            return
        for name in entries:
            if name.startswith(".") or name == "node_modules":
                continue
            full = os.path.join(d, name)
            if os.path.isdir(full):
                walk(full)
                continue
            if include_pattern and not fnmatch.fnmatch(name, include_pattern):
                continue
            try:
                text = Path(full).read_text(errors="replace")
                for i, line in enumerate(text.split("\n")):
                    if regex.search(line):
                        matches.append(f"{full}:{i+1}:{line}")
                        if len(matches) >= 200:
                            return
            except Exception:
                pass

    walk(directory)
    if not matches:
        return "No matches found."
    output = "\n".join(matches[:100])
    if len(matches) > 100:
        output += f"\n... and {len(matches) - 100} more matches"
    return output


def _run_shell(inp: dict) -> str:
    try:
        command = inp["command"]
        preflight = _preflight_interactive_command(command)
        if preflight is not None:
            return preflight
        timeout_ms = inp.get("timeout", 30000)
        timeout_s = timeout_ms / 1000
        return _run_shell_impl(command, timeout_s, timeout_ms)
    except Exception as e:
        return f"Error: {e}"


# ─── Interactive-scaffold preflight (OpenHarness parity) ────
# The shell runs non-interactively (stdin is DEVNULL). Commands that expect
# TTY input (e.g. `npm create foo`, `create-next-app`) would otherwise hang
# until timeout. Fail fast with a hint to use non-interactive flags.

_SCAFFOLD_MARKERS: tuple[str, ...] = (
    "create-next-app",
    "npm create ",
    "pnpm create ",
    "yarn create ",
    "bun create ",
    "pnpm dlx ",
    "npm init ",
    "pnpm init ",
    "yarn init ",
    "bunx create-",
    "npx create-",
)

_NON_INTERACTIVE_MARKERS: tuple[str, ...] = (
    "--yes",
    " -y",
    "--skip-install",
    "--defaults",
    "--non-interactive",
    "--ci",
)


def _preflight_interactive_command(command: str) -> str | None:
    """Return an error message if the command looks like an interactive
    scaffolder without a non-interactive flag, else None."""
    lowered = command.lower()
    if not any(m in lowered for m in _SCAFFOLD_MARKERS):
        return None
    if any(m in lowered for m in _NON_INTERACTIVE_MARKERS):
        return None
    return (
        "Error: This command appears to require interactive input before it can "
        "continue. The shell tool is non-interactive, so it cannot answer "
        "installer/scaffold prompts live. Prefer non-interactive flags "
        "(for example --yes, -y, --skip-install, --defaults, --non-interactive), "
        "or run the scaffolding step once in an external terminal before asking "
        "the agent to continue."
    )


# ─── Persistent shell state ─────────────────────────────────
# Track cwd across run_shell calls so `cd` persists between invocations,
# matching Claude Code's BashTool behaviour. POSIX-only; on Windows the
# state stays at whatever the Python process cwd is.
_shell_cwd: str | None = None
_PWD_MARKER = "__OCS_SHELL_PWD_MARKER__"


def _get_shell_cwd() -> str:
    """Return the persistent shell cwd, re-initialising if it vanished."""
    global _shell_cwd
    if _shell_cwd is None or not os.path.isdir(_shell_cwd):
        _shell_cwd = os.getcwd()
    return _shell_cwd


def reset_shell_state() -> None:
    """Reset persistent shell state — exposed for tests."""
    global _shell_cwd
    _shell_cwd = None


def _extract_pwd(output: str) -> tuple[str | None, str]:
    """Pull the trailing pwd marker line out of the shell output.

    Returns (new_cwd_or_None, cleaned_output).
    """
    needle = "\n" + _PWD_MARKER
    idx = output.rfind(needle)
    if idx == -1:
        return None, output
    start = idx + len(needle)
    end = output.find("\n", start)
    if end == -1:
        return output[start:].strip() or None, output[:idx]
    cwd = output[start:end].strip() or None
    cleaned = output[:idx] + output[end + 1 :]
    return cwd, cleaned


def _kill_process_tree(proc: "subprocess.Popen[str]") -> None:
    """Kill proc and all its descendants — avoids orphaned child processes.

    POSIX: the child was spawned in a new session (setsid), so killpg on
    the process-group id takes out the whole tree. Windows: taskkill /T.
    """
    import signal

    if IS_WIN:
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True,
                timeout=5,
            )
        except Exception:
            pass
    else:
        try:
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, signal.SIGTERM)
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
        except (ProcessLookupError, PermissionError):
            pass
    # Belt-and-braces fallback
    try:
        proc.kill()
    except Exception:
        pass


def _run_shell_impl(command: str, timeout_s: float, timeout_ms: int) -> str:
    global _shell_cwd

    cwd = _get_shell_cwd()

    # On POSIX, wrap the command so we can (1) capture the new pwd after
    # any `cd`, and (2) still propagate the user command's exit code even
    # though we appended a printf. `__rc=$?` snapshots the exit code right
    # after the user command so printf doesn't clobber it.
    if IS_WIN:
        wrapped = command
        popen_kwargs: dict = {}
    else:
        wrapped = (
            f"{command}\n"
            f"__rc=$?\n"
            f'printf "\\n{_PWD_MARKER}%s\\n" "$(pwd)"\n'
            f"exit $__rc"
        )
        # start_new_session => child becomes a process-group leader, so we
        # can killpg the whole tree on timeout.
        popen_kwargs = {"start_new_session": True}

    proc = subprocess.Popen(
        wrapped,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=cwd,
        **popen_kwargs,
    )

    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        _kill_process_tree(proc)
        # Drain any buffered output so fds are closed
        try:
            proc.communicate(timeout=2)
        except Exception:
            pass
        return f"Command timed out after {timeout_ms}ms"

    # Parse the pwd marker (POSIX only) and update persistent state
    if not IS_WIN:
        new_cwd, stdout = _extract_pwd(stdout or "")
        if new_cwd and os.path.isdir(new_cwd):
            _shell_cwd = new_cwd

    returncode = proc.returncode
    output = stdout or ""
    if returncode != 0:
        stderr_part = f"\nStderr: {stderr}" if stderr else ""
        stdout_part = f"\nStdout: {output}" if output else ""
        return f"Command failed (exit code {returncode}){stdout_part}{stderr_part}"
    return output or "(no output)"


def _web_fetch(inp: dict) -> str:
    import urllib.error
    import urllib.request

    url = inp.get("url", "")
    max_length = inp.get("max_length", 50000)
    req = urllib.request.Request(url, headers={"User-Agent": "mini-claude/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            content_type = resp.headers.get("Content-Type", "")
            text = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        return f"HTTP error: {e.code} {e.reason}"
    except urllib.error.URLError as e:
        return f"Error fetching {url}: {e.reason}"
    except Exception as e:
        return f"Error fetching {url}: {e}"

    if "html" in content_type:
        text = re.sub(r"<script[\s\S]*?</script>", "", text, flags=re.IGNORECASE)
        text = re.sub(r"<style[\s\S]*?</style>", "", text, flags=re.IGNORECASE)
        text = re.sub(r"<[^>]*>", " ", text)
        text = text.replace("&nbsp;", " ").replace("&amp;", "&")
        text = text.replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"')
        text = re.sub(r"\s{2,}", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.strip()

    if len(text) > max_length:
        text = text[:max_length] + f"\n\n[... truncated at {max_length} characters]"

    return text or "(empty response)"


# ─── todo_write ────────────────────────────────────────────
# Persistent TODO list for the current session. Stored as a markdown
# checklist next to the session memory so the agent (and the user) can
# resume planning context across turns.


def _todo_file_path() -> Path:
    return get_memory_dir() / "TODO.md"


def _todo_write(inp: dict) -> str:
    try:
        # Validate via the Pydantic model so schema mismatches surface early.
        from .tool_models import TodoWriteInput

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


# ─── web_search (DuckDuckGo HTML endpoint) ─────────────────
# OCS previously only offered web_fetch, which requires a known URL. For
# discovery we hit DuckDuckGo's lightweight HTML endpoint. No API key,
# no external dependency beyond stdlib.


def _web_search(inp: dict) -> str:
    try:
        from .tool_models import WebSearchInput

        parsed = WebSearchInput.model_validate(inp)
    except Exception as e:
        return f"Error: invalid web_search arguments: {e}"

    import html as _html
    import urllib.error
    import urllib.parse
    import urllib.request

    url = "https://html.duckduckgo.com/html/?q=" + urllib.parse.quote_plus(parsed.query)
    req = urllib.request.Request(
        url,
        headers={
            # DDG's HTML endpoint rejects the default urllib UA.
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
            )
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        return f"HTTP error: {e.code} {e.reason}"
    except urllib.error.URLError as e:
        return f"Error fetching search results: {e.reason}"
    except Exception as e:
        return f"Error fetching search results: {e}"

    # DuckDuckGo HTML layout: each result is a <div class="result"> block
    # containing an <a class="result__a" href="..."> title and a
    # <a class="result__snippet"> or <div class="result__snippet"> body.
    result_re = re.compile(
        r'<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>'
        r'.*?class="result__snippet"[^>]*>(.*?)</a|'
        r'<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>'
        r'.*?class="result__snippet"[^>]*>(.*?)</div>',
        re.DOTALL,
    )

    def _clean(frag: str) -> str:
        frag = re.sub(r"<[^>]+>", "", frag)
        frag = _html.unescape(frag)
        return re.sub(r"\s+", " ", frag).strip()

    def _resolve_ddg_url(raw_url: str) -> str:
        # DDG wraps outbound URLs as //duckduckgo.com/l/?uddg=<encoded>.
        # Unwrap if present so the model sees the real target.
        if raw_url.startswith("//"):
            raw_url = "https:" + raw_url
        parsed_url = urllib.parse.urlparse(raw_url)
        if "duckduckgo.com" in parsed_url.netloc and parsed_url.path.startswith("/l/"):
            qs = urllib.parse.parse_qs(parsed_url.query)
            target = qs.get("uddg", [None])[0]
            if target:
                return urllib.parse.unquote(target)
        return raw_url

    items: list[dict] = []
    for m in result_re.finditer(body):
        href = m.group(1) or m.group(4)
        title = m.group(2) or m.group(5)
        snippet = m.group(3) or m.group(6)
        if not href or not title:
            continue
        items.append(
            {
                "url": _resolve_ddg_url(href),
                "title": _clean(title),
                "snippet": _clean(snippet or ""),
            }
        )
        if len(items) >= parsed.max_results:
            break

    if not items:
        return f"No search results found for: {parsed.query}"

    parts = [f"Results for: {parsed.query}", ""]
    for i, it in enumerate(items, 1):
        parts.append(f"{i}. {it['title']}")
        parts.append(f"   {it['url']}")
        if it["snippet"]:
            parts.append(f"   {it['snippet']}")
        parts.append("")
    return "\n".join(parts).rstrip()


# ─── Dangerous command patterns ─────────────────────────────

DANGEROUS_PATTERNS = [
    re.compile(r"\brm\s"),
    re.compile(r"\bgit\s+(push|reset|clean|checkout\s+\.)"),
    re.compile(r"\bsudo\b"),
    re.compile(r"\bmkfs\b"),
    re.compile(r"\bdd\s"),
    re.compile(r">\s*/dev/"),
    re.compile(r"\bkill\b"),
    re.compile(r"\bpkill\b"),
    re.compile(r"\breboot\b"),
    re.compile(r"\bshutdown\b"),
    re.compile(r"\bdel\s", re.IGNORECASE),
    re.compile(r"\brmdir\s", re.IGNORECASE),
    re.compile(r"\bformat\s", re.IGNORECASE),
    re.compile(r"\btaskkill\s", re.IGNORECASE),
    re.compile(r"\bRemove-Item\s", re.IGNORECASE),
    re.compile(r"\bStop-Process\s", re.IGNORECASE),
]


def is_dangerous(command: str) -> bool:
    return any(p.search(command) for p in DANGEROUS_PATTERNS)


# ─── Permission rules (.claude/settings.json) ───────────────


def _parse_rule(rule: str) -> dict:
    m = re.match(r"^([a-z_]+)\((.+)\)$", rule)
    if m:
        return {"tool": m.group(1), "pattern": m.group(2)}
    return {"tool": rule, "pattern": None}


def _load_settings(file_path: Path) -> dict | None:
    if not file_path.exists():
        return None
    try:
        return json.loads(file_path.read_text())
    except Exception:
        return None


_cached_rules: dict | None = None


def load_permission_rules() -> dict:
    global _cached_rules
    if _cached_rules is not None:
        return _cached_rules

    allow: list[dict] = []
    deny: list[dict] = []

    user_settings = _load_settings(user_settings_file())
    project_settings = _load_settings(project_settings_file())

    for settings in [user_settings, project_settings]:
        if not settings or "permissions" not in settings:
            continue
        perms = settings["permissions"]
        for r in perms.get("allow", []):
            allow.append(_parse_rule(r))
        for r in perms.get("deny", []):
            deny.append(_parse_rule(r))

    _cached_rules = {"allow": allow, "deny": deny}
    return _cached_rules


def _matches_rule(rule: dict, tool_name: str, inp: dict) -> bool:
    if rule["tool"] != tool_name:
        return False
    if rule["pattern"] is None:
        return True

    value = ""
    if tool_name == "run_shell":
        value = inp.get("command", "")
    elif "file_path" in inp:
        value = inp["file_path"]
    else:
        return True

    pattern = rule["pattern"]
    if pattern.endswith("*"):
        return value.startswith(pattern[:-1])
    return value == pattern


def _check_permission_rules(tool_name: str, inp: dict) -> str | None:
    rules = load_permission_rules()
    for rule in rules["deny"]:
        if _matches_rule(rule, tool_name, inp):
            return "deny"
    for rule in rules["allow"]:
        if _matches_rule(rule, tool_name, inp):
            return "allow"
    return None


def check_permission(
    tool_name: str,
    inp: dict,
    mode: str = "default",
    plan_file_path: str | None = None,
) -> dict:
    """Returns {"action": "allow"|"deny"|"confirm", "message": ...}"""
    if mode == "bypassPermissions":
        return {"action": "allow"}

    rule_result = _check_permission_rules(tool_name, inp)
    if rule_result == "deny":
        return {
            "action": "deny",
            "message": f"Denied by permission rule for {tool_name}",
        }
    if rule_result == "allow":
        return {"action": "allow"}

    if tool_name in READ_TOOLS:
        return {"action": "allow"}

    if mode == "plan":
        if tool_name in EDIT_TOOLS:
            file_path = inp.get("file_path") or inp.get("path")
            if plan_file_path and file_path == plan_file_path:
                return {"action": "allow"}
            return {"action": "deny", "message": f"Blocked in plan mode: {tool_name}"}
        if tool_name == "run_shell":
            return {"action": "deny", "message": "Shell commands blocked in plan mode"}

    if tool_name in ("enter_plan_mode", "exit_plan_mode"):
        return {"action": "allow"}

    if mode == "acceptEdits" and tool_name in EDIT_TOOLS:
        return {"action": "allow"}

    needs_confirm = False
    confirm_message = ""

    if tool_name == "run_shell" and is_dangerous(inp.get("command", "")):
        needs_confirm = True
        confirm_message = inp.get("command", "")
    elif tool_name == "write_file" and not Path(inp.get("file_path", "")).exists():
        needs_confirm = True
        confirm_message = f"write new file: {inp.get('file_path', '')}"
    elif tool_name == "edit_file" and not Path(inp.get("file_path", "")).exists():
        needs_confirm = True
        confirm_message = f"edit non-existent file: {inp.get('file_path', '')}"

    if needs_confirm:
        if mode == "dontAsk":
            return {
                "action": "deny",
                "message": f"Auto-denied (dontAsk mode): {confirm_message}",
            }
        return {"action": "confirm", "message": confirm_message}

    return {"action": "allow"}


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


# ─── Execute a tool call ────────────────────────────────────
# "agent" and "skill" tools are handled in agent.py to avoid circular deps.
#
# Dispatch order:
#   1. Cross-cutting pre-checks (read-before-edit mtime guard, read_file
#      fast path with mtime recording).
#   2. ``tool_search`` — stays hand-coded because it mutates the
#      module-global ``_activated_tools`` and reflects over
#      ``tool_definitions`` itself.
#   3. Default ``ToolRegistry`` lookup. Migrated tools (PR2) run through
#      their ``BaseTool.execute`` implementation.
#   4. Legacy handlers dict — fallback for the remaining string-returning
#      handlers (``todo_write``, ``web_search``).


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
    # model cannot currently reach (todo_write, web_search are not in
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

    # Registry-backed dispatch for PR2-migrated tools. The registry is
    # the source of truth for the 7 simple tools; legacy dict-based
    # handlers below remain the fallback for unmigrated ones.
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
        "web_search": _web_search,
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


def reset_permission_cache() -> None:
    global _cached_rules
    _cached_rules = None


# ─── Structured result (OpenHarness parity) ─────────────────
# `execute_tool` historically returned a plain string with an "Error: " prefix
# convention. That contract is fragile (callers must string-sniff and typos
# silently swallow errors). `ToolResult` gives callers a reliable is_error
# flag plus room for metadata. The legacy string API is preserved below as
# `execute_tool(...)` for backwards compatibility.


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


# ─── BaseTool subclasses + default registry (PR2) ───────────
# Phase-2 of the tool-system refactor: wrap each side-effect-free handler
# in a ``BaseTool`` subclass and register them with a ``ToolRegistry``.
# These subclasses are additive — ``execute_tool`` still owns dispatch
# for now, ``tool_definitions`` is still the source of truth for the
# Anthropic schema the agent sends to the model, and ``agent.py`` does
# not import from here. PR3 will route ``execute_tool`` through the
# registry and drop the legacy dict list.
#
# Only the seven simple tools land in this PR. ``agent`` / ``skill`` /
# ``enter_plan_mode`` / ``exit_plan_mode`` / ``tool_search`` have
# cross-module dependencies (parent-agent reference, activated-tool
# state) that are better addressed in PR3 together with the
# ``execute_tool`` shim.

from .tool_models import (  # noqa: E402
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
from .tools_base import BaseTool  # noqa: E402
from .tools_base import ToolRegistry  # noqa: E402
from .tools_base import ToolExecutionContext as _ToolExecutionContext  # noqa: E402
from .tools_base import ToolResult as _ToolResult  # noqa: E402


def _read_file_tool_description() -> str:
    # Kept as a function so the ``MAX_FILE_SIZE_BYTES`` value stays the
    # single source of truth for both the legacy dict and the BaseTool
    # subclass description.
    return (
        "Read the contents of a file. Returns the file content with line numbers. "
        f"Files larger than {MAX_FILE_SIZE_BYTES // 1024} KB must be read with "
        "offset and limit to select a specific range."
    )


class ReadFileTool(BaseTool):
    name = "read_file"
    description = _read_file_tool_description()
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
    description = (
        "Write content to a file. Creates the file if it doesn't exist, "
        "overwrites if it does.\n\n"
        "Path rules:\n"
        "- For deliverable files that belong to the user's project, write under the working directory.\n"
        "- For temporary/intermediate artifacts (working notes, captured outputs, throwaway scripts), "
        "write under the session scratchpad directory given in the system prompt (Temporary files section). "
        "Do NOT write temp files to `/tmp` or scatter them across arbitrary paths like `/tmp/foo.md`.\n"
        "- Do not create files unless necessary; prefer edit_file on an existing file when possible."
    )
    input_model = WriteFileInput

    async def execute(
        self, arguments: WriteFileInput, context: _ToolExecutionContext
    ) -> _ToolResult:
        del context
        raw = _write_file(arguments.model_dump())
        return _ToolResult(output=raw, is_error=_derive_is_error(raw))


class EditFileTool(BaseTool):
    name = "edit_file"
    description = (
        "Edit a file by replacing an exact string match with new content. "
        "The old_string must match exactly (including whitespace and indentation)."
    )
    input_model = EditFileInput

    async def execute(
        self, arguments: EditFileInput, context: _ToolExecutionContext
    ) -> _ToolResult:
        del context
        raw = _edit_file(arguments.model_dump())
        return _ToolResult(output=raw, is_error=_derive_is_error(raw))


class ListFilesTool(BaseTool):
    name = "list_files"
    description = "List files matching a glob pattern. Returns matching file paths."
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
    description = (
        "Search for a pattern in files. Returns matching lines with file "
        "paths and line numbers."
    )
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
    # Mirrors the legacy dict's multi-line description verbatim so the
    # Anthropic tool schema is unchanged when PR3 flips the source of
    # truth to the registry.
    description = (
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
    description = (
        "Fetch a URL and return its content as text. For HTML pages, tags "
        "are stripped to return readable text. For JSON/text responses, "
        "content is returned directly."
    )
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
    description = (
        "Search for available tools by name or keyword. Returns full schema "
        "definitions for matching deferred tools so you can use them."
    )
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
    description = (
        "Ask the user one or more multiple-choice questions and return "
        "their answers. Use this when you need specific input from the "
        "user to proceed."
    )
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
        from .ui import console

        answers: dict[str, str] = {}
        for q in arguments.questions:
            console.print()
            if q.header:
                console.print(f"  [cyan][{q.header}][/cyan]")
            console.print(f"  [bold white]{q.question}[/bold white]")
            labels = [opt.label for opt in q.options]
            for i, opt in enumerate(q.options, start=1):
                desc = f"[dim] — {opt.description}[/dim]" if opt.description else ""
                console.print(f"    [white]{i}) {opt.label}[/white]{desc}")
            other_idx = len(q.options) + 1
            console.print(
                f"    [white]{other_idx}) Other[/white][dim] — provide a free-form answer[/dim]"
            )

            if q.multi_select:
                prompt = f"  Choose (comma-separated, 1-{other_idx}): "
            else:
                prompt = f"  Choose (1-{other_idx}): "

            try:
                raw = input(prompt).strip()
            except EOFError:
                return _ToolResult(
                    output="Error: user aborted (EOF while answering questions).",
                    is_error=True,
                )

            selected = _parse_choice_input(raw, len(q.options), q.multi_select)
            if selected is None:
                return _ToolResult(
                    output=(
                        f"Error: invalid selection {raw!r} for question "
                        f"{q.question!r}."
                    ),
                    is_error=True,
                )

            picked: list[str] = []
            for idx in selected:
                if idx == other_idx:
                    try:
                        free = input("  Other — type your answer: ").strip()
                    except EOFError:
                        free = ""
                    picked.append(f"Other: {free}" if free else "Other")
                else:
                    picked.append(labels[idx - 1])
            answers[q.question] = ", ".join(picked)

        return _ToolResult(output=json.dumps({"answers": answers}, indent=2))


def _parse_choice_input(
    raw: str, num_options: int, multi_select: bool
) -> list[int] | None:
    """Parse a user-typed choice string into a list of 1-indexed option IDs.

    Accepts a single integer, or (when ``multi_select``) a comma-separated
    list. The trailing ``Other`` option is ``num_options + 1``. Returns
    ``None`` if any token is not a valid index in ``1..num_options+1``.
    """
    if not raw:
        return None
    tokens = [t.strip() for t in raw.split(",") if t.strip()]
    if not tokens:
        return None
    if not multi_select and len(tokens) > 1:
        return None
    picked: list[int] = []
    for t in tokens:
        if not t.isdigit():
            return None
        idx = int(t)
        if idx < 1 or idx > num_options + 1:
            return None
        picked.append(idx)
    # Dedupe while preserving order
    seen: set[int] = set()
    deduped = [x for x in picked if not (x in seen or seen.add(x))]
    return deduped


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
