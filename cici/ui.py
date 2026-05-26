"""Terminal UI rendering — pluggable backend.

The package's user-visible output (assistant text, tool calls, errors,
cost, sub-agent banners, ``ask_user_question`` prompts) all flows
through this module. Two backends are provided:

* :class:`RichBackend` (default) — streaming rich/console output. Used
  by one-shot CLI invocations (``cici "<prompt>"``).
* :class:`TextualBackend` — installed by :mod:`cici.tui` when the
  interactive TUI starts; routes every call into the running Textual
  App.

Other modules should call the top-level ``print_*`` / ``ask_questions``
helpers below — they always delegate to the active backend.
"""

from __future__ import annotations

import asyncio
import sys
import threading
import time
from typing import Protocol

from rich.console import Console

console = Console(highlight=False)


# ─── Backend protocol ──────────────────────────────────────


class UIBackend(Protocol):
    def print_assistant_text(self, text: str) -> None: ...
    def print_tool_call(self, name: str, inp: dict) -> None: ...
    def print_tool_result(self, name: str, result: str) -> None: ...
    def print_error(self, msg: str) -> None: ...
    def print_confirmation(self, command: str) -> None: ...
    def print_divider(self) -> None: ...
    def print_cost(self, input_tokens: int, output_tokens: int) -> None: ...
    def print_retry(self, attempt: int, max_retries: int, reason: str) -> None: ...
    def print_info(self, msg: str) -> None: ...
    def start_spinner(self, label: str = "Thinking") -> None: ...
    def stop_spinner(self) -> None: ...
    def print_sub_agent_start(self, agent_type: str, description: str) -> None: ...
    def print_sub_agent_end(self, agent_type: str, description: str) -> None: ...
    async def ask_questions(self, questions: list[dict]) -> dict[str, str]: ...


# ─── Tool icons / summaries (shared by all backends) ───────

_TOOL_ICONS = {
    "read_file": "📖",
    "write_file": "✏️",
    "edit_file": "🔧",
    "list_files": "📁",
    "grep_search": "🔍",
    "run_shell": "💻",
    "skill": "⚡",
    "agent": "🤖",
    "agent_result": "📬",
    "web_fetch": "🌐",
    "enter_plan_mode": "📝",
    "exit_plan_mode": "✅",
    "tool_search": "🧰",
    "ask_user_question": "❓",
    "todo_write": "✅",
}


def get_tool_icon(name: str) -> str:
    return _TOOL_ICONS.get(name, "🔨")


def get_tool_summary(name: str, inp: dict) -> str:
    if name == "read_file":
        path = inp.get("file_path", "")
        offset = inp.get("offset")
        limit = inp.get("limit")
        if offset is not None or limit is not None:
            start = (offset or 0) + 1
            if limit is not None:
                return f"{path} (lines {start}-{start + limit - 1})"
            return f"{path} (from line {start})"
        return path
    if name in ("write_file", "edit_file"):
        return inp.get("file_path", "")
    if name == "list_files":
        pattern = inp.get("pattern", "")
        path = inp.get("path", "")
        return f"{pattern} in {path}" if path else pattern
    if name == "grep_search":
        pattern = inp.get("pattern", "")
        path = inp.get("path", ".")
        include = inp.get("include")
        suffix = f" ({include})" if include else ""
        return f'"{pattern}" in {path}{suffix}'
    if name == "run_shell":
        return inp.get("command", "")
    if name == "skill":
        skill_name = inp.get("skill_name", "")
        args = inp.get("args", "")
        return f"{skill_name} {args}".strip()
    if name == "agent":
        bg = " (background)" if inp.get("run_in_background") else ""
        return f'[{inp.get("type", "general")}] {inp.get("description", "")}{bg}'
    if name == "agent_result":
        wait = inp.get("wait_sec")
        suffix = f" (wait {wait}s)" if wait else ""
        return f'{inp.get("task_id", "")}{suffix}'
    if name in ("web_fetch",):
        return inp.get("url", "")
    if name == "enter_plan_mode":
        return "entering plan mode"
    if name == "exit_plan_mode":
        return "exiting plan mode"
    if name == "tool_search":
        return inp.get("query", "")
    if name == "ask_user_question":
        qs = inp.get("questions") or []
        n = len(qs)
        first = qs[0].get("question", "") if n else ""
        suffix = f" (+{n - 1} more)" if n > 1 else ""
        return f"{first}{suffix}"
    if name == "todo_write":
        todos = inp.get("todos") or []
        return f"{len(todos)} item(s)"
    return ""


# ─── Rich backend (default) ────────────────────────────────


class RichBackend:
    """Streaming terminal output via :mod:`rich`. Original behaviour."""

    SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self) -> None:
        self._spinner_thread: threading.Thread | None = None
        self._spinner_stop = threading.Event()

    # — basic output —

    def print_assistant_text(self, text: str) -> None:
        sys.stdout.write(text)
        sys.stdout.flush()

    def print_tool_call(self, name: str, inp: dict) -> None:
        icon = get_tool_icon(name)
        summary = get_tool_summary(name, inp)
        console.print(f"\n  [yellow]{icon} {name}[/yellow][dim] {summary}[/dim]")

    def print_tool_result(self, name: str, result: str) -> None:
        if name in ("edit_file", "write_file") and not result.startswith("Error"):
            self._print_file_change_result(result)
            return
        max_len = 500
        truncated = result
        if len(result) > max_len:
            truncated = result[:max_len] + f"\n  ... ({len(result)} chars total)"
        lines = "\n".join("  " + line for line in truncated.split("\n"))
        console.print(f"[dim]{lines}[/dim]")

    def _print_file_change_result(self, result: str) -> None:
        lines = result.split("\n")
        console.print(f"[dim]  {lines[0]}[/dim]")
        max_display = 40
        content_lines = lines[1:]
        for line in content_lines[:max_display]:
            if not line.strip():
                continue
            if line.startswith("@@"):
                console.print(f"[cyan]  {line}[/cyan]")
            elif line.startswith("- "):
                console.print(f"[red]  {line}[/red]")
            elif line.startswith("+ "):
                console.print(f"[green]  {line}[/green]")
            else:
                console.print(f"[dim]  {line}[/dim]")
        if len(content_lines) > max_display:
            console.print(
                f"[dim]  ... ({len(content_lines) - max_display} more lines)[/dim]"
            )

    def print_error(self, msg: str) -> None:
        console.print(f"\n  [red]Error: {msg}[/red]")

    def print_confirmation(self, command: str) -> None:
        console.print(
            f"\n  [yellow]⚠ Dangerous command:[/yellow] [white]{command}[/white]"
        )

    def print_divider(self) -> None:
        console.print(f"\n[dim]  {'─' * 50}[/dim]")

    def print_cost(self, input_tokens: int, output_tokens: int) -> None:
        cost_in = (input_tokens / 1_000_000) * 3
        cost_out = (output_tokens / 1_000_000) * 15
        total = cost_in + cost_out
        console.print(
            f"\n[dim]  Tokens: {input_tokens} in / {output_tokens} out (~${total:.4f})[/dim]"
        )

    def print_retry(self, attempt: int, max_retries: int, reason: str) -> None:
        console.print(f"\n  [yellow]↻ Retry {attempt}/{max_retries}: {reason}[/yellow]")

    def print_info(self, msg: str) -> None:
        console.print(f"\n  [cyan]ℹ {msg}[/cyan]")

    # — spinner —

    def start_spinner(self, label: str = "Thinking") -> None:
        if self._spinner_thread is not None:
            return
        self._spinner_stop.clear()

        def _run() -> None:
            frame = 0
            sys.stdout.write(f"\n  {self.SPINNER_FRAMES[0]} {label}...")
            sys.stdout.flush()
            while not self._spinner_stop.is_set():
                time.sleep(0.08)
                frame = (frame + 1) % len(self.SPINNER_FRAMES)
                sys.stdout.write(f"\r  {self.SPINNER_FRAMES[frame]} {label}...")
                sys.stdout.flush()

        self._spinner_thread = threading.Thread(target=_run, daemon=True)
        self._spinner_thread.start()

    def stop_spinner(self) -> None:
        if self._spinner_thread is None:
            return
        self._spinner_stop.set()
        self._spinner_thread.join(timeout=1)
        self._spinner_thread = None
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()

    # — sub-agent —

    def print_sub_agent_start(self, agent_type: str, description: str) -> None:
        console.print(
            f"\n  [magenta]┌─ Sub-agent [{agent_type}]: {description}[/magenta]"
        )

    def print_sub_agent_end(self, agent_type: str, description: str) -> None:
        del description
        console.print(f"  [magenta]└─ Sub-agent [{agent_type}] completed[/magenta]")

    # — questions —

    async def ask_questions(self, questions: list[dict]) -> dict[str, str]:
        """Render ``ask_user_question`` prompts using rich + ``input()``.

        Mirrors the legacy AskUserQuestionTool flow. Runs blocking input
        in a thread so callers can ``await`` it without freezing the loop.
        """
        return await asyncio.to_thread(self._ask_questions_blocking, questions)

    def _ask_questions_blocking(self, questions: list[dict]) -> dict[str, str]:
        answers: dict[str, str] = {}
        for q in questions:
            console.print()
            if q.get("header"):
                console.print(f"  [cyan][{q['header']}][/cyan]")
            console.print(f"  [bold white]{q['question']}[/bold white]")
            opts = q.get("options") or []
            labels = [opt.get("label", "") for opt in opts]
            for i, opt in enumerate(opts, start=1):
                desc = (
                    f"[dim] — {opt['description']}[/dim]" if opt.get("description") else ""
                )
                console.print(f"    [white]{i}) {opt.get('label', '')}[/white]{desc}")
            other_idx = len(opts) + 1
            console.print(
                f"    [white]{other_idx}) Other[/white][dim] — provide a free-form answer[/dim]"
            )
            multi = bool(q.get("multi_select"))
            prompt = (
                f"  Choose (comma-separated, 1-{other_idx}): "
                if multi
                else f"  Choose (1-{other_idx}): "
            )
            try:
                raw = input(prompt).strip()
            except EOFError:
                answers[q["question"]] = ""
                continue
            picked = self._parse_choice(raw, len(opts), multi)
            if picked is None:
                answers[q["question"]] = f"Invalid: {raw}"
                continue
            chosen: list[str] = []
            for idx in picked:
                if idx == other_idx:
                    try:
                        free = input("  Other — type your answer: ").strip()
                    except EOFError:
                        free = ""
                    chosen.append(f"Other: {free}" if free else "Other")
                else:
                    chosen.append(labels[idx - 1])
            answers[q["question"]] = ", ".join(chosen)
        return answers

    @staticmethod
    def _parse_choice(raw: str, n: int, multi: bool) -> list[int] | None:
        if not raw:
            return None
        toks = [t.strip() for t in raw.split(",") if t.strip()]
        if not toks or (not multi and len(toks) > 1):
            return None
        out: list[int] = []
        for t in toks:
            if not t.isdigit():
                return None
            i = int(t)
            if i < 1 or i > n + 1:
                return None
            out.append(i)
        seen: set[int] = set()
        return [x for x in out if not (x in seen or seen.add(x))]


# ─── Active backend + facade ───────────────────────────────

_backend: UIBackend = RichBackend()


def set_backend(backend: UIBackend) -> None:
    """Install a different UI backend (e.g. the Textual TUI)."""
    global _backend
    _backend = backend


def get_backend() -> UIBackend:
    return _backend


# Top-level facade — every external caller goes through these.

def print_assistant_text(text: str) -> None:
    _backend.print_assistant_text(text)


def print_tool_call(name: str, inp: dict) -> None:
    _backend.print_tool_call(name, inp)


def print_tool_result(name: str, result: str) -> None:
    _backend.print_tool_result(name, result)


def print_error(msg: str) -> None:
    _backend.print_error(msg)


def print_confirmation(command: str) -> None:
    _backend.print_confirmation(command)


def print_divider() -> None:
    _backend.print_divider()


def print_cost(input_tokens: int, output_tokens: int) -> None:
    _backend.print_cost(input_tokens, output_tokens)


def print_retry(attempt: int, max_retries: int, reason: str) -> None:
    _backend.print_retry(attempt, max_retries, reason)


def print_info(msg: str) -> None:
    _backend.print_info(msg)


def start_spinner(label: str = "Thinking") -> None:
    _backend.start_spinner(label)


def stop_spinner() -> None:
    _backend.stop_spinner()


def print_sub_agent_start(agent_type: str, description: str) -> None:
    _backend.print_sub_agent_start(agent_type, description)


def print_sub_agent_end(agent_type: str, description: str) -> None:
    _backend.print_sub_agent_end(agent_type, description)


async def ask_questions(questions: list[dict]) -> dict[str, str]:
    """Surface ``ask_user_question`` prompts via the active backend."""
    return await _backend.ask_questions(questions)
