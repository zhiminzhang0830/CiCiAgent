"""Terminal UI rendering — colored output, spinner, tool display."""

from __future__ import annotations

import sys
import threading
import time

from rich.console import Console

console = Console(highlight=False)

# ─── Basic output ──────────────────────────────────────────


def print_welcome() -> None:
    console.print(
        "\n  [bold cyan] coco [/bold cyan][dim] — A minimal coding agent[/dim]\n"
    )
    console.print("[dim]  Type your request, or 'exit' to quit.[/dim]")
    console.print(
        "[dim]  Commands: /clear /plan /cost /compact /memory /skills[/dim]\n"
    )


def print_user_prompt() -> None:
    console.print("\n[bold green]> [/bold green]", end="")


def print_assistant_text(text: str) -> None:
    sys.stdout.write(text)
    sys.stdout.flush()


def print_tool_call(name: str, inp: dict) -> None:
    icon = _get_tool_icon(name)
    summary = _get_tool_summary(name, inp)
    console.print(f"\n  [yellow]{icon} {name}[/yellow][dim] {summary}[/dim]")


def print_tool_result(name: str, result: str) -> None:
    if (name in ("edit_file", "write_file")) and not result.startswith("Error"):
        _print_file_change_result(name, result)
        return
    max_len = 500
    truncated = result
    if len(result) > max_len:
        truncated = result[:max_len] + f"\n  ... ({len(result)} chars total)"
    lines = "\n".join("  " + line for line in truncated.split("\n"))
    console.print(f"[dim]{lines}[/dim]")


def _print_file_change_result(_name: str, result: str) -> None:
    lines = result.split("\n")
    console.print(f"[dim]  {lines[0]}[/dim]")

    max_display = 40
    content_lines = lines[1:]
    display_lines = content_lines[:max_display]

    for line in display_lines:
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


def print_error(msg: str) -> None:
    console.print(f"\n  [red]Error: {msg}[/red]")


def print_confirmation(command: str) -> None:
    console.print(f"\n  [yellow]⚠ Dangerous command:[/yellow] [white]{command}[/white]")


def print_divider() -> None:
    console.print(f"\n[dim]  {'─' * 50}[/dim]")


def print_cost(input_tokens: int, output_tokens: int) -> None:
    cost_in = (input_tokens / 1_000_000) * 3
    cost_out = (output_tokens / 1_000_000) * 15
    total = cost_in + cost_out
    console.print(
        f"\n[dim]  Tokens: {input_tokens} in / {output_tokens} out (~${total:.4f})[/dim]"
    )


def print_retry(attempt: int, max_retries: int, reason: str) -> None:
    console.print(f"\n  [yellow]↻ Retry {attempt}/{max_retries}: {reason}[/yellow]")


def print_info(msg: str) -> None:
    console.print(f"\n  [cyan]ℹ {msg}[/cyan]")


# ─── Spinner ──────────────────────────────────────────────

SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

_spinner_thread: threading.Thread | None = None
_spinner_stop = threading.Event()


def start_spinner(label: str = "Thinking") -> None:
    global _spinner_thread
    if _spinner_thread is not None:
        return
    _spinner_stop.clear()

    def _run() -> None:
        frame = 0
        sys.stdout.write(f"\n  {SPINNER_FRAMES[0]} {label}...")
        sys.stdout.flush()
        while not _spinner_stop.is_set():
            time.sleep(0.08)
            frame = (frame + 1) % len(SPINNER_FRAMES)
            sys.stdout.write(f"\r  {SPINNER_FRAMES[frame]} {label}...")
            sys.stdout.flush()

    _spinner_thread = threading.Thread(target=_run, daemon=True)
    _spinner_thread.start()


def stop_spinner() -> None:
    global _spinner_thread
    if _spinner_thread is None:
        return
    _spinner_stop.set()
    _spinner_thread.join(timeout=1)
    _spinner_thread = None
    sys.stdout.write("\r\033[K")
    sys.stdout.flush()


# ─── Plan approval display ──────────────────────────────────


def print_plan_for_approval(plan_content: str) -> None:
    console.print("\n  [cyan]━━━ Plan for Approval ━━━[/cyan]")
    lines = plan_content.split("\n")
    max_lines = 60
    for line in lines[:max_lines]:
        console.print(f"  [white]{line}[/white]")
    if len(lines) > max_lines:
        console.print(f"[dim]  ... ({len(lines) - max_lines} more lines)[/dim]")
    console.print("  [cyan]━━━━━━━━━━━━━━━━━━━━━━━━[/cyan]\n")


def print_plan_approval_options() -> None:
    console.print("  [yellow]Choose an option:[/yellow]")
    console.print(
        "    [white]1) Yes, clear context and execute[/white][dim] — fresh start with auto-accept edits[/dim]"
    )
    console.print(
        "    [white]2) Yes, and execute[/white][dim] — keep context, auto-accept edits[/dim]"
    )
    console.print(
        "    [white]3) Yes, manually approve edits[/white][dim] — keep context, confirm each edit[/dim]"
    )
    console.print(
        "    [white]4) No, keep planning[/white][dim] — provide feedback to revise[/dim]"
    )


# ─── Sub-agent display ──────────────────────────────────────


def print_sub_agent_start(agent_type: str, description: str) -> None:
    console.print(f"\n  [magenta]┌─ Sub-agent [{agent_type}]: {description}[/magenta]")


def print_sub_agent_end(agent_type: str, _description: str) -> None:
    console.print(f"  [magenta]└─ Sub-agent [{agent_type}] completed[/magenta]")


# ─── Tool icons and summaries ───────────────────────────────

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
    "web_search": "🌐",
    "enter_plan_mode": "📝",
    "exit_plan_mode": "✅",
    "tool_search": "🧰",
    "ask_user_question": "❓",
}


def _get_tool_icon(name: str) -> str:
    return _TOOL_ICONS.get(name, "🔨")


def _get_tool_summary(name: str, inp: dict) -> str:
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
    if name == "write_file":
        return inp.get("file_path", "")
    if name == "edit_file":
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
    if name == "web_fetch":
        return inp.get("url", "")
    if name == "web_search":
        return inp.get("query", "")
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
    return ""
