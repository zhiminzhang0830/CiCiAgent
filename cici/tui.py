"""Textual TUI mode for cici.

Three-pane layout:

* Left   — directory tree of the cwd
* Middle — chat history (RichLog) + input + status bar
* Right  — model/cost/turns + collapsible tool-call log

Activated by ``cici --tui``. Bridges into the Agent through a
:class:`TextualBackend` that implements :class:`cici.ui.UIBackend`.
Agent callbacks (confirm, plan approval, ask_user_question) are
surfaced as modal screens that block the agent loop on an
``asyncio.Future``.

This module is imported lazily from ``__main__`` so the ``textual``
dependency stays optional.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Checkbox,
    DirectoryTree,
    Footer,
    Header,
    Input,
    Label,
    LoadingIndicator,
    RadioButton,
    RadioSet,
    RichLog,
    Static,
    TextArea,
    Tree,
)

from .agent import Agent
from .memory import list_memories
from .skills import (
    discover_skills,
    execute_skill,
    get_skill_by_name,
    resolve_skill_prompt,
)
from .ui import get_tool_icon, get_tool_summary, set_backend


# ─── Modals ─────────────────────────────────────────────────


class ConfirmModal(ModalScreen[bool]):
    """y/n permission modal for dangerous tool calls."""

    CSS = """
    ConfirmModal { align: center middle; }
    ConfirmModal > Vertical {
        width: 70%;
        max-width: 100;
        height: auto;
        padding: 1 2;
        background: $surface;
        border: thick $warning;
    }
    ConfirmModal #title { color: $warning; text-style: bold; padding-bottom: 1; }
    ConfirmModal #msg { padding-bottom: 1; }
    ConfirmModal Horizontal { height: auto; align: center middle; }
    ConfirmModal Button { margin: 0 1; }
    """

    BINDINGS = [
        Binding("y", "approve", "Allow"),
        Binding("n,escape", "deny", "Deny"),
    ]

    def __init__(self, message: str) -> None:
        super().__init__()
        self._message = message

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("⚠  Permission required", id="title")
            yield Static(self._message, id="msg")
            with Horizontal():
                yield Button("Allow (y)", variant="success", id="allow")
                yield Button("Deny (n)", variant="error", id="deny")

    def action_approve(self) -> None:
        self.dismiss(True)

    def action_deny(self) -> None:
        self.dismiss(False)

    @on(Button.Pressed, "#allow")
    def _allow(self) -> None:
        self.dismiss(True)

    @on(Button.Pressed, "#deny")
    def _deny(self) -> None:
        self.dismiss(False)


class PlanApprovalModal(ModalScreen[dict]):
    """Plan approval modal with the 4 standard choices + feedback."""

    CSS = """
    PlanApprovalModal { align: center middle; }
    PlanApprovalModal > Vertical {
        width: 90%;
        height: 80%;
        padding: 1 2;
        background: $surface;
        border: thick $primary;
    }
    PlanApprovalModal #title { color: $primary; text-style: bold; padding-bottom: 1; }
    PlanApprovalModal #plan { height: 1fr; border: round $secondary; padding: 0 1; }
    PlanApprovalModal #feedback { display: none; height: 5; margin-top: 1; }
    PlanApprovalModal Horizontal { height: auto; align: center middle; padding-top: 1; }
    PlanApprovalModal Button { margin: 0 1; }
    """

    def __init__(self, plan_content: str) -> None:
        super().__init__()
        self._plan = plan_content

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("━━━ Plan for Approval ━━━", id="title")
            log = RichLog(id="plan", wrap=True, markup=False)
            yield log
            yield TextArea(id="feedback")
            with Horizontal():
                yield Button("1) Clear & execute", id="clear")
                yield Button("2) Execute", id="exec", variant="success")
                yield Button("3) Manual approve", id="manual")
                yield Button("4) Keep planning", id="keep", variant="warning")

    def on_mount(self) -> None:
        log = self.query_one("#plan", RichLog)
        for line in self._plan.split("\n"):
            log.write(line)

    @on(Button.Pressed, "#clear")
    def _clear(self) -> None:
        self.dismiss({"choice": "clear-and-execute"})

    @on(Button.Pressed, "#exec")
    def _exec(self) -> None:
        self.dismiss({"choice": "execute"})

    @on(Button.Pressed, "#manual")
    def _manual(self) -> None:
        self.dismiss({"choice": "manual-execute"})

    @on(Button.Pressed, "#keep")
    def _keep(self) -> None:
        feedback = self.query_one("#feedback", TextArea).text.strip()
        self.dismiss({"choice": "keep-planning", "feedback": feedback or None})


class QuestionsModal(ModalScreen[dict]):
    """Render one or more multiple-choice questions as a form.

    Single-choice questions render as a :class:`RadioSet`; multi-choice
    questions render as a vertical list of :class:`Checkbox` widgets.
    Each question gets an ``Other`` option backed by a free-text
    :class:`Input` that shows itself when ``Other`` is selected.
    """

    CSS = """
    QuestionsModal { align: center middle; }
    QuestionsModal > VerticalScroll {
        width: 80%;
        max-height: 80%;
        padding: 1 2;
        background: $surface;
        border: thick $accent;
    }
    QuestionsModal .qheader { color: $accent; text-style: bold; padding-top: 1; }
    QuestionsModal .qtext { padding-bottom: 1; }
    QuestionsModal RadioSet { padding-bottom: 1; }
    QuestionsModal .multibox { padding-bottom: 1; }
    QuestionsModal Input { display: none; margin-bottom: 1; }
    QuestionsModal #buttons { height: auto; align: center middle; padding-top: 1; }
    QuestionsModal Button { margin: 0 1; }
    """

    def __init__(self, questions: list[dict]) -> None:
        super().__init__()
        self._questions = questions

    def compose(self) -> ComposeResult:
        with VerticalScroll():
            for i, q in enumerate(self._questions):
                if q.get("header"):
                    yield Static(f"[{q['header']}]", classes="qheader")
                yield Static(q.get("question", ""), classes="qtext")
                if q.get("multi_select"):
                    yield Vertical(id=f"rs-{i}", classes="multibox")
                else:
                    yield RadioSet(id=f"rs-{i}")
                yield Input(id=f"in-{i}", placeholder="Other — type here")
            with Horizontal(id="buttons"):
                yield Button("Submit", variant="success", id="submit")
                yield Button("Cancel", variant="error", id="cancel")

    def on_mount(self) -> None:
        for i, q in enumerate(self._questions):
            opts = q.get("options") or []
            container = self.query_one(f"#rs-{i}")
            if q.get("multi_select"):
                for j, opt in enumerate(opts):
                    desc = f" — {opt['description']}" if opt.get("description") else ""
                    container.mount(
                        Checkbox(
                            f"{opt.get('label', '')}{desc}", id=f"opt-{i}-{j}"
                        )
                    )
                container.mount(
                    Checkbox("Other (free text)", id=f"opt-{i}-other")
                )
            else:
                for j, opt in enumerate(opts):
                    desc = f" — {opt['description']}" if opt.get("description") else ""
                    container.mount(
                        RadioButton(
                            f"{opt.get('label', '')}{desc}", id=f"opt-{i}-{j}"
                        )
                    )
                container.mount(
                    RadioButton("Other (free text)", id=f"opt-{i}-other")
                )

    @on(RadioSet.Changed)
    def _on_radio(self, event: RadioSet.Changed) -> None:
        rs_id = event.radio_set.id or ""
        idx = rs_id.split("-", 1)[1]
        free_input = self.query_one(f"#in-{idx}", Input)
        if event.pressed.id and event.pressed.id.endswith("-other"):
            free_input.display = True
            free_input.focus()
        else:
            free_input.display = False

    @on(Checkbox.Changed)
    def _on_checkbox(self, event: Checkbox.Changed) -> None:
        cb_id = event.checkbox.id or ""
        if not cb_id.startswith("opt-"):
            return
        # opt-{i}-other or opt-{i}-{j}
        parts = cb_id.split("-")
        if len(parts) >= 3 and parts[2] == "other":
            idx = parts[1]
            free_input = self.query_one(f"#in-{idx}", Input)
            free_input.display = bool(event.value)
            if event.value:
                free_input.focus()

    @on(Button.Pressed, "#submit")
    def _submit(self) -> None:
        answers: dict[str, str] = {}
        for i, q in enumerate(self._questions):
            opts = q.get("options") or []
            qtext = q.get("question", "")
            free_input = self.query_one(f"#in-{i}", Input)
            if q.get("multi_select"):
                container = self.query_one(f"#rs-{i}")
                picked: list[str] = []
                for cb in container.query(Checkbox):
                    if not cb.value:
                        continue
                    bid = cb.id or ""
                    if bid.endswith("-other"):
                        free = free_input.value.strip()
                        picked.append(f"Other: {free}" if free else "Other")
                    else:
                        j = int(bid.rsplit("-", 1)[1])
                        if 0 <= j < len(opts):
                            picked.append(opts[j].get("label", ""))
                answers[qtext] = ", ".join(picked)
            else:
                rs = self.query_one(f"#rs-{i}", RadioSet)
                if rs.pressed_button is None:
                    answers[qtext] = ""
                    continue
                bid = rs.pressed_button.id or ""
                if bid.endswith("-other"):
                    free = free_input.value.strip()
                    answers[qtext] = f"Other: {free}" if free else "Other"
                else:
                    j = int(bid.rsplit("-", 1)[1])
                    answers[qtext] = (
                        opts[j].get("label", "") if 0 <= j < len(opts) else ""
                    )
        self.dismiss({"answers": answers})

    @on(Button.Pressed, "#cancel")
    def _cancel(self) -> None:
        self.dismiss({"cancelled": True})


# ─── App ────────────────────────────────────────────────────


class CiciApp(App):
    """Three-pane Textual UI for cici."""

    CSS = """
    Screen { layout: vertical; }
    #body { layout: horizontal; height: 1fr; }
    #left  { width: 28; border-right: tall $primary; }
    #mid   { width: 1fr; }
    #right { width: 36; border-left: tall $primary; }

    #chat { height: 1fr; border: none; padding: 0 1; }
    #stream { height: auto; max-height: 20; min-height: 1; padding: 0 1; color: $text; overflow-y: auto; }
    #stream.empty { display: none; }
    #input-row { height: 3; }
    #input { margin: 0 1; }
    #status-bar { height: 1; padding: 0 1; background: $boost; color: $text; }

    #status-panel { height: auto; padding: 1; border-bottom: tall $primary; }
    #tools-log   { height: 1fr; padding: 0 1; }

    #spinner { display: none; height: 3; padding: 1; }
    .spinner-on { display: block !important; }

    DirectoryTree { padding: 0 1; }
    """

    BINDINGS = [
        Binding("ctrl+c", "interrupt", "Interrupt", priority=True),
        Binding("ctrl+d", "quit", "Quit", priority=True),
        Binding("ctrl+l", "clear_chat", "Clear chat"),
    ]

    def __init__(self, agent: Agent, *, model: str, permission_mode: str) -> None:
        super().__init__()
        self._agent = agent
        self._model = model
        self._permission_mode = permission_mode
        self._cwd = Path.cwd()
        self._busy = False
        self._tool_idx = 0
        # Tracks consecutive Ctrl+C presses while idle (for "press again to quit").
        self._idle_interrupts = 0
        # Streaming state: in-progress assistant text waits in a Static
        # widget so users see chars arrive smoothly. Completed lines are
        # committed to the RichLog history above it.
        self._stream_buffer: str = ""
        # Token totals (cumulative)
        self._total_in = 0
        self._total_out = 0
        self._turn_count = 0

    # — layout —

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="body"):
            with Vertical(id="left"):
                yield Label(f"📂 {self._cwd.name}", classes="qheader")
                yield DirectoryTree(str(self._cwd))
            with Vertical(id="mid"):
                yield RichLog(id="chat", wrap=True, markup=True, highlight=False)
                yield Static("", id="stream", classes="empty")
                yield LoadingIndicator(id="spinner")
                with Horizontal(id="input-row"):
                    yield Input(
                        id="input",
                        placeholder="Ask cici anything…  (commands: /clear /plan /cost /compact /memory /skills)",
                        select_on_focus=False,
                    )
                yield Static("", id="status-bar")
            with Vertical(id="right"):
                yield Static("", id="status-panel")
                yield Label("🛠  Tool calls", classes="qheader")
                yield Tree("session", id="tools-log")
        yield Footer()

    def on_mount(self) -> None:
        self.title = "cici"
        self.sub_title = self._model
        self._refresh_status()
        banner = (
            "[bold cyan]"
            "  ██████╗   ██╗      ██████╗   ██╗\n"
            " ██╔════╝   ██║     ██╔════╝   ██║\n"
            " ██║        ██║     ██║        ██║\n"
            " ██║        ██║     ██║        ██║\n"
            " ╚██████╗   ██║     ╚██████╗   ██║\n"
            "  ╚═════╝   ╚═╝      ╚═════╝   ╚═╝"
            "[/bold cyan]"
        )
        self._chat_log(banner)
        self._chat_log("[bold cyan] cici[/bold cyan][dim] — minimal coding agent[/dim]")
        self._chat_log(
            "[dim]Type a request, or /clear /plan /cost /compact /memory /skills.[/dim]"
        )
        self.query_one("#input", Input).focus()
        tree = self.query_one("#tools-log", Tree)
        tree.show_root = False
        tree.root.expand()

    # — bindings —

    def action_interrupt(self) -> None:
        if self._busy:
            self._agent.abort()
            self._chat_log("[yellow]  (interrupted)[/yellow]")
            self._idle_interrupts = 0
            return
        # Idle: first Ctrl+C clears input; second in a row quits.
        inp = self.query_one("#input", Input)
        if inp.value:
            inp.value = ""
            self._idle_interrupts = 0
            return
        self._idle_interrupts += 1
        if self._idle_interrupts >= 2:
            self.exit()
            return
        self._chat_log("[dim]  (press Ctrl+C again to quit, or Ctrl+D)[/dim]")

    def action_clear_chat(self) -> None:
        self.query_one("#chat", RichLog).clear()

    # — chat log helpers —

    def _chat_log(self, markup: str) -> None:
        self.query_one("#chat", RichLog).write(markup)

    def _refresh_status(self) -> None:
        cost_in = (self._total_in / 1_000_000) * 3
        cost_out = (self._total_out / 1_000_000) * 15
        total = cost_in + cost_out
        sb = self.query_one("#status-bar", Static)
        sb.update(
            f" {self._model}  •  mode: {self._permission_mode}  •  "
            f"turns: {self._turn_count}  •  tokens: {self._total_in}↓ {self._total_out}↑  "
            f"•  ~${total:.4f}"
        )
        sp = self.query_one("#status-panel", Static)
        sp.update(
            Text.from_markup(
                f"[bold]model[/bold]   {self._model}\n"
                f"[bold]mode[/bold]    {self._permission_mode}\n"
                f"[bold]cwd[/bold]     {self._cwd.name}\n"
                f"[bold]turns[/bold]   {self._turn_count}\n"
                f"[bold]tokens[/bold]  {self._total_in} in / {self._total_out} out\n"
                f"[bold]cost[/bold]    ${total:.4f}\n"
                f"[dim]started {datetime.now().strftime('%H:%M:%S')}[/dim]"
            )
        )

    # — input —

    @on(Input.Submitted, "#input")
    async def _on_submit(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text or self._busy:
            return
        event.input.value = ""
        self._idle_interrupts = 0
        await self._handle_user_input(text)

    @on(DirectoryTree.FileSelected)
    def _on_file_selected(self, event: DirectoryTree.FileSelected) -> None:
        """Insert ``@<rel_path>`` at the cursor of the input box."""
        try:
            rel = event.path.relative_to(self._cwd)
        except ValueError:
            rel = event.path
        token = f"@{rel} "
        inp = self.query_one("#input", Input)
        inp.focus()
        try:
            inp.insert_text_at_cursor(token)
        except AttributeError:
            cur = inp.cursor_position
            inp.value = inp.value[:cur] + token + inp.value[cur:]
            inp.cursor_position = cur + len(token)

    async def _handle_user_input(self, text: str) -> None:
        self._chat_log(f"\n[bold green]>[/bold green] {text}")
        if text in ("exit", "quit"):
            self.exit()
            return

        # Slash commands
        if text == "/clear":
            self._agent.clear_history()
            self.query_one("#chat", RichLog).clear()
            return
        if text == "/plan":
            self._agent.toggle_plan_mode()
            self._permission_mode = self._agent.permission_mode
            self._refresh_status()
            return
        if text == "/cost":
            self._agent.show_cost()
            return
        if text == "/compact":
            try:
                await self._agent.compact()
            except Exception as e:
                self._chat_log(f"[red]Error: {e}[/red]")
            return
        if text == "/memory":
            ms = list_memories()
            if not ms:
                self._chat_log("[cyan]ℹ[/cyan] No memories saved yet.")
            else:
                self._chat_log(f"[cyan]ℹ[/cyan] {len(ms)} memories:")
                for m in ms:
                    self._chat_log(f"    [dim][{m.type}][/dim] {m.name} — {m.description}")
            return
        if text == "/skills":
            sk = discover_skills()
            if not sk:
                self._chat_log("[cyan]ℹ[/cyan] No skills found.")
            else:
                self._chat_log(f"[cyan]ℹ[/cyan] {len(sk)} skills:")
                for s in sk:
                    tag = f"/{s.name}" if s.user_invocable else s.name
                    self._chat_log(f"    {tag} [dim]({s.source})[/dim] — {s.description}")
            return
        if text.startswith("/"):
            space = text.find(" ")
            cmd = text[1:space] if space > 0 else text[1:]
            args = text[space + 1 :] if space > 0 else ""
            sk = get_skill_by_name(cmd)
            if sk and sk.user_invocable:
                self._chat_log(f"[cyan]ℹ[/cyan] Invoking skill: {sk.name}")
                try:
                    if sk.context == "fork":
                        execute_skill(sk.name, args)
                        await self._run_chat(
                            f'Use the skill tool to invoke "{sk.name}" with args: {args or "(none)"}'
                        )
                    else:
                        resolved = resolve_skill_prompt(sk, args)
                        await self._run_chat(resolved)
                except Exception as e:
                    if "abort" not in str(e).lower():
                        self._chat_log(f"[red]Error: {e}[/red]")
                return

        await self._run_chat(text)

    async def _run_chat(self, message: str) -> None:
        self._busy = True
        self._stream_buffer = ""
        self._render_stream()
        self.query_one("#spinner").add_class("spinner-on")
        try:
            await self._agent.chat(message)
        except Exception as e:
            if "abort" not in str(e).lower():
                self._chat_log(f"[red]Error: {e}[/red]")
        finally:
            self.flush_stream()
            self._busy = False
            self.query_one("#spinner").remove_class("spinner-on")
            self._turn_count += 1
            # Pull cumulative usage from agent if available.
            try:
                self._total_in = getattr(self._agent, "total_input_tokens", self._total_in)
                self._total_out = getattr(
                    self._agent, "total_output_tokens", self._total_out
                )
            except Exception:
                pass
            self._refresh_status()

    # — backend hooks (called from TextualBackend on UI thread) —

    def append_assistant_text(self, text: str) -> None:
        """Stream a text delta from the LLM.

        Strategy: accumulate the full assistant turn in ``_stream_buffer``
        and render it live (raw) into the :class:`Static` (``#stream``)
        widget so users see characters arrive in real time. On turn end
        (``flush_stream``) the buffer is committed to the :class:`RichLog`
        history as a rendered :class:`rich.markdown.Markdown` block, then
        cleared.
        """
        if not text:
            return
        self._stream_buffer += text
        self._render_stream()

    def _render_stream(self) -> None:
        stream = self.query_one("#stream", Static)
        if self._stream_buffer:
            stream.remove_class("empty")
            stream.update(self._stream_buffer)
        else:
            stream.add_class("empty")
            stream.update("")

    def flush_stream(self) -> None:
        """Commit the accumulated assistant turn as rendered Markdown."""
        if self._stream_buffer:
            from rich.markdown import Markdown

            log = self.query_one("#chat", RichLog)
            try:
                log.write(Markdown(self._stream_buffer, code_theme="monokai"))
            except Exception:
                # Fall back to plain text if markdown parsing chokes
                log.write(self._stream_buffer)
            self._stream_buffer = ""
            self._render_stream()

    def append_tool_call(self, name: str, inp: dict) -> None:
        # Commit any in-progress assistant text first so tool call
        # appears below the prose that introduced it.
        self.flush_stream()
        icon = get_tool_icon(name)
        summary = get_tool_summary(name, inp)
        self._chat_log(f"[yellow]  {icon} {name}[/yellow] [dim]{summary}[/dim]")
        tree = self.query_one("#tools-log", Tree)
        self._tool_idx += 1
        node = tree.root.add(f"{icon} {name} {summary[:40]}", expand=False)
        node.add_leaf(f"input: {str(inp)[:200]}")
        # Stash node so result handler can attach output.
        if not hasattr(self, "_pending_tool_nodes"):
            self._pending_tool_nodes = []  # type: ignore[attr-defined]
        self._pending_tool_nodes.append(node)  # type: ignore[attr-defined]

    def append_tool_result(self, name: str, result: str) -> None:
        del name
        nodes = getattr(self, "_pending_tool_nodes", None)
        if nodes:
            node = nodes.pop(0)
            head = result.split("\n", 1)[0][:200]
            node.add_leaf(f"result: {head}")
        # Don't echo result into chat log — keeps middle pane focused on
        # assistant prose. Right pane shows full tool call tree.

    def show_info(self, msg: str) -> None:
        self._chat_log(f"[cyan]  ℹ {msg}[/cyan]")

    def show_error(self, msg: str) -> None:
        self._chat_log(f"[red]  ✗ {msg}[/red]")

    def show_retry(self, attempt: int, max_retries: int, reason: str) -> None:
        self._chat_log(f"[yellow]  ↻ Retry {attempt}/{max_retries}: {reason}[/yellow]")

    def show_cost(self, in_tok: int, out_tok: int) -> None:
        self._total_in = in_tok
        self._total_out = out_tok
        self._refresh_status()


# ─── Backend implementing UIBackend ─────────────────────────


class TextualBackend:
    """Routes every ``ui.print_*`` call to the running CiciApp."""

    def __init__(self, app: CiciApp) -> None:
        self._app = app

    # All sync calls are scheduled on the Textual event loop.
    def _call(self, fn, *args) -> None:
        try:
            self._app.call_from_thread(fn, *args)
        except Exception:
            # call_from_thread fails when running already in the loop
            # thread; fall back to direct invocation.
            try:
                fn(*args)
            except Exception:
                pass

    def print_assistant_text(self, text: str) -> None:
        self._call(self._app.append_assistant_text, text)

    def print_tool_call(self, name: str, inp: dict) -> None:
        self._call(self._app.append_tool_call, name, inp)

    def print_tool_result(self, name: str, result: str) -> None:
        self._call(self._app.append_tool_result, name, result)

    def print_error(self, msg: str) -> None:
        self._call(self._app.show_error, msg)

    def print_confirmation(self, command: str) -> None:
        # The actual modal is triggered via confirm_fn; this is just
        # a chat-log breadcrumb leading up to it.
        self._call(self._app.show_info, f"⚠ Dangerous: {command}")

    def print_divider(self) -> None:
        pass

    def print_cost(self, input_tokens: int, output_tokens: int) -> None:
        self._call(self._app.show_cost, input_tokens, output_tokens)

    def print_retry(self, attempt: int, max_retries: int, reason: str) -> None:
        self._call(self._app.show_retry, attempt, max_retries, reason)

    def print_info(self, msg: str) -> None:
        self._call(self._app.show_info, msg)

    def start_spinner(self, label: str = "Thinking") -> None:
        del label  # LoadingIndicator already running on _busy

    def stop_spinner(self) -> None:
        pass

    def print_sub_agent_start(self, agent_type: str, description: str) -> None:
        self._call(
            self._app._chat_log,
            f"[magenta]  ┌─ Sub-agent [{agent_type}]: {description}[/magenta]",
        )

    def print_sub_agent_end(self, agent_type: str, description: str) -> None:
        del description
        self._call(
            self._app._chat_log,
            f"[magenta]  └─ Sub-agent [{agent_type}] completed[/magenta]",
        )

    async def ask_questions(self, questions: list[dict]) -> dict[str, str]:
        result = await self._push_modal(QuestionsModal(questions))
        if not result or result.get("cancelled"):
            return {}
        return result.get("answers") or {}

    # Modal helpers —

    async def _push_modal(self, modal: ModalScreen) -> Any:
        """Push a modal onto the Textual app and await its result."""
        loop = asyncio.get_event_loop()
        future: asyncio.Future = loop.create_future()

        def _set(value: Any) -> None:
            if not future.done():
                future.set_result(value)

        def _show() -> None:
            self._app.push_screen(modal, _set)

        try:
            self._app.call_from_thread(_show)
        except Exception:
            _show()
        return await future

    async def confirm(self, message: str) -> bool:
        ok = await self._push_modal(ConfirmModal(message))
        return bool(ok)

    async def plan_approval(self, plan_content: str) -> dict:
        return await self._push_modal(PlanApprovalModal(plan_content))


# ─── Entry point ───────────────────────────────────────────


async def run_tui(agent: Agent, *, model: str, permission_mode: str) -> None:
    """Launch the Textual UI bound to ``agent``."""
    app = CiciApp(agent, model=model, permission_mode=permission_mode)
    backend = TextualBackend(app)
    set_backend(backend)

    async def confirm_fn(message: str) -> bool:
        return await backend.confirm(message)

    async def plan_approval_fn(plan: str) -> dict:
        return await backend.plan_approval(plan)

    agent.set_confirm_fn(confirm_fn)
    agent.set_plan_approval_fn(plan_approval_fn)

    await app.run_async()
