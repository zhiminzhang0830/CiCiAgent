"""CLI entry point — TUI is the default interactive UI.

Two run modes:

* **One-shot** — ``cici "prompt"`` runs a single ``agent.chat`` and exits.
  Output streams through the default Rich backend.
* **Interactive** — ``cici`` (no prompt) launches the Textual TUI.

CLI parsing uses :mod:`typer`; environment-driven API configuration is
modelled with :mod:`pydantic_settings.BaseSettings` so the precedence
between ``OPENAI_*`` and ``ANTHROPIC_*`` variables and ``--api-base``
overrides is captured in one place instead of scattered ``os.environ``
lookups.
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
from datetime import datetime
from typing import Optional

import typer
from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from .agent import Agent
from .session import get_latest_session_id, load_session
from .ui import print_error, print_info

load_dotenv(override=True)


# ANSI escape sequence stripper for clean log files
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


class _Tee:
    """Write-through stream that duplicates to terminal + log file.

    Terminal output can contain carriage-return (`\\r`) animations (e.g. the
    "Thinking..." spinner) that overwrite the same line in the TTY. Naively
    teeing those bytes to a log file produces hundreds of duplicated frames.
    This class line-buffers the log side and collapses each `\\r`-overwritten
    run down to its final segment before committing a line to disk.
    """

    def __init__(self, terminal, log_file):
        self._terminal = terminal
        self._log = log_file
        self._log_buf = ""

    @staticmethod
    def _collapse_cr(segment: str) -> str:
        if "\r" in segment:
            return segment.rsplit("\r", 1)[1]
        return segment

    def write(self, data: str) -> int:
        n = self._terminal.write(data)
        try:
            buf = self._log_buf + _ANSI_RE.sub("", data)
            *lines, tail = buf.split("\n")
            if lines:
                self._log.write("\n".join(self._collapse_cr(ln) for ln in lines) + "\n")
                self._log.flush()
            self._log_buf = self._collapse_cr(tail)
        except Exception:
            pass
        return n

    def flush(self) -> None:
        self._terminal.flush()
        try:
            self._log.flush()
        except Exception:
            pass

    def isatty(self) -> bool:
        return getattr(self._terminal, "isatty", lambda: False)()

    def __getattr__(self, name):
        return getattr(self._terminal, name)


def _install_logging(path: str):
    """Tee sys.stdout/stderr and builtins.input() to a log file. Returns the file handle."""
    import builtins

    f = open(path, "a", buffering=1, encoding="utf-8")
    f.write(f"\n===== session started {datetime.now().isoformat()} =====\n")
    sys.stdout = _Tee(sys.stdout, f)
    sys.stderr = _Tee(sys.stderr, f)

    _orig_input = builtins.input

    def _logged_input(prompt: str = "") -> str:
        line = _orig_input(prompt)
        try:
            f.write(f"{prompt}{line}\n")
            f.flush()
        except Exception:
            pass
        return line

    builtins.input = _logged_input
    return f


# ─── Settings ───────────────────────────────────────────────


class APISettings(BaseSettings):
    """API credentials and endpoints, sourced from process env.

    Names mirror the historical environment variables exactly so
    existing user setups keep working. Resolution precedence is applied
    in :meth:`resolve` to match the legacy if/elif ladder in this
    module.
    """

    model_config = SettingsConfigDict(extra="ignore", case_sensitive=True)

    OPENAI_API_KEY: Optional[str] = None
    OPENAI_BASE_URL: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    ANTHROPIC_BASE_URL: Optional[str] = None
    CICI_MODEL: str = Field(default="claude-opus-4-6")

    def resolve(
        self, cli_api_base: Optional[str]
    ) -> tuple[Optional[str], Optional[str], bool]:
        """Return ``(api_key, api_base, use_openai_format)``.

        Mirrors the precedence rules of the original CLI:

        1. Both OpenAI key + base set → OpenAI-compatible.
        2. Anthropic key set → Anthropic.
        3. OpenAI key set → OpenAI-compatible.
        4. ``--api-base`` provided without env match → fall back to any
           available key and force OpenAI-compatible format.
        """
        api_base = cli_api_base
        if self.OPENAI_API_KEY and self.OPENAI_BASE_URL:
            return (
                self.OPENAI_API_KEY,
                api_base or self.OPENAI_BASE_URL,
                True,
            )
        if self.ANTHROPIC_API_KEY:
            return (
                self.ANTHROPIC_API_KEY,
                api_base or self.ANTHROPIC_BASE_URL,
                False,
            )
        if self.OPENAI_API_KEY:
            return (
                self.OPENAI_API_KEY,
                api_base or self.OPENAI_BASE_URL,
                True,
            )
        if cli_api_base:
            # ``--api-base`` was passed but no env match — fall back to
            # whatever key we can find and use the OpenAI-compatible
            # transport.
            return (
                self.OPENAI_API_KEY or self.ANTHROPIC_API_KEY,
                cli_api_base,
                True,
            )
        return (None, api_base, False)


def _resolve_permission_mode(
    yolo: bool, plan: bool, accept_edits: bool, dont_ask: bool
) -> str:
    if yolo:
        return "bypassPermissions"
    if plan:
        return "plan"
    if accept_edits:
        return "acceptEdits"
    if dont_ask:
        return "dontAsk"
    return "default"


# ─── CLI ────────────────────────────────────────────────────

# ``add_completion=False`` and ``no_args_is_help=False`` keep the
# argparse-era UX: ``cici`` with no args launches the TUI rather than
# printing help.
app = typer.Typer(
    name="cici",
    help="cici — a minimal coding agent",
    add_completion=False,
    no_args_is_help=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)


@app.command()
def main(
    prompt: list[str] = typer.Argument(None, help="One-shot prompt"),
    yolo: bool = typer.Option(
        False, "--yolo", "-y", help="Skip all confirmation prompts"
    ),
    plan: bool = typer.Option(False, "--plan", help="Plan mode: read-only"),
    accept_edits: bool = typer.Option(
        False, "--accept-edits", help="Auto-approve file edits"
    ),
    dont_ask: bool = typer.Option(
        False, "--dont-ask", help="Auto-deny confirmations (for CI)"
    ),
    thinking: bool = typer.Option(
        False, "--thinking", help="Enable extended thinking"
    ),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model to use"),
    api_base: Optional[str] = typer.Option(
        None, "--api-base", help="OpenAI-compatible API base URL"
    ),
    resume: bool = typer.Option(False, "--resume", help="Resume last session"),
    max_cost: Optional[float] = typer.Option(
        None, "--max-cost", help="Max USD spend"
    ),
    max_turns: Optional[int] = typer.Option(
        None, "--max-turns", help="Max agentic turns"
    ),
    log: Optional[str] = typer.Option(
        None,
        "--log",
        help="Override log file path (default: ./.cici/logs/cici-<timestamp>.log)",
    ),
) -> None:
    """Run cici either one-shot (with PROMPT) or in interactive TUI mode."""
    # ``prompt: list[str]`` collects positional args. typer hands us
    # ``None`` when none are given.
    prompt_words = list(prompt or [])
    one_shot = bool(prompt_words)

    if one_shot:
        if log:
            log_path = log
        else:
            log_dir = os.path.join(os.getcwd(), ".cici", "logs")
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(
                log_dir,
                f"cici-{datetime.now().strftime('%Y%m%d-%H%M%S')}.log",
            )
        _install_logging(log_path)
        print_info(f"Logging session to {log_path}")
    elif log:
        print_info("--log is ignored in TUI mode.")

    permission_mode = _resolve_permission_mode(yolo, plan, accept_edits, dont_ask)

    settings = APISettings()
    resolved_model = model or settings.CICI_MODEL
    api_key, resolved_api_base, use_openai = settings.resolve(api_base)

    if not api_key:
        print_error(
            "API key is required.\n"
            "  Set ANTHROPIC_API_KEY (+ optional ANTHROPIC_BASE_URL) for Anthropic format,\n"
            "  or OPENAI_API_KEY + OPENAI_BASE_URL for OpenAI-compatible format."
        )
        raise typer.Exit(code=1)

    agent = Agent(
        permission_mode=permission_mode,
        model=resolved_model,
        thinking=thinking,
        max_cost_usd=max_cost,
        max_turns=max_turns,
        api_base=resolved_api_base if use_openai else None,
        anthropic_base_url=resolved_api_base if not use_openai else None,
        api_key=api_key,
    )

    if resume:
        session_id = get_latest_session_id()
        if session_id:
            session = load_session(session_id)
            if session:
                agent.restore_session(
                    {
                        "anthropicMessages": session.get("anthropicMessages"),
                        "openaiMessages": session.get("openaiMessages"),
                    }
                )
            else:
                print_info("No session found to resume.")
        else:
            print_info("No previous sessions found.")

    if one_shot:
        text = " ".join(prompt_words)
        try:
            asyncio.run(agent.chat(text))
        except Exception as e:
            print_error(str(e))
            raise typer.Exit(code=1)
        return

    try:
        from .tui import run_tui
    except ImportError as e:
        print_error(
            f"Could not load the TUI ({e}). Reinstall with: pip install textual"
        )
        raise typer.Exit(code=1)
    asyncio.run(run_tui(agent, model=resolved_model, permission_mode=permission_mode))


def cli() -> None:
    """Setuptools entry point shim."""
    app()


if __name__ == "__main__":
    cli()
