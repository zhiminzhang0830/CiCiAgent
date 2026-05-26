"""CLI entry point and interactive REPL — mirrors cli.ts."""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import signal
import sys
from datetime import datetime

from .agent import Agent
from .memory import list_memories
from .session import get_latest_session_id, load_session
from .skills import (
    discover_skills,
    execute_skill,
    get_skill_by_name,
    resolve_skill_prompt,
)
from .ui import (
    print_error,
    print_info,
    print_plan_approval_options,
    print_plan_for_approval,
    print_user_prompt,
    print_welcome,
)

# from dotenv import load_dotenv
# load_dotenv(".env")


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
        # Keep only the text after the last carriage return in a line-fragment
        # (no newlines inside `segment`). This drops all intermediate spinner
        # frames while preserving whatever was written after the last `\r`.
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
            # Keep the unfinished trailing fragment for the next write, but
            # collapse spinner frames inside it so the buffer cannot grow
            # unboundedly while the spinner is running.
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="coco",
        description="Coco — a minimal coding agent",
        add_help=False,
    )
    parser.add_argument("prompt", nargs="*", help="One-shot prompt")
    parser.add_argument(
        "--yolo", "-y", action="store_true", help="Skip all confirmation prompts"
    )
    parser.add_argument("--plan", action="store_true", help="Plan mode: read-only")
    parser.add_argument(
        "--accept-edits", action="store_true", help="Auto-approve file edits"
    )
    parser.add_argument(
        "--dont-ask", action="store_true", help="Auto-deny confirmations (for CI)"
    )
    parser.add_argument(
        "--thinking", action="store_true", help="Enable extended thinking"
    )
    parser.add_argument("--model", "-m", default=None, help="Model to use")
    parser.add_argument(
        "--api-base", default=None, help="OpenAI-compatible API base URL"
    )
    parser.add_argument("--resume", action="store_true", help="Resume last session")
    parser.add_argument("--max-cost", type=float, default=None, help="Max USD spend")
    parser.add_argument("--max-turns", type=int, default=None, help="Max agentic turns")
    parser.add_argument(
        "--log",
        nargs="?",
        const="",
        default=None,
        help="Override log file path " "(default: ./.coco/logs/coco-<timestamp>.log)",
    )
    parser.add_argument("--help", "-h", action="store_true", help="Show help")
    return parser.parse_args()


def _resolve_permission_mode(args: argparse.Namespace) -> str:
    if args.yolo:
        return "bypassPermissions"
    if args.plan:
        return "plan"
    if args.accept_edits:
        return "acceptEdits"
    if args.dont_ask:
        return "dontAsk"
    return "default"


async def run_repl(agent: Agent) -> None:
    """Interactive REPL loop."""

    async def confirm_fn(message: str) -> bool:
        try:
            answer = input("  Allow? (y/n): ")
            return answer.lower().startswith("y")
        except EOFError:
            return False

    agent.set_confirm_fn(confirm_fn)

    async def plan_approval_fn(plan_content: str) -> dict:
        print_plan_for_approval(plan_content)
        print_plan_approval_options()
        while True:
            try:
                choice = input("  Enter choice (1-4): ").strip()
            except EOFError:
                return {"choice": "manual-execute"}
            if choice == "1":
                return {"choice": "clear-and-execute"}
            elif choice == "2":
                return {"choice": "execute"}
            elif choice == "3":
                return {"choice": "manual-execute"}
            elif choice == "4":
                try:
                    feedback = input("  Feedback (what to change): ").strip()
                except EOFError:
                    feedback = ""
                return {"choice": "keep-planning", "feedback": feedback or None}
            else:
                print("  Invalid choice. Enter 1, 2, 3, or 4.")

    agent.set_plan_approval_fn(plan_approval_fn)

    sigint_count = 0

    def handle_sigint(sig, frame):
        nonlocal sigint_count
        if agent._aborted is False and agent._output_buffer is not None:
            # Agent is processing
            agent.abort()
            print("\n  (interrupted)")
            sigint_count = 0
            print_user_prompt()
        else:
            sigint_count += 1
            if sigint_count >= 2:
                print("\nBye!\n")
                sys.exit(0)
            print("\n  Press Ctrl+C again to exit.")
            print_user_prompt()

    signal.signal(signal.SIGINT, handle_sigint)
    print_welcome()

    while True:
        print_user_prompt()
        try:
            line = input()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!\n")
            break

        inp = line.strip()
        sigint_count = 0

        if not inp:
            continue
        if inp in ("exit", "quit"):
            print("\nBye!\n")
            break

        # REPL commands
        if inp == "/clear":
            agent.clear_history()
            continue
        if inp == "/plan":
            agent.toggle_plan_mode()
            continue
        if inp == "/cost":
            agent.show_cost()
            continue
        if inp == "/compact":
            try:
                await agent.compact()
            except Exception as e:
                print_error(str(e))
            continue
        if inp == "/memory":
            memories = list_memories()
            if not memories:
                print_info("No memories saved yet.")
            else:
                print_info(f"{len(memories)} memories:")
                for m in memories:
                    print(f"    [{m.type}] {m.name} — {m.description}")
            continue
        if inp == "/skills":
            skills = discover_skills()
            if not skills:
                print_info(
                    "No skills found. Add skills to .claude/skills/<name>/SKILL.md"
                )
            else:
                print_info(f"{len(skills)} skills:")
                for s in skills:
                    tag = f"/{s.name}" if s.user_invocable else s.name
                    print(f"    {tag} ({s.source}) — {s.description}")
            continue

        # Skill invocation: /<skill-name> [args]
        if inp.startswith("/"):
            space_idx = inp.find(" ")
            cmd_name = inp[1:space_idx] if space_idx > 0 else inp[1:]
            cmd_args = inp[space_idx + 1 :] if space_idx > 0 else ""
            skill = get_skill_by_name(cmd_name)
            if skill and skill.user_invocable:
                print_info(f"Invoking skill: {skill.name}")
                try:
                    if skill.context == "fork":
                        result = execute_skill(skill.name, cmd_args)
                        if result:
                            await agent.chat(
                                f'Use the skill tool to invoke "{skill.name}" with args: {cmd_args or "(none)"}'
                            )
                    else:
                        resolved = resolve_skill_prompt(skill, cmd_args)
                        await agent.chat(resolved)
                except Exception as e:
                    if "abort" not in str(e).lower():
                        print_error(str(e))
                continue

        # Normal chat
        try:
            await agent.chat(inp)
        except Exception as e:
            if "abort" not in str(e).lower():
                print_error(str(e))


def main() -> None:
    args = parse_args()

    if args.log:
        log_path = args.log
    else:
        log_dir = os.path.join(os.getcwd(), ".coco", "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(
            log_dir,
            f"coco-{datetime.now().strftime('%Y%m%d-%H%M%S')}.log",
        )
    _install_logging(log_path)
    print_info(f"Logging session to {log_path}")

    if args.help:
        print(
            """
Usage: coco [options] [prompt]

Options:
  --yolo, -y          Skip all confirmation prompts (bypassPermissions mode)
  --plan              Plan mode: read-only, describe changes without executing
  --accept-edits      Auto-approve file edits, still confirm dangerous shell
  --dont-ask          Auto-deny anything needing confirmation (for CI)
  --thinking          Enable extended thinking (Anthropic only)
  --model, -m         Model to use (default: claude-opus-4-6, or COCO_MODEL env)
  --api-base URL      Use OpenAI-compatible API endpoint (key via env var)
  --resume            Resume the last session
  --max-cost USD      Stop when estimated cost exceeds this amount
  --max-turns N       Stop after N agentic turns
  --log [PATH]        Override log file path (default: ./.coco/logs/coco-<timestamp>.log)
  --help, -h          Show this help

REPL commands:
  /clear              Clear conversation history
  /plan               Toggle plan mode (read-only <-> normal)
  /cost               Show token usage and cost
  /compact            Manually compact conversation
  /memory             List saved memories
  /skills             List available skills
  /<skill-name>       Invoke a skill (e.g. /commit "fix types")

Examples:
  coco "fix the bug in src/app.ts"
  coco --yolo "run all tests and fix failures"
  coco --plan "how would you refactor this?"
  coco --max-cost 0.50 --max-turns 20 "implement feature X"
  OPENAI_API_KEY=sk-xxx coco --api-base https://aihubmix.com/v1 --model gpt-4o "hello"
  coco --resume
  coco  # starts interactive REPL
"""
        )
        sys.exit(0)

    permission_mode = _resolve_permission_mode(args)
    model = args.model or os.environ.get("MINI_CLAUDE_MODEL", "claude-opus-4-6")
    api_base = args.api_base

    # Resolve API config
    resolved_api_base = api_base
    resolved_api_key: str | None = None
    resolved_use_openai = bool(api_base)

    if os.environ.get("OPENAI_API_KEY") and os.environ.get("OPENAI_BASE_URL"):
        resolved_api_key = os.environ["OPENAI_API_KEY"]
        resolved_api_base = resolved_api_base or os.environ.get("OPENAI_BASE_URL")
        resolved_use_openai = True
    elif os.environ.get("ANTHROPIC_API_KEY"):
        resolved_api_key = os.environ["ANTHROPIC_API_KEY"]
        resolved_api_base = resolved_api_base or os.environ.get("ANTHROPIC_BASE_URL")
        resolved_use_openai = False
    elif os.environ.get("OPENAI_API_KEY"):
        resolved_api_key = os.environ["OPENAI_API_KEY"]
        resolved_api_base = resolved_api_base or os.environ.get("OPENAI_BASE_URL")
        resolved_use_openai = True

    if not resolved_api_key and api_base:
        resolved_api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get(
            "ANTHROPIC_API_KEY"
        )
        resolved_use_openai = True

    if not resolved_api_key:
        print_error(
            "API key is required.\n"
            "  Set ANTHROPIC_API_KEY (+ optional ANTHROPIC_BASE_URL) for Anthropic format,\n"
            "  or OPENAI_API_KEY + OPENAI_BASE_URL for OpenAI-compatible format."
        )
        sys.exit(1)

    agent = Agent(
        permission_mode=permission_mode,
        model=model,
        thinking=args.thinking,
        max_cost_usd=args.max_cost,
        max_turns=args.max_turns,
        api_base=resolved_api_base if resolved_use_openai else None,
        anthropic_base_url=resolved_api_base if not resolved_use_openai else None,
        api_key=resolved_api_key,
    )

    # Resume session
    if args.resume:
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

    prompt = " ".join(args.prompt) if args.prompt else None

    if prompt:
        # One-shot mode
        try:
            asyncio.run(agent.chat(prompt))
        except Exception as e:
            print_error(str(e))
            sys.exit(1)
    else:
        # Interactive REPL
        asyncio.run(run_repl(agent))


if __name__ == "__main__":
    main()
