"""Agent core loop.

Drives a single cici turn-by-turn conversation across either the
Anthropic or an OpenAI-compatible backend, with streaming, sub-agent
fork/return, plan-mode gating, four-tier context compression and
budget controls (turn count + USD cost) layered on top.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any, Awaitable, Callable

import anthropic
import openai

from . import compression
from .mcp_client import McpManager
from .memory import MemoryPrefetch, format_memories_for_injection, start_memory_prefetch
from .paths import plans_dir
from .prompt import build_system_prompt
from .providers import ToolResultBlock, ToolUseBlock
from .session import save_session
from .subagent import get_sub_agent_config
from .tools import (
    ToolDef,
    _derive_is_error,
    check_permission,
    execute_tool,
    is_concurrency_safe_tool,
    tool_definitions,
)
from .ui import (
    print_assistant_text,
    print_confirmation,
    print_cost,
    print_divider,
    print_error,
    print_info,
    print_retry,
    print_sub_agent_end,
    print_sub_agent_start,
    print_tool_call,
    print_tool_result,
    start_spinner,
    stop_spinner,
)

# ─── Retry with exponential backoff ──────────────────────────


def _is_retryable(error: Exception) -> bool:
    status = getattr(error, "status_code", None) or getattr(error, "status", None)
    if status in (429, 503, 529):
        return True
    msg = str(error)
    if "overloaded" in msg or "ECONNRESET" in msg or "ETIMEDOUT" in msg:
        return True
    return False


# Match the variety of provider phrasings for "prompt exceeds context window".
# Needed so reactive compaction can recover without user intervention.
_PROMPT_TOO_LONG_NEEDLES = (
    "prompt too long",
    "context_length_exceeded",
    "context length",
    "maximum context",
    "context window",
    "input tokens exceed",
    "messages resulted in",
    "reduce the length of the messages",
    "configured limit",
    "too many tokens",
    "too large for the model",
    "maximum context length",
    "exceed_context",
    "exceeds the available context size",
    "available context size",
)


def _is_prompt_too_long_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(needle in text for needle in _PROMPT_TOO_LONG_NEEDLES)


def _is_completion_token_limit_error(exc: Exception) -> bool:
    """Provider rejected max_tokens as too large (vs. prompt-length rejection)."""
    text = str(exc).lower()
    return ("max_tokens" in text or "max_completion_tokens" in text) and (
        "too large" in text or "at most" in text or "completion tokens" in text
    )


def _extract_completion_token_limit(exc: Exception) -> int | None:
    """Parse provider errors like 'supports at most 128000 completion tokens'."""
    import re

    text = str(exc).lower().replace(",", "")
    for pattern in (
        r"supports at most\s+(\d+)\s+completion tokens",
        r"at most\s+(\d+)\s+completion tokens",
        r"max(?:imum)?(?:_completion)?[_\s-]tokens.*?(?:<=|less than or equal to|at most)\s+(\d+)",
    ):
        m = re.search(pattern, text)
        if m:
            try:
                return max(1, int(m.group(1)))
            except ValueError:
                return None
    return None


async def _with_retry(fn, max_retries: int = 3):
    for attempt in range(max_retries + 1):
        try:
            return await fn()
        except Exception as error:
            if attempt >= max_retries or not _is_retryable(error):
                raise
            delay = (
                min(1000 * (2**attempt), 30000) / 1000
                + (hash(str(time.time())) % 1000) / 1000
            )
            status = getattr(error, "status_code", None) or getattr(
                error, "status", None
            )
            reason = (
                f"HTTP {status}"
                if status
                else (getattr(error, "code", None) or "network error")
            )
            print_retry(attempt + 1, max_retries, reason)
            await asyncio.sleep(delay)


# ─── Model context windows ──────────────────────────────────

# Local overrides / aliases. Values here take precedence over litellm's
# model registry — use for proxy aliases or to pin a specific window size.
MODEL_CONTEXT = {
    "claude-opus-4-6": 200000,
    "claude-sonnet-4-6": 200000,
    "claude-sonnet-4-20250514": 200000,
    "claude-haiku-4-5-20251001": 200000,
    "claude-opus-4-20250514": 200000,
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
}

_DEFAULT_CONTEXT_WINDOW = 200000


@lru_cache(maxsize=None)
def _get_context_window(model: str) -> int:
    # 1) Local overrides / aliases win.
    if model in MODEL_CONTEXT:
        return MODEL_CONTEXT[model]
    # 2) Query litellm's model registry (offline lookup, no network).
    try:
        import litellm

        info = litellm.get_model_info(model)
        window = info.get("max_input_tokens") or info.get("max_tokens")
        if window:
            return int(window)
    except Exception:
        pass
    # 3) Fallback.
    return _DEFAULT_CONTEXT_WINDOW


# ─── Thinking support detection ─────────────────────────────


def _model_supports_thinking(model: str) -> bool:
    m = model.lower()
    if "claude-3-" in m or "3-5-" in m or "3-7-" in m:
        return False
    if "claude" in m and any(x in m for x in ("opus", "sonnet", "haiku")):
        return True
    return False


def _model_supports_adaptive_thinking(model: str) -> bool:
    m = model.lower()
    return "opus-4-6" in m or "sonnet-4-6" in m


def _get_max_output_tokens(model: str) -> int:
    m = model.lower()
    if "opus-4-6" in m:
        return 64000
    if "sonnet-4-6" in m:
        return 32000
    if any(x in m for x in ("opus-4", "sonnet-4", "haiku-4")):
        return 32000
    return 16384


# ─── Convert tools to OpenAI format ─────────────────────────


def _to_openai_tools(tools: list[ToolDef]) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            },
        }
        for t in tools
    ]


# ─── Multi-tier compression constants ────────────────────────
# Compression logic lives in compression.py. The constants are re-exported
# here so any external import (`from ...agent import SNIP_PLACEHOLDER`)
# continues to resolve.

SNIPPABLE_TOOLS = compression.SNIPPABLE_TOOLS
SNIP_PLACEHOLDER = compression.SNIP_PLACEHOLDER
SNIP_THRESHOLD = compression.SNIP_THRESHOLD
MICROCOMPACT_IDLE_S = compression.MICROCOMPACT_IDLE_S
KEEP_RECENT_RESULTS = compression.KEEP_RECENT_RESULTS


# ─── Agent ───────────────────────────────────────────────────


class Agent:
    def __init__(
        self,
        *,
        permission_mode: str = "default",
        model: str = "claude-opus-4-6",
        api_base: str | None = None,
        anthropic_base_url: str | None = None,
        api_key: str | None = None,
        thinking: bool = False,
        max_cost_usd: float | None = None,
        max_turns: int | None = None,
        confirm_fn: Callable[[str], Awaitable[bool]] | None = None,
        custom_system_prompt: str | None = None,
        custom_tools: list[ToolDef] | None = None,
        is_sub_agent: bool = False,
    ):
        self.permission_mode = permission_mode
        self.thinking = thinking
        self.model = model
        self.use_openai = bool(api_base)
        # Per-provider adapter: every provider-specific operation the
        # chat loop performs routes through this handle. Stateless and
        # cheap, so we pick one at init based on ``use_openai``.
        from .providers import AnthropicAdapter, OpenAIAdapter

        self._provider = OpenAIAdapter() if self.use_openai else AnthropicAdapter()
        self.is_sub_agent = is_sub_agent
        self.tools = custom_tools or tool_definitions
        self.max_cost_usd = max_cost_usd
        self.max_turns = max_turns
        self.confirm_fn = confirm_fn
        self.effective_window = _get_context_window(model) - 20000
        self.session_id = uuid.uuid4().hex[:8]
        self.session_start_time = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.last_input_token_count = 0
        self.current_turns = 0
        self.last_api_call_time = 0.0

        # Abort support
        self._aborted = False
        self._current_task: asyncio.Task | None = None

        # Background sub-agents: track in-flight asyncio.Tasks keyed by a short
        # ID the model can poll via the ``agent_result`` tool. Only populated
        # when the model calls ``agent(run_in_background=True, ...)``.
        self._background_agent_tasks: dict[str, asyncio.Task] = {}
        self._background_agent_meta: dict[str, dict] = {}

        # Runtime output-token cap, lowered when a provider rejects max_tokens
        # as too large. Sticky for the session so subsequent turns don't retry
        # the same failing value.
        self._output_token_cap_override: int | None = None

        # Permission whitelist
        self._confirmed_paths: set[str] = set()

        # Plan mode state
        self._pre_plan_mode: str | None = None
        self._plan_file_path: str | None = None
        self._plan_approval_fn: Callable[[str], Awaitable[dict]] | None = None
        self._context_cleared: bool = False  # Set when plan approval clears context

        # Thinking mode
        self._thinking_mode = self._resolve_thinking_mode()

        # Output buffer (sub-agents capture output)
        self._output_buffer: list[str] | None = None

        # Read-before-edit: track file read timestamps (absolutePath → mtime)
        self._read_file_state: dict[str, float] = {}

        # MCP integration
        self._mcp_manager = McpManager()
        self._mcp_initialized = False

        # Memory recall state — semantic prefetch per user turn
        self._already_surfaced_memories: set[str] = set()
        self._session_memory_bytes = 0

        # ── Structured focus state (survives compaction) ──────────
        # Lives on the Agent instance rather than in conversation messages,
        # so _compact_{anthropic,openai} preserves it automatically. Injected
        # into the system prompt each turn to anchor the model on the current
        # goal + recently-touched artifacts + verified work, reducing drift
        # on long sessions.
        self._focus_state: dict[str, Any] = {
            "goal": "",
            "recent_goals": [],  # list[str], capped
            "active_artifacts": [],  # list[str] — files, URLs, skills
            "work_log": [],  # list[str] — short action summaries
        }

        # Separate message histories
        self._anthropic_messages: list[dict] = []
        self._openai_messages: list[dict] = []

        # Build system prompt
        self._base_system_prompt = custom_system_prompt or build_system_prompt(
            session_id=self.session_id
        )
        if self.permission_mode == "plan":
            self._plan_file_path = self._generate_plan_file_path()
            self._system_prompt = (
                self._base_system_prompt + self._build_plan_mode_prompt()
            )
        else:
            self._system_prompt = self._base_system_prompt

        # Initialize clients
        if self.use_openai:
            self._openai_client = openai.AsyncOpenAI(base_url=api_base, api_key=api_key)
            self._anthropic_client = None
            self._openai_messages.append(
                {"role": "system", "content": self._system_prompt}
            )
        else:
            kwargs: dict[str, Any] = {}
            if api_key:
                kwargs["api_key"] = api_key
            if anthropic_base_url:
                kwargs["base_url"] = anthropic_base_url
            self._anthropic_client = anthropic.AsyncAnthropic(**kwargs)
            self._openai_client = None

    def _resolve_thinking_mode(self) -> str:
        if not self.thinking:
            return "disabled"
        if not _model_supports_thinking(self.model):
            return "disabled"
        if _model_supports_adaptive_thinking(self.model):
            return "adaptive"
        return "enabled"

    @property
    def is_processing(self) -> bool:
        return self._current_task is not None and not self._current_task.done()

    def _build_side_query(self):
        """Build a sideQuery callable for memory recall, works with both backends."""
        if self._anthropic_client:
            client = self._anthropic_client
            model = self.model

            async def _sq(system: str, user_message: str) -> str:
                resp = await client.messages.create(
                    model=model,
                    max_tokens=256,
                    system=system,
                    messages=[{"role": "user", "content": user_message}],
                )
                return "".join(b.text for b in resp.content if b.type == "text")

            return _sq
        if self._openai_client:
            client = self._openai_client
            model = self.model

            async def _sq_oai(system: str, user_message: str) -> str:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user_message},
                    ],
                )
                return resp.choices[0].message.content or "" if resp.choices else ""

            return _sq_oai
        return None

    def abort(self) -> None:
        self._aborted = True
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()

    def set_confirm_fn(self, fn: Callable[[str], Awaitable[bool]]) -> None:
        self.confirm_fn = fn

    def set_plan_approval_fn(self, fn: Callable[[str], Awaitable[dict]]) -> None:
        self._plan_approval_fn = fn

    # ─── Plan mode toggle ────────────────────────────────────

    def toggle_plan_mode(self) -> str:
        if self.permission_mode == "plan":
            self.permission_mode = self._pre_plan_mode or "default"
            self._pre_plan_mode = None
            self._plan_file_path = None
            self._system_prompt = self._base_system_prompt
            if self.use_openai and self._openai_messages:
                self._openai_messages[0]["content"] = self._system_prompt
            print_info(f"Exited plan mode → {self.permission_mode} mode")
            return self.permission_mode
        else:
            self._pre_plan_mode = self.permission_mode
            self.permission_mode = "plan"
            self._plan_file_path = self._generate_plan_file_path()
            self._system_prompt = (
                self._base_system_prompt + self._build_plan_mode_prompt()
            )
            if self.use_openai and self._openai_messages:
                self._openai_messages[0]["content"] = self._system_prompt
            print_info(f"Entered plan mode. Plan file: {self._plan_file_path}")
            return "plan"

    def get_token_usage(self) -> dict:
        return {"input": self.total_input_tokens, "output": self.total_output_tokens}

    # ─── Main entry point ────────────────────────────────────

    async def chat(self, user_message: str) -> None:
        # Lazily connect to MCP servers on first chat (main agent only)
        if not self._mcp_initialized and not self.is_sub_agent:
            self._mcp_initialized = True
            try:
                await self._mcp_manager.load_and_connect()
                mcp_defs = self._mcp_manager.get_tool_definitions()
                if mcp_defs:
                    self.tools = self.tools + mcp_defs
            except Exception as e:
                print_error(f"[mcp] Init failed: {e}")

        self._aborted = False
        self._remember_user_goal(user_message)
        coro = self._chat_loop(user_message)
        self._current_task = asyncio.current_task()
        try:
            await coro
        except asyncio.CancelledError:
            self._aborted = True
        finally:
            self._current_task = None
        if not self.is_sub_agent:
            print_divider()
            self._auto_save()

    # ─── Sub-agent entry point ────────────────────────────────

    async def run_once(self, prompt: str) -> dict:
        self._output_buffer = []
        prev_in = self.total_input_tokens
        prev_out = self.total_output_tokens
        await self.chat(prompt)
        text = "".join(self._output_buffer)
        self._output_buffer = None
        return {
            "text": text,
            "tokens": {
                "input": self.total_input_tokens - prev_in,
                "output": self.total_output_tokens - prev_out,
            },
        }

    # ─── Output helper ────────────────────────────────────────

    def _emit_text(self, text: str) -> None:
        if self._output_buffer is not None:
            self._output_buffer.append(text)
        else:
            print_assistant_text(text)

    # ─── REPL commands ────────────────────────────────────────

    def clear_history(self) -> None:
        self._anthropic_messages = []
        self._openai_messages = []
        if self.use_openai:
            self._openai_messages.append(
                {"role": "system", "content": self._system_prompt}
            )
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.last_input_token_count = 0
        print_info("Conversation cleared.")

    def show_cost(self) -> None:
        total = self._get_current_cost_usd()
        budget_info = f" / ${self.max_cost_usd} budget" if self.max_cost_usd else ""
        turn_info = (
            f" | Turns: {self.current_turns}/{self.max_turns}" if self.max_turns else ""
        )
        print_info(
            f"Tokens: {self.total_input_tokens} in / {self.total_output_tokens} out\n  Estimated cost: ${total:.4f}{budget_info}{turn_info}"
        )

    def _get_current_cost_usd(self) -> float:
        return (self.total_input_tokens / 1_000_000) * 3 + (
            self.total_output_tokens / 1_000_000
        ) * 15

    def _check_budget(self) -> dict:
        if (
            self.max_cost_usd is not None
            and self._get_current_cost_usd() >= self.max_cost_usd
        ):
            return {
                "exceeded": True,
                "reason": f"Cost limit reached (${self._get_current_cost_usd():.4f} >= ${self.max_cost_usd})",
            }
        if self.max_turns is not None and self.current_turns >= self.max_turns:
            return {
                "exceeded": True,
                "reason": f"Turn limit reached ({self.current_turns} >= {self.max_turns})",
            }
        return {"exceeded": False}

    async def compact(self) -> None:
        await self._compact_conversation()

    # ─── Session ──────────────────────────────────────────────

    def restore_session(self, data: dict) -> None:
        if data.get("anthropicMessages"):
            self._anthropic_messages = data["anthropicMessages"]
        if data.get("openaiMessages"):
            self._openai_messages = data["openaiMessages"]
        print_info(f"Session restored ({self._get_message_count()} messages).")

    def _get_message_count(self) -> int:
        return (
            len(self._openai_messages)
            if self.use_openai
            else len(self._anthropic_messages)
        )

    def _auto_save(self) -> None:
        try:
            save_session(
                self.session_id,
                {
                    "metadata": {
                        "id": self.session_id,
                        "model": self.model,
                        "cwd": str(Path.cwd()),
                        "startTime": self.session_start_time,
                        "messageCount": self._get_message_count(),
                    },
                    "anthropicMessages": self._anthropic_messages
                    if not self.use_openai
                    else None,
                    "openaiMessages": self._openai_messages
                    if self.use_openai
                    else None,
                },
            )
        except Exception:
            pass

    # ─── Compression (see compression.py for implementations) ────

    async def _check_and_compact(self) -> None:
        if compression.should_full_compact(
            self.last_input_token_count, self.effective_window
        ):
            print_info("Context window filling up, compacting conversation...")
            await self._compact_conversation()

    async def _compact_conversation(self) -> None:
        if self.use_openai:
            await self._compact_openai()
        else:
            await self._compact_anthropic()
        print_info("Conversation compacted.")

    async def _compact_anthropic(self) -> None:
        new_messages = await compression.compact_anthropic(
            self._anthropic_client, self.model, self._anthropic_messages
        )
        if new_messages is not None:
            self._anthropic_messages = new_messages
            self.last_input_token_count = 0

    async def _compact_openai(self) -> None:
        new_messages = await compression.compact_openai(
            self._openai_client, self.model, self._openai_messages
        )
        if new_messages is not None:
            self._openai_messages = new_messages
            self.last_input_token_count = 0

    def _run_compression_pipeline(self) -> None:
        compression.run_compression_pipeline(
            use_openai=self.use_openai,
            messages=(
                self._openai_messages if self.use_openai else self._anthropic_messages
            ),
            last_input_tokens=self.last_input_token_count,
            effective_window=self.effective_window,
            last_api_call_time=self.last_api_call_time,
        )

    def _find_tool_use_by_id(self, tool_use_id: str) -> dict | None:
        return compression.find_tool_use_by_id(self._anthropic_messages, tool_use_id)

    # ─── Large result persistence ─────────────────────────────────
    # When a tool result exceeds the inline limit, delegate to
    # compression.persist_tool_result which writes the full output to
    # ~/.cici/tool-results/ and returns a preview+path replacement.
    # The same function is called from compression.budget_tool_results_*
    # at the aggregate-budget layer — single implementation, two triggers.
    #
    # read_file is a hard opt-out: it self-bounds via MAX_FILE_SIZE_BYTES
    # and offset/limit slicing. Persisting its output would create a
    # circular Read→file→Read loop (the model reads the saved file, which
    # is still oversized, so it gets persisted again, forever).

    # Chars, not bytes — matches how providers account for tool output.
    _TOOL_OUTPUT_INLINE_LIMIT = 30_000

    def _persist_large_result(self, tool_name: str, result: str) -> str:
        """Per-tool entry point, called at tool-execution time.

        Only persists if `len(result) > _TOOL_OUTPUT_INLINE_LIMIT`. This is
        the proactive, single-block defense: any one oversized tool result
        gets offloaded before it ever enters conversation history. The
        compression-time aggregate budget (tier 1) is a second, independent
        layer that catches the N-parallel-medium-results scenario and
        reuses the same compression.persist_tool_result implementation.
        """
        # Hard opt-out for read_file — see comment above.
        if tool_name == "read_file":
            return result
        if len(result) <= self._TOOL_OUTPUT_INLINE_LIMIT:
            return result
        return compression.persist_tool_result(tool_name, result)

    # ─── Structured focus state ──────────────────────────────────
    # Goal: keep the model anchored on (a) what the user asked, (b) which
    # files/URLs are "active", (c) what has already been done. All buckets
    # are length-capped so the injection stays cheap.

    _FOCUS_MAX_RECENT_GOALS = 3
    _FOCUS_MAX_ARTIFACTS = 6
    _FOCUS_MAX_WORK_LOG = 8

    @staticmethod
    def _append_capped(bucket: list, value: str, *, limit: int) -> None:
        """Move-to-end + dedupe + cap. Mutates in place."""
        value = value.strip()
        if not value:
            return
        if value in bucket:
            bucket.remove(value)
        bucket.append(value)
        if len(bucket) > limit:
            del bucket[:-limit]

    def _remember_user_goal(self, prompt: str) -> None:
        summary = " ".join(prompt.split())[:240]
        if not summary:
            return
        self._append_capped(
            self._focus_state["recent_goals"],
            summary,
            limit=self._FOCUS_MAX_RECENT_GOALS,
        )
        self._focus_state["goal"] = summary

    def _record_tool_carryover(
        self, tool_name: str, tool_input: dict, raw_output: str, is_error: bool
    ) -> None:
        """Extract structured signals from a tool call and stash them in
        focus state. Errors are skipped (they add noise, not signal)."""
        if is_error:
            return
        artifacts = self._focus_state["active_artifacts"]
        work_log = self._focus_state["work_log"]

        def _artifact(val: str) -> None:
            self._append_capped(artifacts, val[:240], limit=self._FOCUS_MAX_ARTIFACTS)

        def _log(val: str) -> None:
            self._append_capped(work_log, val[:200], limit=self._FOCUS_MAX_WORK_LOG)

        # File-path style tools
        for key in ("file_path", "path"):
            fp = tool_input.get(key)
            if isinstance(fp, str) and fp.strip():
                _artifact(fp)
                break

        if tool_name == "read_file":
            fp = tool_input.get("file_path") or tool_input.get("path") or ""
            offset = tool_input.get("offset") or 0
            limit = tool_input.get("limit") or 0
            span = f" lines {offset}-{offset + limit}" if limit else ""
            _log(f"read {fp}{span}".strip())
        elif tool_name in ("edit_file", "write_file"):
            fp = tool_input.get("file_path") or tool_input.get("path") or ""
            _log(f"{tool_name.replace('_', ' ')} {fp}".strip())
        elif tool_name in ("run_shell", "bash"):
            cmd = str(tool_input.get("command") or "").strip().splitlines()[0:1]
            _log(f"ran: {(cmd[0] if cmd else '')[:160]}")
        elif tool_name in ("grep_search", "grep"):
            _log(f"grep: {str(tool_input.get('pattern') or '')[:160]}")
        elif tool_name == "list_files":
            _log(
                f"list: {str(tool_input.get('pattern') or tool_input.get('path') or '')[:160]}"
            )
        elif tool_name == "web_fetch":
            url = str(tool_input.get("url") or "").strip()
            if url:
                _artifact(url)
                _log(f"fetched {url}")
        elif tool_name == "agent":
            _log(
                f"sub-agent: {str(tool_input.get('description') or tool_input.get('type') or '')[:160]}"
            )
        elif tool_name == "skill":
            name = str(
                tool_input.get("skill_name") or tool_input.get("name") or ""
            ).strip()
            if name:
                _artifact(f"skill:{name}")
                _log(f"loaded skill {name}")
        elif tool_name == "enter_plan_mode":
            _log("entered plan mode")
        elif tool_name == "exit_plan_mode":
            _log("exited plan mode")

    def _render_focus_block(self) -> str:
        """Return a short XML-tagged block to append to the system prompt.
        Empty string when nothing is worth injecting."""
        fs = self._focus_state
        sections: list[str] = []
        if fs["goal"]:
            sections.append(f"Current goal: {fs['goal']}")
        recent = [g for g in fs["recent_goals"] if g != fs["goal"]]
        if recent:
            sections.append(
                "Earlier goals this session:\n"
                + "\n".join(f"- {g}" for g in recent[-self._FOCUS_MAX_RECENT_GOALS :])
            )
        if fs["active_artifacts"]:
            sections.append(
                "Active artifacts (recently touched):\n"
                + "\n".join(
                    f"- {a}"
                    for a in fs["active_artifacts"][-self._FOCUS_MAX_ARTIFACTS :]
                )
            )
        if fs["work_log"]:
            sections.append(
                "Recent tool actions:\n"
                + "\n".join(
                    f"- {w}" for w in fs["work_log"][-self._FOCUS_MAX_WORK_LOG :]
                )
            )
        if not sections:
            return ""
        body = "\n\n".join(sections)
        return (
            "\n\n<session_focus>\n"
            "The following session state is preserved across compaction to "
            "anchor you on the user's goal and work already done. Treat it as "
            "a factual summary of the current state.\n\n"
            f"{body}\n"
            "</session_focus>"
        )

    def _effective_system_prompt(self) -> str:
        """System prompt + dynamic focus block. Called at API-call time."""
        return self._system_prompt + self._render_focus_block()

    # ─── Execute tool (handles agent/skill/plan mode internally) ─────

    async def _run_tool_safely(
        self,
        name: str,
        inp: dict,
        *,
        precomputed: asyncio.Task | None = None,
    ) -> tuple[str, bool]:
        """Execute a tool (or await a pre-started task) and persist the result,
        converting any exception into an error string so every tool_use block
        can be matched with a tool_result. Returns (content, is_error).

        Without this guard a raised exception in one tool would either (a) exit
        the Anthropic sequential loop before appending a tool_result, or (b)
        cancel sibling coroutines inside asyncio.gather on the OpenAI path —
        both leave the conversation in a state the provider rejects on the
        next turn (unmatched tool_use ids).
        """
        try:
            raw = await (
                precomputed
                if precomputed is not None
                else self._execute_tool_call(name, inp)
            )
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            return f"Tool {name} failed: {type(exc).__name__}: {exc}", True
        # Detect in-band errors too (handlers return "Error: ..." strings on
        # failure). Without this, a read_file that returned "Error: File too
        # large" would be reported to the model as is_error=False, making it
        # harder for the model to recover.
        in_band_error = _derive_is_error(raw) if isinstance(raw, str) else False
        try:
            persisted = self._persist_large_result(name, raw)
        except Exception as exc:
            return (
                f"Tool {name} completed but result persistence failed: "
                f"{type(exc).__name__}: {exc}"
            ), True
        # Update structured focus state after successful tool execution.
        # Wrapped defensively — a carryover bug must never break the turn.
        try:
            self._record_tool_carryover(
                name, inp if isinstance(inp, dict) else {}, raw, in_band_error
            )
        except Exception:
            pass
        return persisted, in_band_error

    async def _execute_tool_call(self, name: str, inp: dict) -> str:
        if name in ("enter_plan_mode", "exit_plan_mode"):
            return await self._execute_plan_mode_tool(name)
        if name == "agent":
            return await self._execute_agent_tool(inp)
        if name == "agent_result":
            return await self._execute_agent_result_tool(inp)
        if name == "skill":
            return await self._execute_skill_tool(inp)
        # Route MCP tool calls to the MCP manager
        if self._mcp_manager.is_mcp_tool(name):
            return await self._mcp_manager.call_tool(name, inp)
        return await execute_tool(name, inp, self._read_file_state)

    # ─── Skill fork mode ─────────────────────────────────────

    async def _execute_skill_tool(self, inp: dict) -> str:
        from .skills import execute_skill

        result = execute_skill(inp.get("skill_name", ""), inp.get("args", ""))
        if not result:
            return f"Unknown skill: {inp.get('skill_name', '')}"

        if result["context"] == "fork":
            tools = (
                [t for t in self.tools if t["name"] in result["allowed_tools"]]
                if result.get("allowed_tools")
                else [t for t in self.tools if t["name"] != "agent"]
            )
            print_sub_agent_start("skill-fork", inp.get("skill_name", ""))
            sub_agent = Agent(
                model=self.model,
                api_base=str(self._openai_client.base_url)
                if self.use_openai and self._openai_client
                else None,
                custom_system_prompt=result["prompt"],
                custom_tools=tools,
                is_sub_agent=True,
                permission_mode="plan"
                if self.permission_mode == "plan"
                else "bypassPermissions",
            )
            try:
                sub_result = await sub_agent.run_once(
                    inp.get("args") or "Execute this skill task."
                )
                self.total_input_tokens += sub_result["tokens"]["input"]
                self.total_output_tokens += sub_result["tokens"]["output"]
                print_sub_agent_end("skill-fork", inp.get("skill_name", ""))
                return sub_result["text"] or "(Skill produced no output)"
            except Exception as e:
                print_sub_agent_end("skill-fork", inp.get("skill_name", ""))
                return f"Skill fork error: {e}"

        return f'[Skill "{inp.get("skill_name", "")}" activated]\n\n{result["prompt"]}'

    # ─── Plan mode helpers ──────────────────────────────────────

    def _generate_plan_file_path(self) -> str:
        d = plans_dir()
        d.mkdir(parents=True, exist_ok=True)
        return str(d / f"plan-{self.session_id}.md")

    def _build_plan_mode_prompt(self) -> str:
        return f"""

# Plan Mode Active

Plan mode is active. You MUST NOT make any edits (except the plan file below), run non-readonly tools, or make any changes to the system.

## Plan File: {self._plan_file_path}
Write your plan incrementally to this file using write_file or edit_file. This is the ONLY file you are allowed to edit.

## Workflow
1. **Explore**: Read code to understand the task. Use read_file, list_files, grep_search.
2. **Design**: Design your implementation approach. Use the agent tool with type="plan" if the task is complex.
3. **Write Plan**: Write a structured plan to the plan file including:
   - **Context**: Why this change is needed
   - **Steps**: Implementation steps with critical file paths
   - **Verification**: How to test the changes
4. **Exit**: Call exit_plan_mode when your plan is ready for user review.

IMPORTANT: When your plan is complete, you MUST call exit_plan_mode. Do NOT ask the user to approve — exit_plan_mode handles that."""

    async def _execute_plan_mode_tool(self, name: str) -> str:
        if name == "enter_plan_mode":
            if self.permission_mode == "plan":
                return "Already in plan mode."
            self._pre_plan_mode = self.permission_mode
            self.permission_mode = "plan"
            self._plan_file_path = self._generate_plan_file_path()
            self._system_prompt = (
                self._base_system_prompt + self._build_plan_mode_prompt()
            )
            if self.use_openai and self._openai_messages:
                self._openai_messages[0]["content"] = self._system_prompt
            print_info(
                "Entered plan mode (read-only). Plan file: " + self._plan_file_path
            )
            return f"Entered plan mode. You are now in read-only mode.\n\nYour plan file: {self._plan_file_path}\nWrite your plan to this file. This is the only file you can edit.\n\nWhen your plan is complete, call exit_plan_mode."

        if name == "exit_plan_mode":
            if self.permission_mode != "plan":
                return "Not in plan mode."
            plan_content = "(No plan file found)"
            if self._plan_file_path and Path(self._plan_file_path).exists():
                plan_content = Path(self._plan_file_path).read_text()

            # Interactive approval flow
            if self._plan_approval_fn:
                result = await self._plan_approval_fn(plan_content)
                choice = result.get("choice", "manual-execute")

                if choice == "keep-planning":
                    feedback = result.get("feedback") or "Please revise the plan."
                    return (
                        f"User rejected the plan and wants to keep planning.\n\n"
                        f"User feedback: {feedback}\n\n"
                        f"Please revise your plan based on this feedback. When done, call exit_plan_mode again."
                    )

                # User approved — determine target mode
                if choice == "clear-and-execute":
                    target_mode = "acceptEdits"
                elif choice == "execute":
                    target_mode = "acceptEdits"
                else:  # manual-execute
                    target_mode = self._pre_plan_mode or "default"

                # Exit plan mode
                self.permission_mode = target_mode
                self._pre_plan_mode = None
                saved_plan_path = self._plan_file_path
                self._plan_file_path = None
                self._system_prompt = self._base_system_prompt
                if self.use_openai and self._openai_messages:
                    self._openai_messages[0]["content"] = self._system_prompt

                if choice == "clear-and-execute":
                    self._clear_history_keep_system()
                    self._context_cleared = True
                    print_info(
                        f"Plan approved. Context cleared, executing in {target_mode} mode."
                    )
                    return (
                        f"User approved the plan. Context was cleared. Permission mode: {target_mode}\n\n"
                        f"Plan file: {saved_plan_path}\n\n"
                        f"## Approved Plan:\n{plan_content}\n\n"
                        f"Proceed with implementation."
                    )

                print_info(f"Plan approved. Executing in {target_mode} mode.")
                return (
                    f"User approved the plan. Permission mode: {target_mode}\n\n"
                    f"## Approved Plan:\n{plan_content}\n\n"
                    f"Proceed with implementation."
                )

            # Fallback: no approval function (e.g. sub-agents)
            self.permission_mode = self._pre_plan_mode or "default"
            self._pre_plan_mode = None
            self._plan_file_path = None
            self._system_prompt = self._base_system_prompt
            if self.use_openai and self._openai_messages:
                self._openai_messages[0]["content"] = self._system_prompt
            print_info(
                "Exited plan mode. Restored to " + self.permission_mode + " mode."
            )
            return f"Exited plan mode. Permission mode restored to: {self.permission_mode}\n\n## Your Plan:\n{plan_content}"

        return f"Unknown plan mode tool: {name}"

    def _clear_history_keep_system(self) -> None:
        """Clear history but keep system prompt (used for clear-context plan approval)."""
        self._anthropic_messages = []
        self._openai_messages = []
        if self.use_openai:
            self._openai_messages.append(
                {"role": "system", "content": self._system_prompt}
            )
        self.last_input_token_count = 0

    # Default wall-clock ceiling for a single sub-agent. 300s balances "finish
    # a non-trivial exploration" against "catch a runaway loop before it burns
    # budget." Callers override per-call via the ``timeout_sec`` arg.
    _DEFAULT_SUBAGENT_TIMEOUT_SEC = 300.0

    async def _run_subagent(
        self, sub_agent: "Agent", prompt: str, timeout: float | None
    ) -> str:
        """Run a sub-agent with optional timeout, fold its token usage back into
        the parent, and return its text output. Raises ``asyncio.TimeoutError``
        on timeout (caller formats the error message)."""
        coro = sub_agent.run_once(prompt)
        if timeout is not None and timeout > 0:
            sub_result = await asyncio.wait_for(coro, timeout=timeout)
        else:
            sub_result = await coro
        self.total_input_tokens += sub_result["tokens"]["input"]
        self.total_output_tokens += sub_result["tokens"]["output"]
        return sub_result["text"] or "(Sub-agent produced no output)"

    def _resolve_subagent_permission_mode(self, override: str | None) -> str:
        """Pick a permission mode for a spawned sub-agent.

        Parent plan mode is sticky — sub-agents inherit ``plan`` so they can't
        mutate the filesystem while the user is still reviewing a plan. Custom
        agents may override outside of plan mode via their frontmatter.
        """
        if self.permission_mode == "plan":
            return "plan"
        if override:
            return override
        return "bypassPermissions"

    async def _execute_agent_tool(self, inp: dict) -> str:
        agent_type = inp.get("type", "general")
        description = inp.get("description", "sub-agent task")
        prompt = inp.get("prompt", "")
        run_in_background = bool(inp.get("run_in_background", False))
        # ``timeout_sec`` is accepted but never trusted blindly — clamp it
        # so the model can't set it to sys.maxsize and re-create the original
        # "no-timeout" footgun.
        timeout_raw = inp.get("timeout_sec")
        try:
            timeout = (
                float(timeout_raw)
                if timeout_raw is not None
                else self._DEFAULT_SUBAGENT_TIMEOUT_SEC
            )
        except (TypeError, ValueError):
            timeout = self._DEFAULT_SUBAGENT_TIMEOUT_SEC
        if timeout <= 0:
            timeout = self._DEFAULT_SUBAGENT_TIMEOUT_SEC

        config = get_sub_agent_config(agent_type)
        # Custom agent frontmatter can override model + permission_mode;
        # built-ins pass None and fall through to parent defaults.
        sub_model = config.get("model") or self.model
        perm_mode = self._resolve_subagent_permission_mode(
            config.get("permission_mode")
        )

        sub_agent = Agent(
            model=sub_model,
            api_base=str(self._openai_client.base_url)
            if self.use_openai and self._openai_client
            else None,
            custom_system_prompt=config["system_prompt"],
            custom_tools=config["tools"],
            is_sub_agent=True,
            permission_mode=perm_mode,
        )

        if run_in_background:
            # Launch without awaiting. Generate a short opaque handle the
            # model can later pass to ``agent_result`` to retrieve output.
            handle = f"bg-{uuid.uuid4().hex[:12]}"

            async def _runner() -> str:
                print_sub_agent_start(f"{agent_type}:bg", description)
                try:
                    text = await self._run_subagent(sub_agent, prompt, timeout)
                    print_sub_agent_end(f"{agent_type}:bg", description)
                    return text
                except asyncio.TimeoutError:
                    print_sub_agent_end(f"{agent_type}:bg", description)
                    return f"Sub-agent timed out after {timeout:.0f}s"
                except Exception as e:
                    print_sub_agent_end(f"{agent_type}:bg", description)
                    return f"Sub-agent error: {e}"

            task = asyncio.create_task(_runner())
            self._background_agent_tasks[handle] = task
            self._background_agent_meta[handle] = {
                "type": agent_type,
                "description": description,
                "timeout": timeout,
            }
            return (
                f"Sub-agent launched in background. task_id={handle}\n"
                f"Use the agent_result tool with this task_id to retrieve its output."
            )

        print_sub_agent_start(agent_type, description)
        try:
            text = await self._run_subagent(sub_agent, prompt, timeout)
            print_sub_agent_end(agent_type, description)
            return text
        except asyncio.TimeoutError:
            print_sub_agent_end(agent_type, description)
            return f"Sub-agent timed out after {timeout:.0f}s"
        except Exception as e:
            print_sub_agent_end(agent_type, description)
            return f"Sub-agent error: {e}"

    async def _execute_agent_result_tool(self, inp: dict) -> str:
        """Retrieve output from a background sub-agent launched via
        ``agent(run_in_background=True, ...)``."""
        task_id = (inp.get("task_id") or "").strip()
        if not task_id:
            return "agent_result requires a task_id."
        task = self._background_agent_tasks.get(task_id)
        if task is None:
            return f"Unknown task_id: {task_id}"

        wait_raw = inp.get("wait_sec")
        try:
            wait_sec = float(wait_raw) if wait_raw is not None else 0.0
        except (TypeError, ValueError):
            wait_sec = 0.0
        # Cap waits so the model can't accidentally stall its own loop forever.
        wait_sec = max(0.0, min(wait_sec, 120.0))

        if not task.done() and wait_sec > 0:
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=wait_sec)
            except asyncio.TimeoutError:
                pass

        if not task.done():
            meta = self._background_agent_meta.get(task_id, {})
            return (
                f"Sub-agent still running (type={meta.get('type', '?')}, "
                f"description={meta.get('description', '?')}). "
                f"Call agent_result again later, optionally with wait_sec to block briefly."
            )

        # Done — pop from maps so repeated polling doesn't leak memory,
        # but keep the result in the tool output so the model sees it once.
        self._background_agent_tasks.pop(task_id, None)
        self._background_agent_meta.pop(task_id, None)
        try:
            return task.result()
        except Exception as e:
            return f"Sub-agent crashed: {type(e).__name__}: {e}"

    @staticmethod
    def _block_to_dict(block) -> dict:
        """Convert an Anthropic content block to a plain dict for storage."""
        if block.type == "text":
            return {"type": "text", "text": block.text}
        if block.type == "tool_use":
            return {
                "type": "tool_use",
                "id": block.id,
                "name": block.name,
                "input": dict(block.input)
                if hasattr(block.input, "items")
                else block.input,
            }
        # Fallback
        return {"type": block.type}

    # ─── Unified chat loop ────────────────────────────────────────
    #
    # Provider-agnostic control flow. Every provider-specific
    # operation — message encoding, streaming, error classification,
    # reactive compaction, history mutation — routes through
    # :class:`providers.ProviderAdapter`. Adding a new provider means
    # writing one adapter; this loop does not change.

    def _inject_memory_prefetch(self, memory_prefetch: MemoryPrefetch | None) -> None:
        """Drain a settled memory prefetch into the last user message.

        Provider-aware because Anthropic user messages may be strings
        *or* block lists, while OpenAI user messages are always
        strings. Kept here (not on the adapter) because it's a
        one-time-per-turn step that reaches across the loop and would
        bloat the adapter surface for no gain.
        """
        if not (
            memory_prefetch and memory_prefetch.settled and not memory_prefetch.consumed
        ):
            return
        memory_prefetch.consumed = True
        try:
            memories = memory_prefetch.task.result()
        except Exception:
            return  # prefetch errors already logged
        if not memories:
            return
        injection_text = format_memories_for_injection(memories)
        messages = getattr(self, self._provider.messages_attr)
        last = messages[-1] if messages else None
        if last and last.get("role") == "user":
            content = last.get("content", "")
            if isinstance(content, str):
                last["content"] = content + "\n\n" + injection_text
            elif isinstance(content, list):
                # Anthropic-style block list.
                content.append({"type": "text", "text": injection_text})
            else:
                last["content"] = injection_text
        else:
            messages.append({"role": "user", "content": injection_text})
        for m in memories:
            self._already_surfaced_memories.add(m.path)
            self._session_memory_bytes += len(m.content.encode())

    async def _execute_tool_batch(
        self,
        tool_uses: list[ToolUseBlock],
        early_executions: dict[str, asyncio.Task],
    ) -> tuple[list[ToolResultBlock], str | None]:
        """Run all ``tool_uses`` for one assistant turn.

        Returns ``(results, context_break_text)``. When the second
        element is non-None the caller *must* append it as a plain
        user message (via ``adapter.encode_user``) and discard
        ``results`` — a tool cleared the context mid-turn, invalidating
        the ``tool_use`` ids the results would otherwise reference.

        Strategy:

        1. **Permission phase**: classify every tool (allowed / denied
           / confirm-needed) up front. Denied tools skip execution but
           still need a matching result entry.
        2. **Batching phase**: walk the classified list, opening a new
           concurrent batch whenever a safe+allowed tool follows one
           that's also safe+allowed, and a serial batch otherwise.
           Pre-started tools (``early_executions``) count as "safe"
           so they merge into the concurrent group.
        3. **Execution phase**: run each batch — ``asyncio.gather``
           for concurrent batches, serial await for the rest — and
           short-circuit the whole thing the instant
           ``self._context_cleared`` fires.
        """
        results: list[ToolResultBlock] = []
        checked: list[dict] = []

        for tu in tool_uses:
            if self._aborted:
                break
            inp = tu.input if isinstance(tu.input, dict) else dict(tu.input)
            print_tool_call(tu.name, inp)

            early = early_executions.get(tu.id)
            if early is not None:
                checked.append({"tu": tu, "inp": inp, "allowed": True, "early": early})
                continue

            perm = check_permission(
                tu.name, inp, self.permission_mode, self._plan_file_path
            )
            if perm["action"] == "deny":
                print_info(f"Denied: {perm.get('message', '')}")
                checked.append(
                    {
                        "tu": tu,
                        "inp": inp,
                        "allowed": False,
                        "deny_msg": f"Action denied: {perm.get('message', '')}",
                    }
                )
                continue
            if (
                perm["action"] == "confirm"
                and perm.get("message")
                and perm["message"] not in self._confirmed_paths
            ):
                confirmed = await self._confirm_dangerous(perm["message"])
                if not confirmed:
                    checked.append(
                        {
                            "tu": tu,
                            "inp": inp,
                            "allowed": False,
                            "deny_msg": "User denied this action.",
                        }
                    )
                    continue
                self._confirmed_paths.add(perm["message"])
            checked.append({"tu": tu, "inp": inp, "allowed": True, "early": None})

        batches: list[dict] = []
        for ct in checked:
            is_safe = ct["allowed"] and (
                ct.get("early") is not None
                or is_concurrency_safe_tool(ct["tu"].name, ct["inp"])
            )
            if is_safe and batches and batches[-1]["concurrent"]:
                batches[-1]["items"].append(ct)
            else:
                batches.append({"concurrent": is_safe, "items": [ct]})

        async def _run_one(ct: dict) -> tuple[str, bool]:
            res, is_err = await self._run_tool_safely(
                ct["tu"].name, ct["inp"], precomputed=ct.get("early")
            )
            print_tool_result(ct["tu"].name, res)
            return res, is_err

        for batch in batches:
            if self._aborted:
                break
            if batch["concurrent"] and len(batch["items"]) > 1:
                # Gather with return_exceptions so one crashing tool
                # doesn't cancel siblings and leave tool_use ids
                # unmatched on the next turn.
                outcomes = await asyncio.gather(
                    *[_run_one(c) for c in batch["items"]],
                    return_exceptions=True,
                )
                for c, out in zip(batch["items"], outcomes):
                    if isinstance(out, BaseException):
                        results.append(
                            ToolResultBlock(
                                id=c["tu"].id,
                                output=(
                                    f"Tool {c['tu'].name} failed: "
                                    f"{type(out).__name__}: {out}"
                                ),
                                is_error=True,
                            )
                        )
                    else:
                        res, is_err = out
                        results.append(
                            ToolResultBlock(id=c["tu"].id, output=res, is_error=is_err)
                        )
                continue

            for ct in batch["items"]:
                if not ct["allowed"]:
                    results.append(
                        ToolResultBlock(id=ct["tu"].id, output=ct["deny_msg"])
                    )
                    continue
                res, is_err = await _run_one(ct)
                if self._context_cleared:
                    self._context_cleared = False
                    return results, res
                results.append(
                    ToolResultBlock(id=ct["tu"].id, output=res, is_error=is_err)
                )

        return results, None

    async def _chat_loop(self, user_message: str) -> None:
        """Provider-agnostic chat loop.

        Every provider-specific operation routes through
        ``self._provider``:

        * ``encode_user`` / ``append_tool_results``: message shaping.
        * ``run_stream`` + ``finalize_turn``: streaming + normalized
          view of the assistant turn.
        * ``append_assistant_turn``: persist the raw response back into
          history so the next turn passes provider validation.
        * ``is_prompt_too_long`` / ``is_output_token_limit``: recovery
          classification. ``compact`` runs the provider's reactive
          compaction path.
        """
        adapter = self._provider
        messages = getattr(self, adapter.messages_attr)
        messages.append(adapter.encode_user(user_message))

        reactive_compact_attempted = False

        memory_prefetch: MemoryPrefetch | None = None
        if not self.is_sub_agent:
            sq = self._build_side_query()
            if sq:
                memory_prefetch = start_memory_prefetch(
                    user_message,
                    sq,
                    self._already_surfaced_memories,
                    self._session_memory_bytes,
                )

        while True:
            if self._aborted:
                break

            self._run_compression_pipeline()
            self._inject_memory_prefetch(memory_prefetch)

            if not self.is_sub_agent:
                start_spinner()

            # Streaming prefetch: only Anthropic acts on this today,
            # OpenAIAdapter accepts the kwarg and ignores it.
            early_executions: dict[str, asyncio.Task] = {}

            def _on_tool_block(block: dict) -> None:
                if is_concurrency_safe_tool(block["name"], block["input"]):
                    perm = check_permission(
                        block["name"],
                        block["input"],
                        self.permission_mode,
                        self._plan_file_path,
                    )
                    if perm["action"] == "allow":
                        early_executions[block["id"]] = asyncio.create_task(
                            self._execute_tool_call(block["name"], block["input"])
                        )

            try:
                await adapter.run_stream(self, on_tool_block_complete=_on_tool_block)
            except Exception as exc:
                is_cap_err, limit = adapter.is_output_token_limit(exc)
                if is_cap_err:
                    current = self._output_token_cap_override or _get_max_output_tokens(
                        self.model
                    )
                    if limit is not None and limit < current:
                        if not self.is_sub_agent:
                            stop_spinner()
                        self._output_token_cap_override = limit
                        print_info(
                            f"Provider rejected max_tokens; retrying with cap {limit}."
                        )
                        continue
                if not reactive_compact_attempted and adapter.is_prompt_too_long(exc):
                    reactive_compact_attempted = True
                    if not self.is_sub_agent:
                        stop_spinner()
                    print_info("Prompt too long; compacting conversation and retrying.")
                    await adapter.compact(self)
                    continue
                raise

            if not self.is_sub_agent:
                stop_spinner()

            self.last_api_call_time = time.time()
            turn = adapter.finalize_turn(self)
            self.total_input_tokens += turn.usage.input_tokens
            self.total_output_tokens += turn.usage.output_tokens
            self.last_input_token_count = turn.usage.input_tokens

            adapter.append_assistant_turn(self)

            if not turn.tool_uses:
                if not self.is_sub_agent:
                    print_cost(self.total_input_tokens, self.total_output_tokens)
                break

            self.current_turns += 1
            budget = self._check_budget()
            if budget["exceeded"]:
                print_info(f"Budget exceeded: {budget['reason']}")
                break

            results, context_break_text = await self._execute_tool_batch(
                turn.tool_uses, early_executions
            )
            if context_break_text is not None:
                # Context was cleared mid-turn: tool_use ids are gone,
                # so the last tool's output becomes a plain user
                # message instead of a tool_result batch.
                messages.append(adapter.encode_user(context_break_text))
            else:
                adapter.append_tool_results(self, results)

            self._context_cleared = False
            await self._check_and_compact()

    # ─── Shared ──────────────────────────────────────────────────

    async def _confirm_dangerous(self, command: str) -> bool:
        print_confirmation(command)
        if self.confirm_fn:
            return await self.confirm_fn(command)
        # Fallback: blocking input
        try:
            answer = input("  Allow? (y/n): ")
            return answer.lower().startswith("y")
        except EOFError:
            return False
