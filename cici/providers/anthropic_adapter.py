"""Anthropic (Claude) provider adapter.

Implements :class:`providers.base.ProviderAdapter` against Anthropic's
``messages.stream`` API — streaming execution, error classification,
message encoding, and history mutation used by ``agent._chat_loop``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from .base import (
    AssistantTurn,
    ProviderAdapter,
    ToolResultBlock,
    ToolUseBlock,
    UsageReport,
    _stringify_tool_output,
)

if TYPE_CHECKING:
    from ..agent import Agent


class AnthropicAdapter(ProviderAdapter):
    """Adapter for the Anthropic ``messages.stream`` API."""

    messages_attr = "_anthropic_messages"

    # ---- Error classification ---------------------------------------

    def is_prompt_too_long(self, exc: Exception) -> bool:
        # Import lazily so the adapter file has no module-level
        # dependency on ``agent``; keeps import ordering clean and the
        # regex tables live exactly once in ``agent.py``.
        from ..agent import _is_prompt_too_long_error

        return _is_prompt_too_long_error(exc)

    def is_output_token_limit(self, exc: Exception) -> tuple[bool, int | None]:
        from ..agent import (
            _extract_completion_token_limit,
            _is_completion_token_limit_error,
        )

        if not _is_completion_token_limit_error(exc):
            return False, None
        return True, _extract_completion_token_limit(exc)

    # ---- Message encoding -------------------------------------------

    def encode_user(self, content: str) -> dict:
        """Anthropic accepts plain strings for user turns."""
        return {"role": "user", "content": content}

    def encode_tool_results(self, results: list[ToolResultBlock]) -> dict:
        """Pack all tool results into a single user-role message.

        One ``tool_result`` entry per tool, ``is_error`` omitted when
        false to keep the Anthropic schema minimal.
        """
        blocks: list[dict] = []
        for r in results:
            block: dict = {
                "type": "tool_result",
                "tool_use_id": r.id,
                "content": _stringify_tool_output(r.output),
            }
            if r.is_error:
                block["is_error"] = True
            blocks.append(block)
        return {"role": "user", "content": blocks}

    # ---- Streaming --------------------------------------------------

    async def run_stream(
        self,
        agent: "Agent",
        *,
        on_tool_block_complete: Callable[[dict], None] | None = None,
    ) -> Any:
        """Stream one Anthropic turn and return the final ``Message``.

        ``on_tool_block_complete`` fires the instant a ``tool_use``
        content block closes (``content_block_stop``), giving the loop
        a chance to prefetch concurrency-safe tools while the model is
        still generating its remaining output.
        """
        from ..agent import _get_max_output_tokens, _with_retry
        from ..tools import get_active_tool_definitions
        from ..ui import stop_spinner

        async def _do() -> Any:
            max_output = _get_max_output_tokens(agent.model)
            if agent._output_token_cap_override is not None:
                max_output = min(max_output, agent._output_token_cap_override)
            non_thinking_cap = 16384
            if agent._output_token_cap_override is not None:
                non_thinking_cap = min(
                    non_thinking_cap, agent._output_token_cap_override
                )
            create_params: dict[str, Any] = {
                "model": agent.model,
                "max_tokens": (
                    max_output
                    if agent._thinking_mode != "disabled"
                    else non_thinking_cap
                ),
                "system": agent._effective_system_prompt(),
                "tools": get_active_tool_definitions(agent.tools),
                "messages": agent._anthropic_messages,
            }

            if agent._thinking_mode in ("adaptive", "enabled"):
                create_params["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": max_output - 1,
                }

            first_text = True
            # Track in-flight tool_use blocks by index for streaming execution
            tool_blocks_by_index: dict[int, dict] = {}

            async with agent._anthropic_client.messages.stream(
                **create_params
            ) as stream:
                async for event in stream:
                    if not hasattr(event, "type"):
                        continue

                    if event.type == "content_block_start":
                        cb = getattr(event, "content_block", None)
                        if cb and getattr(cb, "type", None) == "tool_use":
                            tool_blocks_by_index[event.index] = {
                                "id": cb.id,
                                "name": cb.name,
                                "input_json": "",
                            }

                    elif event.type == "content_block_delta":
                        delta = event.delta
                        if hasattr(delta, "text"):
                            if first_text:
                                stop_spinner()
                                agent._emit_text("\n")
                                first_text = False
                            agent._emit_text(delta.text)
                        elif hasattr(delta, "thinking"):
                            if first_text:
                                stop_spinner()
                                agent._emit_text("\n  [thinking] ")
                                first_text = False
                            agent._emit_text(delta.thinking)
                        elif hasattr(delta, "partial_json"):
                            tb = tool_blocks_by_index.get(event.index)
                            if tb:
                                tb["input_json"] += delta.partial_json

                    elif event.type == "content_block_stop":
                        tb = tool_blocks_by_index.pop(event.index, None)
                        if tb and on_tool_block_complete:
                            import json as _json

                            try:
                                parsed = _json.loads(tb["input_json"] or "{}")
                            except Exception:
                                parsed = {}
                            on_tool_block_complete(
                                {
                                    "type": "tool_use",
                                    "id": tb["id"],
                                    "name": tb["name"],
                                    "input": parsed,
                                }
                            )

                final_message = await stream.get_final_message()

            # Filter out thinking blocks — they're internal-only and
            # must not be echoed back on the next turn.
            final_message.content = [
                b for b in final_message.content if b.type != "thinking"
            ]
            # Stash on the agent so ``finalize_turn`` and
            # ``append_assistant_turn`` can reach it.
            agent._last_anthropic_response = final_message
            return final_message

        return await _with_retry(_do)

    # ---- Turn normalization ----------------------------------------

    def finalize_turn(self, agent: "Agent") -> AssistantTurn:
        """Normalize the most recent assistant response on ``agent``."""
        response = getattr(agent, "_last_anthropic_response", None)
        if response is None:
            return AssistantTurn()
        text_parts: list[str] = []
        tool_uses: list[ToolUseBlock] = []
        for block in getattr(response, "content", []) or []:
            btype = getattr(block, "type", None)
            if btype == "text":
                text_parts.append(getattr(block, "text", ""))
            elif btype == "tool_use":
                tool_uses.append(
                    ToolUseBlock(
                        id=getattr(block, "id", ""),
                        name=getattr(block, "name", ""),
                        input=getattr(block, "input", {}) or {},
                    )
                )
        usage = getattr(response, "usage", None)
        usage_report = UsageReport(
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
        )
        return AssistantTurn(
            text="".join(text_parts),
            tool_uses=tool_uses,
            usage=usage_report,
        )

    async def compact(self, agent: "Agent") -> None:
        await agent._compact_anthropic()

    # ---- History mutation -------------------------------------------

    def append_assistant_turn(self, agent: "Agent") -> None:
        """Append the most recent assistant response to ``_anthropic_messages``.

        Reconstructs a plain-dict view of each content block via
        ``Agent._block_to_dict`` so the entire history stays JSON-able
        (needed for compaction / session save).
        """
        response = getattr(agent, "_last_anthropic_response", None)
        if response is None:
            return
        from ..agent import Agent

        agent._anthropic_messages.append(
            {
                "role": "assistant",
                "content": [Agent._block_to_dict(b) for b in response.content],
            }
        )

    def append_tool_results(
        self, agent: "Agent", results: list[ToolResultBlock]
    ) -> None:
        """Pack results into one ``user`` message and append."""
        if not results:
            return
        agent._anthropic_messages.append(self.encode_tool_results(results))
