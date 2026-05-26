"""OpenAI-compatible provider adapter.

Implements :class:`providers.base.ProviderAdapter` against the OpenAI
``chat.completions`` streaming API — covers the vendor client, Azure
OpenAI, and any local server exposing the same schema (llama.cpp,
vLLM, LiteLLM proxy, etc.).
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


class OpenAIAdapter(ProviderAdapter):
    """Adapter for any OpenAI-compatible ``chat.completions`` endpoint."""

    messages_attr = "_openai_messages"

    # ---- Error classification ---------------------------------------
    #
    # OpenAI and Anthropic surface provider-side errors with the same
    # ``Exception`` shape by the time they reach us (string
    # interpolation catches both SDKs), so the two classifiers are
    # identical today. Keeping them per-adapter leaves room for
    # provider-specific tweaks without retrofitting.

    def is_prompt_too_long(self, exc: Exception) -> bool:
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
        return {"role": "user", "content": content}

    def encode_tool_results(self, results: list[ToolResultBlock]) -> dict:
        """Return a "_multi" envelope carrying provider-native results.

        OpenAI needs one message per tool result, unlike Anthropic's
        single batched message. The loop today uses
        :meth:`append_tool_results` instead, which handles the
        provider split directly; this method remains for callers that
        want the raw encoded form (e.g. the normalized-types tests).
        """
        return {
            "role": "_multi",
            "messages": [
                {
                    "role": "tool",
                    "tool_call_id": r.id,
                    "content": _stringify_tool_output(r.output),
                }
                for r in results
            ],
        }

    # ---- Streaming --------------------------------------------------

    async def run_stream(
        self,
        agent: "Agent",
        *,
        on_tool_block_complete: Callable[[dict], None] | None = None,
    ) -> Any:
        """Stream one OpenAI turn and return the assembled response dict.

        ``on_tool_block_complete`` is accepted for interface symmetry
        with :class:`AnthropicAdapter` but ignored today — the OpenAI
        delta stream does not expose a reliable per-tool boundary
        (arguments arrive piecewise with no ``content_block_stop``),
        so there is no safe trigger point.
        """
        del on_tool_block_complete  # See docstring.

        from ..agent import _to_openai_tools, _with_retry
        from ..tools import get_active_tool_definitions
        from ..ui import stop_spinner

        async def _do() -> dict:
            # Refresh system message with current focus state each turn.
            if (
                agent._openai_messages
                and agent._openai_messages[0].get("role") == "system"
            ):
                agent._openai_messages[0]["content"] = agent._effective_system_prompt()
            stream_kwargs: dict[str, Any] = dict(
                model=agent.model,
                tools=_to_openai_tools(get_active_tool_definitions(agent.tools)),
                messages=agent._openai_messages,
                stream=True,
                stream_options={"include_usage": True},
            )
            # Only send max_tokens once a provider has forced a cap on
            # us, so users with permissive providers aren't limited
            # unnecessarily.
            if agent._output_token_cap_override is not None:
                stream_kwargs["max_tokens"] = agent._output_token_cap_override
            stream = await agent._openai_client.chat.completions.create(**stream_kwargs)

            content = ""
            first_text = True
            tool_calls: dict[int, dict] = {}
            finish_reason = ""
            usage = None

            async for chunk in stream:
                if chunk.usage:
                    usage = {
                        "prompt_tokens": chunk.usage.prompt_tokens,
                        "completion_tokens": chunk.usage.completion_tokens,
                    }

                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta

                if delta and delta.content:
                    if first_text:
                        stop_spinner()
                        agent._emit_text("\n")
                        first_text = False
                    agent._emit_text(delta.content)
                    content += delta.content

                if delta and delta.tool_calls:
                    for tc in delta.tool_calls:
                        existing = tool_calls.get(tc.index)
                        if existing:
                            if tc.function and tc.function.arguments:
                                existing["arguments"] += tc.function.arguments
                        else:
                            tool_calls[tc.index] = {
                                "id": tc.id or "",
                                "name": (tc.function.name if tc.function else "") or "",
                                "arguments": (
                                    tc.function.arguments if tc.function else ""
                                )
                                or "",
                            }

                if chunk.choices[0].finish_reason:
                    finish_reason = chunk.choices[0].finish_reason

            assembled = None
            if tool_calls:
                assembled = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {"name": tc["name"], "arguments": tc["arguments"]},
                    }
                    for _, tc in sorted(tool_calls.items())
                ]

            response = {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": content or None,
                            "tool_calls": assembled,
                        },
                        "finish_reason": finish_reason or "stop",
                    }
                ],
                "usage": usage or {"prompt_tokens": 0, "completion_tokens": 0},
            }
            # Stash on the agent so ``finalize_turn`` and
            # ``append_assistant_turn`` can reach it.
            agent._last_openai_response = response
            return response

        return await _with_retry(_do)

    def finalize_turn(self, agent: "Agent") -> AssistantTurn:
        """Normalize the most recent response dict stashed on ``agent``."""
        response = getattr(agent, "_last_openai_response", None)
        if not response:
            return AssistantTurn()
        message = (response.get("choices") or [{}])[0].get("message") or {}
        tool_uses: list[ToolUseBlock] = []
        for tc in message.get("tool_calls") or []:
            fn = tc.get("function") or {}
            arguments = fn.get("arguments") or "{}"
            try:
                import json as _json

                parsed = (
                    _json.loads(arguments) if isinstance(arguments, str) else arguments
                )
            except Exception:
                parsed = {}
            tool_uses.append(
                ToolUseBlock(
                    id=tc.get("id", ""),
                    name=fn.get("name", ""),
                    input=parsed or {},
                )
            )
        usage = response.get("usage") or {}
        return AssistantTurn(
            text=message.get("content") or "",
            tool_uses=tool_uses,
            usage=UsageReport(
                input_tokens=int(usage.get("prompt_tokens") or 0),
                output_tokens=int(usage.get("completion_tokens") or 0),
            ),
        )

    async def compact(self, agent: "Agent") -> None:
        await agent._compact_openai()

    # ---- History mutation -------------------------------------------

    def append_assistant_turn(self, agent: "Agent") -> None:
        """Append the most recent assistant response to ``_openai_messages``.

        OpenAI's ``message`` dict can be round-tripped straight back
        into the next request, so no per-block conversion is needed.
        """
        response = getattr(agent, "_last_openai_response", None)
        if not response:
            return
        message = (response.get("choices") or [{}])[0].get("message") or {}
        agent._openai_messages.append(message)

    def append_tool_results(
        self, agent: "Agent", results: list[ToolResultBlock]
    ) -> None:
        """Append one ``tool``-role message per result."""
        for r in results:
            agent._openai_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": r.id,
                    "content": _stringify_tool_output(r.output),
                }
            )
