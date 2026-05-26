"""Provider-agnostic abstractions shared by the coding-agent chat loop.

This module intentionally has **zero** dependencies on ``agent.py``,
``tools.py`` or SDK clients. It only declares the normalized types and
the ``ProviderAdapter`` protocol. Concrete adapters live in sibling
modules (``anthropic_adapter.py`` / ``openai_adapter.py``) and import
the SDKs they need; ``agent.py`` imports everything through
``providers.__init__``.

Keeping the package split this way means:

* Tests can import the types without pulling in a real Anthropic or
  OpenAI client.
* Future providers (Gemini, xAI, local OpenAI-compatible servers) plug
  in without touching ``agent.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, Union, runtime_checkable

if TYPE_CHECKING:
    # Imported only for type hints; avoids a circular import at runtime.
    from ..agent import Agent

# ─── Normalized value types ─────────────────────────────────


@dataclass(frozen=True)
class ToolUseBlock:
    """A single tool invocation requested by the model."""

    id: str
    name: str
    input: dict


@dataclass(frozen=True)
class ToolResultBlock:
    """Result of running one ``ToolUseBlock``."""

    id: str
    output: str
    is_error: bool = False


@dataclass(frozen=True)
class UsageReport:
    """Token accounting for one assistant turn."""

    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class AssistantTurn:
    """Normalized view of one assistant turn.

    Providers differ wildly on how they represent assistant messages
    (Anthropic content blocks vs OpenAI ``tool_calls``); the loop only
    needs the text + tool uses + usage, which live here.
    """

    text: str = ""
    tool_uses: list[ToolUseBlock] = field(default_factory=list)
    usage: UsageReport = field(default_factory=UsageReport)


# ─── Stream events ──────────────────────────────────────────


@dataclass(frozen=True)
class TextDelta:
    """A token (or chunk of tokens) of assistant text."""

    text: str


@dataclass(frozen=True)
class ToolUseReady:
    """A fully assembled tool_use block emitted during streaming.

    Anthropic adapters yield this as soon as the ``content_block_stop``
    event closes a ``tool_use`` block, enabling the loop to prefetch
    concurrency-safe tools before streaming finishes. OpenAI adapters
    may choose to yield these only after the stream closes (OpenAI
    streams ``tool_calls.arguments`` piecewise but does not mark a
    per-tool boundary reliably).
    """

    block: ToolUseBlock


StreamEvent = Union[TextDelta, ToolUseReady]


# ─── Provider adapter protocol ──────────────────────────────


@runtime_checkable
class ProviderAdapter(Protocol):
    """Provider-specific plumbing the unified chat loop depends on.

    Every method is intentionally small and free of side-effects on
    the agent's *logical* state (turn counters, context_cleared flags,
    etc.). Only ``stream`` / ``finalize_turn`` / ``compact`` are allowed
    to mutate ``agent.<messages_attr>``, and they do so in ways
    ``_chat_loop`` would otherwise do itself.
    """

    #: Name of the attribute on ``Agent`` holding the provider's native
    #: message history (``"_anthropic_messages"`` or
    #: ``"_openai_messages"``). Exposed as a string so the loop can
    #: remain ignorant of provider specifics.
    messages_attr: str

    # ---- Error classification ---------------------------------------

    def is_prompt_too_long(self, exc: Exception) -> bool:
        """True if ``exc`` means "prompt exceeds context window"."""
        ...

    def is_output_token_limit(self, exc: Exception) -> tuple[bool, int | None]:
        """True if ``exc`` means "max_tokens too large", with the parsed cap."""
        ...

    # ---- Message encoding -------------------------------------------

    def encode_user(self, content: str) -> dict:
        """Build a provider-native user message dict."""
        ...

    def encode_tool_results(self, results: list[ToolResultBlock]) -> dict:
        """Build the provider-native message that carries tool results.

        Anthropic wraps them all in a single ``{"role":"user","content":[...]}``
        message; OpenAI emits one ``{"role":"tool",...}`` message per
        result. The loop only ever calls this once per turn and appends
        whatever is returned *directly* to ``messages`` — OpenAI's
        return therefore may be a ``list[dict]`` in a future revision,
        but today both adapters return a single dict the loop extends
        vs. appends based on type.
        """
        ...

    # ---- Streaming --------------------------------------------------

    async def run_stream(
        self, agent: "Agent", *, on_tool_block_complete: Any | None = None
    ) -> Any:
        """Execute one streaming turn and return the provider-native response.

        The response is also stashed on ``agent._last_*_response`` so
        :meth:`finalize_turn` and :meth:`append_assistant_turn` can
        reach it without the loop threading it through.

        ``on_tool_block_complete`` fires per completed ``tool_use``
        block during streaming; only the Anthropic adapter emits
        these today (OpenAI's delta stream has no reliable per-tool
        boundary), but both adapters accept the kwarg so the loop
        calls them uniformly.
        """
        ...

    def finalize_turn(self, agent: "Agent") -> AssistantTurn:
        """Normalize the most recent assistant response into an ``AssistantTurn``."""
        ...

    # ---- History mutation -------------------------------------------

    def append_assistant_turn(self, agent: "Agent") -> None:
        """Append the most recent assistant response to the provider's history.

        The unified ``_chat_loop`` introduced in PR5c only sees the
        normalized :class:`AssistantTurn`; the provider-native message
        (Anthropic content blocks vs OpenAI ``message`` dict) is still
        what must land in ``agent.<messages_attr>`` so subsequent turns
        are accepted by the API. Implementations read from the
        ``_last_*_response`` slot populated by :meth:`run_stream`.
        """
        ...

    def append_tool_results(
        self, agent: "Agent", results: list[ToolResultBlock]
    ) -> None:
        """Append tool-result messages to the provider's history.

        Anthropic packs all results into a single user-role message;
        OpenAI emits one ``tool``-role message per result. Hiding that
        asymmetry here lets the loop treat tool results as a single
        operation, freeing it from the ``_multi`` envelope sentinel
        that :meth:`encode_tool_results` still returns for PR5b
        consumers.
        """
        ...

    # ---- Compaction -------------------------------------------------

    async def compact(self, agent: "Agent") -> None:
        """Run the provider's reactive-compaction path (prompt-too-long recovery)."""
        ...


# ─── Small shared utilities ─────────────────────────────────


def _stringify_tool_output(output: Any) -> str:
    """Best-effort coercion of arbitrary tool output to a string.

    Both providers want a string ``content`` for tool results; this
    helper centralises the rule so adapters do not reinvent it.
    """
    if isinstance(output, str):
        return output
    try:
        return str(output)
    except Exception:
        return repr(output)
