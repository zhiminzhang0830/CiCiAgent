"""Provider adapters for the coding-agent chat loop.

The package exposes a provider-agnostic interface (``ProviderAdapter``)
plus concrete implementations for Anthropic and OpenAI-compatible
endpoints. ``agent._chat_loop`` is provider-agnostic; every
provider-specific operation (streaming, message encoding, error
classification, compaction) routes through an adapter chosen at
``Agent`` init.
"""

from .anthropic_adapter import AnthropicAdapter
from .base import (
    AssistantTurn,
    ProviderAdapter,
    TextDelta,
    ToolResultBlock,
    ToolUseBlock,
    ToolUseReady,
    UsageReport,
)
from .openai_adapter import OpenAIAdapter

__all__ = [
    "AnthropicAdapter",
    "AssistantTurn",
    "OpenAIAdapter",
    "ProviderAdapter",
    "TextDelta",
    "ToolResultBlock",
    "ToolUseBlock",
    "ToolUseReady",
    "UsageReport",
]
