"""Multi-tier context compression.

Five layers, applied in order from cheapest to most aggressive:

    Tier 1 — budget_tool_results_*  (per-message aggregate > 200K)
        Pick the largest non-pinned, non-already-processed tool_result
        blocks in a message and persist them to disk (replacing content
        with preview + path) until the message's aggregate size is
        within budget.

    Tier 2 — snip_stale_results_*   (utilization > SNIP_THRESHOLD)
        Replace duplicate/old snippable tool results with a placeholder.

    Tier 3 — microcompact_*         (idle > MICROCOMPACT_IDLE_S)
        Clear all but the most recent tool results.

    Tier 4 — collapse_context_*     (utilization > CONTEXT_COLLAPSE_THRESHOLD)
        Deterministic head+tail truncation of oversized text/tool_result
        blocks in older history. Last cheap resort before tier 0.

    Tier 0 — compact_conversation_* (utilization > 0.85)
        LLM-generated summary that replaces the whole history.

The tier 1-4 functions are pure with respect to their inputs and mutate the
message list in place, matching the prior behaviour of the Agent methods they
replaced. Tier 0 is async because it issues a summarization request; it
returns a fresh message list (the Agent swaps its stored list with the
return value).
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from typing import Any

from .paths import artifacts_dir

logger = logging.getLogger(__name__)

# ─── Constants ──────────────────────────────────────────────

SNIPPABLE_TOOLS = {"read_file", "grep_search", "list_files", "run_shell"}
SNIP_PLACEHOLDER = "[Content snipped - re-read if needed]"
OLD_RESULT_PLACEHOLDER = "[Old result cleared]"
SNIP_THRESHOLD = 0.60
MICROCOMPACT_IDLE_S = 5 * 60  # 5 minutes
KEEP_RECENT_RESULTS = 3
FULL_COMPACT_THRESHOLD = 0.85

# Tier 4 (context collapse) parameters. Mirrors OpenHarness'
# try_context_collapse: deterministic head+tail truncation of oversized
# text / tool_result content in older history. Runs between tier 3 and
# tier 0 so that when tier 2/3 aren't enough we still try a cheap
# LLM-free pass before paying for summarization.
#
# Char-based (not token-based) to match OpenHarness; CJK-heavy sessions
# will hit the limit at fewer tokens than English, which is fine — this
# is a safety net, not a fine-tuned budget.
CONTEXT_COLLAPSE_THRESHOLD = 0.75
CONTEXT_COLLAPSE_TEXT_CHAR_LIMIT = 2_400
CONTEXT_COLLAPSE_HEAD_CHARS = 900
CONTEXT_COLLAPSE_TAIL_CHARS = 500
CONTEXT_COLLAPSE_PRESERVE_RECENT = 6

# Tier 1 (per-message tool-result budget) parameters.
#
# PINNED_TOOLS — tool names whose results are NEVER budgeted / truncated.
# read_file is pinned to avoid a persist -> read_file -> persist loop: if
# we truncate a read_file result, the model will re-read the same path to
# recover, which produces an identical oversized result on the next turn
# that gets truncated again, forever. read_file already self-bounds via
# offset/limit slicing and its own MAX_FILE_SIZE_BYTES, so it does not
# need a second-line defence here.
PINNED_TOOLS: frozenset[str] = frozenset({"read_file"})

# Aggregate budget per user message (Anthropic) or per consecutive tool-role
# message group (OpenAI). 200K chars is a heuristic ceiling that keeps a
# single turn from blowing through the context window when many parallel
# tool calls each return a moderate payload — triggers are per-message
# aggregate, not global utilization, so a turn with 10 parallel tool
# results each at 25K (none individually oversized) will still be caught.
PER_MESSAGE_BUDGET_CHARS = 200_000

# ─── Artifact persistence (public; called by both agent.py per-tool layer
#     and the tier-1 budget inside this module) ───────────────
#
# When a tool_result is too large to keep inline, we write the full content
# to disk and replace it in-context with a preview + absolute path. The
# model can then read the artifact back with `read_file` on demand —
# information is not lost.
ARTIFACT_DIR = artifacts_dir()
TOOL_OUTPUT_PREVIEW_CHARS = 8_000

# Marker prefix that `persist_tool_result` writes at the top of replaced
# content. Also used by `_is_already_processed` to detect blocks that
# have already been persisted (prevents double-persist + prompt-cache
# instability).
PERSIST_MARKER_PREFIX = "[Tool output truncated]"


def _safe_artifact_filename(tool_name: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", tool_name.strip())
    return (normalized or "tool")[:80]


def persist_tool_result(tool_name: str, result: str) -> str:
    """Unconditionally persist `result` to disk, return a preview+path
    replacement suitable for use as the in-context tool_result content.

    Used from two call sites with different triggers but identical output:
      1. agent.py:_persist_large_result — per-tool at execution time, when
         a single result exceeds the inline limit.
      2. compression.budget_tool_results_* — per-message at budget time,
         when the aggregate size of a message's tool_result blocks
         exceeds PER_MESSAGE_BUDGET_CHARS.
    """
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    filename = (
        f"{time.strftime('%Y%m%d-%H%M%S')}-"
        f"{_safe_artifact_filename(tool_name)}-"
        f"{uuid.uuid4().hex[:12]}.txt"
    )
    filepath = ARTIFACT_DIR / filename
    filepath.write_text(result, encoding="utf-8", errors="replace")

    preview = result[:TOOL_OUTPUT_PREVIEW_CHARS]
    omitted = max(0, len(result) - len(preview))
    size_kb = len(result) / 1024
    total_lines = result.count("\n") + 1
    preview_lines = preview.count("\n") + 1

    header = (
        f"{PERSIST_MARKER_PREFIX}\n"
        f"Tool: {tool_name}\n"
        f"Original size: {len(result)} chars ({size_kb:.1f} KB, {total_lines} lines)\n"
        f"Full output saved to: {filepath}\n"
        f"Inline preview: first {len(preview)} chars"
    )
    if omitted:
        header += (
            f" ({omitted} chars omitted — use read_file on the path above "
            "to see more)"
        )
    header += f"\nPreview lines: {preview_lines}"
    return f"{header}\n\nPreview:\n{preview}"


# ─── Utilization helper ─────────────────────────────────────


def utilization(last_input_tokens: int, effective_window: int) -> float:
    if not effective_window:
        return 0.0
    return last_input_tokens / effective_window


# ─── Shared helpers ─────────────────────────────────────────


def find_tool_use_by_id(
    anthropic_messages: list[dict], tool_use_id: str
) -> dict | None:
    for msg in anthropic_messages:
        if msg.get("role") != "assistant" or not isinstance(msg.get("content"), list):
            continue
        for block in msg["content"]:
            if (
                isinstance(block, dict)
                and block.get("type") == "tool_use"
                and block.get("id") == tool_use_id
            ):
                return {"name": block["name"], "input": block.get("input", {})}
    return None


# ─── Tier 1: Per-message aggregate tool-result budget ───────
#
# Design:
#   For each user message whose tool_result blocks together exceed
#   PER_MESSAGE_BUDGET_CHARS, pick the largest BUDGET-ELIGIBLE blocks and
#   replace them via `persist_tool_result` (full content to disk, inline
#   content becomes preview+path) until the per-message aggregate drops
#   under budget.
#
#   A block is budget-eligible iff:
#     - its tool name is not in PINNED_TOOLS (avoids persist/read loops)
#     - its content is not already a placeholder or already persisted
#       (idempotence — same wire content across turns ⇒ prompt-cache stable)
#
#   Messages are evaluated independently. A 150K result in one turn and
#   a 150K result in a later turn are both under budget on their own and
#   are both untouched.
#
# The `util` parameter is accepted for backward-compatibility with callers
# that pass it; the aggregate test makes the utilization gate redundant
# (low-util calls short-circuit on the `total <= budget` check below).


def _is_already_processed(content: str) -> bool:
    """True if this tool_result has already been handled by some tier
    (persist / snip / microcompact). Stable placeholders are preserved
    verbatim across turns for prompt-cache stability."""
    if content in (SNIP_PLACEHOLDER, OLD_RESULT_PLACEHOLDER):
        return True
    if content.startswith(PERSIST_MARKER_PREFIX):
        return True
    return False


def _build_anthropic_tool_name_map(messages: list[dict]) -> dict[str, str]:
    """Index tool_use_id -> tool_name across the whole history (one pass)."""
    out: dict[str, str] = {}
    for msg in messages:
        if msg.get("role") != "assistant" or not isinstance(msg.get("content"), list):
            continue
        for block in msg["content"]:
            if (
                isinstance(block, dict)
                and block.get("type") == "tool_use"
                and isinstance(block.get("id"), str)
                and isinstance(block.get("name"), str)
            ):
                out[block["id"]] = block["name"]
    return out


def budget_tool_results_anthropic(messages: list[dict], util: float = 0.0) -> None:
    """Apply per-message aggregate budget to Anthropic-shaped messages.

    Oversized blocks are replaced by `persist_tool_result` (head preview
    + absolute disk path) so information is not lost — the model can
    re-read the artifact with `read_file`.
    """
    del util  # kept for API compat; aggregate budget is self-gating
    name_by_id = _build_anthropic_tool_name_map(messages)

    for msg in messages:
        if msg.get("role") != "user" or not isinstance(msg.get("content"), list):
            continue

        # Collect this message's tool_result blocks, separating pinned /
        # already-processed ones (counted toward total but not eligible).
        total = 0
        eligible: list[tuple[dict, int, str]] = []  # (block_ref, size, tool_name)
        for block in msg["content"]:
            if (
                not isinstance(block, dict)
                or block.get("type") != "tool_result"
                or not isinstance(block.get("content"), str)
            ):
                continue
            size = len(block["content"])
            total += size
            tool_name = name_by_id.get(block.get("tool_use_id") or "") or ""
            if tool_name in PINNED_TOOLS:
                continue
            if _is_already_processed(block["content"]):
                continue
            eligible.append((block, size, tool_name))

        if total <= PER_MESSAGE_BUDGET_CHARS or not eligible:
            continue

        # Largest first — every persist frees the most bytes per decision.
        eligible.sort(key=lambda bs: bs[1], reverse=True)
        for block, size, tool_name in eligible:
            if total <= PER_MESSAGE_BUDGET_CHARS:
                break
            new_content = persist_tool_result(tool_name, block["content"])
            new_size = len(new_content)
            if new_size < size:
                block["content"] = new_content
                total -= size - new_size


def _build_openai_tool_name_map(messages: list[dict]) -> dict[str, str]:
    """Index tool_call_id -> tool_name from preceding assistant messages."""
    out: dict[str, str] = {}
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls", []) or []:
            if isinstance(tc, dict):
                tc_id = tc.get("id")
                fn = tc.get("function") or {}
                tc_name = fn.get("name") if isinstance(fn, dict) else None
                if isinstance(tc_id, str) and isinstance(tc_name, str):
                    out[tc_id] = tc_name
    return out


def budget_tool_results_openai(messages: list[dict], util: float = 0.0) -> None:
    """Apply aggregate budget to OpenAI-shaped messages.

    OpenAI represents each tool result as its own `role: tool` message, so
    there is no native "message group" to budget against. We group each
    maximal run of consecutive tool-role messages and treat that run as
    one budgeting unit — the closest analogue to Anthropic's single user
    message carrying N tool_result blocks.

    Oversized blocks are replaced by `persist_tool_result` (same as the
    Anthropic path).
    """
    del util
    name_by_id = _build_openai_tool_name_map(messages)

    # Walk and collect each consecutive tool-role run.
    runs: list[list[int]] = []
    current: list[int] = []
    for i, msg in enumerate(messages):
        if msg.get("role") == "tool" and isinstance(msg.get("content"), str):
            current.append(i)
        else:
            if current:
                runs.append(current)
                current = []
    if current:
        runs.append(current)

    for run in runs:
        total = 0
        eligible: list[tuple[int, int, str]] = []  # (msg_index, size, tool_name)
        for i in run:
            content = messages[i]["content"]
            size = len(content)
            total += size
            tool_name = name_by_id.get(messages[i].get("tool_call_id") or "") or ""
            if tool_name in PINNED_TOOLS:
                continue
            if _is_already_processed(content):
                continue
            eligible.append((i, size, tool_name))

        if total <= PER_MESSAGE_BUDGET_CHARS or not eligible:
            continue

        eligible.sort(key=lambda bs: bs[1], reverse=True)
        for i, size, tool_name in eligible:
            if total <= PER_MESSAGE_BUDGET_CHARS:
                break
            new_content = persist_tool_result(tool_name, messages[i]["content"])
            new_size = len(new_content)
            if new_size < size:
                messages[i]["content"] = new_content
                total -= size - new_size


# ─── Tier 2: Snip stale/duplicate results ───────────────────


def snip_stale_results_anthropic(messages: list[dict], util: float) -> None:
    """Tier 2 (Anthropic): replace duplicate/old snippable tool results with
    a stable placeholder once utilization exceeds SNIP_THRESHOLD.

    A block is skipped (never overwritten) if it is already handled by some
    tier — that includes the SNIP/OLD placeholders AND Tier 1's persisted
    preview (whose header contains the disk path). Overwriting a persisted
    preview would permanently lose the recovery path, so `_is_already_processed`
    gates every candidate.
    """
    if util < SNIP_THRESHOLD:
        return

    # Build tool_use_id -> (name, input) once instead of scanning per block.
    tool_use_info: dict[str, dict[str, Any]] = {}
    for msg in messages:
        if msg.get("role") != "assistant" or not isinstance(msg.get("content"), list):
            continue
        for block in msg["content"]:
            if (
                isinstance(block, dict)
                and block.get("type") == "tool_use"
                and isinstance(block.get("id"), str)
            ):
                tool_use_info[block["id"]] = {
                    "name": block.get("name", ""),
                    "input": block.get("input", {}) or {},
                }

    results: list[dict[str, Any]] = []
    for mi, msg in enumerate(messages):
        if msg.get("role") != "user" or not isinstance(msg.get("content"), list):
            continue
        for bi, block in enumerate(msg["content"]):
            if (
                isinstance(block, dict)
                and block.get("type") == "tool_result"
                and isinstance(block.get("content"), str)
                and not _is_already_processed(block["content"])
            ):
                tool_info = tool_use_info.get(block.get("tool_use_id") or "")
                if tool_info and tool_info["name"] in SNIPPABLE_TOOLS:
                    results.append(
                        {
                            "mi": mi,
                            "bi": bi,
                            "name": tool_info["name"],
                            "file_path": tool_info["input"].get("file_path"),
                        }
                    )

    if len(results) <= KEEP_RECENT_RESULTS:
        return

    to_snip: set[int] = set()
    seen_files: dict[str, list[int]] = {}
    for i, r in enumerate(results):
        if r["name"] == "read_file" and r.get("file_path"):
            seen_files.setdefault(r["file_path"], []).append(i)

    for indices in seen_files.values():
        if len(indices) > 1:
            for j in indices[:-1]:
                to_snip.add(j)

    snip_before = len(results) - KEEP_RECENT_RESULTS
    for i in range(snip_before):
        to_snip.add(i)

    for idx in to_snip:
        r = results[idx]
        messages[r["mi"]]["content"][r["bi"]]["content"] = SNIP_PLACEHOLDER


def snip_stale_results_openai(messages: list[dict], util: float) -> None:
    """Tier 2 (OpenAI): same semantics as the Anthropic path.

    Only results from tools in SNIPPABLE_TOOLS are eligible, and results
    that have already been processed by some tier (snip/old placeholders
    or Tier 1's persisted preview) are skipped so we do not destroy the
    disk-path reference written by Tier 1.
    """
    if util < SNIP_THRESHOLD:
        return
    name_by_id = _build_openai_tool_name_map(messages)
    tool_msgs: list[int] = []
    for i, msg in enumerate(messages):
        if (
            msg.get("role") == "tool"
            and isinstance(msg.get("content"), str)
            and not _is_already_processed(msg["content"])
        ):
            tool_name = name_by_id.get(msg.get("tool_call_id") or "") or ""
            if tool_name in SNIPPABLE_TOOLS:
                tool_msgs.append(i)
    if len(tool_msgs) <= KEEP_RECENT_RESULTS:
        return
    snip_count = len(tool_msgs) - KEEP_RECENT_RESULTS
    for i in range(snip_count):
        messages[tool_msgs[i]]["content"] = SNIP_PLACEHOLDER


# ─── Tier 3: Microcompact on idle ───────────────────────────


def microcompact_anthropic(
    messages: list[dict],
    last_api_call_time: float,
    *,
    keep_recent: int = KEEP_RECENT_RESULTS,
) -> int:
    """Tier 3 (Anthropic): after idle > MICROCOMPACT_IDLE_S, replace all but
    the most recent ``keep_recent`` snippable tool results with
    OLD_RESULT_PLACEHOLDER. Placeholders and Tier 1 persisted previews are
    left untouched — the latter is small and holds the recovery path.

    Only results from tools in ``SNIPPABLE_TOOLS`` are eligible, mirroring
    OpenHarness' compactable-tool filter: non-snippable tool outputs (e.g.
    user-question answers, plan results) must not be silently cleared.

    Returns the number of characters freed (0 when no-op). The return value
    is informational — callers may ignore it.
    """
    if (
        not last_api_call_time
        or (time.time() - last_api_call_time) < MICROCOMPACT_IDLE_S
    ):
        return 0
    # Never clear ALL results — floor at 1 (matches OpenHarness).
    keep_recent = max(1, keep_recent)

    name_by_id = _build_anthropic_tool_name_map(messages)
    all_results: list[tuple[int, int, int]] = []  # (mi, bi, size)
    for mi, msg in enumerate(messages):
        if msg.get("role") != "user" or not isinstance(msg.get("content"), list):
            continue
        for bi, block in enumerate(msg["content"]):
            if (
                not isinstance(block, dict)
                or block.get("type") != "tool_result"
                or not isinstance(block.get("content"), str)
                or _is_already_processed(block["content"])
            ):
                continue
            tool_name = name_by_id.get(block.get("tool_use_id") or "") or ""
            if tool_name not in SNIPPABLE_TOOLS:
                continue
            all_results.append((mi, bi, len(block["content"])))

    if len(all_results) <= keep_recent:
        return 0

    clear_count = len(all_results) - keep_recent
    tokens_saved = 0
    for mi, bi, size in all_results[:clear_count]:
        messages[mi]["content"][bi]["content"] = OLD_RESULT_PLACEHOLDER
        tokens_saved += size

    if tokens_saved > 0:
        logger.info(
            "Microcompact (anthropic) cleared %d tool results, freed ~%d chars",
            clear_count,
            tokens_saved,
        )
    return tokens_saved


def microcompact_openai(
    messages: list[dict],
    last_api_call_time: float,
    *,
    keep_recent: int = KEEP_RECENT_RESULTS,
) -> int:
    """Tier 3 (OpenAI): mirrors the Anthropic path — skip content that has
    already been handled by some tier so Tier 1's persisted path survives,
    and only clear results from tools in ``SNIPPABLE_TOOLS``.

    Returns characters freed (0 when no-op).
    """
    if (
        not last_api_call_time
        or (time.time() - last_api_call_time) < MICROCOMPACT_IDLE_S
    ):
        return 0
    keep_recent = max(1, keep_recent)

    name_by_id = _build_openai_tool_name_map(messages)
    tool_msgs: list[tuple[int, int]] = []  # (msg_index, size)
    for i, msg in enumerate(messages):
        if (
            msg.get("role") != "tool"
            or not isinstance(msg.get("content"), str)
            or _is_already_processed(msg["content"])
        ):
            continue
        tool_name = name_by_id.get(msg.get("tool_call_id") or "") or ""
        if tool_name not in SNIPPABLE_TOOLS:
            continue
        tool_msgs.append((i, len(msg["content"])))

    if len(tool_msgs) <= keep_recent:
        return 0

    clear_count = len(tool_msgs) - keep_recent
    tokens_saved = 0
    for i, size in tool_msgs[:clear_count]:
        messages[i]["content"] = OLD_RESULT_PLACEHOLDER
        tokens_saved += size

    if tokens_saved > 0:
        logger.info(
            "Microcompact (openai) cleared %d tool results, freed ~%d chars",
            clear_count,
            tokens_saved,
        )
    return tokens_saved


# ─── Tier 4: Deterministic context collapse ─────────────────
#
# Ported from OpenHarness' try_context_collapse. For each oversized text
# or tool_result block in the OLDER portion of history, replace content
# with head[:HEAD] + marker + tail[-TAIL:]. Last cheap pass before tier 0
# (LLM summarization). Unlike OpenHarness this version:
#   - Respects `_is_already_processed` so tier-1 persisted previews
#     (whose header contains the disk recovery path) are never truncated.
#   - Preserves Anthropic tool_use/tool_result adjacency and OpenAI
#     assistant(tool_calls)/tool runs when splitting — truncating only
#     the strict "older" prefix and never breaking a pair across the
#     split boundary, which the wire protocols both reject.
#   - Does a net-benefit check (chars_freed > 0) and short-circuits
#     when the util gate isn't met.


def _collapse_text(text: str) -> str:
    """Head+tail truncation for oversized text.

    Idempotence: content already handled by another tier
    (SNIP/OLD placeholders, tier-1 persisted preview header) is returned
    unchanged — truncating a persisted preview could bury the disk path
    in the omitted middle. Once collapsed, resulting text length is
    ``HEAD + TAIL + marker`` (~1.4K) which is under the trigger limit,
    so re-invocation is naturally a no-op.
    """
    if _is_already_processed(text):
        return text
    if len(text) <= CONTEXT_COLLAPSE_TEXT_CHAR_LIMIT:
        return text
    omitted = len(text) - CONTEXT_COLLAPSE_HEAD_CHARS - CONTEXT_COLLAPSE_TAIL_CHARS
    head = text[:CONTEXT_COLLAPSE_HEAD_CHARS].rstrip()
    tail = text[-CONTEXT_COLLAPSE_TAIL_CHARS:].lstrip()
    return f"{head}\n...[collapsed {omitted} chars]...\n{tail}"


def _split_anthropic_preserve_tool_pairs(
    messages: list[dict], preserve_recent: int
) -> int:
    """Return the index that starts the `newer` slice.

    Walks the split index backward until it does NOT land on a user
    message carrying tool_result blocks — such a message must stay
    adjacent to the preceding assistant's tool_use, which is (now) in
    `newer`. Returns the final index; `messages[:idx]` is `older`.
    """
    if len(messages) <= preserve_recent:
        return 0
    split = len(messages) - preserve_recent
    while split > 0:
        msg = messages[split]
        if msg.get("role") != "user" or not isinstance(msg.get("content"), list):
            break
        has_tool_result = any(
            isinstance(b, dict) and b.get("type") == "tool_result"
            for b in msg["content"]
        )
        if not has_tool_result:
            break
        split -= 1
    return split


def _split_openai_preserve_tool_pairs(
    messages: list[dict], preserve_recent: int
) -> int:
    """Return the index that starts the `newer` slice, OpenAI shape.

    Walks back through any consecutive ``role == 'tool'`` messages, then
    back off one more if the preceding message is an assistant carrying
    ``tool_calls`` — it must stay paired with the tool messages it
    invoked.
    """
    if len(messages) <= preserve_recent:
        return 0
    split = len(messages) - preserve_recent
    while split > 0 and messages[split].get("role") == "tool":
        split -= 1
    if (
        split > 0
        and messages[split].get("role") == "assistant"
        and messages[split].get("tool_calls")
    ):
        split -= 1
    return split


def collapse_context_anthropic(messages: list[dict], util: float) -> int:
    """Tier 4 (Anthropic): collapse oversized text/tool_result blocks in
    older history once utilization exceeds CONTEXT_COLLAPSE_THRESHOLD.

    Returns characters freed (0 when the util gate doesn't fire or
    nothing was oversized).
    """
    if util < CONTEXT_COLLAPSE_THRESHOLD:
        return 0
    if len(messages) <= CONTEXT_COLLAPSE_PRESERVE_RECENT + 2:
        return 0

    split = _split_anthropic_preserve_tool_pairs(
        messages, CONTEXT_COLLAPSE_PRESERVE_RECENT
    )
    if split <= 0:
        return 0

    chars_freed = 0
    for msg in messages[:split]:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text" and isinstance(block.get("text"), str):
                orig = block["text"]
                new = _collapse_text(orig)
                if new != orig:
                    chars_freed += len(orig) - len(new)
                    block["text"] = new
            elif btype == "tool_result" and isinstance(block.get("content"), str):
                orig = block["content"]
                new = _collapse_text(orig)
                if new != orig:
                    chars_freed += len(orig) - len(new)
                    block["content"] = new

    if chars_freed > 0:
        logger.info(
            "Context collapse (anthropic) truncated older blocks, "
            "freed ~%d chars (over %d messages)",
            chars_freed,
            split,
        )
    return chars_freed


def collapse_context_openai(messages: list[dict], util: float) -> int:
    """Tier 4 (OpenAI): mirrors the Anthropic path — collapse the string
    ``content`` of any message in the older prefix.
    """
    if util < CONTEXT_COLLAPSE_THRESHOLD:
        return 0
    if len(messages) <= CONTEXT_COLLAPSE_PRESERVE_RECENT + 2:
        return 0

    split = _split_openai_preserve_tool_pairs(
        messages, CONTEXT_COLLAPSE_PRESERVE_RECENT
    )
    if split <= 0:
        return 0

    chars_freed = 0
    for msg in messages[:split]:
        content = msg.get("content")
        if not isinstance(content, str):
            continue
        new = _collapse_text(content)
        if new != content:
            chars_freed += len(content) - len(new)
            msg["content"] = new

    if chars_freed > 0:
        logger.info(
            "Context collapse (openai) truncated older messages, "
            "freed ~%d chars (over %d messages)",
            chars_freed,
            split,
        )
    return chars_freed


# ─── Pipeline dispatcher (tiers 1-4) ────────────────────────


def run_compression_pipeline(
    *,
    use_openai: bool,
    messages: list[dict],
    last_input_tokens: int,
    effective_window: int,
    last_api_call_time: float,
) -> None:
    """Apply tiers 1-4 in order. Mutates `messages` in place."""
    util = utilization(last_input_tokens, effective_window)
    if use_openai:
        budget_tool_results_openai(messages, util)
        snip_stale_results_openai(messages, util)
        microcompact_openai(messages, last_api_call_time)
        collapse_context_openai(messages, util)
    else:
        budget_tool_results_anthropic(messages, util)
        snip_stale_results_anthropic(messages, util)
        microcompact_anthropic(messages, last_api_call_time)
        collapse_context_anthropic(messages, util)


# ─── Tier 0: Full conversation summarization ────────────────

_SUMMARY_SYSTEM = (
    "You are a conversation summarizer. Be concise but preserve important details."
)
_SUMMARY_USER = (
    "Summarize the conversation so far in a concise paragraph, preserving key "
    "decisions, file paths, and context needed to continue the work."
)
_POST_COMPACT_USER_PREFIX = "[Previous conversation summary]\n"
_POST_COMPACT_ASSISTANT = (
    "Understood. I have the context from our previous conversation. "
    "How can I continue helping?"
)


def _post_compact_user(summary: str) -> str:
    # Plain concatenation — str.format would raise on literal '{' or '}'
    # occurring naturally in a summary (e.g. quoted JSON or code snippets).
    return _POST_COMPACT_USER_PREFIX + summary


def should_full_compact(last_input_tokens: int, effective_window: int) -> bool:
    return utilization(last_input_tokens, effective_window) > FULL_COMPACT_THRESHOLD


async def compact_anthropic(
    anthropic_client: Any, model: str, messages: list[dict]
) -> list[dict] | None:
    """Return replacement message list, or None if compaction was skipped."""
    if len(messages) < 4:
        return None
    last_user_msg = messages[-1]
    summary_resp = await anthropic_client.messages.create(
        model=model,
        max_tokens=2048,
        system=_SUMMARY_SYSTEM,
        messages=[
            *messages[:-1],
            {"role": "user", "content": _SUMMARY_USER},
        ],
    )
    summary_text = (
        summary_resp.content[0].text
        if summary_resp.content and summary_resp.content[0].type == "text"
        else "No summary available."
    )
    new_messages: list[dict] = [
        {"role": "user", "content": _post_compact_user(summary_text)},
        {"role": "assistant", "content": _POST_COMPACT_ASSISTANT},
    ]
    if last_user_msg.get("role") == "user":
        new_messages.append(last_user_msg)
    return new_messages


async def compact_openai(
    openai_client: Any, model: str, messages: list[dict]
) -> list[dict] | None:
    """Return replacement message list, or None if compaction was skipped.

    The system message at index 0 is preserved.
    """
    if len(messages) < 5:
        return None
    system_msg = messages[0]
    last_user_msg = messages[-1]
    summary_resp = await openai_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SUMMARY_SYSTEM},
            *messages[1:-1],
            {"role": "user", "content": _SUMMARY_USER},
        ],
    )
    summary_text = summary_resp.choices[0].message.content or "No summary available."
    new_messages: list[dict] = [
        system_msg,
        {"role": "user", "content": _post_compact_user(summary_text)},
        {"role": "assistant", "content": _POST_COMPACT_ASSISTANT},
    ]
    if last_user_msg.get("role") == "user":
        new_messages.append(last_user_msg)
    return new_messages
