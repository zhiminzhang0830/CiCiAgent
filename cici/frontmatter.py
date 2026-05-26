"""Lightweight YAML-ish front matter helpers.

Used by the memory and skills modules to round-trip simple ``key: value``
metadata blocks delimited by ``---`` markers at the top of a file. We
deliberately avoid a hard dependency on PyYAML — only flat string maps
are supported, which is all our on-disk formats need.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_FENCE = "---"
_FRONT_MATTER_RE = re.compile(
    r"\A---[ \t]*\r?\n(?P<meta>.*?)\r?\n---[ \t]*(?:\r?\n(?P<body>.*))?\Z",
    re.DOTALL,
)


@dataclass
class FrontMatter:
    """Result of splitting a document into metadata + body."""

    meta: dict[str, str] = field(default_factory=dict)
    body: str = ""


def _parse_meta_block(block: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw_line in block.splitlines():
        line = raw_line.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        key, sep, value = line.partition(":")
        if not sep:
            continue
        key = key.strip()
        if key:
            out[key] = value.strip()
    return out


def parse_frontmatter(content: str) -> FrontMatter:
    """Split ``content`` into front-matter metadata and trailing body.

    If the document does not start with a ``---`` fence, or the fence is
    never closed, the original text is returned unchanged in ``body``.
    """
    match = _FRONT_MATTER_RE.match(content)
    if match is None:
        return FrontMatter(body=content)
    meta = _parse_meta_block(match.group("meta"))
    body = (match.group("body") or "").strip()
    return FrontMatter(meta=meta, body=body)


def format_frontmatter(meta: dict[str, str], body: str) -> str:
    """Serialise ``meta`` + ``body`` back into a fenced front-matter document."""
    parts: list[str] = [_FENCE]
    parts.extend(f"{key}: {value}" for key, value in meta.items())
    parts.append(_FENCE)
    parts.append("")
    parts.append(body)
    return "\n".join(parts)


# Backwards-compatible alias kept for code that imports the old type name.
FrontmatterResult = FrontMatter

__all__ = ["FrontMatter", "FrontmatterResult", "parse_frontmatter", "format_frontmatter"]
