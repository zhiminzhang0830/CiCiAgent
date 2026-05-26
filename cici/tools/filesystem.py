"""File I/O handlers: read / write / edit / list / grep.

Each function takes the raw ``inp`` dict from the model and returns a
plain ``str`` — error responses use the legacy ``Error: ...`` /
``Warning: ...`` prefix convention so :func:`schemas._derive_is_error`
in the dispatcher can flag them without per-handler bookkeeping.

Sensitive-path enforcement happens here at the handler boundary
(:func:`schemas._is_sensitive_path`); the same gate also runs in
:mod:`permission`, so the block holds regardless of which entry point
the call took.
"""

from __future__ import annotations

import fnmatch
import os
import re
import subprocess
from pathlib import Path

from ..memory import get_memory_dir
from .schemas import (
    IS_WIN,
    MAX_FILE_SIZE_BYTES,
    _BINARY_SNIFF_BYTES,
    _is_sensitive_path,
)


def _read_file(inp: dict) -> str:
    try:
        path = Path(inp["file_path"])
        if _is_sensitive_path(str(path)):
            return (
                f"Error: Access denied — {inp['file_path']} matches a "
                f"sensitive-path pattern (credentials/keys). This block "
                f"cannot be overridden."
            )
        offset = inp.get("offset", 0)
        limit = inp.get("limit")

        # Pre-read size check — reject whole-file reads of oversized files.
        # When offset/limit is given, the caller has opted into a bounded
        # slice, so the file-size cap doesn't apply.
        if limit is None and offset == 0:
            try:
                file_size = path.stat().st_size
            except OSError as e:
                return f"Error reading file: {e}"
            if file_size > MAX_FILE_SIZE_BYTES:
                return (
                    f"Error: File too large ({file_size / 1024:.1f} KB > "
                    f"{MAX_FILE_SIZE_BYTES / 1024:.0f} KB). Use offset and limit "
                    f"parameters to read specific portions of the file, or use "
                    f"grep_search to locate specific content."
                )

        # Binary detection — sniff the first few KB for a NUL byte. Reading
        # a binary file as text returns garbled mojibake that wastes the
        # model's context; fail fast instead.
        try:
            with open(path, "rb") as f:
                head = f.read(_BINARY_SNIFF_BYTES)
        except OSError as e:
            return f"Error reading file: {e}"
        if b"\x00" in head:
            return (
                f"Error: {inp['file_path']} appears to be a binary file "
                f"(NUL byte in first {_BINARY_SNIFF_BYTES} bytes). "
                f"read_file only supports text."
            )

        content = path.read_text(errors="replace")
        lines = content.split("\n")
        total_lines = len(lines)

        # Apply offset/limit slicing
        start = max(0, offset)
        if limit is not None:
            selected = lines[start : start + limit]
        else:
            selected = lines[start:]

        # Number lines using absolute line numbers (1-indexed)
        numbered = "\n".join(
            f"{start + i + 1:4d} | {line}" for i, line in enumerate(selected)
        )

        # Add range header when a partial slice was requested
        if limit is not None or offset > 0:
            end = start + len(selected)
            header = (
                f"[Showing lines {start + 1}-{end} of {total_lines} total lines]\n\n"
            )
            return header + numbered
        return numbered
    except Exception as e:
        return f"Error reading file: {e}"


def _write_file(inp: dict) -> str:
    try:
        path = Path(inp["file_path"])
        if _is_sensitive_path(str(path)):
            return (
                f"Error: Write denied — {inp['file_path']} matches a "
                f"sensitive-path pattern (credentials/keys). This block "
                f"cannot be overridden."
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(inp["content"])
        _auto_update_memory_index(str(path))
        lines = inp["content"].split("\n")
        line_count = len(lines)
        preview = "\n".join(f"{i+1:4d} | {line}" for i, line in enumerate(lines[:30]))
        trunc = f"\n  ... ({line_count} lines total)" if line_count > 30 else ""
        return f"Successfully wrote to {inp['file_path']} ({line_count} lines)\n\n{preview}{trunc}"
    except Exception as e:
        return f"Error writing file: {e}"


def _auto_update_memory_index(file_path: str) -> None:
    try:
        mem_dir = str(get_memory_dir())
        if (
            file_path.startswith(mem_dir)
            and file_path.endswith(".md")
            and not file_path.endswith("MEMORY.md")
        ):
            mem_path = Path(mem_dir)
            lines = ["# Memory Index", ""]
            for f in sorted(mem_path.glob("*.md")):
                if f.name == "MEMORY.md":
                    continue
                try:
                    raw = f.read_text()
                    name_match = re.search(r"^name:\s*(.+)$", raw, re.MULTILINE)
                    type_match = re.search(r"^type:\s*(.+)$", raw, re.MULTILINE)
                    desc_match = re.search(r"^description:\s*(.+)$", raw, re.MULTILINE)
                    if name_match and type_match:
                        n = name_match.group(1).strip()
                        t = type_match.group(1).strip()
                        d = desc_match.group(1).strip() if desc_match else ""
                        lines.append(f"- **[{n}]({f.name})** ({t}) — {d}")
                except Exception:
                    pass
            (mem_path / "MEMORY.md").write_text("\n".join(lines))
    except Exception:
        pass


# ─── Edit helpers: quote normalization + diff ───────────────


def _normalize_quotes(s: str) -> str:
    s = re.sub("[\u2018\u2019\u2032]", "'", s)
    s = re.sub("[\u201c\u201d\u2033]", '"', s)
    return s


def _find_actual_string(file_content: str, search_string: str) -> str | None:
    if search_string in file_content:
        return search_string
    norm_search = _normalize_quotes(search_string)
    norm_file = _normalize_quotes(file_content)
    idx = norm_file.find(norm_search)
    if idx != -1:
        return file_content[idx : idx + len(search_string)]
    return None


def _generate_diff(old_content: str, old_string: str, new_string: str) -> str:
    before_change = old_content.split(old_string)[0]
    line_num = before_change.count("\n") + 1
    old_lines = old_string.split("\n")
    new_lines = new_string.split("\n")

    parts = [f"@@ -{line_num},{len(old_lines)} +{line_num},{len(new_lines)} @@"]
    for line in old_lines:
        parts.append(f"- {line}")
    for line in new_lines:
        parts.append(f"+ {line}")
    return "\n".join(parts)


def _edit_file(inp: dict) -> str:
    try:
        path = Path(inp["file_path"])
        if _is_sensitive_path(str(path)):
            return (
                f"Error: Edit denied — {inp['file_path']} matches a "
                f"sensitive-path pattern (credentials/keys). This block "
                f"cannot be overridden."
            )
        content = path.read_text()

        actual = _find_actual_string(content, inp["old_string"])
        if not actual:
            return f"Error: old_string '{inp['old_string']}' not found in {inp['file_path']}"

        count = content.count(actual)
        if count > 1:
            return f"Error: old_string '{inp['old_string']}' found {count} times in {inp['file_path']}. Must be unique."

        new_content = content.replace(actual, inp["new_string"], 1)
        path.write_text(new_content)

        diff = _generate_diff(content, actual, inp["new_string"])
        quote_note = (
            " (matched via quote normalization)" if actual != inp["old_string"] else ""
        )
        return f"Successfully edited {inp['file_path']}{quote_note}\n\n{diff}"
    except Exception as e:
        return f"Error editing file: {e}"


def _list_files(inp: dict) -> str:
    try:
        base = Path(inp.get("path") or ".")
        pattern = inp["pattern"]
        files = []
        for p in base.glob(pattern):
            if p.is_file():
                rel = str(p.relative_to(base) if base != Path(".") else p)
                # Skip node_modules and .git
                if "node_modules" in rel or ".git" in rel.split(os.sep):
                    continue
                files.append(rel)
                if len(files) >= 200:
                    break
        if not files:
            return "No files found matching the pattern."
        result = "\n".join(files[:200])
        if len(files) > 200:
            result += f"\n... and {len(files) - 200} more"
        return result
    except Exception as e:
        return f"Error listing files: {e}"


def _grep_search(inp: dict) -> str:
    pattern = inp["pattern"]
    path = inp.get("path") or "."
    include = inp.get("include")

    # Try system grep first (Linux/macOS)
    if not IS_WIN:
        try:
            args = ["grep", "--line-number", "--color=never", "-r"]
            if include:
                args.append(f"--include={include}")
            args.extend(["--", pattern, path])
            result = subprocess.run(args, capture_output=True, text=True, timeout=10)
            if result.returncode == 1:
                return "No matches found."
            if result.returncode == 0:
                lines = [line for line in result.stdout.split("\n") if line]
                output = "\n".join(lines[:100])
                if len(lines) > 100:
                    output += f"\n... and {len(lines) - 100} more matches"
                return output
            # Non-zero exit (not 1) — fall through to Python fallback
        except Exception:
            pass  # Fall through to Python fallback

    # Pure Python fallback (Windows, or system grep unavailable)
    return _grep_python(pattern, path, include)


def _grep_python(pattern: str, directory: str, include: str | None) -> str:
    regex = re.compile(pattern)
    include_pattern = include
    matches: list[str] = []

    def walk(d: str) -> None:
        if len(matches) >= 200:
            return
        try:
            entries = os.listdir(d)
        except Exception:
            return
        for name in entries:
            if name.startswith(".") or name == "node_modules":
                continue
            full = os.path.join(d, name)
            if os.path.isdir(full):
                walk(full)
                continue
            if include_pattern and not fnmatch.fnmatch(name, include_pattern):
                continue
            try:
                text = Path(full).read_text(errors="replace")
                for i, line in enumerate(text.split("\n")):
                    if regex.search(line):
                        matches.append(f"{full}:{i+1}:{line}")
                        if len(matches) >= 200:
                            return
            except Exception:
                pass

    walk(directory)
    if not matches:
        return "No matches found."
    output = "\n".join(matches[:100])
    if len(matches) > 100:
        output += f"\n... and {len(matches) - 100} more matches"
    return output
