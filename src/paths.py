"""Centralized path resolution for the coding agent.

Everything — agent-owned state (sessions, memory, tool-result artifacts,
plans) and user-authored config (settings.json, rules/, agents/, skills/) —
lives under a single root:

- User root:    `~/.coco`        (override with env var `COCO_HOME`)
- Project root: `<cwd>/.coco`    (project-level overrides)

All other modules should import helpers from this file rather than hardcoding
the root.
"""

from __future__ import annotations

import os
from pathlib import Path

# ─── Roots ──────────────────────────────────────────────────

# Unified directory name used both under $HOME for user-level state/config
# and under the project cwd for project-level overrides.
DIRNAME = ".coco"
CLAUDE_DIRNAME = ".claude"


def user_home() -> Path:
    """User-level root (~/.coco by default, override via COCO_HOME)."""
    override = os.environ.get("COCO_HOME")
    if override:
        return Path(override).expanduser()
    return Path.home() / DIRNAME


def claude_user_home() -> Path:
    """Claude home directory (~/.claude by default)."""
    override = os.environ.get("COCO_HOME")
    if override:
        return Path(override).expanduser()
    return Path.home() / CLAUDE_DIRNAME


def project_dir(directory: Path | None = None) -> Path:
    """Project-level root (<directory or cwd>/.coco)."""
    base = Path(directory) if directory is not None else Path.cwd()
    return base / DIRNAME


def claude_project_dir(directory: Path | None = None) -> Path:
    """Claude project-directory (<directory or cwd>/.claude)."""
    base = Path(directory) if directory is not None else Path.cwd()
    return base / CLAUDE_DIRNAME


# ─── User-level subdirectories ──────────────────────────────


def sessions_dir() -> Path:
    return user_home() / "sessions"


def artifacts_dir() -> Path:
    return user_home() / "tool-results"


def plans_dir() -> Path:
    return user_home() / "plans"


def scratchpad_dir(session_id: str) -> Path:
    """Session-scoped temp directory for the agent's one-off files.

    Model-facing: this is the directory the agent should use instead of
    `/tmp` for intermediate artifacts, working files, or throwaway outputs.
    Layout: ``~/.coco/scratchpad/<session_id>/``.
    """
    return user_home() / "scratchpad" / session_id


def projects_dir() -> Path:
    return user_home() / "projects"


def user_settings_file() -> Path:
    return user_home() / "settings.json"


def claude_user_settings_file() -> Path:
    return claude_user_home() / "settings.json"


def user_agents_dir() -> Path:
    return user_home() / "agents"


def claude_user_agents_dir() -> Path:
    return claude_user_home() / "agents"


def user_skills_dir() -> Path:
    return user_home() / "skills"


def claude_user_skills_dir() -> Path:
    return claude_user_home() / "skills"


# ─── Project-level subpaths ─────────────────────────────────


def project_settings_file(directory: Path | None = None) -> Path:
    return project_dir(directory) / "settings.json"


def claude_project_settings_file(directory: Path | None = None) -> Path:
    return claude_project_dir(directory) / "settings.json"


def project_agents_dir(directory: Path | None = None) -> Path:
    return project_dir(directory) / "agents"


def claude_project_agents_dir(directory: Path | None = None) -> Path:
    return claude_project_dir(directory) / "agents"


def project_rules_dir(directory: Path | None = None) -> Path:
    return project_dir(directory) / "rules"


def claude_project_rules_dir(directory: Path | None = None) -> Path:
    return claude_project_dir(directory) / "rules"


def project_skills_dir(directory: Path | None = None) -> Path:
    return project_dir(directory) / "skills"


def claude_project_skills_dir(directory: Path | None = None) -> Path:
    return claude_project_dir(directory) / "skills"
