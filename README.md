<div align="center">

# cici

**A minimal, hackable coding agent for your terminal.**

[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Status](https://img.shields.io/badge/status-alpha-orange.svg)](#status)

*Small enough to read in an afternoon. Capable enough to ship real changes.*

</div>

---

`cici` is an open-source coding agent that lives in your terminal. It reads your
code, edits files, runs shell commands, talks to MCP servers, and orchestrates
sub-agents — all from a single Python package you can fork and modify.

It is intentionally minimal: ~9k lines of dependency-light Python, no plugin
system, no DSL, no magic. The goal is to be the *clearest* implementation of a
modern coding agent, not the largest.

<p align="center">
  <img src="asserts/cici.png" alt="cici TUI screenshot" width="900" />
</p>



## Features

- **Multi-provider** — Anthropic and any OpenAI-compatible endpoint (DeepSeek,
  Qwen, vLLM, Ollama, …) behind one normalized adapter
- **Real tools** — `read_file`, `write_file`, `edit_file`, `grep_search`,
  `list_files`, `run_shell`, `web_fetch`, `ask_user_question`,
  `todo_write`, `tool_search`, with permission tiers and sensitive-path
  protection
- **Sub-agents** — spawn `explore`, `plan`, `general`, or custom agents in
  foreground or background; results polled via `agent_result`
- **Skills** — drop a `SKILL.md` into `.cici/skills/` (or `.claude/skills/`)
  and it becomes an invocable workflow
- **MCP client** — speak JSON-RPC to any MCP server over stdio; tools are
  auto-discovered and namespaced as `mcp__server__tool`
- **Context engineering** — five-tier compression (artifact spillover, stale
  result snipping, micro-compact, deterministic collapse, LLM summarization)
- **Persistent memory** — file-based `user / feedback / project / reference`
  notes the agent reads on every turn
- **Permission modes** — `default`, `plan` (read-only), `acceptEdits`,
  `dontAsk` (CI), `bypassPermissions` (`--yolo`)
- **Sessions** — resume any prior conversation with `--resume`

## Install

```bash
# from source
git clone https://github.com/zhiminzhang0830/CiCiAgent
cd CiCiAgent
pip install -e .
```

Requires **Python 3.11+**.

## Quick start

```bash
# 1. configure a provider (one of the following)
export ANTHROPIC_API_KEY=sk-ant-...
# or any OpenAI-compatible endpoint:
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=https://api.deepseek.com/v1

# or use .env (auto-loaded from the working directory)
cp .env.example .env
# then edit .env and fill in the keys you need:
#   ANTHROPIC_API_KEY      — for Anthropic
#   OPENAI_API_KEY         — for any OpenAI-compatible endpoint
#   OPENAI_BASE_URL        — provider base URL (OpenAI / DeepSeek / Qwen / vLLM / Ollama / ...)
#   CICI_MODEL             — optional model override
```
Launch the coding agent (named `cici`):

```bash
# 2. one-shot
cici "explain the architecture of this repo"

# 3. or launch the interactive Textual TUI (3-pane: file tree / chat / status & tools)
cici
```

A `.env` file in the working directory is also loaded automatically.

## Usage

```text
cici [PROMPT...] [options]

  -y, --yolo            Skip all confirmation prompts
  --plan                Plan mode (read-only tools only)
  --accept-edits        Auto-approve file edits
  --dont-ask            Auto-deny confirmations (for CI)
  --thinking            Enable extended thinking (Anthropic)
  -m, --model NAME      Override model
  --api-base URL        OpenAI-compatible base URL
  --resume              Resume the last session
  --max-cost USD        Hard budget cap
  --max-turns N         Cap agentic turns
  --log [PATH]          Tee output to a log file
```

Running `cici` with no prompt arg launches the TUI. Inside: `/clear` `/plan`
`/cost` `/compact` `/memory` `/skills` `/<skill-name>`.

## TUI

Run `cici` with no arguments to drop into an interactive [Textual](https://textual.textualize.io/)
terminal UI. The layout is three panes:

```
┌──────────────┬──────────────────────────────────────┬──────────────────────┐
│              │  chat history (markdown)             │  session status      │
│  📂 file     │  ─────────────────────────           │  model / mode / cwd  │
│     tree     │  live stream buffer                  │  turns · tokens · $  │
│              │  ─────────────────────────           │  ─────────────────   │
│  click to    │  > input  (/clear /plan /cost …)     │  🛠  Tool calls      │
│  insert      │  ─────────────────────────           │  ├─ read_file …      │
│  @path       │  status bar: model · mode · cost     │  └─ run_shell …      │
└──────────────┴──────────────────────────────────────┴──────────────────────┘
```

- **Left — file tree.** Live `DirectoryTree` of the cwd. Clicking a file
  inserts an `@<path>` token into the input so you can reference it in your
  next prompt.
- **Middle — chat & input.** Assistant replies render as Markdown; tokens
  stream into a live buffer above the input and are committed to history when
  the turn ends. A spinner shows while the agent is working. The status bar
  along the bottom always shows model, permission mode, turn count, token
  totals (`↓` in / `↑` out) and an estimated USD cost.
- **Right — session & tools.** A summary panel (model, mode, cwd, turns,
  tokens, cost, session start) plus an expandable tree of every tool call
  made in the session, with input args and result snippets nested underneath.

**Slash commands** (typed into the input):

- `/clear` — wipe conversation history and chat pane
- `/plan` — toggle plan mode (read-only tools)
- `/cost` — print cumulative token usage and USD estimate
- `/compact` — compress history to fit the context window
- `/memory` — list saved `user / feedback / project / reference` notes
- `/skills` — list available skills (built-in and from `.cici/skills/`)
- `/<skill-name> [args]` — invoke a user-invocable skill

**Keybindings:**

- `Ctrl+L` clear chat · `Ctrl+C` interrupt agent (twice to quit) · `Ctrl+D` quit

Implementation lives in a single file: [`cici/tui.py`](cici/tui.py).

## Status

Alpha. The core loop and tools work end-to-end; some advanced paths
(background sub-agents, web search, the `BaseTool` migration) are partially
implemented and clearly marked in the source. Contributions welcome.

## Acknowledgements

Inspired by [Claude Code](https://www.anthropic.com/claude-code), [claude-code-from-scratch](https://github.com/Windy3f3f3f3f/claude-code-from-scratch), [aider](https://github.com/Aider-AI/aider), [opencode](https://github.com/sst/opencode), and the broader MCP ecosystem.

## License

MIT — see [LICENSE](LICENSE).
