"""Shell command execution: persistent cwd, dangerous-command preflight, timeout-safe process tree teardown.

Imports run-time constants from :mod:`cici.tools.schemas` so this
module stays decoupled from the registry/runner. ``check_permission``
is intentionally NOT called here — gating happens in :mod:`runner`
before the handler is invoked.
"""

from __future__ import annotations

import os
import subprocess

from .schemas import IS_WIN

def _run_shell(inp: dict) -> str:
    try:
        command = inp["command"]
        preflight = _preflight_interactive_command(command)
        if preflight is not None:
            return preflight
        timeout_ms = inp.get("timeout", 30000)
        timeout_s = timeout_ms / 1000
        return _run_shell_impl(command, timeout_s, timeout_ms)
    except Exception as e:
        return f"Error: {e}"


# ─── Interactive-scaffold preflight (OpenHarness parity) ────
# The shell runs non-interactively (stdin is DEVNULL). Commands that expect
# TTY input (e.g. `npm create foo`, `create-next-app`) would otherwise hang
# until timeout. Fail fast with a hint to use non-interactive flags.

_SCAFFOLD_MARKERS: tuple[str, ...] = (
    "create-next-app",
    "npm create ",
    "pnpm create ",
    "yarn create ",
    "bun create ",
    "pnpm dlx ",
    "npm init ",
    "pnpm init ",
    "yarn init ",
    "bunx create-",
    "npx create-",
)

_NON_INTERACTIVE_MARKERS: tuple[str, ...] = (
    "--yes",
    " -y",
    "--skip-install",
    "--defaults",
    "--non-interactive",
    "--ci",
)


def _preflight_interactive_command(command: str) -> str | None:
    """Return an error message if the command looks like an interactive
    scaffolder without a non-interactive flag, else None."""
    lowered = command.lower()
    if not any(m in lowered for m in _SCAFFOLD_MARKERS):
        return None
    if any(m in lowered for m in _NON_INTERACTIVE_MARKERS):
        return None
    return (
        "Error: This command appears to require interactive input before it can "
        "continue. The shell tool is non-interactive, so it cannot answer "
        "installer/scaffold prompts live. Prefer non-interactive flags "
        "(for example --yes, -y, --skip-install, --defaults, --non-interactive), "
        "or run the scaffolding step once in an external terminal before asking "
        "the agent to continue."
    )


# ─── Persistent shell state ─────────────────────────────────
# Track cwd across run_shell calls so `cd` persists between invocations.
# POSIX-only; on Windows the state stays at whatever the Python process
# cwd is.
_shell_cwd: str | None = None
_PWD_MARKER = "__OCS_SHELL_PWD_MARKER__"


def _get_shell_cwd() -> str:
    """Return the persistent shell cwd, re-initialising if it vanished."""
    global _shell_cwd
    if _shell_cwd is None or not os.path.isdir(_shell_cwd):
        _shell_cwd = os.getcwd()
    return _shell_cwd


def reset_shell_state() -> None:
    """Reset persistent shell state — exposed for tests."""
    global _shell_cwd
    _shell_cwd = None


def _extract_pwd(output: str) -> tuple[str | None, str]:
    """Pull the trailing pwd marker line out of the shell output.

    Returns (new_cwd_or_None, cleaned_output).
    """
    needle = "\n" + _PWD_MARKER
    idx = output.rfind(needle)
    if idx == -1:
        return None, output
    start = idx + len(needle)
    end = output.find("\n", start)
    if end == -1:
        return output[start:].strip() or None, output[:idx]
    cwd = output[start:end].strip() or None
    cleaned = output[:idx] + output[end + 1 :]
    return cwd, cleaned


def _kill_process_tree(proc: "subprocess.Popen[str]") -> None:
    """Kill proc and all its descendants — avoids orphaned child processes.

    POSIX: the child was spawned in a new session (setsid), so killpg on
    the process-group id takes out the whole tree. Windows: taskkill /T.
    """
    import signal

    if IS_WIN:
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True,
                timeout=5,
            )
        except Exception:
            pass
    else:
        try:
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, signal.SIGTERM)
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
        except (ProcessLookupError, PermissionError):
            pass
    # Belt-and-braces fallback
    try:
        proc.kill()
    except Exception:
        pass


def _run_shell_impl(command: str, timeout_s: float, timeout_ms: int) -> str:
    global _shell_cwd

    cwd = _get_shell_cwd()

    # On POSIX, wrap the command so we can (1) capture the new pwd after
    # any `cd`, and (2) still propagate the user command's exit code even
    # though we appended a printf. `__rc=$?` snapshots the exit code right
    # after the user command so printf doesn't clobber it.
    if IS_WIN:
        wrapped = command
        popen_kwargs: dict = {}
    else:
        wrapped = (
            f"{command}\n"
            f"__rc=$?\n"
            f'printf "\\n{_PWD_MARKER}%s\\n" "$(pwd)"\n'
            f"exit $__rc"
        )
        # start_new_session => child becomes a process-group leader, so we
        # can killpg the whole tree on timeout.
        popen_kwargs = {"start_new_session": True}

    proc = subprocess.Popen(
        wrapped,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=cwd,
        **popen_kwargs,
    )

    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        _kill_process_tree(proc)
        # Drain any buffered output so fds are closed
        try:
            proc.communicate(timeout=2)
        except Exception:
            pass
        return f"Command timed out after {timeout_ms}ms"

    # Parse the pwd marker (POSIX only) and update persistent state
    if not IS_WIN:
        new_cwd, stdout = _extract_pwd(stdout or "")
        if new_cwd and os.path.isdir(new_cwd):
            _shell_cwd = new_cwd

    returncode = proc.returncode
    output = stdout or ""
    if returncode != 0:
        stderr_part = f"\nStderr: {stderr}" if stderr else ""
        stdout_part = f"\nStdout: {output}" if output else ""
        return f"Command failed (exit code {returncode}){stdout_part}{stderr_part}"
    return output or "(no output)"

