#!/usr/bin/env python3
"""Run a command with a wall-clock deadline and stop its whole process group."""

from __future__ import annotations

import os
import signal
import subprocess
import sys


TIMEOUT_EXIT_STATUS = 124


def stop_process_group(process: subprocess.Popen[bytes], first_signal: signal.Signals) -> None:
    try:
        os.killpg(process.pid, first_signal)
    except ProcessLookupError:
        return

    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def main(arguments: list[str]) -> int:
    if len(arguments) < 2:
        print("usage: run-with-timeout.py SECONDS COMMAND [ARG ...]", file=sys.stderr)
        return 2

    try:
        timeout = float(arguments[0])
    except ValueError:
        print(f"invalid timeout: {arguments[0]}", file=sys.stderr)
        return 2
    if timeout <= 0:
        print("timeout must be greater than zero", file=sys.stderr)
        return 2

    command = arguments[1:]
    process = subprocess.Popen(command, start_new_session=True)
    try:
        return_code = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        print(
            f"Command exceeded {timeout:g} seconds; terminating process group: "
            + " ".join(command),
            file=sys.stderr,
        )
        stop_process_group(process, signal.SIGTERM)
        return TIMEOUT_EXIT_STATUS
    except KeyboardInterrupt:
        stop_process_group(process, signal.SIGINT)
        return 130

    if return_code < 0:
        return 128 - return_code
    return return_code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
