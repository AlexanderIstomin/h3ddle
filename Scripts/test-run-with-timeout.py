#!/usr/bin/env python3
"""Behavior checks for the process timeout used by CI."""

from __future__ import annotations

import pathlib
import subprocess
import sys
import time
import unittest


RUNNER = pathlib.Path(__file__).with_name("run-with-timeout.py")


class RunWithTimeoutTests(unittest.TestCase):
    def run_wrapper(self, *arguments: str) -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(
            [sys.executable, str(RUNNER), *arguments],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

    def test_propagates_command_status(self) -> None:
        result = self.run_wrapper("2", sys.executable, "-c", "raise SystemExit(7)")
        self.assertEqual(result.returncode, 7)

    def test_times_out_a_stuck_process(self) -> None:
        started = time.monotonic()
        result = self.run_wrapper("0.1", sys.executable, "-c", "import time; time.sleep(5)")
        self.assertEqual(result.returncode, 124)
        self.assertLess(time.monotonic() - started, 2)
        self.assertIn(b"Command exceeded 0.1 seconds", result.stderr)


if __name__ == "__main__":
    unittest.main()
