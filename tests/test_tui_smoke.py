"""Regression tests for PTY ownership and terminal cleanup verification."""

import errno
import os
import signal
import sys
import termios
import unittest
from unittest.mock import patch

from tui_smoke import Terminal


CHILD_SETUP = """
import os
import signal
import termios
import time
import tty

# Behave like an application launched by a shell, with a live session leader.
assert os.getpid() != os.getsid(0)
assert os.getpgrp() == os.tcgetpgrp(0)
original = termios.tcgetattr(0)
tty.setraw(0)

def restore():
    termios.tcsetattr(0, termios.TCSANOW, original)
    os.write(1, b"\\x1b[?1049l\\x1b[?2004l\\x1b[?25h")
"""


class TerminalTests(unittest.TestCase):
    def terminal(self, script):
        terminal = Terminal([sys.executable, "-c", CHILD_SETUP + script])
        self.addCleanup(terminal.close)
        return terminal

    def test_restores_terminal_with_a_separate_session_leader(self):
        self.terminal("restore()\n").finish()

    def test_parent_does_not_inspect_or_restore_a_revoked_terminal(self):
        terminal = self.terminal("restore()\n")
        revoked = termios.error(errno.ENOTTY, "Inappropriate ioctl for device")
        with patch("tui_smoke.termios.tcgetattr", side_effect=revoked), \
                patch("tui_smoke.termios.tcsetattr", side_effect=revoked):
            terminal.finish()
            terminal.close()

    def test_detects_raw_mode_left_by_child_before_harness_cleanup(self):
        terminal = self.terminal('os.write(1, b"\\x1b[?1049l\\x1b[?2004l\\x1b[?25h")\n')
        with self.assertRaisesRegex(AssertionError, "terminal attributes not restored"):
            terminal.finish()

    def test_detects_missing_terminal_escape_cleanup(self):
        terminal = self.terminal("termios.tcsetattr(0, termios.TCSANOW, original)\n")
        with self.assertRaisesRegex(AssertionError, "alternate screen not restored"):
            terminal.finish()

    def test_detects_child_failure_even_when_session_helper_succeeds(self):
        terminal = self.terminal("restore()\nraise SystemExit(7)\n")
        with self.assertRaisesRegex(AssertionError, "PTY child failed"):
            terminal.finish()
        self.assertEqual(terminal.status["returncode"], 7)
        self.assertEqual(terminal.process.returncode, 0)

    def test_exit_signals_reach_child_without_terminating_session_leader(self):
        for number in (signal.SIGTERM, signal.SIGHUP):
            with self.subTest(signal=number):
                terminal = self.terminal("""
def stop(number, frame):
    restore()
    raise SystemExit(0)

signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGHUP, stop)
os.write(1, b"ready")
while True:
    signal.pause()
""")
                terminal.expect("ready")
                terminal.send_signal(number)
                terminal.finish()
                terminal.close()

    def test_close_stops_a_running_child_and_is_idempotent(self):
        terminal = self.terminal('os.write(1, b"ready")\ntime.sleep(60)\n')
        terminal.expect("ready")
        terminal.close()
        terminal.close()
        self.assertIsNotNone(terminal.process.returncode)
        with self.assertRaises(ProcessLookupError):
            os.kill(terminal.status["pid"], 0)


if __name__ == "__main__":
    unittest.main()
