"""Regression tests for PTY ownership and terminal cleanup verification."""

import copy
import errno
import os
import signal
import sys
import termios
import unittest
from unittest.mock import patch

from tui_smoke import Terminal, terminal_attribute_differences


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

    def test_reports_the_setting_that_was_not_restored(self):
        terminal = self.terminal("""
restore()
broken = termios.tcgetattr(0)
broken[3] &= ~termios.ECHO
termios.tcsetattr(0, termios.TCSANOW, broken)
""")
        with self.assertRaisesRegex(AssertionError, r"terminal attributes not restored.*lflag:.*xor"):
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


class TerminalAttributeTests(unittest.TestCase):
    # Darwin values from bsd/sys/termios.h, independent of the host running these tests.
    PENDIN = 0x20000000
    ECHO = 0x8
    ISIG = 0x80
    ICANON = 0x100
    IEXTEN = 0x400

    def setUp(self):
        self.original = [0x100, 0x3, 0x4B00, self.ECHO | self.ISIG | self.ICANON | self.IEXTEN,
                         9600, 9600, [3, 28, 127, 21]]

    def compare_on_darwin(self, restored):
        with patch("tui_smoke.sys.platform", "darwin"), \
                patch("tui_smoke.termios.PENDIN", self.PENDIN, create=True):
            return terminal_attribute_differences(self.original, restored)

    def test_darwin_canonical_restore_can_set_pending_input_state(self):
        restored = copy.deepcopy(self.original)
        restored[3] |= self.PENDIN
        self.assertNotEqual(self.original, restored)  # the former equality check fails
        self.assertEqual(self.compare_on_darwin(restored), [])

    def test_darwin_pending_input_does_not_hide_missing_mode_settings(self):
        for flag in (self.ECHO, self.ISIG, self.ICANON, self.IEXTEN):
            with self.subTest(flag=hex(flag)):
                restored = copy.deepcopy(self.original)
                restored[3] = (restored[3] | self.PENDIN) & ~flag
                differences = self.compare_on_darwin(restored)
                self.assertEqual(len(differences), 1)
                self.assertIn(f"xor {flag:#x}", differences[0])

    def test_darwin_still_checks_other_flags_speeds_and_control_characters(self):
        for index, name in ((0, "iflag"), (1, "oflag"), (2, "cflag"),
                            (4, "ispeed"), (5, "ospeed"), (6, "cc")):
            with self.subTest(field=name):
                restored = copy.deepcopy(self.original)
                restored[3] |= self.PENDIN
                if index == 6:
                    restored[index][0] += 1
                else:
                    restored[index] ^= 1
                differences = self.compare_on_darwin(restored)
                self.assertEqual(len(differences), 1)
                self.assertTrue(differences[0].startswith(f"{name}:"), differences)

    def test_linux_keeps_the_full_flag_comparison(self):
        restored = copy.deepcopy(self.original)
        restored[3] |= self.PENDIN
        with patch("tui_smoke.sys.platform", "linux"):
            differences = terminal_attribute_differences(self.original, restored)
        self.assertEqual(len(differences), 1)
        self.assertIn("lflag:", differences[0])


if __name__ == "__main__":
    unittest.main()
