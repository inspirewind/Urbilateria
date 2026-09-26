"""Linux/macOS PTY smoke tests; Python is a test tool, not a UI dependency.

Usage: python3 tests/tui_smoke.py target/release/urb [--panic-test BIN_TEST_EXECUTABLE]
"""

import argparse
import errno
import fcntl
import json
import math
import os
from pathlib import Path
import pty
import re
import select
import signal
import struct
import subprocess
import sys
import tempfile
import termios
import time


ANSI = re.compile(rb"\x1b\[[0-?]*[ -/]*[@-~]")


def terminal_attributes(fd):
    attributes = termios.tcgetattr(fd)
    # c_cc entries may be bytes or integers, depending on canonical mode and Python version.
    return attributes[:6] + [[value[0] if isinstance(value, bytes) else value
                              for value in attributes[6]]]


def terminal_attribute_differences(expected, actual):
    """Compare settings, excluding Darwin's kernel-maintained pending-input state."""
    differences = []
    for index, name in enumerate(("iflag", "oflag", "cflag", "lflag", "ispeed", "ospeed", "cc")):
        before, after = expected[index], actual[index]
        if name == "lflag" and sys.platform == "darwin":
            # XNU sets PENDIN when tcsetattr restores ICANON without flushing input.
            # It is queue state, not a failure to restore raw-mode settings. Keep every
            # other bit (including ICANON, ECHO, ISIG and IEXTEN) in the comparison.
            # https://github.com/apple-oss-distributions/xnu/blob/main/bsd/kern/tty.c
            before &= ~termios.PENDIN
            after &= ~termios.PENDIN
        if before != after:
            if index < 4:
                differences.append(f"{name}: {before:#x} -> {after:#x} (xor {before ^ after:#x})")
            else:
                differences.append(f"{name}: {before!r} -> {after!r}")
    return differences


def run_pty_session(report_fd, command):
    """Keep the session leader alive until the tested child has exited and been inspected.

    macOS revokes the controlling terminal when its session leader exits. A separate leader
    lets us capture the child's actual terminal state before that happens, just like a shell.
    """
    fcntl.ioctl(0, termios.TIOCSCTTY, 0)
    original = termios.tcgetattr(0)
    try:
        with os.fdopen(report_fd, "w", buffering=1) as report:
            # Both snapshots belong to the initialized controlling terminal, and neither
            # reads the parent's slave descriptor after the session has been revoked.
            baseline = terminal_attributes(0)
            child = subprocess.Popen(command)
            report.write(json.dumps({"pid": child.pid, "original": baseline}) + "\n")
            returncode = child.wait()
            report.write(json.dumps({
                "returncode": returncode,
                "attributes": terminal_attributes(0),
            }) + "\n")
    finally:
        # Capture above must precede harness cleanup, so missing application cleanup still fails.
        termios.tcsetattr(0, termios.TCSANOW, original)


class Terminal:
    def __init__(self, command):
        self.master, self.slave = pty.openpty()
        self.output = bytearray()
        self.status = {}
        self.status_buffer = bytearray()
        self.status_fd, report_fd = os.pipe()
        self.readers = {self.master, self.status_fd}
        self.closed = False
        self.width = 110
        self.height = 36
        fcntl.ioctl(self.slave, termios.TIOCSWINSZ, struct.pack("HHHH", self.height, self.width, 0, 0))
        try:
            self.process = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), "--pty-session", str(report_fd), *command],
                stdin=self.slave, stdout=self.slave, stderr=self.slave,
                start_new_session=True,
                pass_fds=(report_fd,),
                env={**os.environ, "TERM": "xterm-256color"},
            )
        except BaseException:
            for fd in (self.master, self.slave, self.status_fd):
                os.close(fd)
            raise
        finally:
            os.close(report_fd)

    def read(self, timeout=0.05):
        ready = select.select(list(self.readers), [], [], timeout)[0]
        for fd in ready:
            try:
                data = os.read(fd, 65536)
            except OSError as error:
                if error.errno != errno.EIO:
                    raise
                data = b""
            if not data:
                self.readers.discard(fd)
            elif fd == self.master:
                self.output.extend(data)
            else:
                self.status_buffer.extend(data)
                while b"\n" in self.status_buffer:
                    line, _, rest = self.status_buffer.partition(b"\n")
                    self.status.update(json.loads(line))
                    self.status_buffer = bytearray(rest)
        return bool(ready)

    def send(self, data):
        os.write(self.master, data)

    def send_signal(self, number):
        deadline = time.monotonic() + 5
        while "pid" not in self.status and self.process.poll() is None and time.monotonic() < deadline:
            self.read()
        assert "pid" in self.status, ("PTY child did not start", self.output[-3000:])
        os.kill(self.status["pid"], number)

    def resize(self, width, height):
        self.width = width
        self.height = height
        fcntl.ioctl(self.slave, termios.TIOCSWINSZ, struct.pack("HHHH", height, width, 0, 0))

    def expect(self, text, timeout=5):
        needle = re.sub(rb"\s+", b"", text.encode())
        deadline = time.monotonic() + timeout
        redraw_at = time.monotonic() + 0.5
        while time.monotonic() < deadline:
            self.read()
            if needle in re.sub(rb"\s+", b"", ANSI.sub(b"", self.output)):
                return
            if self.process.poll() is not None:
                break
            if time.monotonic() >= redraw_at:
                # A full redraw avoids depending on Ratatui's character-level diff output.
                self.resize(110 if self.width != 110 else 111, self.height)
                redraw_at = time.monotonic() + 0.5
        raise AssertionError(f"missing {text!r}: {self.output[-6000:]!r}")

    def assert_running(self):
        assert self.process.poll() is None, self.output
        assert "returncode" not in self.status, self.status

    def finish(self):
        deadline = time.monotonic() + 5
        while self.process.poll() is None and time.monotonic() < deadline:
            self.read()
        assert self.process.poll() == 0, ("PTY session failed or timed out", self.process.poll(), self.output[-3000:])
        deadline = time.monotonic() + 0.5
        while time.monotonic() < deadline and self.read(0.05):
            pass
        assert self.status.get("returncode") == 0, ("PTY child failed", self.status, self.output[-3000:])
        assert "original" in self.status and "attributes" in self.status, ("missing terminal snapshots", self.status)
        differences = terminal_attribute_differences(self.status["original"], self.status["attributes"])
        assert not differences, (
            f"terminal attributes not restored on {sys.platform}: {'; '.join(differences)}; "
            f"before={self.status['original']!r}; after={self.status['attributes']!r}"
        )
        assert b"\x1b[?1049l" in self.output, "alternate screen not restored"
        assert b"\x1b[?2004l" in self.output, "bracketed paste not disabled"
        assert b"\x1b[?25h" in self.output, "cursor not restored"

    def close(self):
        if self.closed:
            return
        try:
            self.read(0)
            if self.process.poll() is None and "pid" in self.status and "returncode" not in self.status:
                try:
                    os.kill(self.status["pid"], signal.SIGKILL)
                except ProcessLookupError:
                    pass
                deadline = time.monotonic() + 2
                while self.process.poll() is None and time.monotonic() < deadline:
                    self.read()
            if self.process.poll() is None or "returncode" not in self.status:
                # This isolated process group contains only the helper and its test child.
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            self.process.wait(timeout=5)
        finally:
            # macOS may already have revoked the slave; closing descriptors is still valid.
            for fd in (self.master, self.slave, self.status_fd):
                os.close(fd)
            self.closed = True


def fixture(path, complete=False):
    path.mkdir()
    config = {
        "model_type": "glm_moe_dsa", "hidden_size": 8, "num_hidden_layers": 2,
        "num_attention_heads": 2, "vocab_size": 257, "intermediate_size": 12,
        "moe_intermediate_size": 4, "first_k_dense_replace": 1, "n_routed_experts": 8,
        "n_shared_experts": 1, "num_experts_per_tok": 2, "q_lora_rank": 4,
        "kv_lora_rank": 3, "qk_nope_head_dim": 2, "qk_rope_head_dim": 2,
        "qk_head_dim": 4, "v_head_dim": 3, "max_position_embeddings": 64,
    }
    tensors = {"model.embed_tokens.weight": [257, 8]}
    if complete:
        config["num_hidden_layers"] = 1
        tensors.update({
            "model.norm.weight": [8], "lm_head.weight": [257, 8],
            "model.layers.0.input_layernorm.weight": [8],
            "model.layers.0.post_attention_layernorm.weight": [8],
            "model.layers.0.self_attn.q_a_proj.weight": [4, 8],
            "model.layers.0.self_attn.q_b_proj.weight": [8, 4],
            "model.layers.0.self_attn.kv_a_proj_with_mqa.weight": [5, 8],
            "model.layers.0.self_attn.kv_b_proj.weight": [10, 3],
            "model.layers.0.self_attn.o_proj.weight": [8, 6],
            "model.layers.0.self_attn.q_a_layernorm.weight": [4],
            "model.layers.0.self_attn.kv_a_layernorm.weight": [3],
            "model.layers.0.mlp.gate_proj.weight": [12, 8],
            "model.layers.0.mlp.up_proj.weight": [12, 8],
            "model.layers.0.mlp.down_proj.weight": [8, 12],
        })
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    header = {}
    offset = 0
    for name, shape in tensors.items():
        size = 4
        for dimension in shape:
            size *= dimension
        header[name] = {"dtype": "F32", "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size
    header = json.dumps(header).encode()
    header += b" " * (-len(header) % 8)
    (path / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + bytes(offset))
    visible = set(range(33, 127)) | set(range(161, 173)) | set(range(174, 256))
    alphabet = {}
    extra = 256
    for byte in range(256):
        alphabet[chr(byte if byte in visible else extra)] = byte
        if byte not in visible:
            extra += 1
    split_pattern = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
    tokenizer = {
        "version": "1.0", "truncation": None, "padding": None, "normalizer": None,
        "added_tokens": [{"id": 256, "content": "<|user|>", "special": True,
                          "single_word": False, "lstrip": False, "rstrip": False, "normalized": False}],
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": split_pattern}, "behavior": "Isolated", "invert": False},
            {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": False},
        ]},
        "post_processor": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": False, "use_regex": True},
        "decoder": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": True, "use_regex": True},
        "model": {"type": "BPE", "dropout": None, "unk_token": None,
                  "continuing_subword_prefix": None, "end_of_word_suffix": None,
                  "fuse_unk": False, "byte_fallback": False, "ignore_merges": True,
                  "vocab": alphabet, "merges": []},
    }
    (path / "tokenizer.json").write_text(json.dumps(tokenizer), encoding="utf-8")


def browsing_and_text(binary, model):
    terminal = Terminal([binary, "ui"])
    try:
        terminal.resize(110, 64)
        terminal.expect("Welcome to Urbilateria")

        def run(command, *expected, paste=False):
            terminal.send(b"/clear\r")
            terminal.expect("Type a message")
            terminal.output.clear()
            data = command.encode()
            terminal.send((b"\x1b[200~" + data + b"\x1b[201~" if paste else data) + b"\r")
            for text in expected:
                terminal.expect(text)
            terminal.assert_running()

        version = subprocess.check_output([binary, "--version"], text=True).strip()
        run("/ver\t", "Version", version)
        run("/decode 65", "No model selected")
        run(f'/li\tq_a --limit 1 --model "{model}"', "Tensor list", "2 total", "1 returned", "q_a_layernorm.weight")
        run("/ex\t", "Model explanation", "GLM-5.2 token path", "embedding [8]")
        run("/list nonexistent-fragment", "No matching tensors")
        run("/prob\tmodel.norm.weight --samples 4", "Tensor probe", "F32 [8]", "n=4", "nonfinite=0")
        run("/probe nonexistent-tensor", "Error", "nonexistent-tensor")
        run('/to\t"hello"', "Tokenization complete", "104, 101, 108, 108, 111")
        run('/tokenize "中文🙂\n/quit" --chat --no-thinking', "Tokenization complete", "中文🙂", "/quit", paste=True)
        run("/de\t104,101,108,108,111,256", "Decoding complete", "hello<|user|>")
        run("/decode 104,101,108,108,111,256 --skip-special", "Decoding complete", "Text  hello")
        run('/tokenize ""', "Tokenization complete", "Token count  0", "(empty)")
        run("/decode 4294967295", "Error")
        run('/tokenize "hello" --model "/nonexistent/urb-tui-checkpoint"', "Error")
        run('/tokenize "hello"', "Tokenization complete", "GLM-5.2")  # failed selection retained the model
        terminal.send(b"/quit\r")
        terminal.finish()
    finally:
        terminal.close()


def generation_fixture(path, chat=False):
    """A tiny real GLM checkpoint: A -> UTF-8 bytes of 中文🙂 -> EOS.

    Zero attention/MLP leaves the residual embedding unchanged. Unit-circle embeddings
    and matching LM-head rows make the next byte the unique greedy winner.
    """
    fixture(path, complete=True)
    config_file = path / "config.json"
    config = json.loads(config_file.read_text())
    config["eos_token_id"] = [67]
    if chat:
        config["max_position_embeddings"] = 256
    config_file.write_text(json.dumps(config), encoding="utf-8")
    weights = path / "model.safetensors"
    data = bytearray(weights.read_bytes())
    header_length = struct.unpack_from("<Q", data)[0]
    header = json.loads(data[8:8 + header_length])

    def write(name, index, value):
        offset = 8 + header_length + header[name]["data_offsets"][0] + 4 * index
        struct.pack_into("<f", data, offset, value)

    for name, tensor in header.items():
        if "norm.weight" in name:
            for index in range(tensor["shape"][0]):
                write(name, index, 1.0)
    tokens = [65, *"中文🙂".encode(), 67]
    assert len(set(tokens)) == len(tokens)
    for index, (source, target) in enumerate(zip(tokens, tokens[1:])):
        angle = 2 * math.pi * index / (len(tokens) - 1)
        for dimension, value in enumerate((math.cos(angle), math.sin(angle))):
            write("model.embed_tokens.weight", source * 8 + dimension, value)
            write("lm_head.weight", target * 8 + dimension, value)
    if chat:
        # Native GLM assistant prefixes end in ">"; start the same deterministic reply.
        write("model.embed_tokens.weight", ord(">") * 8, 1.0)
    weights.write_bytes(data)


def conversation(binary, root):
    model = root / "chat-model"
    generation_fixture(model, chat=True)
    # The CLI keeps one process and model alive across messages, including /clear.
    persistent_options = ["--ram-gib", "2", "--allow-large-model", "--max-new-tokens", "16",
                          "--threads", "1", "--no-thinking"]
    resident = subprocess.run([binary, "chat", str(model), *persistent_options],
                              input="first\nfollow-up\n/clear\nnew topic\n/quit\n",
                              capture_output=True, text=True, timeout=30)
    assert resident.returncode == 0, resident.stderr
    assert resident.stdout == "中文🙂\n" * 3, (resident.stdout, resident.stderr)
    reused = [int(n) for n in re.findall(r"kv cache: reused=(\d+) tokens", resident.stderr)]
    assert len(reused) == 3 and reused[0] == reused[2] == 0 and reused[1] > 0, resident.stderr
    assert resident.stderr.count("preflight:") == 2, resident.stderr

    # Both structured requests share a single model load and produce separate completion frames.
    requests = [
        {"args": persistent_options, "conversation": {"turns": [], "prompt": "first"}},
        {"args": persistent_options, "conversation": {"turns": [
            {"user": "first", "assistant": "中文🙂", "thinking": False}], "prompt": "follow-up"}},
    ]
    protocol = subprocess.run([binary, "chat", str(model), "--session-json"],
                              input="".join(json.dumps(request) + "\n" for request in requests),
                              capture_output=True, text=True, timeout=30)
    assert protocol.returncode == 0, protocol.stderr
    records = [json.loads(line) for line in protocol.stdout.splitlines()]
    assert [r for r in records if r["event"] == "finished"] == [{"event": "finished", "error": None}] * 2, records
    assert "".join(r["text"] for r in records if r["event"] == "text") == "中文🙂\n" * 2, records
    assert protocol.stderr.count("preflight:") == 1, protocol.stderr
    assert protocol.stderr.count("URB_SESSION_END\n") == 2, protocol.stderr
    options = ["--ram-gib", "2", "--allow-large-model", "--max-new-tokens", "16",
               "--threads", "1", "--no-thinking", "--chat-stdin"]
    chat = {"turns": [{"user": "first", "assistant": "中文🙂", "thinking": False}], "prompt": "follow-up"}
    cli = subprocess.run([binary, "generate", str(model), *options],
                         input=json.dumps(chat), capture_output=True, text=True, timeout=30)
    assert cli.returncode == 0, cli.stderr
    assert cli.stdout == "中文🙂\n", (cli.stdout, cli.stderr)
    assert "chat: history=1 turns, dropped=0 turns" in cli.stderr, cli.stderr
    chat["turns"].insert(0, {"user": "old " * 100, "assistant": "old answer", "thinking": False})
    cli = subprocess.run([binary, "generate", str(model), *options],
                         input=json.dumps(chat), capture_output=True, text=True, timeout=30)
    assert cli.returncode == 0, cli.stderr
    assert "history=1 turns, dropped=1 turns" in cli.stderr, cli.stderr
    assert cli.stdout == "中文🙂\n", cli.stdout
    for invalid in ({"turns": [], "prompt": "x" * 300}, {"turns": [], "prompt": " "}):
        cli = subprocess.run([binary, "generate", str(model), *options],
                             input=json.dumps(invalid), capture_output=True, text=True, timeout=30)
        assert cli.returncode != 0, cli.stderr
        assert not cli.stdout, cli.stdout
        assert "preflight:" not in cli.stderr, cli.stderr  # rejected before loading weights

    terminal = Terminal([binary, "ui"])
    try:
        terminal.resize(110, 64)
        terminal.expect("Welcome to Urbilateria")
        terminal.send(f'/inspect "{model}"\r'.encode())
        terminal.expect("Inspection complete")
        terminal.expect("/inspect for details")
        # Exercise Linux's automatic RAM selection; Darwin uses the explicit budget path.
        ram_option = " --ram-gib 2" if sys.platform == "darwin" else ""
        terminal.send(f"/settings{ram_option} --max-new-tokens 16 --threads 1 --no-thinking\r".encode())
        terminal.expect("Conversation settings")
        terminal.send(b"/clear\r")
        terminal.expect("Type a message")
        terminal.output.clear()
        terminal.send(b'\x1b[200~don\'t close "the quote\n--profile\x1b[201~\r')
        terminal.expect("Generation complete")
        terminal.expect("Text  中文🙂")
        terminal.expect("history=0 turns")
        terminal.expect("1 turns")
        resident_children = None
        if sys.platform == "linux":
            pid = terminal.status["pid"]
            children_path = Path(f"/proc/{pid}/task/{pid}/children")
            resident_children = children_path.read_text().split()
            assert len(resident_children) == 1, resident_children
        terminal.output.clear()
        terminal.send("继续刚才的回答\r".encode())
        terminal.expect("Generation complete")
        terminal.expect("history=1 turns")
        terminal.expect("2 turns")
        assert re.search(rb"reused=[1-9]\d*", ANSI.sub(b"", terminal.output)), terminal.output[-3000:]
        if resident_children is not None:
            assert children_path.read_text().split() == resident_children
        terminal.send(b"/clear\r")
        terminal.expect("Type a message")
        if resident_children is not None:
            assert children_path.read_text().strip() == "", "clear must reap the resident child"
        terminal.output.clear()
        # Force a redraw so presence below proves the model survives clearing scrollback.
        terminal.resize(111, 64)
        terminal.expect("chat-model")
        terminal.expect("/inspect for details")
        terminal.send(b"New topic\r")
        terminal.expect("Generation complete")
        terminal.expect("history=0 turns")
        terminal.send(b"/quit\r")
        terminal.finish()
    finally:
        terminal.close()


def generation(binary, root):
    model = root / "generation model 中文"
    generation_fixture(model)
    options = '--ram-gib 2 --allow-large-model --raw-prompt --max-new-tokens 16'
    cli = subprocess.run([
        binary, "generate", str(model), "--prompt", "[gMASK]<sop>A",
        *options.split(), "--threads", "1",
    ], capture_output=True, text=True, timeout=30)
    assert cli.returncode == 0, cli.stderr
    assert cli.stdout == "中文🙂\n", (cli.stdout, cli.stderr)
    assert "metrics: input=13" in cli.stderr and "output=11" in cli.stderr, cli.stderr
    assert "total=24 tok" in cli.stderr and "TTFT " in cli.stderr, cli.stderr
    assert "URB_PROGRESS " not in cli.stderr, cli.stderr
    metrics = subprocess.run([
        binary, "generate", str(model), "--prompt", "[gMASK]<sop>A",
        *options.split(), "--threads", "1", "--progress-json",
    ], capture_output=True, text=True, timeout=30)
    assert metrics.returncode == 0, metrics.stderr
    assert metrics.stdout == cli.stdout
    snapshots = [json.loads(line.removeprefix("URB_PROGRESS "))
                 for line in metrics.stderr.splitlines() if line.startswith("URB_PROGRESS ")]
    assert len(snapshots) >= 4, metrics.stderr
    assert snapshots[0]["generated_tokens"] == 0 and snapshots[0]["ttft_seconds"] is None
    first = next(snapshot for snapshot in snapshots if snapshot["generated_tokens"])
    assert first["generated_tokens"] == 1, snapshots  # A token ID, even before a full Unicode character.
    final = snapshots[-1]
    assert final["status"] == "complete" and final["generated_tokens"] == 11, final
    assert final["total_tokens"] == 24 and final["prompt_tokens"] == 13, final
    assert 0 <= final["ttft_seconds"] <= final["elapsed_seconds"], final
    assert final["decode_tokens_per_second"] > 0, final
    # Immediate EOS counts as one selected token but has no post-first-token decode rate.
    eos = subprocess.run([
        binary, "generate", str(model), "--prompt", "[gMASK]<sop>🙂",
        *options.split(), "--threads", "1", "--progress-json",
    ], capture_output=True, text=True, timeout=30)
    assert eos.returncode == 0 and eos.stdout == "\n", (eos.stdout, eos.stderr)
    final = json.loads(next(line.removeprefix("URB_PROGRESS ")
                           for line in reversed(eos.stderr.splitlines()) if line.startswith("URB_PROGRESS ")))
    assert final["generated_tokens"] == 1 and final["decode_tokens_per_second"] is None, final
    assert final["ttft_seconds"] is not None, final
    live = subprocess.run([
        binary, "generate", str(model), "--prompt", "[gMASK]<sop>A",
        *options.split(), "--threads", "1", "--progress",
    ], capture_output=True, text=True, timeout=30)
    assert live.returncode == 0 and live.stdout == cli.stdout, (live.stdout, live.stderr)
    assert live.stderr.count("metrics:") >= 4 and "\x1b" not in live.stderr, live.stderr
    terminal = Terminal([binary, "ui"])
    try:
        terminal.resize(110, 64)
        terminal.expect("Welcome to Urbilateria")

        def run(command, *expected):
            terminal.send(b"/clear\r")
            terminal.expect("Type a message")
            terminal.output.clear()
            terminal.send(command.encode() + b"\r")
            for text in expected:
                terminal.expect(text, timeout=15)
            terminal.assert_running()

        run('/generate hi --ram-gib 2', 'requires --allow-large-model')
        run('/generate hi --allow-large-model', 'requires an explicit --ram-gib')
        run(f'/gen\t"[gMASK]<sop>A" --model "{model}" {options} --threads 1',
            "Generation complete", "Text  中文🙂", "preflight:", "new_tokens=11", "stop=eos:67",
            "out 11", "total 24 tok", "tok/s", "TTFT", "F2 expand")
        terminal.send(b"\x1bOQ")  # F2 opens a full runtime view with independent scrolling.
        terminal.expect("F2/Esc close")
        terminal.expect("preflight:")
        terminal.output.clear()
        terminal.send(b"\x1b")
        terminal.expect("Message the model")
        run('/generate hi --model /nonexistent/urb-tui-checkpoint --ram-gib 2 --allow-large-model',
            "Generation failed", "error:", "/nonexistent/urb-tui-checkpoint")
        profile = root / "generation profile.json"
        trace = root / "generation trace.json"
        # A successful command selects the model; a failed one keeps it. Each request can
        # choose a different CPU pool size because generation runs in its own process.
        run(f'/generate --prompt "[gMASK]<sop>A" {options} --threads 2 --profile '
            f'--profile-json "{profile}" --profile-trace "{trace}"',
            "Generation complete", "Text  中文🙂", "out 11")
        terminal.send(b"\x1bOQ\x1b[1;5H")  # F2 + Ctrl+Home: inspect earlier logs above the profile.
        terminal.expect("F2/Esc close")
        terminal.expect("stop=eos:67")
        terminal.output.clear()
        terminal.send(b"\x1bOQ")
        terminal.expect("Message the model")
        assert json.loads(profile.read_text())
        assert "traceEvents" in json.loads(trace.read_text())
        run('/generate "[gMASK]<sop>A" --ram-gib 0.001 --allow-large-model --raw-prompt',
            "Generation failed", "RAM plan is infeasible")
        run('/tokenize hello', "Tokenization complete", "GLM-5.2")
        terminal.send(b"/quit\r")
        terminal.finish()
    finally:
        terminal.close()


def generation_cancel_and_exit(binary, root):
    model = root / "blocked-generation"
    model.mkdir()
    fifo = model / "config.json"
    os.mkfifo(fifo)
    command = b'/generate hello --ram-gib 2 --allow-large-model\r'
    for mode in ("escape", "quit", "ctrl-c", "ctrl-d", "sigterm", "sighup"):
        terminal = Terminal([binary, "ui", str(model)])
        writer = None
        try:
            terminal.resize(110, 64)
            terminal.expect("Welcome to Urbilateria")
            terminal.send(command)
            terminal.expect("Generating")
            terminal.expect("TTFT")
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                try:
                    writer = os.open(fifo, os.O_WRONLY | os.O_NONBLOCK)
                    break
                except OSError as error:
                    if error.errno != errno.ENXIO:
                        raise
                    terminal.read()
            assert writer is not None, "generation child never opened the FIFO"
            terminal.send(b"/clear\r/help\r")
            terminal.send(b"\x1b[1;5H")  # help wraps beside Runtime; scroll to its heading
            terminal.expect("Commands & keys")
            terminal.send(b"/version\r")
            terminal.expect("Version")
            if mode == "escape":
                terminal.send(b"\x1b")
                terminal.expect("Generation cancelled")
                terminal.assert_running()
                terminal.send(b"/quit\r")
            elif mode == "quit":
                terminal.send(b"/quit\r")
            elif mode == "ctrl-c":
                terminal.send(b"\x03")
            elif mode == "ctrl-d":
                terminal.send(b"\x04")
            else:
                terminal.send_signal(signal.SIGTERM if mode == "sigterm" else signal.SIGHUP)
            terminal.finish()
            try:
                os.write(writer, b" ")
            except BrokenPipeError:
                pass
            else:
                raise AssertionError("generation child survived cancellation/exit")
        finally:
            terminal.close()
            if writer is not None:
                os.close(writer)


def planning_and_preflight(binary, model, complete_model):
    terminal = Terminal([binary, "ui", str(model)])
    try:
        terminal.resize(110, 64)
        terminal.expect("Welcome to Urbilateria")
        terminal.send(b"/pl\t--ram-gib 2 --context 16\r")
        terminal.expect("Plan complete")
        terminal.expect("2.00 GiB")
        terminal.expect("missing")
        terminal.send(b"/pre\t--context 16\r")
        terminal.expect("Error")  # the partial fixture must fail strict preflight
        terminal.output.clear()
        terminal.send(f'/preflight "{complete_model}" --context 16 --expert-slots 2\r'.encode())
        terminal.expect("Preflight complete")
        terminal.expect("16 requested")
        terminal.expect("2 persistent slots")
        terminal.send(b"/clear\r")
        terminal.expect("Type a message")
        terminal.output.clear()
        terminal.send(b"/plan --ram-gib 0.001 --context 16\r")
        terminal.expect("NOT FEASIBLE")  # a budget failure is a visible result, not a UI crash
        terminal.send(b"/clear\r")
        terminal.expect("Type a message")
        terminal.output.clear()
        terminal.send(b"/preflight --context 8\r")  # successful preflight selected the complete model
        terminal.expect("Preflight complete")
        terminal.expect("8 requested")
        terminal.send(b"/quit\r")
        terminal.finish()
    finally:
        terminal.close()


def interactive(binary, model):
    terminal = Terminal([binary, "ui", str(model)])
    try:
        terminal.resize(110, 64)
        terminal.expect("Welcome to Urbilateria")
        terminal.send(b"\x1b[200~/quit\n\x1b[201~")
        terminal.expect("/quit")
        terminal.assert_running()  # newline inside bracketed paste did not execute /quit
        terminal.send(b"\x15\x08\x15")  # remove trailing empty line and command
        terminal.send(b"/he\t\r")
        terminal.expect("Commands & keys")
        terminal.send(b"/inspect\r")
        terminal.expect("Inspection complete")
        terminal.expect("GLM-5.2")
        terminal.expect("missing")  # incomplete-checkpoint warning is still visible
        terminal.send(b'/inspect "/nonexistent/urb-tui-checkpoint"\r')
        terminal.expect("Error")
        terminal.expect("/nonexistent/urb-tui-checkpoint")
        terminal.send(b"/inspect\r")  # a failed selection must retain the last successful model
        terminal.output.clear()
        terminal.expect("Inspection complete")
        terminal.resize(30, 8)
        terminal.expect("Enlarge to 36 x 10")
        terminal.resize(100, 30)
        terminal.send(b"\x1b[5~\x1b[6~")  # PageUp / PageDown after a resize
        terminal.send(b"/quit\r")
        terminal.finish()
    finally:
        terminal.close()


def exit_modes(binary):
    for mode in ("ctrl-c", "ctrl-d", "sigterm", "sighup"):
        terminal = Terminal([binary, "ui"])
        try:
            terminal.expect("Welcome to Urbilateria")
            if mode == "ctrl-c":
                terminal.send(b"\x03")
            elif mode == "ctrl-d":
                terminal.send(b"\x04")
            else:
                terminal.send_signal(signal.SIGTERM if mode == "sigterm" else signal.SIGHUP)
            terminal.finish()
        finally:
            terminal.close()


def busy_exit(binary, root):
    model = root / "blocked-read"
    model.mkdir()
    os.mkfifo(model / "config.json")
    os.mkfifo(model / "model.safetensors")
    for command, status in [
        (b"/inspect\r", "Inspecting"),
        (b"/plan --ram-gib 2\r", "Planning"),
        (b"/preflight\r", "Validating"),
        (b"/list\r", "Listing"),
        (b"/explain\r", "Explaining"),
        (b"/probe weight\r", "Sampling"),
        (b'/tokenize "hello"\r', "Tokenizing"),
        (b"/decode 1\r", "Decoding"),
    ]:
        terminal = Terminal([binary, "ui", str(model)])
        try:
            terminal.resize(110, 64)
            terminal.expect("Welcome to Urbilateria")
            terminal.send(command)
            terminal.expect(status)
            terminal.send(b"/help\r")
            terminal.expect("Commands & keys")  # input is responsive during a blocked filesystem read
            terminal.send(b"/version\r")
            terminal.expect("Version")
            terminal.send(b"\x03")
            terminal.finish()  # no blocking join of the read-only worker
        finally:
            terminal.close()


def panic_cleanup(test_binary):
    terminal = Terminal([
        test_binary, "ui::terminal::tests::restores_terminal_on_panic",
        "--exact", "--ignored", "--nocapture",
    ])
    try:
        terminal.finish()
        assert b"intentional terminal cleanup test" in terminal.output
    finally:
        terminal.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary")
    parser.add_argument("--panic-test")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="urb-tui-") as directory:
        root = Path(directory)
        model = root / "model with spaces 中文"
        fixture(model)
        interactive(args.binary, model)
        complete_model = root / "complete model 中文"
        fixture(complete_model, complete=True)
        planning_and_preflight(args.binary, model, complete_model)
        browsing_and_text(args.binary, complete_model)
        generation(args.binary, root)
        conversation(args.binary, root)
        generation_cancel_and_exit(args.binary, root)
        exit_modes(args.binary)
        busy_exit(args.binary, root)
    if args.panic_test:
        panic_cleanup(args.panic_test)
    print("TUI PTY smoke passed: all 14 commands, pinned model, Runtime panel, CLI/TUI token metrics, multi-turn chat, context trimming, generation, cancellation, editing, paste, resize, busy exit, terminal restoration")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--pty-session":
        run_pty_session(int(sys.argv[2]), sys.argv[3:])
    else:
        main()
