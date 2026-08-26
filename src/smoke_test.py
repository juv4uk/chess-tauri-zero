"""One-command setup check -- run this right after `pip install -r
requirements.txt`, before opening the GUI, to catch environment
problems in seconds instead of "click app, see a cryptic Rust error,
go ask someone." Directly motivated by two real bugs found live
during first-ever Windows setup this session (dead `python-chess`
PyPI name, `os.getloadavg()` not existing on Windows) -- both would
have been caught by this script immediately instead of by trial and
error through the GUI.

Usage (from src/, inside the activated venv):
    python3 smoke_test.py
"""
import subprocess
import sys
import time


def check(name, fn):
    try:
        detail = fn()
        print(f"[OK]   {name}" + (f" -- {detail}" if detail else ""))
        return True
    except Exception as e:
        print(f"[FAIL] {name} -- {e}")
        return False


def check_imports():
    import chess  # noqa: F401
    import h5py  # noqa: F401
    import numpy  # noqa: F401
    import torch
    return f"torch {torch.__version__}"


def check_cuda():
    import torch
    if not torch.cuda.is_available():
        return "no CUDA GPU detected -- CPU only, correct and fine, just slower"
    name = torch.cuda.get_device_name(0)
    x = torch.rand(4, 4, device="cuda")
    y = (x @ x).sum().item()  # a real matmul, not just an availability flag
    return f"{name}, real matmul on GPU succeeded (sum={y:.3f})"


def check_weights_file():
    import os
    path = "../data/model/model_best_weight.h5"
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing -- clone did not include it, or wrong cwd (run from src/)")
    size = os.path.getsize(path)
    return f"{path} exists, {size:,} bytes"


def check_uci_handshake():
    """Actually spawns uci_torch.py and speaks real UCI to it -- the
    same protocol Tauri's sidecar uses, but run directly here so a
    failure shows a plain Python traceback instead of a Tauri
    "engine not started" dead end with no context."""
    proc = subprocess.Popen(
        [sys.executable, "chess_zero/play_game/uci_torch.py"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1,
    )
    try:
        proc.stdin.write("uci\nisready\n")
        proc.stdin.flush()
        deadline = time.time() + 60  # model load + first CUDA call can be slow
        seen = set()
        while time.time() < deadline and not {"uciok", "readyok"} <= seen:
            line = proc.stdout.readline()
            if not line:
                break
            seen.add(line.strip())
        missing = {"uciok", "readyok"} - seen
        if missing:
            stderr = proc.stderr.read(2000)
            raise RuntimeError(f"missing {missing} within 60s; stderr: {stderr[:500]}")
        return "uciok + readyok received from a real spawned sidecar process"
    finally:
        proc.stdin.write("quit\n")
        try:
            proc.stdin.flush()
            proc.wait(timeout=5)
        except Exception:
            proc.kill()


def main():
    results = [
        check("Python packages import (chess, h5py, numpy, torch)", check_imports),
        check("Model weights file present", check_weights_file),
        check("CUDA / GPU (informational -- CPU-only is a pass too)", check_cuda),
        check("Real UCI handshake with uci_torch.py", check_uci_handshake),
    ]
    print()
    if all(results):
        print("All checks passed -- safe to open the Tauri app now.")
        sys.exit(0)
    else:
        print("Some checks failed -- fix these before opening the GUI, "
              "the app's own error messages will be much less specific than this.")
        sys.exit(1)


if __name__ == "__main__":
    main()
