"""Cross-platform one-shot venv bootstrap: creates .venv next to the
repo root if it's missing, installs the Python dependencies (auto-
picking the cu118 torch build on older NVIDIA GPUs -- compute
capability < 7.5, e.g. the GTX 1050 Ti this project has actually been
developed and tested against on two separate real machines this
session, where the plain torch pin reports CUDA "available" but can't
really use the GPU), and does nothing (fast, idempotent) if the venv
already has everything it needs.

cu118, not the newer cu126, because that's the CUDA toolkit the owner
has installed on both his real machines (Linux dev box + Windows) as
of 2026-08-27 -- torch has no cu118 build newer than 2.7.1, hence that
older pin below too (cu126 itself would also satisfy the compute-
capability requirement; cu118 was picked to match installed toolkits,
not because cu126 stopped working).

Called by release/run-linux.sh and release/run-windows.bat before
launching the app, so a fresh clone can go from "just cloned" to "app
is open" with one command instead of a separate manual venv-setup
step first. Those scripts still handle the CWD-relative-path nuance
documented in release/README.md -- this script only touches .venv/,
never the app's own runtime working directory.
"""
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENV_DIR = os.path.join(REPO_ROOT, ".venv")
VENV_PYTHON = os.path.join(
    VENV_DIR, "Scripts" if os.name == "nt" else "bin",
    "python.exe" if os.name == "nt" else "python3",
)


def venv_has_torch():
    if not os.path.exists(VENV_PYTHON):
        return False
    result = subprocess.run([VENV_PYTHON, "-c", "import torch"], capture_output=True)
    return result.returncode == 0


def needs_cu118():
    """True if an NVIDIA GPU with compute capability < 7.5 is visible.
    No nvidia-smi (no GPU, or AMD/Intel/CPU-only) -> False, plain
    requirements.txt is the right, safe default there."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False
    if out.returncode != 0:
        return False
    caps = []
    for line in out.stdout.strip().splitlines():
        try:
            caps.append(float(line.strip()))
        except ValueError:
            pass
    return bool(caps) and min(caps) < 7.5


def main():
    if not os.path.exists(VENV_PYTHON):
        print("Створюю .venv...")
        subprocess.run([sys.executable, "-m", "venv", VENV_DIR], check=True)

    if venv_has_torch():
        print(".venv вже готовий, залежності на місці.")
        return

    print("Встановлюю Python-залежності (перший запуск -- torch може зайняти кілька хвилин)...")
    if needs_cu118():
        print("Виявлено GPU зі старішою compute capability (<7.5) -- ставлю torch з cu118-індексу для сумісності.")
        subprocess.run(
            [VENV_PYTHON, "-m", "pip", "install", "-q",
             "torch==2.7.1", "--index-url", "https://download.pytorch.org/whl/cu118"],
            check=True,
        )
        subprocess.run(
            [VENV_PYTHON, "-m", "pip", "install", "-q",
             "numpy==2.5.2", "h5py==3.16.0", "chess==1.11.2"],
            check=True,
        )
    else:
        subprocess.run(
            [VENV_PYTHON, "-m", "pip", "install", "-q", "-r",
             os.path.join(REPO_ROOT, "requirements.txt")],
            check=True,
        )
    print("Готово.")


if __name__ == "__main__":
    main()
