"""Cross-platform one-shot venv bootstrap: creates .venv next to the
repo root if it's missing, installs the Python dependencies (auto-
picking a compatible torch build on older NVIDIA GPUs -- compute
capability < 7.5, e.g. the GTX 1050 Ti this project has actually been
developed and tested against on two separate real machines this
session, where the plain torch pin reports CUDA "available" but can't
really use the GPU), and does nothing (fast, idempotent) if the venv
already has everything it needs.

The pin differs by OS because the owner's two real machines use different
toolkits: Windows keeps CUDA 11.8 -> torch==2.7.1
+cu118 (torch has no cu118 build newer than 2.7.1). Linux/WSL moved to
CUDA 12.6 -> torch==2.8.0+cu126 -- cu126, not cu121, because the cu121
wheel index tops out at torch 2.5.1 (a real downgrade from what was
already running), while cu126 still ships working Pascal/sm_61 kernels
at 2.8.0 (empirically confirmed live on this machine: real GPU matmul
succeeds, though torch.cuda.get_arch_list() lists sm_60, not sm_61
literally -- sm_61 runs via same-major forward compatibility with the
sm_60 kernel, not a listed exact match). The wheel's bundled CUDA
runtime does not need to match the installed toolkit version to run
torch itself -- only building native CUDA extensions with nvcc would.

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


def needs_old_gpu_pin():
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
    if needs_old_gpu_pin():
        if os.name == "nt":
            torch_pin, index_url = "torch==2.7.1", "https://download.pytorch.org/whl/cu118"
            print("Виявлено GPU зі старішою compute capability (<7.5) на Windows -- ставлю torch з cu118-індексу (CUDA 11.8 toolkit) для сумісності.")
        else:
            torch_pin, index_url = "torch==2.8.0", "https://download.pytorch.org/whl/cu126"
            print("Виявлено GPU зі старішою compute capability (<7.5) на Linux/WSL -- ставлю torch з cu126-індексу (CUDA 12.6 toolkit) для сумісності.")
        subprocess.run(
            [VENV_PYTHON, "-m", "pip", "install", "-q", torch_pin, "--index-url", index_url],
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
