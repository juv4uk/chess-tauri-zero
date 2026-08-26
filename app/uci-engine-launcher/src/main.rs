//! Windows sidecar launcher -- see Cargo.toml's description for why
//! this exists as a compiled binary rather than a .bat file (Tauri's
//! sidecar spawn uses CreateProcess directly, which does not dispatch
//! .bat/.cmd the way cmd.exe's own PATHEXT resolution does).
//!
//! Mirrors src-tauri/binaries/uci-engine-x86_64-unknown-linux-gnu's
//! logic exactly: find this executable's own directory, go up three
//! levels (binaries/ -> src-tauri/ -> app/ -> repo root) to the repo
//! root, run python.exe from a venv the user set up there themselves
//! (`.venv\Scripts\python.exe` on Windows, vs `.venv/bin/python3` on
//! Linux), pointed at chess_zero/play_game/uci_torch.py from `src/`.
use std::env;
use std::process::{Command, Stdio};

fn main() {
    let exe_path = env::current_exe().expect("could not determine own executable path");
    let binaries_dir = exe_path.parent().expect("no parent dir for exe path");
    let repo_root = binaries_dir
        .parent() // src-tauri/
        .and_then(|p| p.parent()) // app/
        .and_then(|p| p.parent()) // repo root
        .expect("could not walk up to repo root from binaries/ dir")
        .to_path_buf();

    let python = repo_root.join(".venv").join("Scripts").join("python.exe");
    let script = repo_root
        .join("src")
        .join("chess_zero")
        .join("play_game")
        .join("uci_torch.py");
    let src_dir = repo_root.join("src");

    let status = Command::new(&python)
        .arg(&script)
        .current_dir(&src_dir)
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status();

    match status {
        Ok(s) => std::process::exit(s.code().unwrap_or(1)),
        Err(e) => {
            eprintln!(
                "uci-engine-launcher: failed to run {} {}: {e}",
                python.display(),
                script.display()
            );
            std::process::exit(1);
        }
    }
}
