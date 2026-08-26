//! Bridges the JS frontend to uci_torch.py (chess-tauri-zero's real
//! PyTorch AlphaZero engine, verified as a correct UCI speaker earlier
//! in the same session), spawned as a Tauri sidecar
//! (binaries/uci-engine-x86_64-unknown-linux-gnu -- a shell wrapper
//! around the existing venv + uci_torch.py, not a frozen binary; see
//! that script's own header comment for why).
//!
//! Deliberately simple protocol instead of matching request/response:
//! `engine_send` writes a raw UCI line to the sidecar's stdin,
//! `engine_drain` returns every stdout line collected since the last
//! drain. The frontend polls `engine_drain` after sending "go" and
//! watches for a line starting with "bestmove" -- exactly how a real
//! UCI GUI treats an engine's stdout, just polled instead of matched
//! against a specific pending request.

use std::collections::VecDeque;
use std::sync::Mutex;

use tauri::{AppHandle, Manager, State};
use tauri_plugin_shell::process::{CommandChild, CommandEvent};
use tauri_plugin_shell::ShellExt;

#[derive(Default)]
pub struct EngineState {
    child: Mutex<Option<CommandChild>>,
    output: Mutex<VecDeque<String>>,
}

#[tauri::command]
pub fn engine_start(app: AppHandle, state: State<'_, EngineState>) -> Result<(), String> {
    let mut child_slot = state.child.lock().map_err(|e| e.to_string())?;
    if child_slot.is_some() {
        return Ok(()); // already running
    }

    let sidecar = app
        .shell()
        .sidecar("uci-engine")
        .map_err(|e| e.to_string())?;
    let (mut rx, child) = sidecar.spawn().map_err(|e| e.to_string())?;
    *child_slot = Some(child);
    drop(child_slot);

    let app_handle = app.clone();
    tauri::async_runtime::spawn(async move {
        while let Some(event) = rx.recv().await {
            // Terminated/Error mean the sidecar is gone -- clear the
            // slot so the NEXT engine_start actually respawns it
            // instead of engine_send silently (or now visibly) hitting
            // a broken pipe forever. This was the real bug: once set,
            // child_slot.is_some() never went back to false on its
            // own, so a sidecar that died for any reason (observed:
            // "Broken pipe (os error 32)" from a live run) left the
            // app permanently stuck with no way to recover without a
            // full restart.
            let is_terminal = matches!(event, CommandEvent::Terminated(_) | CommandEvent::Error(_));
            let line = match event {
                CommandEvent::Stdout(bytes) => Some(String::from_utf8_lossy(&bytes).to_string()),
                CommandEvent::Stderr(bytes) => {
                    Some(format!("[stderr] {}", String::from_utf8_lossy(&bytes)))
                }
                CommandEvent::Error(err) => Some(format!("[error] {err}")),
                CommandEvent::Terminated(status) => {
                    Some(format!("[terminated] {status:?}"))
                }
                _ => None,
            };
            let engine_state = app_handle.state::<EngineState>();
            if let Some(line) = line {
                if let Ok(mut buf) = engine_state.output.lock() {
                    buf.push_back(line.trim_end().to_string());
                }
            }
            if is_terminal {
                if let Ok(mut child_slot) = engine_state.child.lock() {
                    *child_slot = None;
                }
                break;
            }
        }
    });

    Ok(())
}

#[tauri::command]
pub fn engine_send(line: String, state: State<'_, EngineState>) -> Result<(), String> {
    let mut child_slot = state.child.lock().map_err(|e| e.to_string())?;
    let child = child_slot.as_mut().ok_or("engine not started")?;
    let result = child.write((line + "\n").as_bytes());
    if result.is_err() {
        // Write failed (e.g. broken pipe) -- the process is dead even
        // though we haven't received its Terminated event yet. Clear
        // the slot now so the caller's respawn-and-retry can succeed
        // immediately instead of racing the event loop.
        *child_slot = None;
    }
    result.map_err(|e| e.to_string())
}

#[tauri::command]
pub fn engine_kill(state: State<'_, EngineState>) -> Result<(), String> {
    // engine_start is idempotent -- if the sidecar is ALIVE but just
    // slow/stuck (mid-search, mid-training-cycle; neither has an
    // interrupt point in the Python code), calling engine_start again
    // does nothing (child_slot.is_some() short-circuits it). This is
    // the actual "force stop": kills the live process outright so the
    // next engine_start spawns a fresh one. Used by the frontend's
    // Escape handler for the two modes (thinking/train) that have no
    // graceful stop of their own, unlike selfplay's `selfplay stop`.
    //
    // Windows nuance, not fully solved here: the Windows sidecar
    // (uci-engine-launcher.exe) spawns python.exe as a CHILD process
    // rather than exec-replacing itself (no exec() on Windows) --
    // killing the launcher does not directly kill that grandchild.
    // In practice python.exe should still exit on its own once the
    // stdin pipe this closes reaches EOF (caught by uci_torch.py's
    // own `except EOFError: break`), but this is not an immediate,
    // guaranteed kill the way it is on Linux (the sidecar there execs
    // python directly, so the same PID IS python).
    let mut child_slot = state.child.lock().map_err(|e| e.to_string())?;
    if let Some(child) = child_slot.take() {
        child.kill().map_err(|e| e.to_string())?;
    }
    Ok(())
}

#[tauri::command]
pub fn engine_is_running(state: State<'_, EngineState>) -> Result<bool, String> {
    let child_slot = state.child.lock().map_err(|e| e.to_string())?;
    Ok(child_slot.is_some())
}

#[tauri::command]
pub fn engine_drain(state: State<'_, EngineState>) -> Result<Vec<String>, String> {
    let mut buf = state.output.lock().map_err(|e| e.to_string())?;
    Ok(buf.drain(..).collect())
}
