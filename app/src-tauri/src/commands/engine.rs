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
            if let Some(line) = line {
                if let Ok(mut buf) = app_handle.state::<EngineState>().output.lock() {
                    buf.push_back(line.trim_end().to_string());
                }
            }
        }
    });

    Ok(())
}

#[tauri::command]
pub fn engine_send(line: String, state: State<'_, EngineState>) -> Result<(), String> {
    let mut child_slot = state.child.lock().map_err(|e| e.to_string())?;
    let child = child_slot.as_mut().ok_or("engine not started")?;
    child
        .write((line + "\n").as_bytes())
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub fn engine_drain(state: State<'_, EngineState>) -> Result<Vec<String>, String> {
    let mut buf = state.output.lock().map_err(|e| e.to_string())?;
    Ok(buf.drain(..).collect())
}
