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
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use tauri::{AppHandle, Manager, State};
use tauri_plugin_shell::process::{CommandChild, CommandEvent};
use tauri_plugin_shell::ShellExt;

#[derive(Default)]
pub struct EngineState {
    child: Mutex<Option<CommandChild>>,
    output: Mutex<VecDeque<String>>,
    // Bumped every time a new child is installed into `child`. Each
    // spawn's own event-loop task below captures the value it was
    // given at spawn time and only clears the slot on termination if
    // that's STILL the current generation -- see engine_start's own
    // comment for the real race this closes.
    generation: AtomicU64,
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
    // Incremented while still holding child_slot's lock, so this is
    // correctly ordered against any concurrent engine_start call (they
    // serialize on the same lock) -- this generation value uniquely
    // identifies THIS spawn for the event-loop task below.
    let my_generation = state.generation.fetch_add(1, Ordering::SeqCst) + 1;
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
            //
            // Real, CONFIRMED race found in today's 4-agent audit
            // (Rust/Tauri Auditor): engine_kill() (added the same
            // session as this fix) kills the live child out-of-band --
            // if THIS task's Terminated event for that kill arrives
            // AFTER a subsequent engine_start already installed a
            // fresh, live child in the slot, the unconditional
            // `*child_slot = None` below used to wipe out that new,
            // healthy child -- an orphaned, untrackable sidecar
            // process with no way to kill or respawn it short of a
            // full app restart. Gated on the generation check so a
            // stale task can only ever clear the slot for ITS OWN
            // spawn, never a newer one.
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
                    if engine_state.generation.load(Ordering::SeqCst) == my_generation {
                        *child_slot = None;
                    }
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
    // Escape handler for all three busy modes (thinking/train/selfplay)
    // as a hard panic-key stop, distinct from selfplay's own graceful
    // `selfplay stop` UCI command (still used by its dedicated button).
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
    drop(child_slot);
    // Also drop whatever's still buffered from the killed session --
    // partial cross-check finding from today's audit (Rust/Tauri
    // Auditor, prompted by the Frontend Auditor's own finding): a
    // stale JS poll loop left over from before the kill can otherwise
    // keep draining leftover lines (or, on Windows specifically, lines
    // an orphaned grandchild python.exe is still writing -- see this
    // function's own doc comment above) and misread them as belonging
    // to the NEW session. This does not fully close that race (the
    // frontend's own poll loop isn't cancelled by this alone -- a
    // separate, not-yet-done fix), but removes the specific stale data
    // this command can reach.
    if let Ok(mut buf) = state.output.lock() {
        buf.clear();
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

#[cfg(test)]
mod tests {
    use super::*;

    // engine_start/engine_kill themselves need a real Tauri AppHandle
    // (to spawn a real sidecar process) and aren't unit-testable
    // without that -- these tests instead cover the EngineState
    // primitives directly (this module's own private fields, visible
    // here since `tests` is a child module), specifically the
    // generation-counter arithmetic that was the actual site of a
    // real, CONFIRMED race found and fixed earlier this session (a
    // stale spawn's event-loop task could clear a NEWER child's slot).

    #[test]
    fn engine_state_starts_empty_at_generation_zero() {
        let state = EngineState::default();
        assert!(state.child.lock().unwrap().is_none());
        assert!(state.output.lock().unwrap().is_empty());
        assert_eq!(state.generation.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn generation_increments_are_monotonic_and_unique_per_caller() {
        // Mirrors engine_start's own `fetch_add(1, ...) + 1` pattern
        // exactly (fetch_add returns the OLD value) -- an off-by-one
        // here would silently make two consecutive spawns share a
        // generation id and reopen the exact race this counter exists
        // to close.
        let state = EngineState::default();
        let gen1 = state.generation.fetch_add(1, Ordering::SeqCst) + 1;
        let gen2 = state.generation.fetch_add(1, Ordering::SeqCst) + 1;
        assert_eq!(gen1, 1);
        assert_eq!(gen2, 2);
        assert_ne!(gen1, gen2);
    }

    #[test]
    fn stale_generation_no_longer_matches_after_a_newer_spawn() {
        // Regression test for the real race: a stale spawn's captured
        // generation must NOT equal the current one after a newer
        // spawn has happened -- engine_start's event-loop task relies
        // on exactly this check before clearing child_slot.
        let state = EngineState::default();
        let stale_generation = state.generation.fetch_add(1, Ordering::SeqCst) + 1; // 1st engine_start
        let _current_generation = state.generation.fetch_add(1, Ordering::SeqCst) + 1; // 2nd engine_start
        assert_ne!(
            stale_generation,
            state.generation.load(Ordering::SeqCst),
            "a stale spawn's generation must not match the current one after a newer spawn"
        );
    }

    #[test]
    fn output_buffer_drains_in_fifo_order_and_empties() {
        let state = EngineState::default();
        {
            let mut buf = state.output.lock().unwrap();
            buf.push_back("uciok".to_string());
            buf.push_back("readyok".to_string());
        }
        let drained: Vec<String> = state.output.lock().unwrap().drain(..).collect();
        assert_eq!(drained, vec!["uciok".to_string(), "readyok".to_string()]);
        assert!(state.output.lock().unwrap().is_empty());
    }
}
