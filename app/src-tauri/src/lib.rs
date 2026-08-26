mod commands;

use commands::engine::{engine_drain, engine_is_running, engine_send, engine_start, EngineState};

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .manage(EngineState::default())
        .invoke_handler(tauri::generate_handler![
            engine_start,
            engine_send,
            engine_drain,
            engine_is_running
        ])
        .run(tauri::generate_context!())
        .expect("error while running Chess Tauri Zero");
}
