mod commands;

use commands::engine::{
    engine_drain, engine_is_running, engine_kill, engine_send, engine_start, EngineState,
};
use tauri::Manager;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .manage(EngineState::default())
        .invoke_handler(tauri::generate_handler![
            engine_start,
            engine_send,
            engine_drain,
            engine_is_running,
            engine_kill
        ])
        .setup(|app| {
            // Version shown in the window title comes from tauri.conf.json's
            // own "version" field (via Tauri's package_info, not a hardcoded
            // string) -- stays correct automatically as long as the binary
            // itself is rebuilt from a given tauri.conf.json, which is the
            // whole point of embedding it here rather than relying on a
            // filename convention alone.
            let version = app.package_info().version.to_string();
            if let Some(window) = app.get_webview_window("main") {
                let _ = window.set_title(&format!("Chess Tauri Zero v{version}"));
            }
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running Chess Tauri Zero");
}
