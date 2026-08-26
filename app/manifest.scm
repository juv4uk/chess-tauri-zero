;; guix shell -m manifest.scm
;; Build environment for the Tauri desktop shell (app/src-tauri). Rust
;; itself and the JS tooling are already available via the top-level
;; toolchain used throughout this session; what's missing specifically
;; for `cargo check`/`cargo build` here is the GTK/WebKit stack Tauri's
;; Linux backend links against (confirmed by the real pkg-config error:
;; glib-2.0 not found). All three packages below have 100% substitute
;; availability on bordeaux.guix.gnu.org as of 2026-08-26 (checked via
;; `guix weather` before adding this) -- a download, not a local build.
(specifications->manifest
 '("pkg-config"
   "glib"
   "gtk+"
   "webkitgtk-for-gtk3"))
