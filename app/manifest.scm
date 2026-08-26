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
   "webkitgtk-for-gtk3"
   ;; Windows cross-compile experiment (x86_64-pc-windows-gnu target):
   ;; cargo correctly pulls the Windows-specific dependency graph
   ;; (webview2-com, windows-core -- no gtk/webkit at all, confirmed by
   ;; a real build attempt) but needs dlltool, which lives in the FULL
   ;; cross gcc toolchain, not the smaller mingw-w64-tools package
   ;; (that one only has widl/gendef/genidl/genpeimg -- checked its
   ;; bin/ directly after a first attempt failed with "dlltool: No
   ;; such file or directory").
   "mingw-w64-x86_64"
   "gcc-cross-x86_64-w64-mingw32-toolchain"))
