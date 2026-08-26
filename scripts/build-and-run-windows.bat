@echo off
rem Builds chess-tauri-zero locally with the GNU target (matching the
rem committed sidecar binary's ABI, app/src-tauri/binaries/
rem uci-engine-x86_64-pc-windows-gnu.exe) and runs it with the correct
rem working directory. See docs/windows-local-build-uk.md for the
rem one-time toolchain setup (winget install Rustlang.Rustup winlibs;
rem rustup target add x86_64-pc-windows-gnu) this script assumes is
rem already done.
setlocal
set REPO_ROOT=%~dp0..
set CARGO_TARGET_X86_64_PC_WINDOWS_GNU_LINKER=x86_64-w64-mingw32-gcc

where cargo >nul 2>nul
if errorlevel 1 (
    echo cargo not found on PATH. Install Rust first: winget install Rustlang.Rustup winlibs
    echo See docs\windows-local-build-uk.md for details.
    exit /b 1
)

cd /d "%REPO_ROOT%\app\src-tauri"
cargo build --release --target x86_64-pc-windows-gnu
if errorlevel 1 (
    echo Build failed -- see the cargo output above.
    exit /b 1
)

rem Run from app\src-tauri (not target\...\release\) -- frontendDist
rem resolves relative to the working directory at launch, not to
rem wherever the .exe file sits (real bug found and documented in
rem release/README.md).
"%REPO_ROOT%\app\src-tauri\target\x86_64-pc-windows-gnu\release\chess-tauri-zero-app.exe"
