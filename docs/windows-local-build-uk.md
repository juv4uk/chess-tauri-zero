# Збірка на власній Windows-машині (без готового бінарника)

Найнадійніший шлях, якщо готовий `.exe` з `release/`/GitHub Releases
блокується політикою безпеки (Device Guard/WDAC на керованих машинах)
— зібрати самому, локальним інструментарієм. Локально зібраний
бінарник не гарантовано пройде ту саму політику (вона зазвичай працює
за allowlist, не за "хто зібрав"), але це найдешевший наступний крок
для перевірки, і в будь-якому разі корисний навик для подальшого
розвитку.

**Важливо:** наш sidecar (`app/src-tauri/binaries/uci-engine-x86_64-pc-windows-gnu.exe`)
скомпільований під **GNU**-ABI target, не типовий для Windows MSVC.
Якщо зібрати додаток стандартним MSVC-тулчейном (типова установка
Rust на Windows), Tauri шукатиме sidecar під іншою назвою
(`...msvc.exe`), якого немає — команда нижче явно каже зібрати під
той самий GNU-target, що вже є.

## 1. Встановити Rust + GNU-тулчейн (один рядок, через winget)

```powershell
winget install Rustlang.Rustup winlibs
rustup target add x86_64-pc-windows-gnu
```

Перевір:
```powershell
rustc --version
x86_64-w64-mingw32-gcc --version
```

## 2. Зібрати

```powershell
cd app\src-tauri
$env:CARGO_TARGET_X86_64_PC_WINDOWS_GNU_LINKER = "x86_64-w64-mingw32-gcc"
cargo build --release --target x86_64-pc-windows-gnu
```

Перша збірка компілює весь Tauri-стек (webview2-com, windows-core) —
кілька хвилин. Результат:
`target\x86_64-pc-windows-gnu\release\chess-tauri-zero-app.exe`.

## 3. Запустити (той самий CWD-нюанс, що й для готового бінарника)

**Не** запускай `.exe` напряму з `target\...\release\` — той самий
баг, що описаний у `release/README.md`: `frontendDist` резолвиться
відносно робочої директорії, не місця файлу. Або скористайся готовим
`scripts\build-and-run-windows.bat` (нижче), або вручну:

```powershell
cd app\src-tauri   # саме звідси, не з target\...\release\
.\target\x86_64-pc-windows-gnu\release\chess-tauri-zero-app.exe
```

## Автоматизований варіант

`scripts/build-and-run-windows.bat` у корені репо робить кроки 2-3
самостійно (крок 1 — інструменти — усе одно треба встановити руками
один раз):

```powershell
scripts\build-and-run-windows.bat
```
