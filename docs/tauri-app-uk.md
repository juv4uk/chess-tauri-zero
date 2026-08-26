# Tauri-обгортка (app/)

Desktop GUI над існуючим PyTorch-рушієм, шлях "A" (sidecar) з
обговорення: нуль переписування Python-коду, `uci_torch.py` лишається
джерелом істини для гри, Tauri лише малює дошку і говорить з ним по
UCI через stdin/stdout.

## Структура

- `app/src-tauri/binaries/uci-engine-x86_64-unknown-linux-gnu` — shell-обгортка
  навколо `../src/chess_zero/play_game/uci_torch.py` у своєму venv.
  **Не** PyInstaller-бінарник (torch+cuda в один портативний файл —
  окрема, важча задача пакування) — мінімум, що доводить: міст
  Rust↔Python по UCI реально працює.
  Перевірено напряму (без Tauri): `echo -e "uci\nisready\nposition
  startpos\ngo\nquit" | ./uci-engine-x86_64-unknown-linux-gnu` →
  `uciok`, `readyok`, `bestmove c2c4`.
- `app/src-tauri/src/commands/engine.rs` — Rust-міст: `engine_start`
  (спавнить sidecar через `tauri-plugin-shell`), `engine_send` (пише
  рядок у stdin), `engine_drain` (віддає накопичені рядки stdout).
  Простий poll-протокол, а не match request↔response — фронтенд сам
  чекає рядок `bestmove ...` після `go`.
- `app/web/` — фронтенд без фреймворка (як у tauricode): чиста
  HTML/CSS/JS, `chess.js` (BSD-ліцензія, провенанс:
  `web/vendor/chess.js.LICENSE`) заведмплено локально одним файлом
  для легальності ходів на клієнті (рушій сам генерує лише ХІД, не
  перевіряє легальність твоїх кліків).

## Реальний блокер, знайдений і виправлюваний зараз

`cargo check` у `app/src-tauri` впав на реальній помилці:
`glib-2.0` не знайдено через pkg-config. Це та сама стіна залежностей
(`glib`/`gtk+`/`webkitgtk`), що згадувалась у попередньому
Tauricode donor-build досвіді. Різниця цього разу: `guix weather
webkitgtk-for-gtk3` показав **100% substitute available** — усі три
пакунки завантажуються готовими (~78MB), не компілюються локально.

`app/manifest.scm` (ефемерний `guix shell -m manifest.scm`, не
глобальна інсталяція) додає `pkg-config`, `glib`, `gtk+`,
`webkitgtk-for-gtk3`. Машина була сильно навантажена іншими агентами
(load 13+ на 4 ядра) в момент спроби — відкладено до спадання
навантаження, той самий preflight-принцип, що й для self-play
верифікації раніше в цій сесії.

## Статус чесно

- **empirically confirmed**: sidecar-скрипт коректно говорить UCI сам
  по собі (без Tauri).
- **source-confirmed, НЕ empirically confirmed**: весь Rust-код
  (`engine.rs`, `lib.rs`, `Cargo.toml`, `tauri.conf.json`,
  `capabilities/default.json`) написаний за реальним, підтвердженим
  патерном з `tauricode`, але жодного разу не компілювався успішно —
  `cargo check` не дійшов до кінця через відсутність `glib`/`gtk+`/
  `webkitgtk`. Можливі помилки типів/сигнатур `tauri-plugin-shell` API,
  які виявляться лише після реальної компіляції.
- **не перевірено взагалі**: JS-фронтенд (`app/web/app.js`) — жодного
  разу не відкривався у вікні (та сама причина: без `webkitgtk` немає
  чим рендерити).

## Оновлення: реально запускалось (2026-08-26)

`cargo check`/`cargo build` — **empirically confirmed, компілюється
чисто** після `guix shell -m manifest.scm` (glib/gtk+/webkitgtk-for-gtk3,
100% substitute, ніякого локального білду). Один реальний баг по дорозі:
`tauri::generate_context!()` вимагає `icons/icon.png` навіть без явного
`bundle.icon` в конфізі — додано плейсхолдер-іконку (PIL).

Вікно **реально відкривається** через WSLg (`DISPLAY=:0`) — підтверджено
скріншотом (`xwd` + `netpbm`): заголовок "Chess Tauri Zero", дошка
рендериться коректно, статус українською. Знадобились
env-обхідні прапори для WebKitGTK під WSLg (Zink/EGL не може вибрати
GPU-пристрій): `WEBKIT_DISABLE_COMPOSITING_MODE=1
LIBGL_ALWAYS_SOFTWARE=1 GDK_BACKEND=x11
WEBKIT_DISABLE_DMABUF_RENDERER=1`.

**Один реальний клік дійшов до кінця**: e2e4 з'явився на дошці,
chess.js підтвердив легальність, статус коректно перейшов на "Рушій
думає...". **Не підтверджено**: повний round-trip людина→рушій→відповідь.
У спостереженому прогоні sidecar-процес (`uci_torch.py`) так і не
з'явився в `ps` після цього — або `engine_start`/`engine_send` тихо
впав (немає видимого JS-консольного виводу з headless X11-скріншотів,
це реальне обмеження цього способу перевірки), або вікно, яке я
скріншотив вдруге, показувало застарілий кадр від попереднього
процесу, що впав (WebKitGTK під WSLg+software-rendering падав мовчки,
без виводу в лог, після ~10-15с в одному з прогонів).

**Чесний висновок**: архітектура правильна і код компілюється й
запускається, але стабільність WebKitGTK під WSLg+software-rendering
для повного інтерактивного циклу (клік → рушій → відповідь на дошці)
**не підтверджена емпірично до кінця**. Найкраща наступна перевірка —
не headless-скріншоти (обмежений спосіб, немає доступу до JS-консолі),
а реальний живий запуск і клік власноруч, де видно консоль
розробника (F12 у webview) для реальних помилок.

## Реліз v0.0.1: Linux + Windows портативні бінарники (2026-08-26)

За рішенням власника: **не пакувати Python/torch у реліз** — лише
скомпільований Tauri-двигун (GUI + Rust-код). Людина клонує репо і сама
налаштовує venv (`.venv/bin/python3` на Linux, `.venv\Scripts\python.exe`
на Windows) поруч, як описано в головному README.

- **Linux** (`x86_64-unknown-linux-gnu`) — `cargo build --release`,
  15MB ELF-бінарник. **empirically confirmed**: та сама збірка вже
  реально відкривала вікно й малювала дошку раніше в цій сесії
  (перевірено скріншотами через WSLg).
- **Windows** (`x86_64-pc-windows-gnu`) — крос-компільовано ПРЯМО з
  цієї Linux-машини через `gcc-cross-x86_64-w64-mingw32-toolchain`
  (Guix). Реальний результат: **весь Rust/Tauri-код (включно з
  webview2-com, windows-core) скомпілювався чисто** — cargo сам
  підтягнув правильний Windows-специфічний граф залежностей, без
  жодного gtk/webkit. 22.9MB `.exe`, підтверджено `file` як справжній
  `PE32+ executable ... for MS Windows`.
- **uci-engine-launcher** (`app/uci-engine-launcher/`) — новий,
  маленький Rust-бінарник, потрібен був для Windows-версії sidecar:
  `.bat`-файл НЕ підійшов би (Tauri спавнить sidecar через
  `CreateProcess` напряму, без `cmd.exe`-обробки `.bat`), тож замість
  цього написано мінімальний Rust-лаунчер, що повторює логіку
  Linux-скрипта (знайти свою директорію, піднятись на 3 рівні до
  кореня репо, запустити `.venv\Scripts\python.exe uci_torch.py` зі
  своїм stdio, успадкованим напряму для UCI-моста). Скомпільовано тим
  самим mingw-крос-тулчейном, `binaries/uci-engine-x86_64-pc-windows-gnu.exe`
  закомічений як готовий артефакт (перегенерувати з
  `uci-engine-launcher/src/main.rs` при зміні логіки).

**Чесно НЕ підтверджено**: жоден з двох Windows-компонентів
(`chess-tauri-zero-app.exe`, `uci-engine.exe`) жодного разу не
запускався на реальній Windows чи під wine — тут немає ні Windows
машини, ні встановленого wine для смоук-тесту. Компіляція для
Windows-таргету — реальний, сильний сигнал коректності (типи, ABI,
лінковка вже перевірені), але не заміна реального запуску.
