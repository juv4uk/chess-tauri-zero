# Межа ліцензій у chess-tauri-zero

`chess-tauri-zero` є форком
[`kmader/chess-alpha-zero`](https://github.com/kmader/chess-alpha-zero), а не
повністю оригінальним репозиторієм Володимира. Наявність репозиторію в акаунті
`juv4uk`, перейменування або значна подальша переробка не змінюють походження
успадкованого коду.

## Яка ліцензія до чого застосовується

- [`LICENSE.txt`](LICENSE.txt) — первісна ліцензія успадкованого
  `chess-alpha-zero` та похідних змін у тих частинах, де робота лишається
  продовженням upstream-коду.
- [`LICENSE`](LICENSE) — ВОЛЬНІСТЬ лише для оригінальних, юридично віддільних
  внесків Володимира, створених у цьому форку.
- [`app/web/vendor/chess.js.LICENSE`](app/web/vendor/chess.js.LICENSE) — окрема
  BSD-ліцензія vendored `chess.js`.

Якщо конкретний файл або фрагмент неможливо впевнено відділити від upstream,
до нього не можна автоматично застосовувати ВОЛЬНІСТЬ: треба зберегти
upstream-умови й дослідити provenance через історію Git. ВОЛЬНІСТЬ не скасовує
атрибуцію, copyright notices чи умови сторонніх залежностей.

## License boundary in English

`chess-tauri-zero` is a fork of `kmader/chess-alpha-zero`, not a wholly
original repository by Volodymyr. `LICENSE.txt` remains the license for the
inherited upstream code and derivative changes that cannot be separated from
it. `LICENSE` (VOLNOST) applies only to Volodymyr's original, legally separable
contributions. Vendored `chess.js` remains under its own BSD license. When
provenance is unclear, preserve the upstream terms until the boundary is
verified from Git history.
