# План: замкнений цикл self-play → тренування → arena (PyTorch)

Що вже було (roadmap crатегія): модель + MCTS + UCI повністю
портовані й перевірені; `optimize_torch.py` мав лише механізм
(loss/backward/step) на синтетичних даних; `self_play_torch.py` вмів
зіграти одну обмежену партію, але нічого не зберігав і не тренував.

Цей план закриває решту: реальний цикл self-play → дані на диску →
тренування на реальних даних → arena (нова модель проти поточної
найкращої) → промоушн переможця.

## Формат даних і версіювання моделі (рішення)

Оригінал зберігав ваги в Keras `.h5` + окремий `model_config.json`.
Для нових (тренованих тут) поколінь моделі це зайве — PyTorch-модель
має фіксовану архітектуру (`ChessResNet`), тож досить власного
`state_dict` у `.pt`. Оригінальний `model_best_weight.h5` лишається
єдиним джерелом для *початкової* моделі (через `load_weights.py`,
без змін); усі наступні покоління зберігаються і завантажуються вже
як `.pt` — не потрібно відтворювати Keras JSON-конфіг.

Шляхи (нові, паралельно до оригінальних `data/play_data`,
`data/model/next_generation`, щоб нічого оригінальне не зламати):

- `data/play_data_torch/play_<timestamp>.json` — self-play партії
  (той самий `[fen, policy, value]` формат, що й оригінал —
  сумісний з `data_helper.py`'s `write_game_data_to_file`, просто без
  залежності від `pyperclip`).
- `data/model_torch/model_best.pt` — поточна найкраща модель
  (спочатку — конвертовані ваги з `model_best_weight.h5`).
- `data/model_torch/next_generation/model_<timestamp>.pt` —
  кандидати, що чекають на arena-перевірку.

## Фази

1. **`lib/data_helper_torch.py`** — `write_game_data_to_file`,
   `read_game_data_from_file`, `get_game_data_filenames` (glob по
   `play_data_torch/`). Прямий порт `data_helper.py` без
   `pretty_print`/`pyperclip`/PGN (той функціонал реально не потрібен
   для тренувального циклу, лише для людського перегляду партій).
2. **`worker/self_play_torch.py`** (розширення) — `self_play_loop(model,
   play_config, data_dir, num_games, max_halfmoves)`: грає N партій,
   кожну зберігає окремим файлом через `data_helper_torch`.
3. **`worker/optimize_torch.py`** (розширення) — `load_dataset(data_dir)`
   збирає всі збережені партії в тензори (state/policy/value);
   `train_epochs(model, optimizer, data_dir, epochs, batch_size)` —
   реальний цикл з батчуванням і shuffle; `save_checkpoint`/
   `load_checkpoint` — `torch.save`/`torch.load` на `state_dict`.
4. **`worker/evaluate_torch.py`** (новий) — прямий порт
   `evaluate.py`'s arena-логіки: кандидат проти поточної найкращої,
   гравці міняються кольором по черзі, `replace_rate=0.55`
   (`configs/mini.py`'s `EvaluateConfig`, як в оригіналі) — якщо
   кандидат набирає ≥55% очок, він стає новою `model_best.pt`.
5. **`worker/pipeline_torch.py`** (новий) — тонкий драйвер, що
   зв'язує 2-4: N self-play партій → тренувати на всіх накопичених
   даних → arena з поточним best → промоушн/відкидання. Один прогін
   циклу — не безкінечний воркер-пул оригіналу (`SelfPlayWorker`,
   `ProcessPoolExecutor`) — це окрема, набагато більша інфраструктурна
   робота, свідомо не робиться зараз.

## Перевірка (реалістичні масштаби)

Слабка GPU (GTX 1050 Ti) і 4 ядра CPU роблять справжнє AlphaZero-масштабне
тренування (тисячі партій, 1200 симуляцій/хід) нереалістичним у межах
цієї сесії. Перевірка тут — це **реальний, повний прохід циклу на
малому масштабі** (кілька партій, зменшена кількість симуляцій), що
доводить: механізм цілісний і дані з self-play дійсно покращують
(або принаймні коректно впливають на) модель — не production-тренування.
