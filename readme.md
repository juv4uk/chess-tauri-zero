[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/kmader/chess-alpha-zero/master?urlpath=lab)
[![Demo Notebook](https://img.shields.io/badge/launch-demo_notebook-red.svg)](https://mybinder.org/v2/gh/kmader/chess-alpha-zero/master?filepath=notebooks%2Fdemo.ipynb)

PyTorch-порт (2026)
====================

Цей форк ([juv4uk/chess-tauri-zero](https://github.com/juv4uk/chess-tauri-zero),
з оригіналу [kmader/chess-alpha-zero](https://github.com/kmader/chess-alpha-zero))
переписаний з мертвого стеку `tensorflow-gpu==1.15.2`/`keras==2.0.8`/Python 3.6
на **PyTorch 2.13**, з реально працюючими вагами (`data/model/model_best_weight.h5`)
та повним, перевіреним циклом self-play → тренування → arena. Методологія
лишається чистою AlphaZero (нуль людських шахових знань — жодної дистиляції
від Stockfish чи іншого рушія), просто адаптована під слабке залізо
(GTX 1050 Ti, 4 ядра CPU) через batched MCTS замість масштабної self-play
інфраструктури оригіналу.

Детальніше: [`docs/pytorch-port-uk.md`](docs/pytorch-port-uk.md) (що і чому),
[`docs/pytorch-port-roadmap-uk.md`](docs/pytorch-port-roadmap-uk.md) (план портування),
[`docs/pytorch-self-play-loop-plan-uk.md`](docs/pytorch-self-play-loop-plan-uk.md)
(план self-play→train→evaluate циклу).

## Встановлення

```bash
cd chess-tauri-zero
python3 -m venv .venv
source .venv/bin/activate
pip install h5py numpy chess

# GPU: якщо карта старша (compute capability < 7.5, напр. GTX 1050 Ti / sm_61) --
# звичайний "pip install torch" ставить збірку без ядер під таку архітектуру
# (тиха помилка CUBLAS_STATUS_ARCH_MISMATCH при першому реальному виклику).
# Потрібна саме cu126-збірка:
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu126

# Або без GPU (повільніше, але коректно):
pip install torch==2.13.0
```

Усі команди нижче виконуються **з каталогу `src/`** (скрипти читають ваги за
відносним шляхом `../data/model/model_best_weight.h5`):

```bash
cd src
source ../.venv/bin/activate
```

## Пограти інтерактивно (з консолі)

Одна команда = один хід. Стан партії зберігається між викликами у `/tmp/chess_turn_state.fen`.

```bash
python3 chess_zero/agent/play_turn.py           # модель ходить першою (за білих)
python3 chess_zero/agent/play_turn.py e2e4      # твій хід (UCI або SAN: "e4", "Nf3" теж підійде)
```

## UCI-рушій (для GUI на кшталт Arena/CuteChess, або python-chess)

```bash
python3 chess_zero/play_game/uci_torch.py
# або перевірка вручну:
echo -e "uci\nisready\nposition startpos\ngo\nquit" | python3 chess_zero/play_game/uci_torch.py
```

## Self-play (генерація партій моделлю самою проти себе)

```bash
python3 chess_zero/worker/self_play_torch.py 5   # 5 партій, зберігаються в ../data/play_data_torch/
```

MCTS-пошук усередині кожної партії використовує batched-предиктор
(`agent/batched_predictor.py`) — паралельні гілки пошуку діляться GPU
батчами замість окремого forward pass на кожен лист дерева
(~4x швидше на GTX 1050 Ti, підтверджено вимірюванням).

## Тренування на накопичених self-play даних

```bash
python3 chess_zero/worker/optimize_torch.py   # запущений сам по собі -- лише
                                                # синтетична перевірка механізму
```

Реальне тренування на справжніх self-play даних відбувається через
`optimize_torch.load_dataset`/`train_epochs` — викликається `pipeline_torch.py`
(нижче), не напряму.

## Evaluate (кандидат проти поточної найкращої моделі)

```bash
python3 chess_zero/worker/evaluate_torch.py                       # кандидат = базова модель (перевірка механізму, ~50%)
python3 chess_zero/worker/evaluate_torch.py ../data/model_torch/next_generation/model_cycle0.pt  # реальний кандидат
```

## Повний цикл: self-play → train → evaluate → promote

```bash
python3 chess_zero/worker/pipeline_torch.py 3   # 3 цикли поспіль
```

Кожен цикл: 2 self-play партії → 1 епоха тренування кандидата на всіх
накопичених партіях (`../data/play_data_torch/`) → 2 arena-партії
кандидат vs найкраща модель → якщо кандидат набрав ≥55% — стає новою
`../data/model_torch/model_best.pt`, інакше відкидається.

Дані й моделі, згенеровані цими скриптами, лежать у `data/play_data_torch/`
і `data/model_torch/` (в `.gitignore`, не комітяться) -- окремо від
оригінальних `data/play_data/`/`data/model/next_generation/`, щоб нічого
з оригінального Keras-стеку не зачепити.

## Tauri-додаток: панель керування всіма можливостями

Desktop GUI (`app/`, UCI-sidecar над цим же Python-рушієм) має кнопки
на все: колір, складність, нова партія/скасувати/здатись, режим
спостереження за самогрою рушія, запуск одного циклу тренування, і
hot-reload натренованої моделі — прямо з інтерфейсу. Деталі
протоколу й що саме перевірено — [`docs/tauri-app-uk.md`](docs/tauri-app-uk.md).

### Запустити без збірки (готові бінарники)

Клонуй репо, налаштуй `.venv` (вище), тоді:

```bash
./release/run-linux.sh          # Linux
release\run-windows.bat         # Windows (PowerShell/cmd)
```

**Не запускай файли з `release/` напряму** (подвійним кліком чи
з іншої директорії) — реальний, знайдений баг: бінарник шукає
фронтенд-файли відносно робочої директорії при старті, не відносно
свого власного розташування, і без правильного CWD вікно просто не
з'явиться, без жодної помилки. Скрипти-обгортки це вирішують — деталі
в [`release/README.md`](release/README.md). Спочатку перевір усе
середовище одною командою: `python src/smoke_test.py`.

**Якщо готовий `.exe` блокує Windows з повідомленням "blocked by your
organization's Device Guard policy"** — це формулювання оманливе:
на звичайній персональній (не корпоративній) машині на Windows 11 це
майже завжди не справжній корпоративний Device Guard/WDAC, а **Smart
App Control** — фіча, увімкнена за замовчуванням на свіжих інсталяціях
Windows 11, яка використовує те саме повідомлення. **Перевірено й
підтверджено власником репо, реально спрацювало:**

1. Пуск → `Windows Security` → `App & browser control` → `Smart App
   Control` → `Off`.
2. Перезавантаження не обов'язкове. На Windows 11 24H2/25H2 це можна
   вимкнути й увімкнути назад без переустановки ОС (на старіших
   білдах вимкнення було одноразовим — назад тільки через
   перевстановлення Windows, тож на них вимикай усвідомлено).

Якщо після цього `.exe` все одно блокується (або машина справді
корпоративно керована, і Smart App Control недоступний/сірий) — це
вже, ймовірно, реальний Device Guard/WDAC, готовий бінарник тут не
допоможе незалежно від підпису. Збери локально своїм інструментарієм:
[`docs/windows-local-build-uk.md`](docs/windows-local-build-uk.md)
(`scripts\build-and-run-windows.bat` робить це одною командою після
одноразового встановлення Rust).

## Що ще НЕ реалізовано (чесно)

- **Масштабна self-play інфраструктура** — оригінал ганяв нескінченний пул
  процесів (`ProcessPoolExecutor`), генеруючи тисячі партій паралельно.
  Тут — послідовний, обмежений цикл (кілька партій за раз).
- **Реальне повноцінне тренування** — оригінал тренував на корпусі до
  100 000 позицій з регулярними чекпоінтами. Тут — невелика кількість
  self-play партій за прогін; щоб модель реально стала сильнішою за вже
  натреновану `model_best_weight.h5`, потрібні тисячі партій і багато
  циклів `pipeline_torch.py` — довгий прогін, не одноразовий запуск.
- **FTP-розподілена генерація** оригіналу — свідомо не портовано (мертвий
  сервер 2017 року, пароль у відкритому тексті в старому конфігу).
- **Дистиляція від зовнішнього рушія (Stockfish тощо)** — свідомо НЕ
  зроблено: це дало б сильнішу гру швидше, але порушило б "zero" —
  принцип нуля людських/зовнішніх знань, на якому стоїть AlphaZero.

About
=====

Chess reinforcement learning by [AlphaGo Zero](https://deepmind.com/blog/alphago-zero-learning-scratch/) methods.

This project is based on these main resources:
1) DeepMind's Oct 19th publication: [Mastering the Game of Go without Human Knowledge](https://www.nature.com/articles/nature24270.epdf?author_access_token=VJXbVjaSHxFoctQQ4p2k4tRgN0jAjWel9jnR3ZoTv0PVW4gB86EEpGqTRDtpIz-2rmo8-KG06gqVobU5NSCFeHILHcVFUeMsbvwS-lxjqQGg98faovwjxeTUgZAUMnRQ).
2) The <b>great</b> Reversi development of the DeepMind ideas that @mokemokechicken did in his repo: https://github.com/mokemokechicken/reversi-alpha-zero
3) DeepMind just released a new version of AlphaGo Zero (named now AlphaZero) where they master chess from scratch:
https://arxiv.org/pdf/1712.01815.pdf. In fact, in chess AlphaZero outperformed Stockfish after just 4 hours (300k steps) Wow!

See the [wiki](https://github.com/Akababa/Chess-Zero/wiki) for more details.

Note
----

I'm the creator of this repo. I (and some others collaborators did our best: https://github.com/Zeta36/chess-alpha-zero/graphs/contributors) but we found the self-play is too much costed for an only machine. Supervised learning worked fine but we never try the self-play by itself.

Anyway I want to mention we have moved to a new repo where lot of people is working in a distributed version of AZ for chess (MCTS in C++): https://github.com/glinscott/leela-chess

Project is almost done and everybody will be able to participate just by executing a pre-compiled windows (or Linux) application. A really great job and effort has been done is this project and I'm pretty sure we'll be able to simulate the DeepMind results in not too long time of distributed cooperation.

So, I ask everybody that wish to see a UCI engine running a neural network to beat Stockfish go into that repo and help with his machine power.

Environment
-----------

* Python 3.6.3
* tensorflow-gpu: 1.3.0
* Keras: 2.0.8

### New results (after a great number of modifications due to @Akababa)

Using supervised learning on about 10k games, I trained a model (7 residual blocks of 256 filters) to a guesstimate of 1200 elo with 1200 sims/move. One of the strengths of MCTS is it scales quite well with computing power.

Here you can see an example where I (black) played against the model in the repo (white):

![img](https://user-images.githubusercontent.com/4205182/34333105-ada817c6-e8fe-11e7-8c01-5958aaf264c1.gif)

Here you can see an example of a game where I (white, ~2000 elo) played against the model in this repo (black):

![img](https://user-images.githubusercontent.com/4205182/34323276-ecd2a7b6-e806-11e7-856a-4e2394bd75df.gif)

### First "good" results

Using the new supervised learning step I created, I've been able to train a model to the point that seems to be learning the openings of chess. Also it seems the model starts to avoid losing naively pieces.

Here you can see an example of a game played for me against this model (AI plays black):

![partida1](https://user-images.githubusercontent.com/17341905/33597844-ea53c8ae-d9a0-11e7-8564-4b9b0f35a221.gif)

Here we have a game trained by @bame55 (AI plays white):

![partida3](https://user-images.githubusercontent.com/17341905/34030278-8796f7c6-e16c-11e7-9ba4-97af15f2cde5.gif)

This model plays in this way after only 5 epoch iterations of the 'opt' worker, the 'eval' worker changed 4 times the best model (4 of 5). At this moment the loss of the 'opt' worker is 5.1 (and still seems to be converging very well).

Modules
-------

### Supervised Learning

I've done a supervised learning new pipeline step (to use those human games files "PGN" we can find in internet as play-data generator).
This SL step was also used in the first and original version of AlphaGo and maybe chess is a some complex game that we have to pre-train first the policy model before starting the self-play process (i.e., maybe chess is too much complicated for a self training alone).

To use the new SL process is as simple as running in the beginning instead of the worker "self" the new worker "sl".
Once the model converges enough with SL play-data we just stop the worker "sl" and start the worker "self" so the model will start improving now due to self-play data.

```bash
python src/chess_zero/run.py sl
```
If you want to use this new SL step you will have to download big PGN files (chess files) and paste them into the `data/play_data` folder ([FICS](http://ficsgames.org/download.html) is a good source of data). You can also use the [SCID program](http://scid.sourceforge.net/) to filter by headers like player ELO, game result and more.

**To avoid overfitting, I recommend using data sets of at least 3000 games and running at most 3-4 epochs.**

### Reinforcement Learning

This AlphaGo Zero implementation consists of three workers: `self`, `opt` and `eval`.

* `self` is Self-Play to generate training data by self-play using BestModel.
* `opt` is Trainer to train model, and generate next-generation models.
* `eval` is Evaluator to evaluate whether the next-generation model is better than BestModel. If better, replace BestModel.


### Distributed Training

Now it's possible to train the model in a distributed way. The only thing needed is to use the new parameter:

* `--type distributed`: use mini config for testing, (see `src/chess_zero/configs/distributed.py`)

So, in order to contribute to the distributed team you just need to run the three workers locally like this:

```bash
python src/chess_zero/run.py self --type distributed (or python src/chess_zero/run.py sl --type distributed)
python src/chess_zero/run.py opt --type distributed
python src/chess_zero/run.py eval --type distributed
```

### GUI
* `uci` launches the Universal Chess Interface, for use in a GUI.

To set up ChessZero with a GUI, point it to `C0uci.bat` (or rename to .sh).
For example, this is screenshot of the random model using Arena's self-play feature:
![capture](https://user-images.githubusercontent.com/4205182/34057277-e9c99118-e19b-11e7-91ee-dd717f7efe9d.PNG)

Data
-----

* `data/model/model_best_*`: BestModel.
* `data/model/next_generation/*`: next-generation models.
* `data/play_data/play_*.json`: generated training data.
* `logs/main.log`: log file.

If you want to train the model from the beginning, delete the above directories.

How to use
==========

> **2026 PyTorch port note:** `requirements.txt` now pins the PyTorch
> stack (see "Встановлення" at the top of this file), not the original
> tensorflow-gpu/keras below -- this section is preserved as-is for
> historical/upstream provenance, but following it literally today
> installs torch, not TensorFlow.

Setup
-------
### install libraries
```bash
pip install -r requirements.txt
```

If you want to use GPU, follow [these instructions](https://www.tensorflow.org/install/) to install with pip3.

Make sure Keras is using Tensorflow and you have Python 3.6.3+. Depending on your environment, you may have to run python3/pip3 instead of python/pip.


Basic Usage
------------

For training model, execute `Self-Play`, `Trainer` and `Evaluator`.

**Note**: Make sure you are running the scripts from the top-level directory of this repo, i.e. `python src/chess_zero/run.py opt`, not `python run.py opt`.


Self-Play
--------

```bash
python src/chess_zero/run.py self
```

When executed, Self-Play will start using BestModel.
If the BestModel does not exist, new random model will be created and become BestModel.

### options
* `--new`: create new BestModel
* `--type mini`: use mini config for testing, (see `src/chess_zero/configs/mini.py`)

Trainer
-------

```bash
python src/chess_zero/run.py opt
```

When executed, Training will start.
A base model will be loaded from latest saved next-generation model. If not existed, BestModel is used.
Trained model will be saved every epoch.

### options
* `--type mini`: use mini config for testing, (see `src/chess_zero/configs/mini.py`)
* `--total-step`: specify total step(mini-batch) numbers. The total step affects learning rate of training.

Evaluator
---------

```bash
python src/chess_zero/run.py eval
```

When executed, Evaluation will start.
It evaluates BestModel and the latest next-generation model by playing about 200 games.
If next-generation model wins, it becomes BestModel.

### options
* `--type mini`: use mini config for testing, (see `src/chess_zero/configs/mini.py`)


Tips and Memory
====

GPU Memory
----------

Usually the lack of memory cause warnings, not error.
If error happens, try to change `vram_frac` in `src/configs/mini.py`,

```python
self.vram_frac = 1.0
```

Smaller batch_size will reduce memory usage of `opt`.
Try to change `TrainerConfig#batch_size` in `MiniConfig`.
