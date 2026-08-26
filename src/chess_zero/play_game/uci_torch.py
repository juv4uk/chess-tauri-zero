"""PyTorch-backed UCI engine loop -- direct port of uci.py.

The UCI protocol loop itself never touched Keras/TF (confirmed by
reading uci.py directly) -- the only place that did was get_player(),
which built a Keras ChessModel + ChessPlayer. That's the only real
change here: get_player() now loads the PyTorch model and returns a
TorchChessPlayer (player_chess_torch.py, roadmap step 1) instead.

Usage as a UCI engine (e.g. with python-chess's SimpleEngine, or any
UCI-speaking GUI like Arena/CuteChess pointed at this script via a
wrapper that runs `python3 uci_torch.py`):

    echo -e "uci\nisready\nposition startpos\ngo\nquit" | python3 uci_torch.py
"""
import chess
import os
import shutil
import sys
import threading
import time
from datetime import datetime

sys.path.insert(0, ".")
from chess_zero.agent.load_weights import load_torch_model
from chess_zero.agent.player_chess_torch import TorchChessPlayer, PlayConfig
from chess_zero.agent.batched_predictor import BatchedPredictor
from chess_zero.agent.torch_model import ChessResNet
from chess_zero.env.chess_env import ChessEnv
from chess_zero.lib.data_helper_torch import write_game_data_to_file
from chess_zero.worker.optimize_torch import save_checkpoint
import json

from chess_zero.worker.pipeline_torch import run_cycle, model_history, BEST_MODEL_PATH, NEXT_GEN_DIR


def human_play_config() -> PlayConfig:
    pc = PlayConfig()
    pc.simulation_num_per_move = 200
    pc.noise_eps = 0
    pc.tau_decay_rate = 0
    pc.resign_threshold = None
    return pc


def get_player():
    model = load_torch_model("../data/model/model_best_weight.h5")
    return TorchChessPlayer(model, human_play_config())


def machine_busy():
    """os.getloadavg() is POSIX-only -- raises on Windows (confirmed
    live: the owner's own Windows run would have crashed the whole UCI
    process here, uncaught, the moment `train start` was sent). No
    stdlib equivalent exists for Windows, so this degrades to "never
    busy" there rather than adding a new dependency (psutil) just for
    a nice-to-have safety check -- the check still protects the Linux
    dev machine, which is where it was actually needed."""
    try:
        load1 = os.getloadavg()[0]
    except (AttributeError, OSError):
        return False
    cores = os.cpu_count() or 1
    return load1 > cores * 1.5


# --- shared model for selfplay/train/play. Loaded once from the
# original h5 weights, then reloaded in place from
# data/model_torch/model_best.pt (state_dict) whenever `reload` is
# received or a `train start` promotes a new candidate -- see reload().
# get_player()'s own model is a SEPARATE instance still loaded fresh
# from the h5 file each time (existing, verified behavior, untouched);
# reload() updates get_player()'s cached instance too if one exists,
# so a human game in progress also picks up a promoted model without
# restarting the whole process.
_shared_model = None
_shared_model_lock = threading.Lock()

# The live player used for normal `go` moves -- module-level (not
# start()-local) so a background thread (train promotion) can hot-reload
# its .model in place. None until the first `isready`/`go`/`setoption`.
me_player = None

# Guards me_player.model specifically: a human `go` reads it (forward
# pass inside action()) while a training promotion's hot-reload writes
# it in place (load_state_dict()) from a DIFFERENT thread. Before the
# "let training keep running detached in the background" feature, this
# was only latently possible (the GUI blocked all other actions while
# training ran); that feature makes a human move genuinely concurrent
# with a real promotion a real, reachable scenario, not just a
# theoretical one -- so the two now share this lock instead of racing
# on live tensor data mid-copy.
_me_player_lock = threading.Lock()


# --- human game recording: docs/development-plan-uk.md P1 item 7,
# "перший, дешевий крок" -- a human's own move has no MCTS-derived
# policy distribution (the human just plays it, no search happens), so
# it's recorded as a one-hot policy (1.0 on the move actually played,
# 0 everywhere else) -- the standard way to represent a move without a
# real search distribution in this same (fen, policy, value) training
# format optimize_torch.py's data pipeline already reads unchanged.
# Written to its OWN directory (never data/play_data_torch/, which is
# self-play only) -- the directory itself IS the "source: human" tag
# the plan asked for; no per-record field was added since the shared
# JSON shape (a flat [fen, policy, value] list) has no field for one
# without changing what optimize_torch.py/self-play both already read.
HUMAN_PLAY_DATA_DIR = "../data/play_data_human"
_human_game_moves = []  # [[fen_before_move, one_hot_policy], ...] for the CURRENT game in progress


def _one_hot_policy(uci_move):
    """labels_n-length policy with a single 1.0 at the played move's
    index, 0 elsewhere -- reuses me_player's own label encoding
    (identical across every TorchChessPlayer instance, same LABELS
    table) so this stays consistent with whatever policy vectors
    self-play/training already produce, with no separate table to
    keep in sync."""
    policy = [0.0] * me_player.labels_n
    idx = me_player.move_lookup.get(chess.Move.from_uci(uci_move))
    if idx is not None:
        policy[idx] = 1.0
    return policy


def _record_human_move(fen_before, uci_move):
    """fen_before comes from the FRONTEND (its own chess.js `game`
    object), not from this process's own `env` -- a real bug caught by
    testing this end-to-end rather than trusting it from reading the
    code: `go` computes the engine's OWN reply move but never applies
    it to `env` (only an explicit `position ... moves ...` command
    does, and that only arrives later, as part of the NEXT human
    move's own round-trip) -- so by the time a second/later humanmove
    arrives, this process's `env` is stale by exactly one ply (missing
    the engine's most recent reply). The frontend's chess.js state has
    no such lag, since it applies every move (its own and the
    engine's) the instant it happens."""
    _human_game_moves.append([fen_before, _one_hot_policy(uci_move)])


def _save_human_game(result):
    """result: "human-win" | "human-loss" | "draw" (or anything else,
    treated as an unknown/drawn outcome -- see uci_torch.py's own
    `humangameover` handler for exactly what the frontend sends and
    when). Every recorded position in this one game gets the SAME z,
    from the human's own perspective throughout -- unlike self-play's
    white/black split, there's only one side here that matters."""
    global _human_game_moves
    if not _human_game_moves:
        return
    z = {"human-win": 1.0, "human-loss": -1.0}.get(result, 0.0)
    data = [[fen, policy, z] for fen, policy in _human_game_moves]
    os.makedirs(HUMAN_PLAY_DATA_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    path = os.path.join(HUMAN_PLAY_DATA_DIR, f"human_{ts}.json")
    write_game_data_to_file(path, data)
    print(f"humangameoverresult saved {len(data)} moves to {path}")
    sys.stdout.flush()
    _human_game_moves = []


def get_shared_model():
    global _shared_model
    with _shared_model_lock:
        if _shared_model is None:
            _shared_model = load_torch_model("../data/model/model_best_weight.h5")
    return _shared_model


def current_best_checkpoint():
    """data/model_torch/model_best.pt if a promoted checkpoint exists
    (torch.save(state_dict) format, written by optimize_torch.save_checkpoint),
    else None -- caller falls back to the original h5 weights."""
    import os as _os
    path = "../data/model_torch/model_best.pt"
    return path if _os.path.exists(path) else None


def reload_models(*models):
    """Loads the current best checkpoint's state_dict into every given
    model in place (torch.load + load_state_dict) -- same architecture
    (ChessResNet) whether it came from the h5 weights or a previous
    checkpoint, so this works uniformly. Returns True if a checkpoint
    was found and loaded, False if there was nothing newer than the
    original h5 weights to load."""
    import torch
    ckpt = current_best_checkpoint()
    if ckpt is None:
        return False
    for model in models:
        if model is None:
            continue
        device = next(model.parameters()).device
        model.load_state_dict(torch.load(ckpt, map_location=device))
    return True


def reset_model_to_scratch():
    """"Справжній нуль" -- owner's explicit choice over the alternative
    of resetting to the shipped h5 baseline: a freshly, randomly
    initialized ChessResNet, no supervised-learning h5 weights loaded
    at all. Applies AlphaZero's zero-external-knowledge principle to
    the WHOLE lineage, not just the self-play loop on top of it --
    readme.md's own honest note is that model_best_weight.h5 itself
    came from 2017-era supervised learning on ~10k human games, which
    this bypasses entirely.

    Genuinely destructive, by design: also clears the ENTIRE
    generation journal (NEXT_GEN_DIR), since every existing record's
    win_rate/parent chain was measured against the OLD baseline and
    would be actively misleading plotted against a fresh one -- a
    "cycle 12: 61%" from the old lineage means nothing once the
    baseline itself has changed underneath it.

    Caller (`resetmodel` UCI command) is responsible for refusing this
    while selfplay/train is running -- mutating _shared_model/
    me_player.model out from under a live search or training thread
    would be exactly the kind of unsynchronized-tensor-write race an
    earlier fix this session (_me_player_lock) closed for promotion;
    this function itself only takes that lock for the brief, already-
    serialized case of a `go` running concurrently, not for
    selfplay/train (structurally prevented by the caller instead,
    matching fix #7's own selfplay/train mutual exclusion)."""
    fresh = ChessResNet()
    fresh.eval()
    fresh_state = fresh.state_dict()

    global _shared_model
    with _shared_model_lock:
        if _shared_model is None:
            _shared_model = fresh
        else:
            _shared_model.load_state_dict(fresh_state)
    with _me_player_lock:
        if me_player is not None:
            me_player.model.load_state_dict(fresh_state)

    save_checkpoint(fresh, BEST_MODEL_PATH)
    if os.path.isdir(NEXT_GEN_DIR):
        shutil.rmtree(NEXT_GEN_DIR)


# --- selfplay: two TorchChessPlayers sharing one model/predictor play
# each other continuously, streaming each move as it happens.
_selfplay_thread = None
_selfplay_stop_event = threading.Event()


def _selfplay_worker(model, stop_event):
    predictor = BatchedPredictor(model)
    pc = PlayConfig()
    pc.simulation_num_per_move = 50  # spectator-mode default, same choice self_play_torch.py's own __main__ uses
    pc.resign_threshold = None
    max_halfmoves = 200  # generous cap for a mode meant to be watched/stopped by the user, not a training bound
    try:
        env = ChessEnv().reset()
        white = TorchChessPlayer(model, pc, predictor=predictor)
        black = TorchChessPlayer(model, pc, predictor=predictor)
        while not env.done and env.num_halfmoves < max_halfmoves:
            if stop_event.is_set():
                break
            player = white if env.white_to_move else black
            action = player.action(env)
            print(f"info mctsroot {json.dumps(player.root_stats(env))}")
            env.step(action)
            print(f"selfplaymove {action}")
            sys.stdout.flush()
        if not env.done:
            env.adjudicate()  # covers both the halfmove cap and an explicit stop mid-game
        print(f"selfplayresult {env.result}")
    except Exception as e:
        # Real, confirmed bug this responds to: this whole function runs
        # on its own daemon thread, outside the main loop's `_dispatch`
        # try/except added earlier -- an uncaught exception here used to
        # just kill the thread silently (traceback to stderr only), so
        # `selfplayresult` was NEVER printed and the frontend's
        # `pollUntil` waiting on it hung for its full 600s timeout with
        # zero feedback. "*" is the standard PGN "unknown/unfinished
        # result" marker -- distinct from a real 1-0/0-1/1/2-1/2 -- so a
        # client that DOES start caring about the value later can tell
        # "crashed" apart from "actually finished".
        print(f"info error [selfplay] {e}")
        print("selfplayresult *")
    finally:
        sys.stdout.flush()
        predictor.stop()


# --- train: one bounded pipeline_torch.run_cycle(), streaming phase
# progress. On promotion, hot-reloads _shared_model AND the live
# me_player (module-level global, see start()) from the checkpoint
# run_cycle just wrote to disk -- both a human game in progress and
# future selfplay/train pick up the promoted weights without
# restarting the process. Uses reload_models() rather than swapping in
# run_cycle's own returned candidate object, so this is the exact same
# code path the explicit `reload` UCI command uses.
_train_thread = None
_train_lock = threading.Lock()


def _train_worker(background=False):
    """background=True is used by _background_train_loop() (no UCI
    `train start` command was sent) -- prints info-prefixed
    bgtrainprogress/bgtrainresult/bgtrainerror lines instead of the
    plain trainprogress/trainresult/trainerror lines a manually
    triggered run uses, so a client watching stdout can tell the two
    apart, but everything else (the run_cycle call, the hot-reload on
    promotion, _train_lock/_train_thread bookkeeping) is identical --
    both paths share this one worker rather than duplicating it."""
    global _train_thread
    progress_word = "bgtrainprogress" if background else "trainprogress"
    result_line = "info bgtrainresult" if background else "trainresult"
    error_line = "info bgtrainerror" if background else "trainerror"
    try:
        model = get_shared_model()
        device = next(model.parameters()).device

        def on_progress(stage):
            print(f"info {progress_word} {stage}")
            sys.stdout.flush()

        result_model = run_cycle(model, device, on_progress=on_progress)
        promoted = result_model is not model
        if promoted:
            with _me_player_lock:
                reload_models(_shared_model, me_player.model if me_player else None)
        print(f"{result_line} {'promoted' if promoted else 'not-promoted'}")
        sys.stdout.flush()
    except Exception as e:
        print(f"{error_line} {e}")
        sys.stdout.flush()
    finally:
        with _train_lock:
            _train_thread = None


# --- background self-training: when the machine is idle AND no human
# game appears to be in progress AND nothing else (selfplay/manual
# train) is already running, automatically run one unattended
# run_cycle every so often. The engine has no direct signal from the
# frontend for "a human is mid-game" (the frontend tracks its own
# `mode` state client-side; nothing communicates that to the engine
# process) -- the proxy used here is elapsed time since the last
# `go`/`position` command. BG_IDLE_THRESHOLD_SEC=300 (5 min) is a
# judgment call: MCTS moves in this app resolve in single-digit
# seconds even at the "hard" difficulty (confirmed earlier this
# session: 200 sims ~9.6s/move), so 5 minutes of silence is already
# many multiples of a real thinking pause; shorter would risk firing
# while a human is still mid-game between moves, longer would just
# mean background training runs less often than it safely could.
BG_CHECK_INTERVAL_SEC = 120
BG_IDLE_THRESHOLD_SEC = 300

_last_human_activity_lock = threading.Lock()
_last_human_activity = None  # None = no go/position seen yet this process -> treated as idle


def _mark_human_activity():
    global _last_human_activity
    with _last_human_activity_lock:
        _last_human_activity = time.time()


def _human_possibly_active():
    with _last_human_activity_lock:
        last = _last_human_activity
    return last is not None and (time.time() - last) < BG_IDLE_THRESHOLD_SEC


def _background_train_loop():
    global _train_thread
    while True:
        time.sleep(BG_CHECK_INTERVAL_SEC)
        # PLAUSIBLE bug from today's 4-agent audit (Python Logic
        # Auditor, not reproduced but structurally real): this whole
        # loop had no exception guard -- any uncaught exception on any
        # iteration (in _human_possibly_active/machine_busy/thread
        # creation) would kill this daemon thread silently, with no
        # traceback anywhere a client could see, permanently ending
        # background training for the rest of the process's life with
        # zero indication anything went wrong. Reusing the [tag] error
        # format from the main crash-guard fixed earlier today.
        try:
            if _human_possibly_active():
                continue
            if _selfplay_thread is not None and _selfplay_thread.is_alive():
                continue
            if machine_busy():
                continue
            with _train_lock:
                if _train_thread is not None and _train_thread.is_alive():
                    continue
                _train_thread = threading.Thread(target=_train_worker, kwargs={"background": True}, daemon=True)
                _train_thread.start()
        except Exception as e:
            print(f"info error [background-train] {e}")
            sys.stdout.flush()


def start():
    # me_player is module-level (not a plain local) so _train_worker's
    # background thread can hot-reload it on promotion -- see reload_models().
    global _selfplay_thread, _train_thread, me_player
    env = ChessEnv().reset()
    threading.Thread(target=_background_train_loop, daemon=True).start()

    while True:
        try:
            line = input()
        except EOFError:
            break
        words = line.rstrip().split(" ", 1)
        try:
            _dispatch(words, env)
        except _Quit:
            break
        except Exception as e:
            # A real, live-reported bug this fix responds to: any
            # uncaught exception ANYWHERE in command handling used to
            # kill the WHOLE process (input() loop had no guard at
            # all), which the frontend then reported as "Двигун
            # відключився, перезапускаю..." (broken pipe) and silently
            # respawned -- masking the real error instead of surfacing
            # it. Now one bad command reports itself and the engine
            # keeps running.
            # Tagged with the failing command's own name (words[0], the
            # same token _dispatch() switched on) so a client watching
            # for "info error [<cmd>] " can match it to whatever it was
            # actually waiting for -- part of the same audit fix as the
            # [selfplay] tags elsewhere in this file.
            print(f"info error [{words[0]}] {e}")
            sys.stdout.flush()


class _Quit(Exception):
    pass


def _dispatch(words, env):
    global _selfplay_thread, _train_thread, me_player
    if words[0] == "uci":
        print("id name ChessZeroTorch")
        print("id author ChessZero (PyTorch port)")
        print("uciok")
    elif words[0] == "isready":
        if not me_player:
            me_player = get_player()
        print("readyok")
    elif words[0] == "ucinewgame":
        env.reset()
        _human_game_moves.clear()  # abandon any unfinished game's recording rather than bleed into the next one
    elif words[0] == "position":
        _mark_human_activity()
        words = words[1].split(" ", 1)
        if words[0] == "startpos":
            env.reset()
        else:
            if words[0] == "fen":
                words = words[1].split(' ', 1)
            fen = words[0]
            for _ in range(5):
                words = words[1].split(' ', 1)
                fen += " " + words[0]
            env.update(fen)
        if len(words) > 1:
            words = words[1].split(" ", 1)
            if words[0] == "moves":
                for w in words[1].split(" "):
                    env.step(w, False)
    elif words[0] == "go":
        _mark_human_activity()
        if not me_player:
            me_player = get_player()
        with _me_player_lock:
            action = me_player.action(env, False)
            mcts_stats = me_player.root_stats(env)
        print(f"info mctsroot {json.dumps(mcts_stats)}")
        print(f"bestmove {action}")
        sys.stdout.flush()
    elif words[0] == "setoption":
        # "setoption name Simulations value <N>" -- difficulty control,
        # only Simulations is recognized, anything else is ignored.
        rest = words[1] if len(words) > 1 else ""
        if rest.startswith("name "):
            rest = rest[len("name "):]
            if " value " in rest:
                name_part, value_part = rest.split(" value ", 1)
                if name_part.strip() == "Simulations":
                    try:
                        n = int(value_part.strip())
                    except ValueError:
                        n = None
                    if n is not None:
                        if not me_player:
                            me_player = get_player()
                        me_player.play_config.simulation_num_per_move = n
    elif words[0] == "selfplay":
        sub = words[1].strip() if len(words) > 1 else ""
        if sub == "start":
            # Real, CONFIRMED race from today's 4-agent audit (Python
            # Logic Auditor, reproduced live): selfplay and train both
            # use get_shared_model() -- the SAME model object -- but
            # neither used to check the other's running state, unlike
            # _background_train_loop, which already guards against
            # _selfplay_thread. A promotion's reload_models() does an
            # in-place load_state_dict() on that shared model while
            # selfplay could still be mid-forward-pass on it: a real,
            # unsynchronized read/write race on live weights, not just
            # a UX inconsistency. Refusing here (mirroring train
            # start's own already-established trainerror pattern) is
            # cheaper and safer than trying to make concurrent
            # self-play/training on one shared model actually correct.
            if _train_thread is not None and _train_thread.is_alive():
                print("info error [selfplay] training already running")
                print("selfplayresult *")
                sys.stdout.flush()
            elif _selfplay_thread is None or not _selfplay_thread.is_alive():
                _selfplay_stop_event.clear()
                model = get_shared_model()
                _selfplay_thread = threading.Thread(
                    target=_selfplay_worker, args=(model, _selfplay_stop_event), daemon=True)
                _selfplay_thread.start()
            # already running: ignored, per spec (no second instance)
        elif sub == "stop":
            _selfplay_stop_event.set()
    elif words[0] == "train":
        sub = words[1].strip() if len(words) > 1 else ""
        if sub == "start":
            with _train_lock:
                running = _train_thread is not None and _train_thread.is_alive()
                if running:
                    print("trainerror already running")
                    sys.stdout.flush()
                elif _selfplay_thread is not None and _selfplay_thread.is_alive():
                    # Symmetric side of the same fix -- see the
                    # "selfplay start" branch's comment above for the
                    # real race this closes.
                    print("trainerror selfplay running")
                    sys.stdout.flush()
                elif machine_busy():
                    print("trainerror machine busy, try again later")
                    sys.stdout.flush()
                else:
                    _train_thread = threading.Thread(target=_train_worker, daemon=True)
                    _train_thread.start()
    elif words[0] == "resetmodel":
        # "Справжній нуль": discard all training progress, start from
        # a fresh random-weight model. Refused while selfplay/train is
        # live -- mirrors that fix's own mutual-exclusion pattern,
        # since mutating shared model weights mid-search/mid-training
        # would be exactly the race that fix closed for promotion.
        #
        # Real TOCTOU race found in today's round-2 audit (Python
        # Logic Auditor): the check-then-reset used to run WITHOUT
        # _train_lock, but _background_train_loop does its own
        # independent check-and-start of _train_thread under
        # _train_lock every BG_CHECK_INTERVAL_SEC on a separate
        # thread -- a background cycle could start in the narrow
        # window between this handler's check and reset_model_to_
        # scratch() actually running, mutating the SAME _shared_model
        # object reset_model_to_scratch() is concurrently rewriting.
        # Holding _train_lock for the whole check+reset here closes
        # it: whichever side acquires the lock first is the one that
        # sees accurate state, the other correctly detects a
        # conflict instead of racing.
        with _train_lock:
            if _train_thread is not None and _train_thread.is_alive():
                print("resetmodelresult error training running")
            elif _selfplay_thread is not None and _selfplay_thread.is_alive():
                print("resetmodelresult error selfplay running")
            else:
                reset_model_to_scratch()
                print("resetmodelresult ok")
        sys.stdout.flush()
    elif words[0] == "humanmove":
        # "humanmove <fen-before-move, 6 space-separated fields> <uci
        # move>" -- fen comes from the frontend's own chess.js state,
        # not this process's `env` (see _record_human_move's docstring
        # for the real staleness bug that forced this design). Split on
        # the LAST space only, since the FEN itself is multi-token.
        rest = words[1] if len(words) > 1 else ""
        if not me_player:
            me_player = get_player()
        if " " in rest:
            fen_before, uci_move = rest.rsplit(" ", 1)
            _record_human_move(fen_before, uci_move)
    elif words[0] == "humangameover":
        result = words[1].strip() if len(words) > 1 else ""
        _save_human_game(result)
    elif words[0] == "history":
        # P0 "generation journal + quality curve": one line per
        # evaluated cycle (promoted or rejected), oldest first.
        for record in model_history():
            print(f"historyentry {json.dumps(record)}")
        print("historyresult ok")
        sys.stdout.flush()
    elif words[0] == "reload":
        # Hot-reload the live player (and shared selfplay/train model)
        # from data/model_torch/model_best.pt if a promoted checkpoint
        # exists, without restarting the process.
        if not me_player:
            me_player = get_player()
        with _me_player_lock:
            found = reload_models(get_shared_model(), me_player.model)
        print(f"reloadresult {'ok' if found else 'nothing-to-reload'}")
        sys.stdout.flush()
    elif words[0] == "stop":
        pass
    elif words[0] == "quit":
        raise _Quit()


if __name__ == "__main__":
    start()
