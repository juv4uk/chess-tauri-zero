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
import os
import sys
import threading

sys.path.insert(0, ".")
from chess_zero.agent.load_weights import load_torch_model
from chess_zero.agent.player_chess_torch import TorchChessPlayer, PlayConfig
from chess_zero.agent.batched_predictor import BatchedPredictor
from chess_zero.env.chess_env import ChessEnv
import json

from chess_zero.worker.pipeline_torch import run_cycle, model_history


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
            env.step(action)
            print(f"selfplaymove {action}")
            sys.stdout.flush()
        if not env.done:
            env.adjudicate()  # covers both the halfmove cap and an explicit stop mid-game
        print(f"selfplayresult {env.result}")
        sys.stdout.flush()
    finally:
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


def _train_worker():
    global _train_thread
    try:
        model = get_shared_model()
        device = next(model.parameters()).device

        def on_progress(stage):
            print(f"info trainprogress {stage}")
            sys.stdout.flush()

        result_model = run_cycle(model, device, on_progress=on_progress)
        promoted = result_model is not model
        if promoted:
            reload_models(_shared_model, me_player.model if me_player else None)
        print(f"trainresult {'promoted' if promoted else 'not-promoted'}")
        sys.stdout.flush()
    except Exception as e:
        print(f"trainerror {e}")
        sys.stdout.flush()
    finally:
        with _train_lock:
            _train_thread = None


def start():
    # me_player is module-level (not a plain local) so _train_worker's
    # background thread can hot-reload it on promotion -- see reload_models().
    global _selfplay_thread, _train_thread, me_player
    env = ChessEnv().reset()

    while True:
        line = input()
        words = line.rstrip().split(" ", 1)
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
        elif words[0] == "position":
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
            if not me_player:
                me_player = get_player()
            action = me_player.action(env, False)
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
                if _selfplay_thread is None or not _selfplay_thread.is_alive():
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
                    elif machine_busy():
                        print("trainerror machine busy, try again later")
                        sys.stdout.flush()
                    else:
                        _train_thread = threading.Thread(target=_train_worker, daemon=True)
                        _train_thread.start()
        elif words[0] == "history":
            # P0 "generation journal + quality curve": one line per
            # promoted cycle (cycle/win_rate/promoted_at), oldest first.
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
            found = reload_models(get_shared_model(), me_player.model)
            print(f"reloadresult {'ok' if found else 'nothing-to-reload'}")
            sys.stdout.flush()
        elif words[0] == "stop":
            pass
        elif words[0] == "quit":
            break


if __name__ == "__main__":
    start()
