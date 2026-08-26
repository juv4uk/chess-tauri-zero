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
from chess_zero.worker.pipeline_torch import run_cycle


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


# --- shared model for selfplay/train, separate from get_player()'s own
# model -- get_player() is left untouched (existing, verified behavior),
# so selfplay/train load+cache their own reference instead of reusing it.
_shared_model = None
_shared_model_lock = threading.Lock()


def get_shared_model():
    global _shared_model
    with _shared_model_lock:
        if _shared_model is None:
            _shared_model = load_torch_model("../data/model/model_best_weight.h5")
    return _shared_model


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
# progress. Does NOT hot-swap _shared_model or get_player()'s model on
# promotion -- a promoted candidate updates data/model_torch/model_best.pt
# on disk, but this running process keeps whatever model it already
# loaded until restarted. Known follow-up, out of scope here.
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
        print(f"trainresult {'promoted' if promoted else 'not-promoted'}")
        sys.stdout.flush()
    except Exception as e:
        print(f"trainerror {e}")
        sys.stdout.flush()
    finally:
        with _train_lock:
            _train_thread = None


def start():
    global _selfplay_thread, _train_thread
    me_player = None
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
                    else:
                        load1 = os.getloadavg()[0]
                        cores = os.cpu_count() or 1
                        if load1 > cores * 1.5:
                            print("trainerror machine busy, try again later")
                            sys.stdout.flush()
                        else:
                            _train_thread = threading.Thread(target=_train_worker, daemon=True)
                            _train_thread.start()
        elif words[0] == "stop":
            pass
        elif words[0] == "quit":
            break


if __name__ == "__main__":
    start()
