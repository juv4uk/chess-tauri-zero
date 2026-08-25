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
import sys

sys.path.insert(0, ".")
from chess_zero.agent.load_weights import load_torch_model
from chess_zero.agent.player_chess_torch import TorchChessPlayer, PlayConfig
from chess_zero.env.chess_env import ChessEnv


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


def start():
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
        elif words[0] == "stop":
            pass
        elif words[0] == "quit":
            break


if __name__ == "__main__":
    start()
