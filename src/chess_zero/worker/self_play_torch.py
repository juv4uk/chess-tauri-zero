"""PyTorch port of self_play.py -- roadmap "restore all capabilities"
follow-up. Direct port of self_play_buffer(): two TorchChessPlayer
instances (player_chess_torch.py) alternate action(env) until the
game ends, collect (fen, policy, value) triples, write them as JSON
in the same shape write_game_data_to_file used
(list of [fen, policy_list, value]), so optimize_torch.py's data
pipeline can consume them unchanged.

Simplified from the original's ProcessPoolExecutor-based endless
worker pool (self_play.py's SelfPlayWorker) to a single-process,
single-game function -- the endless multi-process loop is production
infrastructure, not part of "does self-play actually work", and
running it for real is a genuinely long, resource-heavy job on its
own (not attempted here, same discipline as the rest of this port).

Uses lib/data_helper_torch.py's write_game_data_to_file (a straight
port of data_helper.py's, minus pretty_print/pyperclip, which fails
headless) so saved games are consumable by optimize_torch.py's
data pipeline without adaptation.
"""
import os
import sys
import time
from datetime import datetime

import torch

sys.path.insert(0, ".")
from chess_zero.agent.load_weights import load_torch_model
from chess_zero.agent.player_chess_torch import TorchChessPlayer, PlayConfig
from chess_zero.env.chess_env import ChessEnv, Winner
from chess_zero.lib.data_helper_torch import write_game_data_to_file, PLAY_DATA_DIR, PLAY_DATA_FILENAME_TMPL


def self_play_game(model, play_config: PlayConfig, max_halfmoves: int = 40):
    """Plays one game, both sides using the same model. max_halfmoves
    caps game length for a bounded verification run -- the original
    had no such cap (max_game_length=1000, i.e. play to a real result);
    a full game to checkmate/draw can take a very long time even on
    GPU with MCTS, so this defaults to a short, honest partial game
    unless the caller explicitly asks for more."""
    env = ChessEnv().reset()
    white = TorchChessPlayer(model, play_config)
    black = TorchChessPlayer(model, play_config)

    while not env.done and env.num_halfmoves < max_halfmoves:
        player = white if env.white_to_move else black
        action = player.action(env)
        if action is None:
            env.step(None)  # resignation
        else:
            env.step(action)

    if not env.done:
        env.adjudicate()  # material-based verdict for the truncated game, honestly labeled as such below

    if env.winner == Winner.white:
        black_win = -1
    elif env.winner == Winner.black:
        black_win = 1
    else:
        black_win = 0

    black.finish_game(black_win)
    white.finish_game(-black_win)

    data = []
    for i in range(len(white.moves)):
        data.append(white.moves[i])
        if i < len(black.moves):
            data.append(black.moves[i])

    return env, data


def self_play_loop(model, play_config: PlayConfig, num_games: int, max_halfmoves: int = 40,
                    data_dir: str = PLAY_DATA_DIR):
    """Plays num_games games and saves each as its own file via
    data_helper_torch, in the same on-disk shape the original
    self_play.py's SelfPlayWorker produced (one JSON file per game).
    Sequential, single-process -- the original ran many of these
    concurrently via ProcessPoolExecutor; that concurrency is
    infrastructure, not part of "does self-play produce usable data",
    and is not ported here (see docs/pytorch-self-play-loop-plan-uk.md)."""
    os.makedirs(data_dir, exist_ok=True)
    saved_paths = []
    for i in range(num_games):
        t0 = time.time()
        env, data = self_play_game(model, play_config, max_halfmoves=max_halfmoves)
        dt = time.time() - t0

        ts = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        path = os.path.join(data_dir, PLAY_DATA_FILENAME_TMPL % ts)
        write_game_data_to_file(path, data)
        saved_paths.append(path)

        truncated = env.num_halfmoves >= max_halfmoves and not env.done
        print(f"  game {i+1}/{num_games}: {env.num_halfmoves} halfmoves, result={env.result}"
              f"{' (truncated, adjudicated)' if truncated else ''}, {len(data)} moves, {dt:.1f}s -> {path}")
    return saved_paths


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_torch_model("../data/model/model_best_weight.h5").to(device)
    print(f"Model on: {device}")

    pc = PlayConfig()
    pc.simulation_num_per_move = 50  # reduced for a quick verification run, not a real self-play session
    pc.resign_threshold = None

    num_games = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    max_halfmoves = 20  # bounded verification, not full games -- see module docstring

    t0 = time.time()
    paths = self_play_loop(model, pc, num_games=num_games, max_halfmoves=max_halfmoves)
    dt = time.time() - t0
    print(f"Saved {len(paths)} game(s) to {PLAY_DATA_DIR}/ in {dt:.1f}s -- small-scale verification, not a real training corpus")
