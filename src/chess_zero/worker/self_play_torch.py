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

Avoids data_helper.py's pretty_print/write_game_data_to_file, both of
which import pyperclip (clipboard access, fails headless) -- writes
plain JSON directly instead, same on-disk shape.
"""
import json
import os
import sys
import time

import torch

sys.path.insert(0, ".")
from chess_zero.agent.load_weights import load_torch_model
from chess_zero.agent.player_chess_torch import TorchChessPlayer, PlayConfig
from chess_zero.env.chess_env import ChessEnv, Winner


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


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_torch_model("../data/model/model_best_weight.h5").to(device)
    print(f"Model on: {device}")

    pc = PlayConfig()
    pc.simulation_num_per_move = 50  # reduced for a quick verification run, not a real self-play session
    pc.resign_threshold = None

    max_halfmoves = 20  # bounded verification, not a full game -- see module docstring

    t0 = time.time()
    env, data = self_play_game(model, pc, max_halfmoves=max_halfmoves)
    dt = time.time() - t0

    truncated = env.num_halfmoves >= max_halfmoves and not env.done
    print(f"Game {'(truncated at cap, adjudicated)' if truncated else '(reached a real result)'}: "
          f"{env.num_halfmoves} halfmoves, result={env.result}, {dt:.1f}s")
    print(f"Collected {len(data)} training triples (fen, policy, value)")

    out_path = "/tmp/self_play_verify.json"
    with open(out_path, "w") as f:
        json.dump(data, f)
    print(f"Saved to {out_path} ({os.path.getsize(out_path):,} bytes) -- verification output, not a real training corpus")
