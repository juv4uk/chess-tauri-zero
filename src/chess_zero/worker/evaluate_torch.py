"""PyTorch port of evaluate.py's arena logic -- candidate model vs
current best, alternating colors, promote the candidate to best if it
wins >= replace_rate of games. Direct port of EvaluateWorker's
evaluate_model()/play_game(), simplified from its
ProcessPoolExecutor-based concurrent game batch to sequential games
(same reasoning as self_play_torch.py's self_play_loop: concurrency
is infrastructure, not part of "does promotion logic work").

replace_rate=0.55 matches configs/mini.py's EvaluateConfig (read
directly) -- same threshold the original project used.
"""
import sys

import torch

sys.path.insert(0, ".")
from chess_zero.agent.player_chess_torch import TorchChessPlayer, PlayConfig
from chess_zero.agent.batched_predictor import BatchedPredictor
from chess_zero.env.chess_env import ChessEnv, Winner

REPLACE_RATE = 0.55


def eval_play_config():
    """Matches configs/mini.py's EvaluateConfig.play_config: noise_eps=0
    (no exploration noise -- we want the model's real preference, not
    self-play's exploration), lower c_puct than self-play's default."""
    pc = PlayConfig()
    pc.simulation_num_per_move = 100
    pc.c_puct = 1
    pc.tau_decay_rate = 0.6
    pc.noise_eps = 0
    pc.resign_threshold = None
    return pc


def play_one_game(candidate_model, best_model, candidate_white: bool, play_config: PlayConfig, max_halfmoves=40,
                   candidate_predictor=None, best_predictor=None):
    """Plays one game between candidate and best. Returns candidate's
    score: 1.0 win, 0.5 draw, 0.0 loss.

    candidate/best are different models, so they need separate
    BatchedPredictors (a batch can only mix leaves headed for the same
    model) -- each player's own search threads still batch together
    within their own predictor."""
    env = ChessEnv().reset()
    white_predictor = candidate_predictor if candidate_white else best_predictor
    black_predictor = best_predictor if candidate_white else candidate_predictor
    white_player = TorchChessPlayer(candidate_model if candidate_white else best_model, play_config, predictor=white_predictor)
    black_player = TorchChessPlayer(best_model if candidate_white else candidate_model, play_config, predictor=black_predictor)

    while not env.done and env.num_halfmoves < max_halfmoves:
        player = white_player if env.white_to_move else black_player
        action = player.action(env, can_stop=False)
        env.step(action)

    if not env.done:
        env.adjudicate()

    if env.winner == Winner.draw:
        return 0.5
    candidate_won_as_white = candidate_white and env.winner == Winner.white
    candidate_won_as_black = (not candidate_white) and env.winner == Winner.black
    return 1.0 if (candidate_won_as_white or candidate_won_as_black) else 0.0


def evaluate_candidate(candidate_model, best_model, game_num=4, max_halfmoves=40):
    """Plays game_num games, alternating who's white, and returns
    (candidate_win_rate, should_promote). game_num defaults small (4,
    vs the original's 50) for the same small-scale-verification reason
    as the rest of this loop."""
    pc = eval_play_config()
    candidate_predictor = BatchedPredictor(candidate_model)
    best_predictor = BatchedPredictor(best_model)
    scores = []
    try:
        for i in range(game_num):
            candidate_white = (i % 2 == 0)
            score = play_one_game(candidate_model, best_model, candidate_white, pc, max_halfmoves,
                                   candidate_predictor=candidate_predictor, best_predictor=best_predictor)
            scores.append(score)
            win_rate = sum(scores) / len(scores)
            print(f"  game {i+1}/{game_num}: candidate as {'white' if candidate_white else 'black'}, "
                  f"score={score}, running win_rate={win_rate*100:.1f}%")
    finally:
        candidate_predictor.stop()
        best_predictor.stop()

    win_rate = sum(scores) / len(scores)
    return win_rate, win_rate >= REPLACE_RATE


if __name__ == "__main__":
    from chess_zero.agent.load_weights import load_torch_model
    from chess_zero.worker.optimize_torch import load_checkpoint

    device = "cuda" if torch.cuda.is_available() else "cpu"
    best_model = load_torch_model("../data/model/model_best_weight.h5").to(device)

    if len(sys.argv) > 1:
        candidate_model = load_torch_model("../data/model/model_best_weight.h5").to(device)
        load_checkpoint(candidate_model, sys.argv[1])
        print(f"Candidate loaded from {sys.argv[1]}")
    else:
        # No trained checkpoint given: evaluate the base model against
        # itself as a mechanism smoke test (should land near 50%).
        candidate_model = load_torch_model("../data/model/model_best_weight.h5").to(device)
        print("No checkpoint given -- running base-model-vs-itself as a mechanism check.")

    win_rate, promote = evaluate_candidate(candidate_model, best_model, game_num=4, max_halfmoves=20)
    print(f"Final win_rate={win_rate*100:.1f}%, promote={'YES' if promote else 'no'} (threshold {REPLACE_RATE*100:.0f}%)")
