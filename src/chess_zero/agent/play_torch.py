"""Minimal play script for the PyTorch-ported model -- no MCTS, just
greedy policy-head move selection, to verify the port actually
produces sane chess (not to reach the original project's playing
strength, which relied on 1200 MCTS simulations per move).

Reuses chess_env.py's canon_input_planes / maybe_flip_fen (pure
numpy + python-chess, no Keras/TF dependency -- ported nothing there,
it already worked as-is) and config.py's create_uci_labels (pure
Python, imported directly rather than re-transcribed after the
earlier 1968-vs-1840 transcription bug).
"""
import sys
import importlib.util

import chess
import torch

sys.path.insert(0, ".")
from chess_zero.agent.load_weights import load_torch_model
from chess_zero.env.chess_env import canon_input_planes, maybe_flip_fen, is_black_turn

_spec = importlib.util.spec_from_file_location("config", "chess_zero/config.py")
_config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_config)
LABELS = _config.create_uci_labels()
LABEL_INDEX = {l: i for i, l in enumerate(LABELS)}


def flip_uci(move: str) -> str:
    """Mirrors a uci move vertically (rank r -> 9-r), matching how
    canon_input_planes always presents the side-to-move as if it were
    white moving up the board."""
    def flip_sq(sq):
        return sq[0] + str(9 - int(sq[1]))
    src, dst = move[:2], move[2:4]
    promo = move[4:]
    return flip_sq(src) + flip_sq(dst) + promo


def best_move(model, board: chess.Board) -> chess.Move:
    fen = board.fen()
    flip = is_black_turn(fen)
    planes = canon_input_planes(fen)
    x = torch.from_numpy(planes).unsqueeze(0).float()
    with torch.no_grad():
        policy, value = model(x)
    policy = policy.squeeze(0).numpy()

    legal = list(board.legal_moves)
    scored = []
    for mv in legal:
        uci = mv.uci()
        label = flip_uci(uci) if flip else uci
        # underpromotions/queen-promotions not always literally in LABELS
        # (e.g. a plain queen promotion move from python-chess is "a7a8q",
        # which IS in LABELS; non-promotion moves need no suffix trimming)
        idx = LABEL_INDEX.get(label)
        if idx is None:
            idx = LABEL_INDEX.get(label[:4])
        p = policy[idx] if idx is not None else 0.0
        scored.append((p, mv))
    scored.sort(key=lambda t: -t[0])
    return scored, float(value.item())


if __name__ == "__main__":
    model = load_torch_model("../data/model/model_best_weight.h5")
    board = chess.Board()
    scored, value = best_move(model, board)
    print(f"Position: startpos, value estimate (side to move's perspective): {value:+.3f}")
    print("Top 5 moves by policy probability:")
    for p, mv in scored[:5]:
        print(f"  {mv.uci():6s}  p={p:.4f}")
