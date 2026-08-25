"""One-turn CLI helper for playing against the PyTorch-ported model
interactively from the shell: apply a human move (if given), have the
model respond, save state to a FEN file between calls.

Usage:
    python3 play_turn.py                  # model plays first move (as white)
    python3 play_turn.py e2e4              # apply human move e2e4, model responds
"""
import sys
import chess

sys.path.insert(0, ".")
from chess_zero.agent.load_weights import load_torch_model
from chess_zero.agent.play_torch import best_move

STATE_FILE = "/tmp/chess_turn_state.fen"
MODEL = None


def get_model():
    global MODEL
    if MODEL is None:
        MODEL = load_torch_model("../data/model/model_best_weight.h5")
    return MODEL


def load_board() -> chess.Board:
    try:
        with open(STATE_FILE) as f:
            fen = f.read().strip()
        return chess.Board(fen)
    except FileNotFoundError:
        return chess.Board()


def save_board(board: chess.Board) -> None:
    with open(STATE_FILE, "w") as f:
        f.write(board.fen())


def render_board(board: chess.Board) -> str:
    """Unicode board with file/rank labels, white at the bottom."""
    lines = [str(board.unicode(borders=False, empty_square="·"))]
    labeled = []
    for i, row in enumerate(lines[0].split("\n")):
        rank = 8 - i
        labeled.append(f"{rank} {row}")
    labeled.append("  a b c d e f g h")
    return "\n".join(labeled)


def main():
    board = load_board()

    if len(sys.argv) > 1:
        human_input = sys.argv[1]
        mv = None
        try:
            candidate = chess.Move.from_uci(human_input)
            if candidate in board.legal_moves:
                mv = candidate
        except ValueError:
            pass
        if mv is None:
            try:
                mv = board.parse_san(human_input)  # short/algebraic notation, e.g. "e6", "Nf3"
            except ValueError:
                print(f"Невірний або нелегальний хід: {human_input}")
                print("Легальні ходи (UCI):", ", ".join(m.uci() for m in board.legal_moves))
                return
        board.push(mv)
        print(f"Твій хід: {human_input} ({mv.uci()})")

    if board.is_game_over():
        print(f"Гра закінчена: {board.result()}")
        print(board)
        save_board(board)
        return

    model = get_model()
    scored, value = best_move(model, board)
    model_move = scored[0][1]
    board.push(model_move)
    print(f"Хід моделі: {model_move.uci()}  (оцінка позиції до ходу: {value:+.3f})")
    print()
    print(render_board(board))
    print()
    print("FEN:", board.fen())

    if board.is_game_over():
        print(f"\nГра закінчена: {board.result()}")

    save_board(board)


if __name__ == "__main__":
    main()
