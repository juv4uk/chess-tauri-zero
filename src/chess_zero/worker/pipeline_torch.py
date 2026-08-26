"""Thin driver linking self-play -> training -> evaluation into one
real, closed cycle: self_play_torch.self_play_loop generates games,
optimize_torch.train_epochs trains a candidate on them, evaluate_torch
decides whether the candidate replaces the current best.

Runs ONE cycle by default (num_cycles=1), not the original project's
endless per-worker loops (self_play.py/optimize.py/evaluate.py each
ran forever as separate long-lived processes) -- that's real
production infrastructure and a separate, much larger task from
proving the cycle itself is correct end-to-end. See
docs/pytorch-self-play-loop-plan-uk.md.
"""
import copy
import datetime
import glob
import json
import os
import sys
import time

import torch

sys.path.insert(0, ".")
from chess_zero.agent.load_weights import load_torch_model
from chess_zero.agent.player_chess_torch import PlayConfig
from chess_zero.worker.self_play_torch import self_play_loop
from chess_zero.worker.optimize_torch import make_optimizer, train_epochs, save_checkpoint
from chess_zero.worker.evaluate_torch import evaluate_candidate

BEST_MODEL_PATH = "../data/model_torch/model_best.pt"
NEXT_GEN_DIR = "../data/model_torch/next_generation"
PLAY_DATA_DIR = "../data/play_data_torch"


def run_cycle(best_model, device, games_per_cycle=2, sims_per_move=30, max_halfmoves=16,
              train_epochs_n=1, eval_games=2, on_progress=None):
    """on_progress: optional callable(stage: str) -- called at the start
    of each phase ("selfplay"/"train"/"evaluate") so a caller (e.g. the
    UCI sidecar's `train start` command) can stream progress over a
    line-based protocol instead of only seeing this function's own
    print() output.

    cycle_idx is no longer a caller-supplied parameter -- every real
    caller (the UCI `train start`/background-training path, and this
    module's own __main__) either passed a constant 0 or restarted
    from 0 on every fresh process, which silently overwrote the SAME
    model_cycle0.pt/.json on every run (a real bug: only the very last
    cycle's record ever survived, promoted or not). _next_attempt_number()
    now derives a real monotonic id from what's already on disk instead."""
    cycle_idx = _next_attempt_number()
    if on_progress:
        on_progress("selfplay")
    print(f"\n=== Cycle {cycle_idx}: self-play ({games_per_cycle} games) ===")
    pc = PlayConfig()
    pc.simulation_num_per_move = sims_per_move
    pc.resign_threshold = None
    t0 = time.time()
    self_play_loop(best_model, pc, num_games=games_per_cycle, max_halfmoves=max_halfmoves, data_dir=PLAY_DATA_DIR)
    print(f"self-play done in {time.time()-t0:.1f}s")

    if on_progress:
        on_progress("train")
    print(f"=== Cycle {cycle_idx}: train candidate ({train_epochs_n} epoch(s)) ===")
    candidate_model = copy.deepcopy(best_model)
    optimizer = make_optimizer(candidate_model)
    t0 = time.time()
    history = train_epochs(candidate_model, optimizer, PLAY_DATA_DIR, epochs=train_epochs_n, batch_size=16)
    print(f"training done in {time.time()-t0:.1f}s, loss history={history}")

    if on_progress:
        on_progress("evaluate")
    print(f"=== Cycle {cycle_idx}: evaluate candidate vs best ({eval_games} games) ===")
    t0 = time.time()
    win_rate, promote = evaluate_candidate(candidate_model, best_model, game_num=eval_games, max_halfmoves=max_halfmoves)
    print(f"evaluation done in {time.time()-t0:.1f}s: win_rate={win_rate*100:.1f}%, promote={promote}")

    record_cycle_result(cycle_idx, win_rate, promote)
    if promote:
        gen_path = os.path.join(NEXT_GEN_DIR, f"model_cycle{cycle_idx}.pt")
        save_checkpoint(candidate_model, gen_path)
        save_checkpoint(candidate_model, BEST_MODEL_PATH)
        print(f"PROMOTED: candidate saved to {gen_path} and {BEST_MODEL_PATH}")
        return candidate_model
    else:
        print("Candidate not promoted, keeping current best.")
        return best_model


def _next_attempt_number():
    """One higher than the highest cycle id already recorded in
    NEXT_GEN_DIR (0 if the directory doesn't exist yet, or has no
    records) -- gives every run_cycle() call, across process restarts,
    a unique id instead of every caller re-using the same constant."""
    if not os.path.isdir(NEXT_GEN_DIR):
        return 0
    existing = []
    for path in glob.glob(os.path.join(NEXT_GEN_DIR, "model_cycle*.json")):
        name = os.path.basename(path)[len("model_cycle"):-len(".json")]
        if name.isdigit():
            existing.append(int(name))
    return (max(existing) + 1) if existing else 0


def record_cycle_result(cycle_idx, win_rate, promoted):
    """Writes model_cycle{N}.json for EVERY cycle's evaluation result,
    promoted or not -- P0 "generation journal + quality curve" from
    docs/development-plan-uk.md originally only recorded promotions,
    which silently discarded every rejected candidate's result (real
    gap: no way to tell "never tried" from "tried and rejected", and
    no data for a future lineage view). Rejected candidates' WEIGHTS
    still aren't kept (that candidate_model is discarded when run_cycle
    returns best_model unchanged) -- only the small win_rate/promoted
    record, which is what "history" actually needs to show."""
    os.makedirs(NEXT_GEN_DIR, exist_ok=True)
    record = {
        "cycle": cycle_idx,
        "win_rate": win_rate,
        "promoted": promoted,
        "evaluated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    record_path = os.path.join(NEXT_GEN_DIR, f"model_cycle{cycle_idx}.json")
    with open(record_path, "w") as f:
        json.dump(record, f)


def model_history():
    """Every cycle's evaluation record found in NEXT_GEN_DIR (promoted
    and rejected alike), sorted by cycle number. NEXT_GEN_DIR itself
    may not exist at all yet -- os.makedirs is lazy, it's only created
    on the first recorded cycle -- so a missing directory is treated
    as "no history yet", not an error."""
    if not os.path.isdir(NEXT_GEN_DIR):
        return []
    records = []
    for path in glob.glob(os.path.join(NEXT_GEN_DIR, "model_cycle*.json")):
        with open(path) as f:
            records.append(json.load(f))
    records.sort(key=lambda r: r["cycle"])
    return records


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if os.path.exists(BEST_MODEL_PATH):
        best_model = load_torch_model("../data/model/model_best_weight.h5").to(device)
        best_model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
        print(f"Loaded existing best from {BEST_MODEL_PATH}")
    else:
        best_model = load_torch_model("../data/model/model_best_weight.h5").to(device)
        os.makedirs(os.path.dirname(BEST_MODEL_PATH), exist_ok=True)
        torch.save(best_model.state_dict(), BEST_MODEL_PATH)
        print(f"Initialized {BEST_MODEL_PATH} from the pretrained h5 weights")

    num_cycles = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    for c in range(num_cycles):
        best_model = run_cycle(best_model, device)

    print("\nPipeline run complete.")
