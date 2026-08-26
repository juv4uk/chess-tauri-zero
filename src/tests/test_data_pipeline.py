"""Regression tests for the training data pipeline (optimize_torch.py).

BH-01, a real P0 bug found by an external audit (2026-08-26) and
independently re-verified against this exact source before being
fixed: load_dataset() canonicalized the board FEN (white-to-move
perspective, via canon_input_planes) for every black-to-move self-play
position, but the recorded policy target stayed in raw board
orientation -- silently misaligning the policy head's training target
for every black ply. flip_policy() (already used at inference time to
convert the network's canonical output back to raw orientation) is the
exact inverse transform needed here too.

Usage (from src/, inside the venv):
    python3 -m unittest discover -s tests -p "test_*.py" -v
"""
import json
import os
import shutil
import tempfile
import unittest

import numpy as np

from chess_zero.agent.player_chess_torch import LABELS
from chess_zero.worker.optimize_torch import load_dataset


class TestLoadDatasetPolicyOrientation(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _write_game(self, filename, records):
        path = os.path.join(self.tmpdir, filename)
        with open(path, "w") as f:
            json.dump(records, f)

    def _one_hot(self, move):
        policy = [0.0] * len(LABELS)
        policy[LABELS.index(move)] = 1.0
        return policy

    def test_white_move_target_is_unchanged(self):
        # White-to-move: canon_input_planes leaves the board as-is, so
        # the policy target must also stay untouched -- e2e4 in, e2e4 out.
        white_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        self._write_game("play_white.json", [[white_fen, self._one_hot("e2e4"), 1.0]])

        _, policies, _ = load_dataset(self.tmpdir)
        target_label = LABELS[int(np.argmax(policies[0].numpy()))]
        self.assertEqual(target_label, "e2e4")

    def test_black_move_target_is_canonicalized_to_match_the_flipped_board(self):
        # Real witness from the BH-01 audit, independently reproduced:
        # raw black e7e5 is label index 1097; its canonical equivalent
        # (matching the white-perspective board canon_input_planes
        # produces) is e2e4, label index 930. Before the fix, this test
        # would see the RAW e7e5 index here -- misaligned with the
        # canonicalized board the state tensor actually contains.
        black_fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
        self._write_game("play_black.json", [[black_fen, self._one_hot("e7e5"), -1.0]])

        _, policies, _ = load_dataset(self.tmpdir)
        target_label = LABELS[int(np.argmax(policies[0].numpy()))]
        self.assertEqual(
            target_label,
            "e2e4",
            "black policy target must be canonicalized to match the flipped board state",
        )

    def test_policy_distribution_shape_and_mass_preserved_by_the_flip(self):
        # flip_policy is a pure permutation (see player_chess_torch.py) --
        # confirms the fix doesn't drop or duplicate probability mass,
        # not just that the argmax happens to land right.
        black_fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
        policy = [0.0] * len(LABELS)
        policy[LABELS.index("e7e5")] = 0.7
        policy[LABELS.index("g8f6")] = 0.3
        self._write_game("play_mixed.json", [[black_fen, policy, 0.0]])

        _, policies, _ = load_dataset(self.tmpdir)
        result = policies[0].numpy()
        self.assertAlmostEqual(float(result.sum()), 1.0, places=5)
        self.assertEqual((result > 0).sum(), 2)


if __name__ == "__main__":
    unittest.main()
