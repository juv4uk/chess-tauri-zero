"""Regression tests for the generation journal's crash-safety
(pipeline_torch.py's record_cycle_result/model_history).

BH-04, a real bug found by an external audit (2026-08-26) and
independently re-verified before fixing: record_cycle_result() wrote
model_cycle{N}.json directly with open(..., "w"), leaving a window
where a crash or power loss mid-write produces a truncated,
unparseable file -- and model_history() read every matching file
directly, so one malformed record could crash the whole history read
(or, before this fix, would have crashed json.load() with no recovery
path at all).

Usage (from src/, inside the venv):
    python3 -m unittest discover -s tests -p "test_*.py" -v
"""
import os
import shutil
import tempfile
import unittest

import chess_zero.worker.pipeline_torch as pipeline_torch


class TestJournalAtomicity(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self._orig_next_gen_dir = pipeline_torch.NEXT_GEN_DIR
        pipeline_torch.NEXT_GEN_DIR = self.tmpdir

    def tearDown(self):
        pipeline_torch.NEXT_GEN_DIR = self._orig_next_gen_dir
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_record_cycle_result_leaves_no_leftover_tmp_file(self):
        pipeline_torch.record_cycle_result(0, 0.6, True)
        files = os.listdir(self.tmpdir)
        self.assertIn("model_cycle0.json", files)
        self.assertFalse(any(f.endswith(".tmp") for f in files))

    def test_model_history_quarantines_a_malformed_record_instead_of_crashing(self):
        pipeline_torch.record_cycle_result(0, 0.6, True)
        # Simulate a crash mid-write: a real record_cycle_result() call
        # would never leave this on disk (write-to-temp + os.replace),
        # but a pre-existing file from before that fix, or damage from
        # something else entirely, must not crash the whole read.
        corrupt_path = os.path.join(self.tmpdir, "model_cycle1.json")
        with open(corrupt_path, "w") as f:
            f.write('{"cycle": 1, "win_rate":')  # truncated JSON

        history = pipeline_torch.model_history()

        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["cycle"], 0)
        self.assertTrue(os.path.exists(corrupt_path + ".corrupt"), "corrupt record must be quarantined")
        self.assertFalse(os.path.exists(corrupt_path), "corrupt record must be renamed aside, not left in place")

    def test_model_history_survives_a_second_call_after_quarantine(self):
        # A quarantined file must not be picked up (and fail) again on
        # a later call -- it no longer matches the model_cycle*.json glob.
        corrupt_path = os.path.join(self.tmpdir, "model_cycle0.json")
        with open(corrupt_path, "w") as f:
            f.write("not json at all")

        first = pipeline_torch.model_history()
        second = pipeline_torch.model_history()
        self.assertEqual(first, [])
        self.assertEqual(second, [])


if __name__ == "__main__":
    unittest.main()
