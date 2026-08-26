"""Real UCI protocol tests for uci_torch.py -- spawn the actual engine
process and talk UCI to it over its real stdin/stdout, the same
surface Tauri's sidecar uses. Deliberately stdlib unittest, not
pytest -- this project keeps its runtime dependency list minimal
(torch/numpy/h5py/chess only), and unittest needs nothing extra.

Every TestCase here is a regression test for a real, empirically
verified bug found and fixed during actual development (crashes,
races, silent data loss) -- not speculative coverage. A one-off manual
verification is real evidence, but it stops being reproducible the
moment nobody reruns it; committing it here turns it into a check
CI reruns on every future change, closing exactly the gap that let
this session's real bugs (a killed process on a bad command, a
selfplay/train weight race, a stale-FEN bug) go unnoticed until they
were hit live.

Usage (from src/, inside the venv):
    python3 -m unittest discover -s tests -p "test_*.py" -v
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import time
import unittest


class UciEngineProcess:
    """Spawns a real uci_torch.py process and gives tests a simple
    send/expect API over its real stdin/stdout."""

    def __init__(self, timeout=90):
        self.timeout = timeout
        self.proc = subprocess.Popen(
            [sys.executable, "chess_zero/play_game/uci_torch.py"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

    def send(self, line):
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def expect(self, predicate, timeout=None):
        deadline = time.time() + (timeout or self.timeout)
        while time.time() < deadline:
            line = self.proc.stdout.readline()
            if not line:
                break
            line = line.rstrip()
            if predicate(line):
                return line
        raise TimeoutError(f"never matched predicate within {timeout or self.timeout}s")

    def close(self):
        try:
            self.send("quit")
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()
        finally:
            self.proc.stdin.close()
            self.proc.stdout.close()


class UciTorchTestCase(unittest.TestCase):
    """Base class: spawns a fresh engine per test and handshakes it,
    tears it down after -- matches the real engine_start/quit
    lifecycle instead of sharing one process across tests, so a bug
    in one test can't leave stray state for the next."""

    def setUp(self):
        self.engine = UciEngineProcess()
        self.engine.send("uci")
        self.engine.expect(lambda l: l == "uciok")

    def tearDown(self):
        self.engine.close()


class TestBasicHandshake(UciTorchTestCase):
    def test_isready_returns_readyok(self):
        self.engine.send("isready")
        self.engine.expect(lambda l: l == "readyok")

    def test_go_returns_a_bestmove(self):
        self.engine.send("position startpos")
        self.engine.send("go")
        line = self.engine.expect(lambda l: l.startswith("bestmove "))
        uci_move = line.split(" ")[1]
        self.assertGreaterEqual(len(uci_move), 4)  # e.g. "e2e4", or 5 chars for a promotion


class TestCrashGuard(UciTorchTestCase):
    """Regression test for a real, live-reported bug: an uncaught
    exception in ANY command used to kill the whole engine process
    silently -- the frontend only saw it as a broken pipe and quietly
    respawned, hiding the real error."""

    def test_malformed_position_does_not_kill_the_engine(self):
        self.engine.send("position")  # missing required argument -- the actual crash trigger found live
        self.engine.expect(lambda l: l.startswith("info error [position]"))
        self.engine.send("isready")
        self.engine.expect(lambda l: l == "readyok")


class ModelStateBackupMixin:
    """Backs up data/model_torch/model_best.pt before a destructive
    test and restores it after -- same discipline used manually
    throughout this session for resetmodel/history tests. A no-op on
    a fresh CI checkout (the file is gitignored, won't exist there);
    meaningful on a dev machine with real accumulated training state."""

    MODEL_PATH = "../data/model_torch/model_best.pt"

    def backup_model(self):
        self._backup_path = None
        if os.path.exists(self.MODEL_PATH):
            self._backup_path = self.MODEL_PATH + ".test-backup"
            shutil.copy(self.MODEL_PATH, self._backup_path)

    def restore_model(self):
        if self._backup_path:
            shutil.copy(self._backup_path, self.MODEL_PATH)
            os.remove(self._backup_path)


class PlayDataBackupMixin:
    """Moves data/play_data_torch/ aside before a destructive test and
    restores it after -- independent of whatever archiving resetmodel
    itself does internally (a play_data_torch.archived-* sibling), so
    multiple resetmodel-calling tests in the same class can't leave
    each other's cleanup half-done. A no-op backup (None) on a fresh
    CI checkout where the directory doesn't exist yet."""

    PLAY_DATA_DIR = "../data/play_data_torch"

    def backup_play_data(self):
        self._play_data_backup = None
        if os.path.isdir(self.PLAY_DATA_DIR):
            self._play_data_backup = self.PLAY_DATA_DIR + ".test-backup"
            shutil.move(self.PLAY_DATA_DIR, self._play_data_backup)

    def restore_play_data(self):
        # Whatever the test run left behind (a fresh play_data_torch/
        # a test created, or a play_data_torch.archived-* resetmodel
        # produced) is a test artifact, not real data -- discard it
        # before restoring the real backup, if there was one.
        for stray in glob.glob(self.PLAY_DATA_DIR + "*"):
            if stray != self._play_data_backup and os.path.isdir(stray):
                shutil.rmtree(stray)
        if self._play_data_backup:
            shutil.move(self._play_data_backup, self.PLAY_DATA_DIR)


class TestHistoryAndReset(UciTorchTestCase, ModelStateBackupMixin, PlayDataBackupMixin):
    def setUp(self):
        super().setUp()
        self.backup_model()
        self.backup_play_data()

    def tearDown(self):
        self.restore_play_data()
        self.restore_model()
        super().tearDown()

    def test_history_completes_cleanly(self):
        self.engine.send("history")
        self.engine.expect(lambda l: l == "historyresult ok")

    def test_resetmodel_ok_and_engine_stays_usable(self):
        self.engine.send("resetmodel")
        result = self.engine.expect(lambda l: l.startswith("resetmodelresult"))
        self.assertEqual(result, "resetmodelresult ok")

        self.engine.send("position startpos")
        self.engine.send("go")
        self.engine.expect(lambda l: l.startswith("bestmove "))

    def test_resetmodel_archives_old_selfplay_corpus_instead_of_leaving_it_active(self):
        # Regression test for BH-02, a real P0 bug found by an
        # external audit (2026-08-26) and independently verified: this
        # used to clear the generation journal but leave
        # data/play_data_torch/ completely untouched -- the NEXT
        # `train start` would silently train a brand-new "справжній
        # нуль" model on self-play games the OLD, discarded baseline
        # generated, contradicting the whole point of the reset.
        os.makedirs(self.PLAY_DATA_DIR, exist_ok=True)
        marker_path = os.path.join(self.PLAY_DATA_DIR, "play_test-marker.json")
        with open(marker_path, "w") as f:
            json.dump([["fen", [0.0], 0.0]], f)

        self.engine.send("resetmodel")
        result = self.engine.expect(lambda l: l.startswith("resetmodelresult"))
        self.assertEqual(result, "resetmodelresult ok")

        self.assertFalse(
            os.path.isdir(self.PLAY_DATA_DIR),
            "old self-play corpus must not remain active after resetmodel",
        )
        archives = glob.glob(self.PLAY_DATA_DIR + ".archived-*")
        self.assertEqual(len(archives), 1, "expected exactly one archive directory")
        archived_marker = os.path.join(archives[0], "play_test-marker.json")
        self.assertTrue(os.path.exists(archived_marker), "marker file must survive inside the archive")


class TestSelfplayTrainMutualExclusion(UciTorchTestCase):
    """Regression test for a real, reproduced race: selfplay and
    train used to have no guard against each other, both touching the
    same shared model object -- a promotion's hot-reload could mutate
    live weights while selfplay was still reading them."""

    def test_train_start_refused_while_selfplay_running(self):
        self.engine.send("selfplay start")
        self.engine.expect(lambda l: l.startswith("selfplaymove ") or l.startswith("info mctsroot"))
        self.engine.send("train start")
        self.engine.expect(lambda l: l == "trainerror selfplay running")
        self.engine.send("selfplay stop")
        self.engine.expect(lambda l: l.startswith("selfplayresult "), timeout=60)


class TestHumanGameRecording(UciTorchTestCase):
    """Regression test for a real bug caught while building this
    feature: the FEN used to come from this process's own (stale-by-
    one-ply) `env`, not the frontend's authoritative chess.js state."""

    HUMAN_DATA_DIR = "../data/play_data_human"

    def setUp(self):
        super().setUp()
        self._existing = set(os.listdir(self.HUMAN_DATA_DIR)) if os.path.isdir(self.HUMAN_DATA_DIR) else set()

    def test_humanmove_and_gameover_writes_correct_data(self):
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        self.engine.send(f"humanmove {fen} e2e4")
        self.engine.send("humangameover human-win")
        self.engine.expect(lambda l: l.startswith("humangameoverresult"))

        after = set(os.listdir(self.HUMAN_DATA_DIR)) if os.path.isdir(self.HUMAN_DATA_DIR) else set()
        new_files = after - self._existing
        self.assertEqual(len(new_files), 1, "expected exactly one new game file")
        path = os.path.join(self.HUMAN_DATA_DIR, new_files.pop())
        try:
            with open(path) as f:
                data = json.load(f)
            self.assertEqual(len(data), 1)
            recorded_fen, policy, z = data[0]
            self.assertEqual(recorded_fen, fen)
            self.assertAlmostEqual(sum(policy), 1.0)  # one-hot
            self.assertEqual(z, 1.0)  # human-win -> +1.0
        finally:
            os.remove(path)  # don't leave test data in the real directory


if __name__ == "__main__":
    unittest.main()
