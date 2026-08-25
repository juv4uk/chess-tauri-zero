"""Batches concurrent MCTS leaf evaluations into single GPU forward
passes. Pure efficiency fix, zero change to the self-play/MCTS
algorithm or its zero-human-knowledge premise: TorchChessPlayer's
search already runs simulation_num_per_move leaf expansions across
search_threads=16 threads (virtual loss lets them overlap), but until
now each thread called model(x) with batch size 1 -- 16 separate GPU
round-trips instead of one. This collects whatever's pending across a
short window into one batch, exactly the technique the original
project used multiprocess pipes for (see model_chess.py's pipe-based
serving loop, read directly) -- here done in-process with a queue,
since PyTorch inference from multiple threads is safe without that.
"""
import queue
import threading
import time
from concurrent.futures import Future

import numpy as np
import torch


class BatchedPredictor:
    def __init__(self, model, max_batch_size=16, timeout=0.02):
        self.model = model
        self.max_batch_size = max_batch_size
        self.timeout = timeout
        self.device = next(model.parameters()).device
        self._queue = queue.Queue()
        self._stop = threading.Event()
        self._batches_run = 0
        self._items_run = 0
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while not self._stop.is_set():
            try:
                item = self._queue.get(timeout=0.05)
            except queue.Empty:
                continue
            batch = [item]
            deadline = time.time() + self.timeout
            while len(batch) < self.max_batch_size:
                remaining = deadline - time.time()
                if remaining <= 0:
                    break
                try:
                    batch.append(self._queue.get(timeout=remaining))
                except queue.Empty:
                    break
            self._run_batch(batch)

    def _run_batch(self, batch):
        state_planes_list, futures = zip(*batch)
        x = torch.from_numpy(np.stack(state_planes_list)).float().to(self.device)
        with torch.no_grad():
            policy, value = self.model(x)
        policy = policy.cpu().numpy()
        value = value.squeeze(-1).cpu().numpy()  # (batch, 1) -> (batch,)
        for i, fut in enumerate(futures):
            fut.set_result((policy[i], float(value[i])))
        self._batches_run += 1
        self._items_run += len(batch)

    def predict(self, state_planes):
        fut = Future()
        self._queue.put((state_planes, fut))
        return fut.result()

    def stats(self):
        """Mean batch size actually achieved -- 1.0 means batching
        bought nothing (every leaf arrived alone); the closer to
        max_batch_size, the more the GPU round-trips were amortized."""
        if self._batches_run == 0:
            return {"batches": 0, "items": 0, "mean_batch_size": 0.0}
        return {
            "batches": self._batches_run,
            "items": self._items_run,
            "mean_batch_size": self._items_run / self._batches_run,
        }

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=1)
