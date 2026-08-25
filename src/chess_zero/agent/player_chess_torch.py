"""PyTorch-backed port of player_chess.py's AGZ MCTS.

The tree/PUCT logic below is a direct, unmodified port of the
original ChessPlayer (VisitStats/ActionStats, select_action_q_and_u,
apply_temperature, calc_policy) -- none of that touches Keras/TF, as
confirmed by reading the original file directly. The ONLY change is
predict(): the original sent state planes through a multiprocessing
pipe to a separate Keras-serving process (needed for TF1's session
threading model); here it's a direct call into the PyTorch model
under torch.no_grad(), since PyTorch inference is thread-safe for a
read-only forward pass.

PlayConfig values below match configs/mini.py (the config family
whose ModelConfig -- 256 filters, 7 residual blocks -- matches this
checkpoint's actual architecture), except cnn_first_filter_size,
which mini.py sets to 5 but the real saved weights use 3 (confirmed
against model_best_config.json directly, same finding already
recorded in torch_model.py/pytorch-port-uk.md) -- search
hyperparameters (c_puct, noise_eps, ...) are unaffected by that and
are copied as-is from mini.py's PlayConfig.
"""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import importlib.util

import chess
import numpy as np
import torch

from chess_zero.env.chess_env import canon_input_planes, maybe_flip_fen, is_black_turn, ChessEnv, Winner

_spec = importlib.util.spec_from_file_location("config", "chess_zero/config.py")
_config_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_config_mod)
LABELS = _config_mod.create_uci_labels()
N_LABELS = len(LABELS)
FLIPPED_LABELS = _config_mod.flipped_uci_labels()
UNFLIPPED_INDEX = [LABELS.index(x) for x in FLIPPED_LABELS]


def flip_policy(pol):
    return np.asarray([pol[ind] for ind in UNFLIPPED_INDEX])


class PlayConfig:
    """Matches configs/mini.py's PlayConfig -- see module docstring."""
    def __init__(self):
        self.search_threads = 16
        self.simulation_num_per_move = 100
        self.c_puct = 1.5
        self.noise_eps = 0.25
        self.dirichlet_alpha = 0.3
        self.tau_decay_rate = 0.99
        self.virtual_loss = 3
        self.resign_threshold = None  # disabled by default for interactive play
        self.min_resign_turn = 5


class VisitStats:
    def __init__(self):
        self.a = defaultdict(ActionStats)
        self.sum_n = 0


class ActionStats:
    def __init__(self):
        self.n = 0
        self.w = 0
        self.q = 0
        self.p = 0


def state_key(env: ChessEnv) -> str:
    fen = env.board.fen().rsplit(' ', 1)
    return fen[0]


class TorchChessPlayer:
    def __init__(self, model, play_config: PlayConfig = None):
        self.model = model
        self.play_config = play_config or PlayConfig()
        self.labels_n = N_LABELS
        self.labels = LABELS
        self.move_lookup = {chess.Move.from_uci(move): i for move, i in zip(self.labels, range(self.labels_n))}
        self.tree = defaultdict(VisitStats)
        self.node_lock = defaultdict(Lock)
        self.moves = []

    def reset(self):
        self.tree = defaultdict(VisitStats)

    def finish_game(self, z):
        """z: win=1, lose=-1, draw=0 -- appended to every recorded move
        so optimize_torch.py's data pipeline can build (state, policy,
        value) training triples. Direct port of the original
        ChessPlayer.finish_game, which this port had not carried over
        yet (self_play_torch.py is what actually needs it)."""
        for move in self.moves:
            move += [z]

    def action(self, env: ChessEnv, can_stop=True):
        self.reset()
        root_value, _ = self.search_moves(env)
        policy = self.calc_policy(env)
        my_action = int(np.random.choice(range(self.labels_n), p=self.apply_temperature(policy, env.num_halfmoves)))

        if can_stop and self.play_config.resign_threshold is not None and \
                root_value <= self.play_config.resign_threshold and \
                env.num_halfmoves > self.play_config.min_resign_turn:
            return None
        self.moves.append([env.observation, list(policy)])
        return self.labels[my_action]

    def search_moves(self, env: ChessEnv):
        futures = []
        with ThreadPoolExecutor(max_workers=self.play_config.search_threads) as executor:
            for _ in range(self.play_config.simulation_num_per_move):
                futures.append(executor.submit(self.search_my_move, env=env.copy(), is_root_node=True))
        vals = [f.result() for f in futures]
        return np.max(vals), vals[0]

    def search_my_move(self, env: ChessEnv, is_root_node=False) -> float:
        if env.done:
            if env.winner == Winner.draw:
                return 0
            return -1

        state = state_key(env)

        with self.node_lock[state]:
            if state not in self.tree:
                leaf_p, leaf_v = self.expand_and_evaluate(env)
                self.tree[state].p = leaf_p
                return leaf_v

            action_t = self.select_action_q_and_u(env, is_root_node)
            virtual_loss = self.play_config.virtual_loss

            my_visit_stats = self.tree[state]
            my_stats = my_visit_stats.a[action_t]

            my_visit_stats.sum_n += virtual_loss
            my_stats.n += virtual_loss
            my_stats.w += -virtual_loss
            my_stats.q = my_stats.w / my_stats.n

        env.step(action_t.uci())
        leaf_v = self.search_my_move(env)
        leaf_v = -leaf_v

        with self.node_lock[state]:
            my_visit_stats.sum_n += -virtual_loss + 1
            my_stats.n += -virtual_loss + 1
            my_stats.w += virtual_loss + leaf_v
            my_stats.q = my_stats.w / my_stats.n

        return leaf_v

    def expand_and_evaluate(self, env: ChessEnv):
        state_planes = env.canonical_input_planes()
        leaf_p, leaf_v = self.predict(state_planes)
        if not env.white_to_move:
            leaf_p = flip_policy(leaf_p)
        return leaf_p, leaf_v

    def predict(self, state_planes):
        device = next(self.model.parameters()).device
        x = torch.from_numpy(state_planes).unsqueeze(0).float().to(device)
        with torch.no_grad():
            policy, value = self.model(x)
        return policy.squeeze(0).numpy(), float(value.item())

    def select_action_q_and_u(self, env: ChessEnv, is_root_node):
        state = state_key(env)
        my_visitstats = self.tree[state]

        if my_visitstats.p is not None:
            tot_p = 1e-8
            for mov in env.board.legal_moves:
                mov_p = my_visitstats.p[self.move_lookup[mov]]
                my_visitstats.a[mov].p = mov_p
                tot_p += mov_p
            for a_s in my_visitstats.a.values():
                a_s.p /= tot_p
            my_visitstats.p = None

        xx_ = np.sqrt(my_visitstats.sum_n + 1)
        e = self.play_config.noise_eps
        c_puct = self.play_config.c_puct
        dir_alpha = self.play_config.dirichlet_alpha

        best_s = -999
        best_a = None
        if is_root_node:
            noise = np.random.dirichlet([dir_alpha] * max(len(my_visitstats.a), 1))

        i = 0
        for action, a_s in my_visitstats.a.items():
            p_ = a_s.p
            if is_root_node:
                p_ = (1 - e) * p_ + e * noise[i]
                i += 1
            b = a_s.q + c_puct * p_ * xx_ / (1 + a_s.n)
            if b > best_s:
                best_s = b
                best_a = action
        return best_a

    def apply_temperature(self, policy, turn):
        tau = np.power(self.play_config.tau_decay_rate, turn + 1)
        if tau < 0.1:
            tau = 0
        if tau == 0:
            action = np.argmax(policy)
            ret = np.zeros(self.labels_n)
            ret[action] = 1.0
            return ret
        ret = np.power(policy, 1 / tau)
        ret /= np.sum(ret)
        return ret

    def calc_policy(self, env: ChessEnv):
        state = state_key(env)
        my_visitstats = self.tree[state]
        policy = np.zeros(self.labels_n)
        for action, a_s in my_visitstats.a.items():
            policy[self.move_lookup[action]] = a_s.n
        policy /= np.sum(policy)
        return policy
