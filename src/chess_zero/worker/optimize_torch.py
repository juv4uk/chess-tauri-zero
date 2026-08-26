"""PyTorch port of optimize.py's training step, since extended
(docs/pytorch-self-play-loop-plan-uk.md) with a real dataset loader
and multi-epoch training loop that consumes actual self-play data
(worker/self_play_torch.py's saved games), not just the original
synthetic mechanism check. load_dataset/train_epochs are still scoped
to the small-scale runs this machine can actually produce (weak GPU,
4 CPU cores) -- not production-scale AlphaZero training.

Original (optimize.py, read directly): Adam optimizer, losses
['categorical_crossentropy', 'mean_squared_error'], loss_weights
[1.25, 1.0] (policy, value) from configs/mini.py's TrainerConfig --
"prevent value overfit in SL" per that file's own comment. Keras'
categorical_crossentropy accepts a full target DISTRIBUTION (the MCTS
visit-count policy), not just a one-hot class index -- the PyTorch
equivalent for a soft-label target is the standard AlphaZero policy
loss -(target * log(pred)).sum(dim=1).mean(), not
nn.CrossEntropyLoss (which expects integer class indices by default).
"""
import random

import numpy as np
import torch
import torch.nn.functional as F

POLICY_LOSS_WEIGHT = 1.25
VALUE_LOSS_WEIGHT = 1.0
N_LABELS = 1968  # create_uci_labels() -- see player_chess_torch.py


def policy_loss(pred_policy: torch.Tensor, target_policy: torch.Tensor) -> torch.Tensor:
    """AlphaZero policy loss: cross-entropy against a full probability
    distribution (MCTS visit counts, normalized), not a single label --
    the soft-target equivalent of Keras' categorical_crossentropy here.
    pred_policy is already softmax'd (ChessResNet.forward's output),
    so this is -(sum(target * log(pred))), not log_softmax+nll."""
    eps = 1e-8
    return -(target_policy * torch.log(pred_policy + eps)).sum(dim=1).mean()


def value_loss(pred_value: torch.Tensor, target_value: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred_value.squeeze(-1), target_value)


def train_step(model, optimizer, state_batch, policy_batch, value_batch):
    """One optimizer step on one batch. Returns (total_loss, policy_loss, value_loss) as floats."""
    model.train()
    optimizer.zero_grad()

    pred_policy, pred_value = model(state_batch)
    p_loss = policy_loss(pred_policy, policy_batch)
    v_loss = value_loss(pred_value, value_batch)
    total = POLICY_LOSS_WEIGHT * p_loss + VALUE_LOSS_WEIGHT * v_loss

    total.backward()
    optimizer.step()
    model.eval()

    return float(total.item()), float(p_loss.item()), float(v_loss.item())


def make_optimizer(model, lr=1e-3):
    """Adam with PyTorch's defaults -- matches the original's bare
    Adam() (Keras' own Adam defaults: lr=1e-3, beta1=0.9, beta2=0.999),
    not tuned further since this is the mechanism check, not a real
    training run."""
    return torch.optim.Adam(model.parameters(), lr=lr)


def load_dataset(data_dir):
    """Reads every saved self-play game (data_helper_torch's
    [fen, policy, value] triples, see self_play_torch.py) and turns
    them into (state, policy, value) tensors ready for train_step.

    Each record stores the raw board FEN (ChessEnv.observation), not
    planes -- canon_input_planes() does the same side-to-move flip
    used at inference time (expand_and_evaluate), so training sees
    exactly the encoding the model is actually queried with.

    Real P0 bug found by an external audit and independently verified
    against this exact source before fixing (witness reproduced here,
    not just trusted): canon_input_planes() flips the BOARD to white's
    perspective for a black-to-move FEN, but the recorded policy
    vector (self.moves in player_chess_torch.py) is written in RAW
    board orientation -- expand_and_evaluate() already flips the
    network's canonical policy output back to raw orientation via
    flip_policy() before it ever reaches the MCTS tree stats, so
    every self-play (fen, policy) pair is internally consistent in
    RAW orientation. Canonicalizing only the FEN here, without also
    flipping the policy, silently misaligns every black-to-move
    training example: the state tensor is in white's perspective, the
    policy target stays indexed for the actual (black) board. Half of
    all self-play training data was training the policy head against
    the wrong move index. flip_policy() is the exact inverse of the
    inference-time flip and is reused here rather than re-derived."""
    import sys
    sys.path.insert(0, ".")
    from chess_zero.env.chess_env import canon_input_planes, is_black_turn
    from chess_zero.agent.player_chess_torch import flip_policy
    from chess_zero.lib.data_helper_torch import get_game_data_filenames, read_game_data_from_file

    states, policies, values = [], [], []
    for path in get_game_data_filenames(data_dir):
        for fen, policy, value in read_game_data_from_file(path):
            policy = np.asarray(policy, dtype=np.float32)
            if is_black_turn(fen):
                policy = flip_policy(policy)
            states.append(canon_input_planes(fen))
            policies.append(policy)
            values.append(value)

    if not states:
        return None, None, None

    state_t = torch.from_numpy(np.stack(states)).float()
    policy_t = torch.from_numpy(np.stack(policies)).float()
    value_t = torch.tensor(values, dtype=torch.float32)
    return state_t, policy_t, value_t


def train_epochs(model, optimizer, data_dir, epochs=1, batch_size=32):
    """Real training loop over saved self-play data: shuffles indices
    each epoch, runs train_step per batch, prints per-epoch mean loss.
    batch_size default is small (32, vs the original's 384) since
    load_dataset's game counts here are necessarily small-scale (see
    docs/pytorch-self-play-loop-plan-uk.md) -- a large batch would
    just be one under-full batch."""
    state_t, policy_t, value_t = load_dataset(data_dir)
    if state_t is None:
        print(f"No games found in {data_dir} -- nothing to train on.")
        return []

    n = state_t.shape[0]
    device = next(model.parameters()).device
    history = []

    for epoch in range(epochs):
        idx = list(range(n))
        random.shuffle(idx)
        epoch_losses = []
        for start in range(0, n, batch_size):
            batch_idx = idx[start:start + batch_size]
            sb = state_t[batch_idx].to(device)
            pb = policy_t[batch_idx].to(device)
            vb = value_t[batch_idx].to(device)
            total, p, v = train_step(model, optimizer, sb, pb, vb)
            epoch_losses.append(total)
        mean_loss = sum(epoch_losses) / len(epoch_losses)
        history.append(mean_loss)
        print(f"  epoch {epoch+1}/{epochs}: mean loss={mean_loss:.4f} over {len(epoch_losses)} batch(es), {n} positions")

    return history


def save_checkpoint(model, path):
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(model.state_dict(), path)


def load_checkpoint(model, path):
    model.load_state_dict(torch.load(path, map_location=next(model.parameters()).device))
    return model


if __name__ == "__main__":
    # Mechanism smoke test: NOT real training data, NOT self-play --
    # a tiny synthetic batch, just to prove the loss/backward/optimizer
    # step actually runs and produces finite, decreasing loss.
    import sys
    sys.path.insert(0, ".")
    from chess_zero.agent.load_weights import load_torch_model

    torch.manual_seed(0)
    model = load_torch_model("../data/model/model_best_weight.h5")
    optimizer = make_optimizer(model)

    batch_size = 4
    state = torch.rand(batch_size, 18, 8, 8)
    policy_target = torch.rand(batch_size, 1968)
    policy_target = policy_target / policy_target.sum(dim=1, keepdim=True)  # normalize to a real distribution
    value_target = torch.rand(batch_size) * 2 - 1  # in [-1, 1], matching tanh's range

    print("Synthetic mechanism check (not real training data):")
    for step in range(5):
        total, p, v = train_step(model, optimizer, state, policy_target, value_target)
        print(f"  step {step}: total={total:.4f}  policy={p:.4f}  value={v:.4f}")
