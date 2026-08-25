"""PyTorch port of optimize.py's training step -- roadmap step 3,
scoped deliberately: this ports the TRAINING MECHANISM (loss functions,
optimizer, one train_step) and verifies it actually runs a real
backward pass + parameter update. It does NOT run self-play data
generation (worker/self_play.py) -- that's a genuinely long-running,
CPU-heavy process (many full MCTS games) and deliberately not started
here, per the same resource-pacing discipline used for GUIX-WITNESS-01
earlier. Running real self-play + multi-epoch training is a separate,
much bigger task than porting the step itself.

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
import torch
import torch.nn.functional as F

POLICY_LOSS_WEIGHT = 1.25
VALUE_LOSS_WEIGHT = 1.0


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
