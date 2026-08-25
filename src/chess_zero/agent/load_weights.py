"""Loads the real Keras HDF5 weights (data/model/model_best_weight.h5)
into the PyTorch ChessResNet defined in torch_model.py.

Keras/TF stores Conv2D kernels as (kh, kw, in_ch, out_ch); PyTorch
expects (out_ch, in_ch, kh, kw) -- permuted (3,2,0,1) below. Keras
Dense kernels are (in_features, out_features); PyTorch Linear wants
(out_features, in_features) -- transposed below. BatchNorm's
beta/gamma/moving_mean/moving_variance map directly to PyTorch's
bias/weight/running_mean/running_var, no reshaping needed. Verified
against the actual h5py group structure (`f.visititems`), not
assumed from the Keras docs alone.
"""
import h5py
import numpy as np
import torch

from chess_zero.agent.torch_model import ChessResNet


def _conv(f, name):
    w = f[name][name]["kernel:0"][()]
    return torch.from_numpy(np.transpose(w, (3, 2, 0, 1)).copy())


def _bn(f, name):
    g = f[name][name]
    return (
        torch.from_numpy(g["gamma:0"][()].copy()),
        torch.from_numpy(g["beta:0"][()].copy()),
        torch.from_numpy(g["moving_mean:0"][()].copy()),
        torch.from_numpy(g["moving_variance:0"][()].copy()),
    )


def _dense(f, name):
    g = f[name][name]
    w = torch.from_numpy(g["kernel:0"][()].T.copy())
    b = torch.from_numpy(g["bias:0"][()].copy())
    return w, b


def load_torch_model(weight_path: str, filters=256, res_blocks=7, n_labels=1968, value_fc_size=256) -> ChessResNet:
    model = ChessResNet(filters=filters, res_blocks=res_blocks, n_labels=n_labels, value_fc_size=value_fc_size)
    with h5py.File(weight_path, "r") as f:
        model.input_conv.weight.data = _conv(f, "input_conv-3-256")
        g, b, rm, rv = _bn(f, "input_batchnorm")
        model.input_bn.weight.data, model.input_bn.bias.data = g, b
        model.input_bn.running_mean.data, model.input_bn.running_var.data = rm, rv

        for i, block in enumerate(model.res_blocks, start=1):
            block.conv1.weight.data = _conv(f, f"res{i}_conv1-3-256")
            g, b, rm, rv = _bn(f, f"res{i}_batchnorm1")
            block.bn1.weight.data, block.bn1.bias.data = g, b
            block.bn1.running_mean.data, block.bn1.running_var.data = rm, rv

            block.conv2.weight.data = _conv(f, f"res{i}_conv2-3-256")
            g, b, rm, rv = _bn(f, f"res{i}_batchnorm2")
            block.bn2.weight.data, block.bn2.bias.data = g, b
            block.bn2.running_mean.data, block.bn2.running_var.data = rm, rv

        model.policy_conv.weight.data = _conv(f, "policy_conv-1-2")
        g, b, rm, rv = _bn(f, "policy_batchnorm")
        model.policy_bn.weight.data, model.policy_bn.bias.data = g, b
        model.policy_bn.running_mean.data, model.policy_bn.running_var.data = rm, rv
        w, b_ = _dense(f, "policy_out")
        model.policy_out.weight.data, model.policy_out.bias.data = w, b_

        model.value_conv.weight.data = _conv(f, "value_conv-1-1")
        g, b, rm, rv = _bn(f, "value_batchnorm")
        model.value_bn.weight.data, model.value_bn.bias.data = g, b
        model.value_bn.running_mean.data, model.value_bn.running_var.data = rm, rv
        w, b_ = _dense(f, "value_dense")
        model.value_dense.weight.data, model.value_dense.bias.data = w, b_
        w, b_ = _dense(f, "value_out")
        model.value_out.weight.data, model.value_out.bias.data = w, b_

    model.eval()
    return model


if __name__ == "__main__":
    m = load_torch_model("data/model/model_best_weight.h5" if __package__ else "../../data/model/model_best_weight.h5")
    total_params = sum(p.numel() for p in m.parameters())
    print(f"Loaded ChessResNet: {total_params:,} parameters")
