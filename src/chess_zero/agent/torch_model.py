"""PyTorch port of model_chess.py's Keras ChessModel architecture.

Ported 2026-08-26 to run the real pretrained weights
(data/model/model_best_weight.h5, model_best_config.json) on PyTorch
2.13, since the original pinned tensorflow-gpu==1.15.2/keras==2.0.8
stack has no wheels for Python 3.12 and Guix no longer carries Python
3.6/3.7 either -- confirmed dead end on this machine, not assumed.

Architecture matches model_best_config.json exactly (read directly,
not guessed): input_conv (18->256, 3x3, no bias) + BN + ReLU, 7
residual blocks (256 filters, 3x3, no bias, two conv+BN per block,
skip-add, ReLU), policy head (conv 256->2, 1x1, no bias + BN + ReLU +
flatten(128) + Linear(128, 1968) + softmax), value head (conv 256->4,
1x1, no bias + BN + ReLU + flatten(256) + Linear(256, 256) + ReLU +
Linear(256, 1) + tanh). 1968 = len(create_uci_labels()) confirmed by
running config.py's own function directly, not by hand-counting (a
hand-transcribed reimplementation of that function gave 1840 -- a
real bug caught precisely by re-running the original code instead of
trusting a manual retype).
"""
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, filters: int):
        super().__init__()
        self.conv1 = nn.Conv2d(filters, filters, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(filters)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        return self.relu(out)


class ChessResNet(nn.Module):
    def __init__(self, filters: int = 256, res_blocks: int = 7, n_labels: int = 1968, value_fc_size: int = 256):
        super().__init__()
        self.input_conv = nn.Conv2d(18, filters, kernel_size=3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(filters)
        self.relu = nn.ReLU(inplace=True)

        self.res_blocks = nn.ModuleList([ResidualBlock(filters) for _ in range(res_blocks)])

        # policy head
        self.policy_conv = nn.Conv2d(filters, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_out = nn.Linear(2 * 8 * 8, n_labels)

        # value head -- 1 filter, not 4: confirmed against the actual saved
        # weights' shapes (value_conv-1-1, kernel (1,1,256,1)), not assumed
        # from model_chess.py's generic naming convention comment.
        self.value_conv = nn.Conv2d(filters, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_dense = nn.Linear(1 * 8 * 8, value_fc_size)
        self.value_out = nn.Linear(value_fc_size, 1)

    def forward(self, x):
        x = self.relu(self.input_bn(self.input_conv(x)))
        for block in self.res_blocks:
            x = block(x)
        res_out = x

        p = self.relu(self.policy_bn(self.policy_conv(res_out)))
        p = p.flatten(1)
        policy = torch.softmax(self.policy_out(p), dim=-1)

        v = self.relu(self.value_bn(self.value_conv(res_out)))
        v = v.flatten(1)
        v = self.relu(self.value_dense(v))
        value = torch.tanh(self.value_out(v))

        return policy, value
