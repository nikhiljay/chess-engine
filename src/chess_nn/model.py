"""NNUE-style evaluation network and its training loss."""

import torch
from torch import nn

from chess_nn.encoding import HALF_SIZE


class EvalNet(nn.Module):
    """Shared feature transformer over both perspective halves → 2H → head → 1.

    `head` lists the hidden layer widths after the accumulator; empty means the
    accumulator feeds the output layer directly. Output is an evaluation in
    pawns from the side to move's perspective.
    """

    def __init__(self, hidden: int = 256, head: tuple[int, ...] = (32, 32)) -> None:
        super().__init__()
        self.hidden = hidden
        self.ft = nn.Linear(HALF_SIZE, hidden)
        sizes = [2 * hidden, *head]
        self.head = nn.ModuleList(nn.Linear(a, b) for a, b in zip(sizes, sizes[1:]))
        self.out = nn.Linear(sizes[-1], 1)

    def forward(self, board: torch.Tensor) -> torch.Tensor:
        stm = torch.clamp(self.ft(board[:, :HALF_SIZE]), 0.0, 1.0)
        nstm = torch.clamp(self.ft(board[:, HALF_SIZE:]), 0.0, 1.0)
        x = torch.cat([stm, nstm], dim=1)
        for layer in self.head:
            x = torch.relu(layer(x))
        return self.out(x)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def wdl_loss(eval_pawns: torch.Tensor, score_cp: torch.Tensor, scale_cp: float = 400.0) -> torch.Tensor:
    """MSE between predicted and target win probabilities.

    A larger scale flattens the sigmoid, keeping gradient between large
    advantages (e.g. +8 vs +25 pawns) that a small scale treats as equal.
    """
    pred = torch.sigmoid(eval_pawns * 100.0 / scale_cp)
    target = torch.sigmoid(score_cp / scale_cp)
    return torch.mean((pred - target) ** 2)
