"""NNUE-style evaluation network and its training loss."""

import torch
from torch import nn

from chess_nn.encoding import HALF_SIZE

# Centipawn scale of the win-probability sigmoid used by the loss.
WDL_SCALE_CP = 400.0


class EvalNet(nn.Module):
    """Shared feature transformer over both perspective halves → 2H → 32 → 32 → 1.

    Output is an evaluation in pawns from the side to move's perspective.
    """

    def __init__(self, hidden: int = 256) -> None:
        super().__init__()
        self.hidden = hidden
        self.ft = nn.Linear(HALF_SIZE, hidden)
        self.l1 = nn.Linear(2 * hidden, 32)
        self.l2 = nn.Linear(32, 32)
        self.out = nn.Linear(32, 1)

    def forward(self, board: torch.Tensor) -> torch.Tensor:
        stm = torch.clamp(self.ft(board[:, :HALF_SIZE]), 0.0, 1.0)
        nstm = torch.clamp(self.ft(board[:, HALF_SIZE:]), 0.0, 1.0)
        x = torch.cat([stm, nstm], dim=1)
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        return self.out(x)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def wdl_loss(eval_pawns: torch.Tensor, score_cp: torch.Tensor) -> torch.Tensor:
    """MSE between predicted and target win probabilities."""
    pred = torch.sigmoid(eval_pawns * 100.0 / WDL_SCALE_CP)
    target = torch.sigmoid(score_cp / WDL_SCALE_CP)
    return torch.mean((pred - target) ** 2)
