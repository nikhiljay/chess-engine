"""NNUE-style evaluation network and its training loss."""

import torch
from torch import nn

from chess_nn.encoding import HALF_SIZE
from chess_nn.threats import GEOMETRY_NAMES, THREAT_FEATURES, geometry, threat_features


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


# Offset of a perspective half's own-king plane (channel 5) within that half.
KING_PLANE = 5 * 64


def default_king_buckets() -> list[int]:
    """Bucket for each own-king square (a1 = 0, from that side's view).

    Queenside (files a-d) or kingside (e-h), crossed with back rank, second
    rank, ranks 3-4 and ranks 5-8: 8 buckets.
    """
    buckets = []
    for square in range(64):
        file, rank = square % 8, square // 8
        band = 0 if rank == 0 else 1 if rank == 1 else 2 if rank < 4 else 3
        buckets.append((0 if file < 4 else 4) + band)
    return buckets


class KingBucketNet(EvalNet):
    """EvalNet whose feature transformer is chosen by the perspective's own king square.

    Each bucket's transformer is the shared one plus a per-bucket delta, so rare
    king squares start from, and are pulled toward, what every position
    teaches. Export merges shared + delta into one weight per bucket.
    """

    def __init__(self, hidden: int = 256, head: tuple[int, ...] = (32, 32), king_buckets: list[int] | None = None) -> None:
        super().__init__(hidden, head)
        buckets = list(king_buckets) if king_buckets is not None else default_king_buckets()
        self.num_buckets = max(buckets) + 1
        self.register_buffer("bucket_of_square", torch.tensor(buckets, dtype=torch.long))
        self.ft_delta = nn.Parameter(torch.zeros(self.num_buckets, hidden, HALF_SIZE))
        self.ft_bias_delta = nn.Parameter(torch.zeros(self.num_buckets, hidden))

    def buckets(self, half: torch.Tensor) -> torch.Tensor:
        """Bucket index [N] from a perspective half's own-king plane."""
        return self.bucket_of_square[half[:, KING_PLANE : KING_PLANE + 64].argmax(dim=1)]

    def transform(self, half: torch.Tensor) -> torch.Tensor:
        idx = self.buckets(half)
        delta = torch.zeros(half.shape[0], self.hidden, device=half.device, dtype=half.dtype)
        for b in range(self.num_buckets):
            rows = (idx == b).nonzero(as_tuple=True)[0]
            if rows.numel():
                delta = delta.index_add(0, rows, half[rows] @ self.ft_delta[b].T + self.ft_bias_delta[b])
        return torch.clamp(self.ft(half) + delta, 0.0, 1.0)

    def forward(self, board: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.transform(board[:, :HALF_SIZE]), self.transform(board[:, HALF_SIZE:])], dim=1)
        for layer in self.head:
            x = torch.relu(layer(x))
        return self.out(x)

    def merged_transformer(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-bucket weight [B, 770, H] (row per input feature) and bias [B, H]."""
        weight = (self.ft.weight.unsqueeze(0) + self.ft_delta).transpose(1, 2)
        bias = self.ft.bias.unsqueeze(0) + self.ft_bias_delta
        return weight, bias

    def exported_params(self) -> int:
        """Parameters the harness counts in the exported graph (see export.export_onnx)."""
        head = sum(p.numel() for module in (*self.head, self.out) for p in module.parameters())
        # bucket weights and biases, the 64-entry bucket map, 4 index constants, 2 split sizes
        return self.num_buckets * (HALF_SIZE + 1) * self.hidden + head + 64 + 4 + 2


class ThreatNet(EvalNet):
    """EvalNet plus a second feature transformer over attack and threat features.

    Each perspective's accumulator is [clamp(ft(half)), clamp(threat_ft(threat_features(half)))];
    the fixed attack geometry lives in non-persistent buffers (chess_nn.threats).
    """

    def __init__(self, hidden: int = 256, head: tuple[int, ...] = (32, 32), threat_hidden: int = 64) -> None:
        super().__init__(hidden, head)
        self.threat_hidden = threat_hidden
        self.threat_ft = nn.Linear(THREAT_FEATURES, threat_hidden)
        sizes = [2 * (hidden + threat_hidden), *head]
        self.head = nn.ModuleList(nn.Linear(a, b) for a, b in zip(sizes, sizes[1:]))
        self.out = nn.Linear(sizes[-1], 1)
        for name, matrix in geometry().items():
            self.register_buffer(name, torch.from_numpy(matrix), persistent=False)

    def geometry_tensors(self) -> dict[str, torch.Tensor]:
        return {name: getattr(self, name) for name in GEOMETRY_NAMES}

    def transform(self, half: torch.Tensor) -> torch.Tensor:
        base = torch.clamp(self.ft(half), 0.0, 1.0)
        threats = torch.clamp(self.threat_ft(threat_features(half, self.geometry_tensors())), 0.0, 1.0)
        return torch.cat([base, threats], dim=1)

    def forward(self, board: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.transform(board[:, :HALF_SIZE]), self.transform(board[:, HALF_SIZE:])], dim=1)
        for layer in self.head:
            x = torch.relu(layer(x))
        return self.out(x)


def model_from_config(config: dict) -> EvalNet:
    """The architecture a run's config.json describes, with fresh weights."""
    head = tuple(config["head"])
    if config.get("threat_hidden"):
        return ThreatNet(config["hidden"], head, config["threat_hidden"])
    if config.get("king_buckets"):
        return KingBucketNet(config["hidden"], head)
    return EvalNet(config["hidden"], head)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def cp_loss(eval_pawns: torch.Tensor, score_cp: torch.Tensor, clip_pawns: float = 50.0) -> torch.Tensor:
    """Huber loss in pawns against targets clipped to ±clip_pawns.

    Unlike the WDL loss, this keeps full gradient between large advantages
    (e.g. +8 pawns vs a laddered mate), which a depth-1 search needs in order
    to rank progress moves above shuffles in won positions.
    """
    target = torch.clamp(score_cp / 100.0, -clip_pawns, clip_pawns)
    return torch.nn.functional.smooth_l1_loss(eval_pawns, target)


def rank_loss(good_child: torch.Tensor, bad_child: torch.Tensor, margin_pawns: float = 0.2, temperature_pawns: float = 0.5) -> torch.Tensor:
    """Logistic ranking loss over the positions two moves lead to.

    The net scores a child from its side to move's view (the opponent), so the
    mover prefers the child scored lower: this pushes eval(good child) at least
    `margin_pawns` below eval(bad child).
    """
    return torch.nn.functional.softplus((good_child - bad_child + margin_pawns) / temperature_pawns).mean()


def wdl_loss(eval_pawns: torch.Tensor, score_cp: torch.Tensor, scale_cp: float = 400.0) -> torch.Tensor:
    """MSE between predicted and target win probabilities.

    A larger scale flattens the sigmoid, keeping gradient between large
    advantages (e.g. +8 vs +25 pawns) that a small scale treats as equal.
    """
    pred = torch.sigmoid(eval_pawns * 100.0 / scale_cp)
    target = torch.sigmoid(score_cp / scale_cp)
    return torch.mean((pred - target) ** 2)
