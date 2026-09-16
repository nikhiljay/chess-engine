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


class ResidualBlock(nn.Module):
    """Two 3x3 convolutions over the board with a skip connection."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.first = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm_first = nn.BatchNorm2d(channels)
        self.second = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm_second = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.relu(self.norm_first(self.first(x)))
        y = self.norm_second(self.second(y))
        return torch.relu(x + y)


class ConvNet(nn.Module):
    """Residual CNN over the 8x8 board, for the local patterns tactics are made of.

    Every NNUE-style net in this project sits at the same accuracy in level positions
    (finding 71: r ~0.35 against Stockfish static's 0.561), across an 8x parameter range
    and a 4x data range, because a linear map over piece-square features has to judge a
    tactic from what is on which square. A convolution sees a square together with its
    neighbours and shares weights across the board, which is the bias those positions want.

    Input is the harness's flat [N, 1540] tensor: two 770-float perspective halves, each
    12 piece planes of 64 squares plus 2 castling flags. Both halves become planes, so the
    net sees the position from both sides exactly as EvalNet does. Output is pawns from
    the side to move.
    """

    def __init__(self, channels: int = 32, blocks: int = 6) -> None:
        super().__init__()
        self.channels, self.blocks_count = channels, blocks
        self.stem = nn.Conv2d(28, channels, 3, padding=1)
        self.norm_stem = nn.BatchNorm2d(channels)
        self.blocks = nn.ModuleList(ResidualBlock(channels) for _ in range(blocks))
        self.reduce = nn.Conv2d(channels, 8, 1)
        self.fc = nn.Linear(8 * 64, 64)
        self.out = nn.Linear(64, 1)

    @staticmethod
    def planes(board: torch.Tensor) -> torch.Tensor:
        stm = board[:, :768].reshape(-1, 12, 8, 8)
        nstm = board[:, HALF_SIZE : HALF_SIZE + 768].reshape(-1, 12, 8, 8)
        castling = torch.cat([board[:, 768:770], board[:, HALF_SIZE + 768 : HALF_SIZE + 770]], dim=1)
        return torch.cat([stm, nstm, castling.reshape(-1, 4, 1, 1).expand(-1, 4, 8, 8)], dim=1)

    def forward(self, board: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.norm_stem(self.stem(self.planes(board))))
        for block in self.blocks:
            x = block(x)
        x = torch.relu(self.reduce(x)).flatten(1)
        return self.out(torch.relu(self.fc(x)))


class TransformerBlock(nn.Module):
    """Pre-norm self-attention over the squares, then a position-wise MLP."""

    def __init__(self, dim: int, heads: int, mlp_ratio: int) -> None:
        super().__init__()
        self.norm_attention = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Linear(dim * mlp_ratio, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.norm_attention(x)
        x = x + self.attention(normed, normed, normed, need_weights=False)[0]
        return x + self.mlp(self.norm_mlp(x))


class TransformerNet(nn.Module):
    """Self-attention encoder over the 64 squares.

    Where a convolution sees a square with its neighbours, attention lets any square consult any
    other directly, which is the relation a pin, a skewer or a long diagonal actually is. Each
    square becomes a token carrying its 24 piece-plane indicators (both perspectives) and the
    castling flags, plus a learned square embedding; the pooled encoding scores the position in
    pawns from the side to move.

    Input is the harness's flat [N, 1540] tensor, reshaped the same way ConvNet does it.
    """

    def __init__(self, dim: int = 64, layers: int = 4, heads: int = 4, mlp_ratio: int = 2) -> None:
        super().__init__()
        self.dim, self.layers_count = dim, layers
        self.embed = nn.Linear(28, dim)
        self.square = nn.Parameter(torch.zeros(1, 64, dim))
        nn.init.normal_(self.square, std=0.02)
        self.blocks = nn.ModuleList(TransformerBlock(dim, heads, mlp_ratio) for _ in range(layers))
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, 1)

    def forward(self, board: torch.Tensor) -> torch.Tensor:
        tokens = ConvNet.planes(board).flatten(2).transpose(1, 2)  # [N, 64 squares, 28 channels]
        x = self.embed(tokens) + self.square
        for block in self.blocks:
            x = block(x)
        pooled = self.norm(x).mean(dim=1)
        return self.out(torch.relu(self.head(pooled)))


def model_from_config(config: dict) -> nn.Module:
    """The architecture a run's config.json describes, with fresh weights."""
    if config.get("transformer_dim"):
        return TransformerNet(config["transformer_dim"], config.get("transformer_layers", 4),
                              config.get("transformer_heads", 4))
    if config.get("conv_channels"):
        return ConvNet(config["conv_channels"], config.get("conv_blocks", 6))
    head = tuple(config["head"])
    if config.get("threat_hidden"):
        return ThreatNet(config["hidden"], head, config["threat_hidden"])
    if config.get("king_buckets"):
        return KingBucketNet(config["hidden"], head)
    return EvalNet(config["hidden"], head)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def cp_loss(eval_pawns: torch.Tensor, score_cp: torch.Tensor, clip_pawns: float = 50.0, kind: str = "huber") -> torch.Tensor:
    """Loss in pawns against targets clipped to ±clip_pawns.

    Unlike the WDL loss, this keeps full gradient between large advantages
    (e.g. +8 pawns vs a laddered mate), which a depth-1 search needs in order
    to rank progress moves above shuffles in won positions.

    "huber" is linear past one pawn, so a large miss costs little; finding 70
    found the nets over-dispersed in level positions (predicted sd 3.26 against
    a true sd of 1.42), which is what makes the search walk into a leaf it rates
    three pawns too high. "mse" prices those misses quadratically instead.
    """
    target = torch.clamp(score_cp / 100.0, -clip_pawns, clip_pawns)
    if kind == "mse":
        return torch.nn.functional.mse_loss(eval_pawns, target)
    return torch.nn.functional.smooth_l1_loss(eval_pawns, target)


def group_consistency_loss(eval_pawns: torch.Tensor, score_cp: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean within-group variance of the error in pawns, over [G, M] sibling leaves.

    Targets arrive in centipawns, like cp_loss and wdl_loss; padding is masked out.
    The search compares a decision's leaves against each other, so error shared by
    every sibling cancels and only the variation inside a group changes the move
    played. Unlike rank_loss this leaves the absolute targets in charge: a group
    whose evaluations are all wrong by the same amount is not penalised here.
    """
    error = (eval_pawns - score_cp / 100.0) * mask
    count = mask.sum(dim=1).clamp(min=1)
    mean = (error.sum(dim=1) / count).unsqueeze(1)
    centered = (error - mean) * mask
    return ((centered**2).sum(dim=1) / count).mean()


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
