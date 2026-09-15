"""Attack maps and threat features computed from the board encoding itself.

A depth-1 search follows only captures, so quiet threats — a piece left
attacked, an undefended piece, a piece attacked by a pawn — have to come from
the evaluation. These features are fixed linear algebra over one perspective
half of the 1540-input encoding: lookup matrices for knight, king and pawn
attacks, and one-square shift matrices stepped along rays (continuing only
through empty squares) for sliders. The same operations export to plain ONNX
(see export._threat_transformer), so the harness needs no changes.

Squares follow the encoding: a1 = 0 from that half's own side, which always
moves up the board.
"""

import numpy as np
import torch

STEPS = {"n": (0, 1), "s": (0, -1), "e": (1, 0), "w": (-1, 0), "ne": (1, 1), "nw": (-1, 1), "se": (1, -1), "sw": (-1, -1)}
ORTHOGONAL = ("n", "s", "e", "w")
DIAGONAL = ("ne", "nw", "se", "sw")
KNIGHT_JUMPS = [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)]
RAY_LENGTH = 7

# 34 piece planes × 64 (own/their pieces attacked, attacked and undefended,
# attacked by pawns) plus 2 control maps × 64.
THREAT_FEATURES = 36 * 64


def jump_matrix(jumps: list[tuple[int, int]]) -> np.ndarray:
    """[64, 64] with m[from, to] = 1 for each on-board (file, rank) jump; rows are sources."""
    matrix = np.zeros((64, 64), dtype=np.float32)
    for square in range(64):
        file, rank = square % 8, square // 8
        for df, dr in jumps:
            f, r = file + df, rank + dr
            if 0 <= f < 8 and 0 <= r < 8:
                matrix[square, r * 8 + f] = 1.0
    return matrix


def geometry() -> dict[str, np.ndarray]:
    """Fixed attack matrices, named as ONNX initializers."""
    matrices = {
        "geo_knight": jump_matrix(KNIGHT_JUMPS),
        "geo_king": jump_matrix(list(STEPS.values())),
        "geo_pawn_up": jump_matrix([(-1, 1), (1, 1)]),
        "geo_pawn_down": jump_matrix([(-1, -1), (1, -1)]),
    }
    for name, step in STEPS.items():
        matrices[f"geo_step_{name}"] = jump_matrix([step])
    return matrices


GEOMETRY_NAMES = tuple(geometry())


def _rays(sources: torch.Tensor, empty: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
    total = torch.zeros_like(sources)
    frontier = sources
    for _ in range(RAY_LENGTH):
        frontier = frontier @ step
        total = total + frontier
        frontier = frontier * empty
    return total


def _side_attacks(stack: torch.Tensor, empty: torch.Tensor, pawn: torch.Tensor, geo: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Attacker counts per square [N, 64] for one side's 6 planes, and its pawn attacks alone."""
    pawn_attacks = stack[:, 0] @ pawn
    attacks = pawn_attacks + stack[:, 1] @ geo["geo_knight"] + stack[:, 5] @ geo["geo_king"]
    diagonal = stack[:, 2] + stack[:, 4]
    orthogonal = stack[:, 3] + stack[:, 4]
    for direction in DIAGONAL:
        attacks = attacks + _rays(diagonal, empty, geo[f"geo_step_{direction}"])
    for direction in ORTHOGONAL:
        attacks = attacks + _rays(orthogonal, empty, geo[f"geo_step_{direction}"])
    return attacks, pawn_attacks


def attack_maps(half: torch.Tensor, geo: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Planes and attacker counts for one perspective half [N, 770]."""
    planes = half[:, :768].reshape(half.shape[0], 12, 64)
    own, their = planes[:, :6], planes[:, 6:]
    empty = 1.0 - planes.sum(dim=1)
    own_attacks, own_pawn_attacks = _side_attacks(own, empty, geo["geo_pawn_up"], geo)
    their_attacks, their_pawn_attacks = _side_attacks(their, empty, geo["geo_pawn_down"], geo)
    return {
        "own": own,
        "their": their,
        "own_attacks": own_attacks,
        "their_attacks": their_attacks,
        "own_pawn_attacks": own_pawn_attacks,
        "their_pawn_attacks": their_pawn_attacks,
    }


def threat_features(half: torch.Tensor, geo: dict[str, torch.Tensor]) -> torch.Tensor:
    """[N, THREAT_FEATURES] threat features for one perspective half."""
    maps = attack_maps(half, geo)
    own, their = maps["own"], maps["their"]
    own_hit = torch.clamp(maps["own_attacks"], 0.0, 1.0)
    their_hit = torch.clamp(maps["their_attacks"], 0.0, 1.0)
    own_pawn_hit = torch.clamp(maps["own_pawn_attacks"], 0.0, 1.0)
    their_pawn_hit = torch.clamp(maps["their_pawn_attacks"], 0.0, 1.0)
    n = half.shape[0]
    features = [
        own * their_hit[:, None],  # my pieces they attack
        own * (their_hit * (1.0 - own_hit))[:, None],  # ... and I don't defend
        own[:, 1:] * their_pawn_hit[:, None],  # my pieces their pawns attack
        their * own_hit[:, None],  # their pieces I attack
        their * (own_hit * (1.0 - their_hit))[:, None],  # ... and they don't defend
        their[:, 1:] * own_pawn_hit[:, None],  # their pieces my pawns attack
        own_hit[:, None],  # squares I control
        their_hit[:, None],  # squares they control
    ]
    return torch.cat([f.reshape(n, -1) for f in features], dim=1)
