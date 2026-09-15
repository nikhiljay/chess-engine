"""Move-ranking pairs from chess_nn.diagnose decisions.

For each decision where the net's move was clearly worse than Stockfish's best
(scored with the game history, so a shuffle into a repetition counts as a
draw), keep the two positions those moves lead to. The net scores a position
from its side to move's view, so the mover should see the better child scored
lower; model.rank_loss trains exactly that ordering.

Pairs are skipped when either child is in check or the game is over there —
the harness resolves those with a search rather than the net's static eval —
and when the position was already lost. Pairs are split by game into train and
validation files, so validation accuracy measures positions from unseen games.

Usage:
  uv run python -m chess_nn.rank_pairs data/diagnosis_h1024_L2 data/diagnosis_h1024_L3 --out data/rank_pairs_a
"""

import argparse
import json
import zlib
from pathlib import Path

import chess
import numpy as np
import torch

from chess_nn.diagnose import WIN_CP
from chess_nn.encoding import MAX_FEATURES, feature_indices
from chess_nn.harness import TRAINING_ROOT

MIN_GAP_CP = 100
VAL_EVERY = 10  # one game in ten goes to validation


def child_features(fen: str, uci: str) -> list[int] | None:
    """Feature indices of the position after `uci`, or None if the net's static eval would not score it."""
    board = chess.Board(fen)
    move = chess.Move.from_uci(uci)
    if move not in board.legal_moves:
        return None
    board.push(move)
    if board.is_check() or board.is_game_over():
        return None
    return feature_indices(board)


def build_pairs(decisions: list[dict], source: str, min_gap_cp: int = MIN_GAP_CP) -> list[dict]:
    pairs = []
    for d in decisions:
        gap = d["best_cp"] - d["played_cp"]
        if d["best"] == d["played"] or gap < min_gap_cp or d["best_cp"] <= -WIN_CP:
            continue
        good = child_features(d["fen"], d["best"])
        bad = child_features(d["fen"], d["played"])
        if good is None or bad is None:
            continue
        pairs.append({"game": f"{source}:{d['game']}", "good": good, "bad": bad, "gap_cp": gap, "repeats": d["repeats"]})
    return pairs


def to_arrays(pairs: list[dict]) -> dict[str, np.ndarray]:
    good = np.full((len(pairs), MAX_FEATURES), -1, dtype=np.int16)
    bad = np.full((len(pairs), MAX_FEATURES), -1, dtype=np.int16)
    for i, pair in enumerate(pairs):
        good[i, : len(pair["good"])] = pair["good"]
        bad[i, : len(pair["bad"])] = pair["bad"]
    return {
        "good": good,
        "bad": bad,
        "gap_cp": np.clip([p["gap_cp"] for p in pairs], 0, 32000).astype(np.int16),
        "repeats": np.array([p["repeats"] for p in pairs], dtype=bool),
    }


def is_validation(game_key: str) -> bool:
    return zlib.crc32(game_key.encode()) % VAL_EVERY == 0


def load_pairs(paths: list[str], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Concatenated (good, bad) feature rows [P, 68] on `device`; relative paths are under training/."""
    goods, bads = [], []
    for p in paths:
        path = Path(p) if Path(p).is_absolute() else TRAINING_ROOT / p
        with np.load(path) as data:
            goods.append(data["good"])
            bads.append(data["bad"])
    return torch.from_numpy(np.concatenate(goods)).to(device), torch.from_numpy(np.concatenate(bads)).to(device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("diagnoses", type=Path, nargs="+", help="chess_nn.diagnose output directories")
    parser.add_argument("--out", type=Path, required=True, help="writes <out>_train.npz and <out>_val.npz")
    parser.add_argument("--min-gap-cp", type=int, default=MIN_GAP_CP)
    args = parser.parse_args()

    pairs = []
    for directory in args.diagnoses:
        decisions = [json.loads(line) for line in (directory / "decisions.jsonl").read_text().splitlines() if line.strip()]
        pairs += build_pairs(decisions, directory.name, args.min_gap_cp)
    val = [p for p in pairs if is_validation(p["game"])]
    train = [p for p in pairs if not is_validation(p["game"])]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for name, subset in (("train", train), ("val", val)):
        np.savez(f"{args.out}_{name}.npz", **to_arrays(subset))
    repeats = sum(p["repeats"] for p in pairs)
    print(f"{len(pairs):,} pairs ({repeats:,} where the net's move repeated a position): {len(train):,} train, {len(val):,} val → {args.out}_*.npz")


if __name__ == "__main__":
    main()
