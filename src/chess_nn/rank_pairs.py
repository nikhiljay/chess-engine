"""Move-ranking pairs from chess_nn.diagnose decisions.

Three kinds of pair, each holding the positions two moves lead to:

  - mate-in-one pairs: the net's move allowed a mate in one that the harness
    search never examines (chess_nn.mate_in_one) while moves allowing no mate
    existed; up to MATE_ALTERNATIVES of those safe moves (Stockfish's best
    first) are each paired against it. Kept even in lost positions, since the
    search cannot see this and avoiding the mate is always the better move.
  - score pairs: the net's move was clearly worse than Stockfish's best (by at
    least --min-gap-cp, scored with the game history).
  - repetition pairs: the net was winning (best >= WIN_CP) and its move went
    back into a position already seen in the game while Stockfish's best move
    did not. Stockfish only scores the move that completes a threefold as a
    draw, so these shuffles often look nearly as good as progress; the rule
    labels them anyway, whatever the centipawn gap.

The net scores a position from its side to move's view, so the mover should see
the better child scored lower; model.rank_loss trains exactly that ordering.
Score and repetition pairs are skipped when either child is in check or the
game is over there — the harness resolves those with a search rather than the
net's static eval — and when the position was already lost. Pairs are split by
game into train and validation files, so validation accuracy measures
positions from unseen games.

Usage:
  uv run python -m chess_nn.rank_pairs data/diagnosis_h1024_L2 data/diagnosis_h1024_L3 --out data/rank_pairs_a
"""

import argparse
import json
import random
import zlib
from pathlib import Path

import chess
import numpy as np
import torch

from chess_nn.diagnose import WIN_CP
from chess_nn.encoding import MAX_FEATURES, feature_indices
from chess_nn.harness import TRAINING_ROOT
from chess_nn.mate_in_one import safe_moves, unseen_mate

MIN_GAP_CP = 100
MATE_ALTERNATIVES = 4
VAL_EVERY = 10  # one game in ten goes to validation
KINDS = ("mate_in_one", "score", "repetition")


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


def repeats_position(decision: dict, uci: str) -> bool:
    """Whether playing `uci` at this decision returns to a position already seen in the game."""
    board = chess.Board(decision["start_fen"])
    seen = {board._transposition_key()}
    for move in decision["history"]:
        board.push_uci(move)
        seen.add(board._transposition_key())
    board.push_uci(uci)
    return board._transposition_key() in seen


def mate_in_one_pairs(decision: dict, game_key: str) -> list[dict]:
    """Safe alternatives paired against a move that allowed an unseen mate in one."""
    board = chess.Board(decision["fen"])
    if unseen_mate(board, chess.Move.from_uci(decision["played"])) is None:
        return []
    bad = child_features(decision["fen"], decision["played"])
    safe = [m.uci() for m in safe_moves(board)]
    if bad is None or not safe:
        return []
    rest = [m for m in safe if m != decision["best"]]
    random.Random(zlib.crc32(f"{game_key}:{decision['ply']}".encode())).shuffle(rest)
    goods = ([decision["best"]] if decision["best"] in safe else []) + rest
    pairs = []
    for uci in goods:
        good = child_features(decision["fen"], uci)
        if good is not None:
            gap = decision["best_cp"] - decision["played_cp"] if uci == decision["best"] else 0
            pairs.append({"game": game_key, "good": good, "bad": bad, "gap_cp": max(gap, 0), "repeats": decision["repeats"], "kind": "mate_in_one"})
        if len(pairs) == MATE_ALTERNATIVES:
            break
    return pairs


def build_pairs(
    decisions: list[dict],
    source: str,
    min_gap_cp: int = MIN_GAP_CP,
    repetition_pairs: bool = True,
    mate_pairs: bool = True,
) -> list[dict]:
    pairs = []
    for d in decisions:
        if d["best"] == d["played"]:
            continue
        game_key = f"{source}:{d['game']}"
        if mate_pairs:
            mates = mate_in_one_pairs(d, game_key)
            if mates:
                pairs += mates
                continue
        if d["best_cp"] <= -WIN_CP:
            continue
        gap = d["best_cp"] - d["played_cp"]
        if gap >= min_gap_cp:
            kind = "score"
        elif repetition_pairs and d["repeats"] and d["best_cp"] >= WIN_CP and not repeats_position(d, d["best"]):
            kind = "repetition"
        else:
            continue
        good = child_features(d["fen"], d["best"])
        bad = child_features(d["fen"], d["played"])
        if good is None or bad is None:
            continue
        pairs.append({"game": game_key, "good": good, "bad": bad, "gap_cp": max(gap, 0), "repeats": d["repeats"], "kind": kind})
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
        "repetition_rule": np.array([p["kind"] == "repetition" for p in pairs], dtype=bool),
        "mate_rule": np.array([p["kind"] == "mate_in_one" for p in pairs], dtype=bool),
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
    parser.add_argument("--no-repetition-pairs", action="store_true")
    parser.add_argument("--no-mate-pairs", action="store_true")
    args = parser.parse_args()

    pairs = []
    for directory in args.diagnoses:
        decisions = [json.loads(line) for line in (directory / "decisions.jsonl").read_text().splitlines() if line.strip()]
        pairs += build_pairs(decisions, directory.name, args.min_gap_cp, not args.no_repetition_pairs, not args.no_mate_pairs)
    val = [p for p in pairs if is_validation(p["game"])]
    train = [p for p in pairs if not is_validation(p["game"])]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for name, subset in (("train", train), ("val", val)):
        np.savez(f"{args.out}_{name}.npz", **to_arrays(subset))
    counts = ", ".join(f"{sum(p['kind'] == kind for p in pairs):,} {kind}" for kind in KINDS)
    print(f"{len(pairs):,} pairs ({counts}): {len(train):,} train, {len(val):,} val → {args.out}_*.npz")


if __name__ == "__main__":
    main()
