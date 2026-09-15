"""Candidate positions around genuine defensible-to-lost decisions, for Stockfish labeling.

A turning decision is one where Stockfish (with the game history) rated the
position defensible (best >= DEFENSIBLE_CP) and the net's move losing (<= -WIN_CP).
Move selection scores every legal move with the harness search: depth 1, then
captures until the position is quiet, with the net evaluating that quiet leaf.
So for each turning decision this collects, for every legal move, both the
position the move leads to and the quiescence leaf the search would actually
evaluate there (under --net), plus the decision position itself. The positions
go to chess_nn.label like any other candidates, so they get the same depth and
quiet-best-move filter as the broad training data.

Only training-source diagnoses feed this; positions that occur in any --exclude
diagnosis (held-out games) are dropped, and duplicates are removed.

Usage:
  uv run python -m chess_nn.decision_data data/diagnosis_h1024_L2 data/diagnosis_b_h1024 \\
      --net runs/jj6-h1024-s1000-cp20/model.onnx --exclude data/diagnosis_heldout_h1024 --out data/decision_candidates_a.txt
  uv run python -m chess_nn.label --candidates data/decision_candidates_a.txt \\
      --out-dir data/shards_decision --prefix shard_decision_a --depth 9
"""

import argparse
import json
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import chess

from chess_nn.rank_pairs import is_decisive
from chess_nn.search_check import MATE, NetEval, harness_captures


def position_key(board: chess.Board) -> str:
    return " ".join(board.fen().split()[:4])


def quiescence_leaf(net, board: chess.Board, alpha: float, beta: float) -> tuple[float, str | None]:
    """The harness quiescence value and the FEN of the quiet leaf its principal line ends in.

    The leaf is None when the line ends in mate, stalemate or the fifty-move
    rule (no net evaluation happens there).
    """
    if not any(board.generate_legal_moves()):
        return (-MATE if board.is_check() else 0.0), None
    if board.halfmove_clock >= 100:
        return 0.0, None
    in_check = board.is_check()
    leaf = None
    if not in_check:
        stand_pat = net(board)
        leaf = board.fen()
        if stand_pat >= beta:
            return beta, leaf
        if stand_pat > alpha:
            alpha = stand_pat
    for move in list(board.legal_moves) if in_check else harness_captures(board):
        board.push(move)
        score, child_leaf = quiescence_leaf(net, board, -beta, -alpha)
        score = -score
        board.pop()
        if score >= beta:
            return beta, child_leaf
        if score > alpha:
            alpha, leaf = score, child_leaf
    return alpha, leaf


def decision_positions(net, fen: str) -> tuple[list[str], Counter]:
    """The positions whose static evals decide this move, and how each was reached.

    For every legal move that is the quiet leaf quiescence scores the move by:
    the child itself when quiescence stands pat there ("child"), otherwise the
    leaf at the end of the capture line ("leaf"). A child quiescence searches
    past is left out: its searched score reflects captures the search already
    plays out, so it is not a target for the static evaluator. The decision
    position itself is not scored by this decision and is left out too.
    """
    board = chess.Board(fen)
    positions, kinds = [], Counter()
    for move in list(board.legal_moves):
        board.push(move)
        child = board.fen()
        _, leaf = quiescence_leaf(net, board, -float("inf"), float("inf"))
        if leaf is None:
            kinds["terminal"] += 1
        else:
            kind = "child" if leaf == child else "leaf"
            kinds[kind] += 1
            positions.append(leaf)
        board.pop()
    return positions, kinds


_net: NetEval | None = None


def _init(model: str) -> None:
    global _net
    _net = NetEval(Path(model))


def _positions(fen: str) -> list[str]:
    return decision_positions(_net, fen)


def load_decisions(directory: Path) -> list[dict]:
    return [json.loads(line) for line in (directory / "decisions.jsonl").read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("diagnoses", type=Path, nargs="+", help="training-source chess_nn.diagnose directories")
    parser.add_argument("--net", type=Path, required=True, help="model whose quiescence leaves to collect")
    parser.add_argument("--exclude", type=Path, nargs="*", default=[], help="held-out diagnosis directories")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    excluded = set()
    for directory in args.exclude:
        for d in load_decisions(directory):
            excluded.add(position_key(chess.Board(d["fen"])))
    turning = {d["fen"] for directory in args.diagnoses for d in load_decisions(directory) if is_decisive(d)}
    start = time.time()
    with Pool(args.workers, initializer=_init, initargs=(str(args.net),)) as pool:
        collected = pool.map(_positions, sorted(turning), chunksize=4)
    kinds = sum((k for _, k in collected), Counter())
    seen, kept, dropped = set(), [], 0
    for fen in (f for group, _ in collected for f in group):
        key = position_key(chess.Board(fen))
        if key in excluded:
            dropped += 1
        elif key not in seen:
            seen.add(key)
            kept.append(fen)
    args.out.write_text("\n".join(kept) + "\n")
    print(f"{len(turning):,} turning decisions; moves scored by {dict(kinds)} → {len(kept):,} unique positions "
          f"({dropped:,} held-out duplicates dropped, {time.time() - start:.0f}s) → {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
