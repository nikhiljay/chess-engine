"""Do static move rankings survive the harness's quiescence search?

The ranking loss trains the order of two child positions' static evals. Move
selection instead scores each move with the harness search: depth 1, then only
captures, promotions and castling (all moves when in check), with the net at
the quiet leaves. For each held-out pair (position, better move, worse move)
this compares, per run:

  - static: the net scores the better move's child lower (prefers it),
  - search: the better move's quiescence value beats the worse move's,
  - chosen: the move the harness's play-move binary actually selects —
    the better move, the worse move, or another one.

The quiescence here mirrors engine/src/nn.rs move for move; `--verify` checks
its choices against play-move.

Usage:
  uv run python -m chess_nn.search_check data/rank_pairs_b_val.jsonl --runs runs/rank-a-h512-w0 runs/threat-a-h512-t64-w0
"""

import argparse
import json
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import chess
import onnxruntime as ort

from chess_nn.encoding import dense_numpy, encode_sparse
from chess_nn.play import net_move

MATE = 100_000.0
VALUE = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 100}


class NetEval:
    """Single-position ONNX evals from the side to move's view, cached by position."""

    def __init__(self, model: Path, threads: int = 1) -> None:
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model), options, providers=["CPUExecutionProvider"])
        self.cache: dict[str, float] = {}

    def __call__(self, board: chess.Board) -> float:
        key = board.fen()
        value = self.cache.get(key)
        if value is None:
            value = float(self.session.run(None, {"board": dense_numpy(encode_sparse([board]))})[0][0, 0])
            self.cache[key] = value
        return value


def harness_captures(board: chess.Board) -> list[chess.Move]:
    """The harness's capture list (onto an occupied square, promotions, castling) in its MVV-LVA order."""
    def key(move: chess.Move) -> tuple[int, int]:
        promo = 0 if move.promotion is None else (9 if move.promotion == chess.QUEEN else 3)
        victim = board.piece_at(move.to_square)
        victim_value = VALUE[victim.piece_type] if victim else (VALUE[chess.ROOK] if board.is_castling(move) else 0)
        return promo + victim_value, -VALUE[board.piece_at(move.from_square).piece_type]

    moves = [m for m in board.legal_moves if board.piece_at(m.to_square) is not None or m.promotion is not None or board.is_castling(m)]
    return sorted(moves, key=key, reverse=True)


def quiescence(net, board: chess.Board, alpha: float, beta: float) -> float:
    if not any(board.generate_legal_moves()):
        return -MATE if board.is_check() else 0.0
    if board.halfmove_clock >= 100:
        return 0.0
    in_check = board.is_check()
    if not in_check:
        stand_pat = net(board)
        if stand_pat >= beta:
            return beta
        alpha = max(alpha, stand_pat)
    for move in list(board.legal_moves) if in_check else harness_captures(board):
        board.push(move)
        score = -quiescence(net, board, -beta, -alpha)
        board.pop()
        if score >= beta:
            return beta
        alpha = max(alpha, score)
    return alpha


def move_value(net, board: chess.Board, move: chess.Move) -> float:
    """The harness's exact value of a root move for the side to move (full window)."""
    board.push(move)
    try:
        return -quiescence(net, board, -float("inf"), float("inf"))
    finally:
        board.pop()


def search_choice(net, board: chess.Board) -> chess.Move:
    """First move with the highest search value (the harness breaks exact ties by its own move order)."""
    best, best_value = None, -float("inf")
    for move in board.legal_moves:
        value = move_value(net, board, move)
        if best is None or value > best_value:
            best, best_value = move, value
    return best


_net: NetEval | None = None
_model: Path | None = None


def _init(model: str) -> None:
    global _net, _model
    _model = Path(model)
    _net = NetEval(_model)


def _check(item: dict) -> dict:
    board = chess.Board(item["fen"])
    good, bad = chess.Move.from_uci(item["good_move"]), chess.Move.from_uci(item["bad_move"])
    after_good, after_bad = board.copy(stack=False), board.copy(stack=False)
    after_good.push(good)
    after_bad.push(bad)
    chosen = net_move(_model, board)
    return {
        "kind": item["kind"],
        "decisive": item.get("decisive", False),
        "static": _net(after_good) < _net(after_bad),
        "search": move_value(_net, board, good) > move_value(_net, board, bad),
        "chosen": "better" if chosen == good else "worse" if chosen == bad else "other",
    }


def _verify(fen: str) -> tuple[str, str]:
    board = chess.Board(fen)
    return search_choice(_net, board).uci(), net_move(_model, board).uci()


def summarize(rows: list[dict]) -> dict:
    groups = defaultdict(list)
    for row in rows:
        groups[row["kind"]].append(row)
        groups["all"].append(row)
        if row["decisive"]:
            groups["decisive"].append(row)
    out = {}
    for name, group in groups.items():
        n = len(group)
        chosen = Counter(r["chosen"] for r in group)
        out[name] = {
            "pairs": n,
            "static_accuracy": round(sum(r["static"] for r in group) / n, 3),
            "search_accuracy": round(sum(r["search"] for r in group) / n, 3),
            "static_right_search_wrong": sum(r["static"] and not r["search"] for r in group),
            "static_wrong_search_right": sum(r["search"] and not r["static"] for r in group),
            "chose_better": round(chosen["better"] / n, 3),
            "chose_worse": round(chosen["worse"] / n, 3),
            "chose_other": round(chosen["other"] / n, 3),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pairs", type=Path, help="rank_pairs .jsonl (positions and moves)")
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--verify", type=int, default=0, help="also compare this many search choices with play-move")
    args = parser.parse_args()

    items = [json.loads(line) for line in args.pairs.read_text().splitlines() if line.strip()][: args.limit]
    for run in args.runs:
        with Pool(args.workers, initializer=_init, initargs=(str(run / "model.onnx"),)) as pool:
            rows = pool.map(_check, items, chunksize=8)
            verified = pool.map(_verify, [i["fen"] for i in items[: args.verify]]) if args.verify else []
        result = summarize(rows)
        if verified:
            result["verify_search_matches_play_move"] = f"{sum(a == b for a, b in verified)}/{len(verified)}"
        (run / f"search_check_{args.pairs.stem}.json").write_text(json.dumps(result, indent=2))
        print(run.name, json.dumps(result))


if __name__ == "__main__":
    main()
