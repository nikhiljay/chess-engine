"""Which static evaluations decide a move, and which of them candidate extraction keeps.

For each decision this builds two views of the search engine/src/nn.rs runs:

1. Harness-evaluated positions: a replica of try_choose_move with its root
   alpha window, fail-hard cutoffs and batched stand-pats (every ongoing child
   not in check is evaluated before the loop over it, even when a cutoff then
   skips its subtree). Python-chess generates root and in-check moves in a
   different order than cozy-chess, so cutoffs, and with them the evaluated set,
   can differ slightly; the chosen move is the same up to exact ties.
2. Sensitivity: the same capture trees searched with a full window and no
   pruning. For every static evaluation in them, the smallest error at that
   one position (net units, pawns) that changes the chosen move with every
   other evaluation held fixed. Raising a value propagates through a node once
   it passes that node's value (cost: the gap), and lowering one propagates
   only while its line is the node's unique best, by at most the margin to the
   second best; negamax alternates the two at each ply. At the root the chosen
   move's value has to fall below the runner-up's, or another move's rise
   above the chosen one's.

decision_data keeps, per legal move, the quiet leaf of that move's principal
line: the child itself when quiescence stands pat there, else the end of the
capture line. The report splits every position in the unpruned trees into
kept, dropped move children (quiescence searched past them), and dropped
deeper positions, with how many the harness evaluates and how many could
change the decision with an error of at most 0.5, 1, 2 or 4 pawns.

With --meta-out it also writes, for every legal move of every decision, the
kept position and whether the move is the one the net played, for joining
with chess_nn.label --record.

Usage:
  uv run python -m chess_nn.quiescence_audit data/diagnosis_h1024_L2 data/diagnosis_b_h1024 \\
      --net runs/jj6-h1024-s1000-cp20/model.onnx --sample 300 --out data/quiescence_audit_a.json
"""

import argparse
import json
import random
import statistics
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import chess

from chess_nn.decision_data import load_decisions, quiescence_leaf
from chess_nn.rank_pairs import is_decisive
from chess_nn.search_check import MATE, NetEval, harness_captures

INF = float("inf")
ERROR_BINS = (0.5, 1.0, 2.0, 4.0)
NODE_BUDGET = 50_000


class TreeTooLarge(Exception):
    pass


def status(board: chess.Board) -> str:
    """cozy-chess's GameStatus as the harness sees it: mated, drawn (stalemate, fifty moves) or ongoing."""
    if not any(board.generate_legal_moves()):
        return "mated" if board.is_check() else "drawn"
    return "drawn" if board.halfmove_clock >= 100 else "ongoing"


def children(board: chess.Board, moves: list[chess.Move]) -> list[chess.Board]:
    out = []
    for move in moves:
        board.push(move)
        out.append(board.copy(stack=False))
        board.pop()
    return out


def harness_evaluated(net, board: chess.Board) -> tuple[chess.Move, set[str]]:
    """The move try_choose_move picks and every position whose evaluation it requests."""
    evaluated: set[str] = set()

    def batch(kids: list[chess.Board]) -> list[float | None]:
        hints = []
        for kid in kids:
            if status(kid) == "ongoing" and not kid.is_check():
                evaluated.add(kid.fen())
                hints.append(net(kid))
            else:
                hints.append(None)
        return hints

    def quiescence(node: chess.Board, alpha: float, beta: float, hint: float | None) -> float:
        state = status(node)
        if state != "ongoing":
            return -MATE if state == "mated" else 0.0
        in_check = node.is_check()
        if not in_check:
            if hint is None:
                evaluated.add(node.fen())
                hint = net(node)
            if hint >= beta:
                return beta
            alpha = max(alpha, hint)
        kids = children(node, list(node.legal_moves) if in_check else harness_captures(node))
        for kid, kid_hint in zip(kids, batch(kids)):
            score = -quiescence(kid, -beta, -alpha, kid_hint)
            if score >= beta:
                return beta
            alpha = max(alpha, score)
        return alpha

    moves = list(board.legal_moves)
    kids = children(board, moves)
    best, alpha = None, -INF
    for move, kid, hint in zip(moves, kids, batch(kids)):
        state = status(kid)
        value = MATE if state == "mated" else 0.0 if state == "drawn" else -quiescence(kid, -INF, -alpha, hint)
        if value > alpha:
            best, alpha = move, value
        if value >= MATE:
            break
    return best or moves[0], evaluated


def full_tree(net, board: chess.Board, ply: int, budget: list[int]) -> dict:
    """Unpruned full-window quiescence tree: each node's stand-pat (None in check or terminal) and value."""
    budget[0] -= 1
    if budget[0] < 0:
        raise TreeTooLarge
    state = status(board)
    if state != "ongoing":
        return {"fen": None, "ply": ply, "stand_pat": None, "children": [], "value": -MATE if state == "mated" else 0.0}
    in_check = board.is_check()
    stand_pat = None if in_check else net(board)
    kids = [full_tree(net, kid, ply + 1, budget) for kid in children(board, list(board.legal_moves) if in_check else harness_captures(board))]
    options = ([stand_pat] if stand_pat is not None else []) + [-kid["value"] for kid in kids]
    return {"fen": board.fen(), "ply": ply, "stand_pat": stand_pat, "children": kids, "value": max(options)}


def thresholds(node: dict, plus_cost: float, minus_cost: float, out: list[tuple[str, int, float]]) -> None:
    """Append (fen, ply, smallest eval error that changes the chosen move) for every stand-pat in the tree.

    plus_cost / minus_cost: how far this node's value (for its side to move) must rise / fall to change the
    chosen move; INF when no change in that direction can.
    """
    options = ([node["stand_pat"]] if node["stand_pat"] is not None else []) + [-kid["value"] for kid in node["children"]]
    value = node["value"]

    def lowering_propagates(option: float) -> bool:
        if option != value or options.count(value) > 1:
            return False
        rest = [o for o in options if o != value]
        return not rest or value - max(rest) > minus_cost

    if node["stand_pat"] is not None:
        up = value - node["stand_pat"] + plus_cost
        down = minus_cost if lowering_propagates(node["stand_pat"]) else INF
        out.append((node["fen"], node["ply"], min(up, down)))
    for kid in node["children"]:
        option = -kid["value"]
        # Raising the child's value lowers this option; lowering it raises the option past the gap.
        kid_plus = minus_cost if lowering_propagates(option) else INF
        kid_minus = value - option + plus_cost
        thresholds(kid, kid_plus, kid_minus, out)


def audit_decision(net, fen: str) -> dict | None:
    board = chess.Board(fen)
    chosen, evaluated = harness_evaluated(net, board)
    kept: set[str] = set()
    trees = []
    budget = [NODE_BUDGET]
    try:
        for move, kid in zip(list(board.legal_moves), children(board, list(board.legal_moves))):
            _, leaf = quiescence_leaf(net, kid.copy(stack=False), -INF, INF)
            if leaf is not None:
                kept.add(leaf)
            trees.append((move, full_tree(net, kid, 1, budget)))
    except TreeTooLarge:
        return None
    values = {move: -tree["value"] for move, tree in trees}
    best_value = max(values.values())
    runner_up = max((v for m, v in values.items() if m != chosen), default=-INF)
    positions: list[tuple[str, int, float]] = []
    for move, tree in trees:
        if move == chosen:
            thresholds(tree, best_value - runner_up if runner_up > -INF else INF, INF, positions)
        else:
            thresholds(tree, INF, best_value - values[move], positions)
    rows = defaultdict(list)
    for position, ply, error in positions:
        role = "kept" if position in kept else "dropped move child" if ply == 1 else "dropped deeper"
        rows[role].append({"error": error, "harness": position in evaluated})
    return {
        "fen": fen,
        "chosen_matches_full_window": values[chosen] == best_value,
        "evaluated_by_harness": len(evaluated),
        "kept_positions": len(kept),
        "roles": rows,
    }


def summarize(audits: list[dict], skipped: int) -> dict:
    roles: dict[str, list[dict]] = defaultdict(list)
    for a in audits:
        for role, rows in a["roles"].items():
            roles[role] += rows
    report = {
        "decisions": len(audits),
        "skipped_tree_too_large": skipped,
        "chosen_matches_full_window": sum(a["chosen_matches_full_window"] for a in audits),
        "median_evaluated_by_harness": statistics.median(a["evaluated_by_harness"] for a in audits),
        "median_kept_positions": statistics.median(a["kept_positions"] for a in audits),
        "roles": {},
    }
    for role, rows in sorted(roles.items()):
        finite = [r["error"] for r in rows if r["error"] < INF]
        report["roles"][role] = {
            "positions": len(rows),
            "evaluated_by_harness": sum(r["harness"] for r in rows),
            "can_change_decision": len(finite),
            **{f"error_le_{b}": sum(e <= b for e in finite) for b in ERROR_BINS},
            **{f"harness_evaluated_error_le_{b}": sum(r["harness"] and r["error"] <= b for r in rows) for b in ERROR_BINS},
        }
    return report


def move_meta(net, decision: dict) -> list[dict]:
    """Per legal move: the kept position, how quiescence reached it, and whether the net played the move."""
    board = chess.Board(decision["fen"])
    out = []
    for move in list(board.legal_moves):
        board.push(move)
        child = board.fen()
        value, leaf = quiescence_leaf(net, board, -INF, INF)
        board.pop()
        out.append({
            "decision": decision["fen"],
            "move": move.uci(),
            "played": move.uci() == decision["played"],
            "best": move.uci() == decision["best"],
            "value": -value,
            "kind": "terminal" if leaf is None else "child" if leaf == child else "leaf",
            "position": leaf,
        })
    return out


_net: NetEval | None = None


def _init(model: str) -> None:
    global _net
    _net = NetEval(Path(model))


def _audit(fen: str) -> dict | None:
    _net.cache.clear()
    return audit_decision(_net, fen)


def _meta(decision: dict) -> list[dict]:
    _net.cache.clear()
    return move_meta(_net, decision)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("diagnoses", type=Path, nargs="+", help="training-source chess_nn.diagnose directories")
    parser.add_argument("--net", type=Path, required=True)
    parser.add_argument("--sample", type=int, default=300, help="decisions to audit")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--meta-out", type=Path, help="per-move kept positions for every turning decision (jsonl)")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    decisions = {}
    for directory in args.diagnoses:
        for d in load_decisions(directory):
            if is_decisive(d):
                decisions.setdefault(d["fen"], d)
    fens = sorted(decisions)
    sample = random.Random(args.seed).sample(fens, min(args.sample, len(fens)))
    with Pool(args.workers, initializer=_init, initargs=(str(args.net),)) as pool:
        audits = pool.map(_audit, sample, chunksize=1)
        if args.meta_out:
            with open(args.meta_out, "w") as out:
                for rows in pool.imap(_meta, [decisions[f] for f in fens], chunksize=4):
                    out.writelines(json.dumps(r) + "\n" for r in rows)
    done = [a for a in audits if a is not None]
    report = summarize(done, skipped=len(audits) - len(done))
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), file=sys.stderr)


if __name__ == "__main__":
    main()
