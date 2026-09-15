"""Where a candidate's new turning moves come from: static child evals or quiescence.

For each matched decision (chess_nn.move_quality rows) where the control's move
was safe and the candidate's move turned a defensible position into a losing
one, this replays the harness search with both nets and records:

  - origin: where the candidate's value for its losing move came from — the
    net's static eval of the immediate child (quiescence stood pat there, so
    the error arose before any capture search), a quiet leaf at the end of a
    capture line (the error arose during quiescence), or a position in check
    (quiescence searched every reply);
  - shift: how much each net's search value moved between control and
    candidate, for the losing move and for the control's safe move — did the
    candidate overrate the losing move or underrate the safe one?
  - refutation: Stockfish's reply to the losing move (with the game history),
    and whether it is in the harness capture list (visible to quiescence) or a
    quiet move no capture search can see.

Usage:
  uv run python -m chess_nn.rank_failure data/diagnosis_heldout_h1024 \\
      runs/rank-b-h1024-w005=runs/jj6-h1024-s1000-cp20 runs/rank-b-h1024-w005-seed1=runs/rank-b-h1024-w0-seed1
"""

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

import chess

from chess_nn.decision_data import load_decisions, quiescence_leaf
from chess_nn.diagnose import search
from chess_nn.label import Stockfish
from chess_nn.paired import parse_pairs, turning
from chess_nn.search_check import NetEval, harness_captures, search_choice


def load_rows(run: Path, quality: str) -> dict[tuple[int, int], dict]:
    lines = (run / f"move_quality_{quality}_rows.jsonl").read_text().splitlines()
    return {(r["game"], r["ply"]): r for r in map(json.loads, filter(str.strip, lines))}


def searched_move(net, board: chess.Board, move: chess.Move) -> dict:
    """The harness value of a root move and where quiescence took it from."""
    board.push(move)
    try:
        child = board.fen()
        value, leaf = quiescence_leaf(net, board, -float("inf"), float("inf"))
        if board.is_check():
            origin = "check"
        elif leaf == child:
            origin = "static child"
        elif leaf is None:
            origin = "terminal"
        else:
            origin = "capture line"
        return {"value": -value, "origin": origin}
    finally:
        board.pop()


def analyse(decision: dict, cand_row: dict, ctrl_row: dict, cand: NetEval, ctrl: NetEval, engine: Stockfish, depth: int) -> dict:
    board = chess.Board(decision["fen"])
    losing, safe = board.parse_uci(cand_row["chosen"]), board.parse_uci(ctrl_row["chosen"])
    cand_losing, ctrl_losing = searched_move(cand, board, losing), searched_move(ctrl, board, losing)
    cand_safe, ctrl_safe = searched_move(cand, board, safe), searched_move(ctrl, board, safe)
    history = decision["history"] + [losing.uci()]
    _, reply = search(engine, f"position fen {decision['start_fen']} moves {' '.join(history)}", depth)
    after = board.copy(stack=False)
    after.push(losing)
    reply_move = after.parse_uci(reply) if reply != "(none)" else None
    return {
        "game": decision["game"],
        "ply": decision["ply"],
        "fen": decision["fen"],
        "losing": losing.uci(),
        "safe": safe.uci(),
        "best_cp": cand_row["best_cp"],
        "losing_cp": cand_row["chosen_cp"],
        "safe_cp": ctrl_row["chosen_cp"],
        "replica_matches_play_move": search_choice(cand, board.copy(stack=False)) == losing,
        "origin": cand_losing["origin"],
        "control_origin": ctrl_losing["origin"],
        "losing_shift": round(cand_losing["value"] - ctrl_losing["value"], 3),
        "safe_shift": round(cand_safe["value"] - ctrl_safe["value"], 3),
        "control_prefers_safe_by": round(ctrl_safe["value"] - ctrl_losing["value"], 3),
        "candidate_prefers_losing_by": round(cand_losing["value"] - cand_safe["value"], 3),
        "refutation": reply,
        "refutation_kind": "none" if reply_move is None
        else "capture list" if reply_move in harness_captures(after)
        else "quiet check" if after.gives_check(reply_move)
        else "quiet",
    }


def summarize(rows: list[dict], matched: int, prevented: int) -> dict:
    cause = Counter()
    for r in rows:
        if r["losing_shift"] > 0 and r["losing_shift"] >= -r["safe_shift"]:
            cause["overrated losing move"] += 1
        else:
            cause["underrated safe move"] += 1
    median = lambda key: round(statistics.median(r[key] for r in rows), 3) if rows else None
    return {
        "matched_decisions": matched,
        "turning_introduced": len(rows),
        "turning_prevented": prevented,
        "origin": dict(Counter(r["origin"] for r in rows)),
        "origin_by_refutation": dict(Counter(f"{r['origin']} / {r['refutation_kind']}" for r in rows)),
        "cause": dict(cause),
        "median_losing_shift_pawns": median("losing_shift"),
        "median_safe_shift_pawns": median("safe_shift"),
        "replica_matches_play_move": sum(r["replica_matches_play_move"] for r in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("diagnosis", type=Path, help="the chess_nn.diagnose directory move_quality sampled")
    parser.add_argument("pairs", nargs="+", help="CANDIDATE_RUN=CONTROL_RUN")
    parser.add_argument("--depth", type=int, default=12)
    args = parser.parse_args()

    quality = args.diagnosis.name
    decisions = {(d["game"], d["ply"]): d for d in load_decisions(args.diagnosis)}
    engine = Stockfish(hash_mb=64)
    for candidate, control in parse_pairs(args.pairs):
        cand_rows, ctrl_rows = load_rows(candidate, quality), load_rows(control, quality)
        keys = [k for k in cand_rows if k in ctrl_rows]
        introduced = [k for k in keys if turning(cand_rows[k]) and not turning(ctrl_rows[k])]
        prevented = sum(turning(ctrl_rows[k]) and not turning(cand_rows[k]) for k in keys)
        cand, ctrl = NetEval(candidate / "model.onnx"), NetEval(control / "model.onnx")
        rows = [analyse(decisions[k], cand_rows[k], ctrl_rows[k], cand, ctrl, engine, args.depth) for k in introduced]
        result = {"control": control.name, "diagnosis": quality, **summarize(rows, len(keys), prevented), "decisions": rows}
        (candidate / f"rank_failure_{quality}.json").write_text(json.dumps(result, indent=2))
        print(candidate.name, "vs", control.name, json.dumps({k: v for k, v in result.items() if k != "decisions"}))
    engine.close()


if __name__ == "__main__":
    main()
