"""Mate-in-one blind spots of the challenge's search, and a held-out benchmark.

The harness searches every legal move to depth 1, then follows only captures,
promotions and castling (its board library encodes castling as the king taking
its own rook) — or every move when in check. A mate the opponent gives with
any other move, like 33. Rh8# after 32...Nxe4 in the diagnostic game, is never
examined from a position that is not in check, so avoiding it rests on the
static evaluation alone.

build: scan recorded games (datagen --record) for positions where the net's
move allowed such an unseen mate while some move allowing no mate in one
existed. Stockfish then scores the best of those safe moves, separating
positions where avoiding mate also saves the game (not clearly lost) from ones
where it only delays the loss. Positions that appear in the given diagnosis
directories — the sources of ranking pairs — are excluded, so the benchmark
stays held out.

evaluate: for each run, take the move the harness's play-move binary chooses in
every benchmark position and report how often it still allows mate in one.

Usage:
  uv run python -m chess_nn.mate_in_one build data/bench_games_*.jsonl --out data/mate_bench.jsonl \\
      --exclude data/diagnosis_h1024_L2 data/diagnosis_h1024_L3 data/diagnosis_b_h1024 data/diagnosis_b_kb256
  uv run python -m chess_nn.mate_in_one evaluate data/mate_bench.jsonl runs/rank-a-h512-w0 runs/threat-a-h512-t64-w0
"""

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import chess

from chess_nn.diagnose import WIN_CP, search
from chess_nn.label import Stockfish
from chess_nn.play import net_move


def searched_reply(board: chess.Board, move: chess.Move) -> bool:
    """Whether the harness's quiescence search examines `move` in a position that is not in check.

    Its capture list is moves onto an occupied square (castling lands on the
    rook in its encoding) or promotions; en passant lands on an empty square.
    """
    return board.piece_at(move.to_square) is not None or move.promotion is not None or board.is_castling(move)


def mating_replies(board: chess.Board) -> list[chess.Move]:
    replies = []
    for reply in board.legal_moves:
        if not board.gives_check(reply):  # only a checking move can mate; skips most full checkmate tests
            continue
        board.push(reply)
        mate = board.is_checkmate()
        board.pop()
        if mate:
            replies.append(reply)
    return replies


def allows_mate(board: chess.Board, move: chess.Move) -> bool:
    """Whether the opponent has any mate in one after `move`."""
    after = board.copy(stack=False)
    after.push(move)
    return not after.is_game_over() and bool(mating_replies(after))


def unseen_mate(board: chess.Board, move: chess.Move) -> chess.Move | None:
    """The opponent's mate in one after `move`, if the harness search examines none of the mating replies.

    If the move gives check, the search examines every reply; if any mating
    reply is a capture, promotion or castling, it sees the mate and the net
    scores the move as lost.
    """
    after = board.copy(stack=False)
    after.push(move)
    if after.is_game_over() or after.is_check():
        return None
    mates = mating_replies(after)
    if not mates or any(searched_reply(after, m) for m in mates):
        return None
    return mates[0]


def safe_moves(board: chess.Board) -> list[chess.Move]:
    """Legal moves after which the opponent has no mate in one."""
    return [m for m in board.legal_moves if not allows_mate(board, m)]


def position_key(fen: str) -> str:
    return " ".join(fen.split()[:4])


def scan_game(args: tuple[dict, str]) -> list[dict]:
    game, source = args
    board = chess.Board(game["start_fen"])
    net_color = chess.WHITE if game["net_white"] else chess.BLACK
    found = []
    for ply, raw in enumerate(game["moves"]):
        move = board.parse_uci(raw)  # the harness writes castling as king-takes-rook
        if board.turn == net_color:
            mate = unseen_mate(board, move)
            if mate is not None:
                safe = safe_moves(board)
                if safe:
                    found.append({
                        "fen": board.fen(),
                        "played": move.uci(),
                        "mate": mate.uci(),
                        "safe": [m.uci() for m in safe],
                        "ply": ply,
                        "level": game.get("level"),
                        "source": source,
                    })
        board.push(move)
    return found


_engine: Stockfish | None = None


def _init_engine() -> None:
    global _engine
    _engine = Stockfish(hash_mb=64)


def score_safe_moves(args: tuple[dict, int]) -> dict:
    item, depth = args
    cp, best = search(_engine, f"position fen {item['fen']}", depth, " ".join(item["safe"]))
    return {**item, "best_safe": best, "best_safe_cp": cp, "saves_game": cp > -WIN_CP}


def _choose(args: tuple[str, str]) -> str:
    model, fen = args
    return net_move(Path(model), chess.Board(fen)).uci()


def build(args: argparse.Namespace) -> None:
    excluded = set()
    for directory in args.exclude:
        for line in (directory / "decisions.jsonl").read_text().splitlines():
            if line.strip():
                excluded.add(position_key(json.loads(line)["fen"]))
    jobs = []
    for path in args.games:
        for i, line in enumerate(path.read_text().splitlines()):
            if line.strip():
                jobs.append((json.loads(line), f"{path.stem}:{i}"))
    start = time.time()
    with Pool(args.workers) as pool:
        found = [item for items in pool.imap_unordered(scan_game, jobs, chunksize=16) for item in items]
    items, seen, dropped = [], set(), 0
    for item in found:
        key = position_key(item["fen"])
        if key in excluded:
            dropped += 1
        elif key not in seen:
            seen.add(key)
            items.append(item)
    print(f"{len(jobs):,} games → {len(found):,} positions allowing an unseen mate with a safe alternative; "
          f"{dropped} excluded as ranking-pair sources, {len(items):,} unique kept ({time.time() - start:.0f}s)", file=sys.stderr)
    with Pool(args.workers, initializer=_init_engine) as pool:
        scored = pool.map(score_safe_moves, [(item, args.depth) for item in items])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(item) + "\n" for item in scored))
    saves = sum(item["saves_game"] for item in scored)
    print(f"{len(scored):,} benchmark positions; the best safe move saves the game (Stockfish > {-WIN_CP} cp) in {saves:,}")


def evaluate(args: argparse.Namespace) -> None:
    for bench in args.bench:
        items = [json.loads(line) for line in bench.read_text().splitlines() if line.strip()]
        for run in args.runs:
            with Pool(args.workers) as pool:
                chosen = pool.map(_choose, [(str(run / "model.onnx"), item["fen"]) for item in items])
            rows = []
            for item, uci in zip(items, chosen):
                board = chess.Board(item["fen"])
                move = chess.Move.from_uci(uci)
                rows.append({
                    "fen": item["fen"],
                    "chosen": uci,
                    "allows_unseen_mate": unseen_mate(board, move) is not None,
                    "allows_mate": allows_mate(board, move),
                    "saves_game": item.get("saves_game"),
                })
            savable = [r for r in rows if r["saves_game"]]
            lost = [r for r in rows if r["saves_game"] is False]
            report = {
                "bench": bench.name,
                "positions": len(rows),
                "allowed_unseen_mate": sum(r["allows_unseen_mate"] for r in rows),
                "allowed_any_mate": sum(r["allows_mate"] for r in rows),
                "savable_positions": len(savable),
                "allowed_unseen_mate_when_savable": sum(r["allows_unseen_mate"] for r in savable),
                "lost_positions": len(lost),
                "allowed_unseen_mate_when_lost": sum(r["allows_unseen_mate"] for r in lost),
            }
            report["allowed_unseen_mate_rate"] = round(report["allowed_unseen_mate"] / max(len(rows), 1), 3)
            report["allowed_unseen_mate_rate_when_savable"] = round(report["allowed_unseen_mate_when_savable"] / max(len(savable), 1), 3)
            report["allowed_unseen_mate_rate_when_lost"] = round(report["allowed_unseen_mate_when_lost"] / max(len(lost), 1), 3)
            (run / f"mate_bench_{bench.stem}.json").write_text(json.dumps({**report, "rows": rows}, indent=2))
            print(f"{run.name:34s} {bench.stem}: allowed unseen mate {report['allowed_unseen_mate']}/{len(rows)} "
                  f"({report['allowed_unseen_mate_rate']:.1%}); when savable {report['allowed_unseen_mate_when_savable']}/{len(savable)} "
                  f"({report['allowed_unseen_mate_rate_when_savable']:.1%}); any mate {report['allowed_any_mate']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    b = commands.add_parser("build", help="scan recorded games into a benchmark")
    b.add_argument("games", type=Path, nargs="+")
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--exclude", type=Path, nargs="*", default=[], help="diagnosis directories whose positions to exclude")
    b.add_argument("--depth", type=int, default=16)
    b.add_argument("--workers", type=int, default=8)
    e = commands.add_parser("evaluate", help="measure runs' move choices on benchmark files")
    e.add_argument("bench", type=Path, nargs="+", help="benchmark .jsonl files, then --runs")
    e.add_argument("--runs", type=Path, nargs="+", required=True)
    e.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    build(args) if args.command == "build" else evaluate(args)


if __name__ == "__main__":
    main()
