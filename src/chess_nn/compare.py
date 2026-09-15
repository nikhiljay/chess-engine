"""One comparison table across runs.

Per run: Level 1-3 scores and W/D/L on the dev openings; move quality on
sampled held-out decisions (chess_nn.move_quality: agreement, mean loss,
turning moves); mate-in-one allowances through the actual search, split into
defensible and already-lost positions (chess_nn.mate_in_one evaluate);
held-out pair accuracy by kind with counts (chess_nn.pair_report) and after
the search (chess_nn.search_check); and inference time — ONNX Runtime on one
CPU thread per position, and play-move wall time per move including model load.

Usage:
  uv run python -m chess_nn.compare --runs runs/jj6-h1024-s1000-cp20 runs/cap-h8192-s1000-cp20 \\
      --bench data/mate_bench_heldout.jsonl --pairs data/rank_pairs_b_val \\
      --search data/rank_pairs_b_val_sample --quality diagnosis_heldout_h1024 --out data/comparison.md
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import chess
import onnxruntime as ort

from chess_nn.encoding import dense_numpy, encode_sparse
from chess_nn.play import net_move


def read_json(path: Path) -> dict | list | None:
    return json.loads(path.read_text()) if path.exists() else None


def inference_times(model: Path, fens: list[str], repeats: int = 3) -> dict:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(model), options, providers=["CPUExecutionProvider"])
    dense = dense_numpy(encode_sparse([chess.Board(f) for f in fens]))
    session.run(None, {"board": dense[:1]})
    single, batched = [], []
    for _ in range(repeats):
        start = time.perf_counter()
        for row in dense:
            session.run(None, {"board": row[None, :]})
        single.append((time.perf_counter() - start) / len(dense))
        start = time.perf_counter()
        for i in range(0, len(dense), 32):
            session.run(None, {"board": dense[i : i + 32]})
        batched.append((time.perf_counter() - start) / len(dense))
    moves = []
    for fen in fens[:20]:
        start = time.perf_counter()
        net_move(model, chess.Board(fen))
        moves.append(time.perf_counter() - start)
    return {
        "ort_ms_per_position_batch1": round(min(single) * 1000, 3),
        "ort_ms_per_position_batch32": round(min(batched) * 1000, 3),
        "play_move_ms_median": round(statistics.median(moves) * 1000, 1),
    }


def row_for(run: Path, args: argparse.Namespace, fens: list[str]) -> dict:
    config = read_json(run / "config.json") or {}
    summary = read_json(run / "summary.json") or {}
    metrics = read_json(run / "metrics.json") or []
    levels = {f"L{l['level']}": l for l in read_json(run / "dev_100_summary.json") or []}
    return {
        "run": run.name,
        "params": summary.get("params"),
        "hidden": config.get("hidden"),
        "threat_hidden": config.get("threat_hidden", 0),
        "rank_weight": config.get("rank_weight", 0.0),
        "seed": config.get("seed", 0),
        "train_minutes": round(sum(m["seconds"] for m in metrics) / 60, 1) if metrics else None,
        "val_loss": metrics[-1]["val_loss"] if metrics else None,
        "levels": {k: {"score_pct": v["score_pct"], "wdl": f"{v['wins']}/{v['draws']}/{v['losses']}"} for k, v in levels.items()},
        "move_quality": read_json(run / f"move_quality_{args.quality}.json") if args.quality else None,
        "mate_bench": read_json(run / f"mate_bench_{args.bench.stem}.json") if args.bench else None,
        "pair_report": read_json(run / f"pair_report_{args.pairs.stem}.json") if args.pairs else None,
        "search_check": read_json(run / f"search_check_{args.search.stem}.json") if args.search else None,
        "inference": inference_times(run / "model.onnx", fens),
    }


def markdown(rows: list[dict]) -> str:
    def level(r: dict, k: str) -> str:
        v = r["levels"].get(k)
        return f"{v['score_pct']:.1f}% ({v['wdl']})" if v else "—"

    def quality(r: dict) -> str:
        q = r["move_quality"]
        return f"{q['agreement']:.3f} · {q['mean_loss_cp']:.0f} cp · {q['turning_moves']}/{q['defensible_decisions']}" if q else "—"

    def mate(r: dict) -> str:
        m = r["mate_bench"]
        if not m:
            return "—"
        return f"{m['allowed_unseen_mate_when_savable']}/{m['savable_positions']} · {m.get('allowed_unseen_mate_when_lost', '?')}/{m.get('lost_positions', '?')}"

    def pairs(r: dict) -> str:
        p, s = r["pair_report"], r["search_check"] or {}
        if not p:
            return "—"
        cells = []
        for short, key in (("mate", "mate-in-one rule"), ("tact", "tactical score gap (>= 300 cp)"), ("mod", "moderate score gap (100-299 cp)"), ("rep", "repetition rule")):
            if p.get(key, {}).get("pairs"):
                cells.append(f"{short} {p[key]['accuracy']:.2f} (n={p[key]['pairs']})")
        if s.get("all"):
            cells.append(f"after search {s['all']['search_accuracy']:.2f}; picks better {s['all']['chose_better']:.0%} / worse {s['all']['chose_worse']:.0%}")
        return "; ".join(cells)

    lines = [
        "| run | params | train min | L1 | L2 | L3 | move quality: agree · loss · turning | unseen mate allowed: defensible · lost | held-out pairs (static by kind; after search) | ms/pos ORT b1 / b32 · play-move ms |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        inf = r["inference"]
        params = f"{r['params']:,}" if r["params"] else "—"
        lines.append(
            f"| {r['run']} | {params} | {r['train_minutes']} | {level(r, 'L1')} | {level(r, 'L2')} | {level(r, 'L3')} | {quality(r)} | {mate(r)} | {pairs(r)} | "
            f"{inf['ort_ms_per_position_batch1']} / {inf['ort_ms_per_position_batch32']} · {inf['play_move_ms_median']} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--bench", type=Path, help="mate_in_one benchmark .jsonl (also the timing positions)")
    parser.add_argument("--pairs", type=Path, help="pair_report input prefix, e.g. data/rank_pairs_b_val")
    parser.add_argument("--search", type=Path, help="search_check input prefix, e.g. data/rank_pairs_b_val_sample")
    parser.add_argument("--quality", help="move_quality diagnosis name, e.g. diagnosis_heldout_h1024")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    source = args.bench if args.bench and args.bench.exists() else None
    fens = [json.loads(l)["fen"] for l in source.read_text().splitlines()[:256]] if source else [chess.STARTING_FEN] * 64
    rows = [row_for(run, args, fens) for run in args.runs]
    table = markdown(rows)
    print(table)
    if args.out:
        args.out.write_text(table + "\n\n```json\n" + json.dumps(rows, indent=2) + "\n```\n")


if __name__ == "__main__":
    main()
