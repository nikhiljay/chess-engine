#!/bin/zsh
# leaf_a: train on the distribution the search actually evaluates.
#
# The net trains on quiet human positions but is deployed on the leaves its own search reaches,
# and findings 68/71 measured the gap: on those leaves it is 3x more often badly wrong than
# Stockfish's static eval, and over half of them are winning positions the human mix barely
# contains. tact_a tried targeted data but its leaves landed in the bucket where the net is
# already accurate (48.8% in -10..-3, where MAE is 1.12).
#
# Here the leaves come from 1,400 recorded h1024 games at Levels 2 and 3, taken exactly as the
# search asked about them, with d9 values as targets under the usual transform. Openings are
# datagen's random plies, so they do not overlap the dev set.
# Exit status lands in runs/leaf_a_pipeline.status.
T=/Users/nikhildsouza/workspace/chess-challenge/training
C=$T/runs/jj6-h1024-s1000-cp20
cd $T
set -e
trap 'echo $? > $T/runs/leaf_a_pipeline.status' EXIT
rm -f $T/runs/leaf_a_pipeline.status

# 1. Recorded games in the replay format the teacher tool reads.
if [[ ! -s $T/runs/teacher/train_games.txt ]]; then
  uv run --project $T python - <<'EOF'
import json
from pathlib import Path
root = Path("/Users/nikhildsouza/workspace/chess-challenge/training")
lines = []
for name in ["bench_games_h1024_L2.jsonl", "bench_games_h1024_L3.jsonl"]:
    for line in (root / "data" / name).read_text().splitlines():
        if not line.strip():
            continue
        game = json.loads(line)
        colour = "white" if game["net_white"] else "black"
        lines.append(f"{game['level']} {colour} {game['start_fen']}|{' '.join(game['moves'])}")
(root / "runs/teacher/train_games.txt").write_text("\n".join(lines) + "\n")
print(f"{len(lines)} recorded games")
EOF
fi

# 2. The leaves those searches evaluated, with d9 values (hash kept between queries, as
#    chess_nn.label does, for speed; these are labels, not measurements).
if [[ ! -s $T/runs/teacher/leaves_train.jsonl ]]; then
  $T/datagen/target/release/teacher leaves --model $C/model.onnx --teacher d9 --keep-hash \
      --sample 1500000 --seed 0 --threads 6 \
      < $T/runs/teacher/train_games.txt > $T/runs/teacher/leaves_train.jsonl 2> $T/runs/teacher/leaves_train.log
fi
tail -2 $T/runs/teacher/leaves_train.log

# 3. Shard, then train with the broad mix plus those leaves.
if [[ ! -d $T/data/shards_leaf ]] || [[ -z "$(ls -A $T/data/shards_leaf 2>/dev/null)" ]]; then
  uv run --project $T python -m chess_nn.leaf_shard $T/runs/teacher/leaves_train.jsonl \
      --out-dir $T/data/shards_leaf --prefix shard_leaf_a
fi

uv run --project $T python -m chess_nn.sweep $T/sweeps/leaf_a.json > $T/runs/leaf_a_sweep.log 2>&1
grep -E '^\| leaf-a' $T/runs/leaf_a_sweep.log | tail -2

# 4. Start-position games, separately from match strength.
[[ -f $T/runs/leaf-a-h1024/startpos_summary.json ]] || uv run --project $T python -m chess_nn.evaluate \
    $T/runs/leaf-a-h1024/model.onnx --levels 1 2 3 --openings $T/data/startpos_openings.txt \
    > $T/runs/leaf-a-h1024/startpos_eval.log 2>&1

# 5. Paired against the standing default, plus the leaf-error mechanism check.
echo "== leaf-a-h1024 vs jj6-h1024-s1000-cp20"
uv run --project $T python -m chess_nn.paired games $T/runs/leaf-a-h1024=$C --tag dev_100 --levels 1 2 3 | python3 -c "
import json, sys
for level, v in json.load(sys.stdin).items():
    p = v['pairs'][0]; c, k, pr = p['candidate_outcomes'], p['control_outcomes'], p['paired']
    print(' ', level, 'cand', c['score_pct'], c['wdl'], 'rep', c['repetition_draws'], 'rep-win', c['repetition_while_winning'],
          '| ctrl', k['score_pct'], k['wdl'], 'rep', k['repetition_draws'], 'rep-win', k['repetition_while_winning'],
          '| diff', round(pr['mean_points_per_game'], 4), '+-', round(pr['se'], 4), pr['ci95'],
          'better/worse', pr['openings_better'], pr['openings_worse'])
"
uv run --project $T python -m chess_nn.leaf_error $T/runs/teacher/leaves_d9_dev5.jsonl \
    $T/runs/leaf-a-h1024 $C | grep -E '^[a-z]|level \('

uv run --project $T python - <<'EOF'
import json
from pathlib import Path
root = Path("/Users/nikhildsouza/workspace/chess-challenge/training/runs")
for run in ['leaf-a-h1024', 'jj6-h1024-s1000-cp20']:
    for level in (1, 2, 3):
        path = root / run / f"startpos_L{level}.json"
        if path.exists():
            games = json.loads(path.read_text())["levels"][0]["games"]
            print(run, f"startpos L{level}", sum(g["points"] for g in games), [(g["color"], g["points"], g["reason"]) for g in games])
EOF
echo "leaf_a pipeline: done"
