#!/bin/zsh
# tact_a: the h1024 recipe plus every legal move's quiescence leaf at 20,000 sampled winning decisions
# (best >= +300) from training-source diagnoses, labeled like broad data (sweeps/tact_a.json); seed 0,
# paired with jj6-h1024-s1000-cp20, which differs only in data. Steps run one at a time and skip or resume
# finished work, so rerunning continues after an interruption. Exit status lands in runs/tact_a_pipeline.status.
T=/Users/nikhildsouza/workspace/chess-challenge/training
R=$T/runs/tact-a-h1024
C=$T/runs/jj6-h1024-s1000-cp20
L=$T/data/tactics_a.log
cd $T
set -e
trap 'echo $? > $T/runs/tact_a_pipeline.status' EXIT
rm -f $T/runs/tact_a_pipeline.status

# 1. Collect (decision_data writes only once every worker finishes, so write aside and rename).
if [[ ! -f $T/data/tactics_candidates_a.txt ]]; then
  uv run --project $T python -m chess_nn.decision_data $T/data/diagnosis_h1024_L2 $T/data/diagnosis_h1024_L3 \
      $T/data/diagnosis_b_h1024 $T/data/diagnosis_b_kb256 --net $C/model.onnx \
      --exclude $T/data/diagnosis_heldout_h1024 $T/data/diagnosis_fresh_h1024 \
      --out $T/data/tactics_candidates_a.part.txt --select winning --min-cp 300 --sample 20000 --seed 0 --workers 8 >> $L 2>&1
  mv $T/data/tactics_candidates_a.part.txt $T/data/tactics_candidates_a.txt
fi

# 2. Label (skips finished shards).
mkdir -p $T/data/shards_tactics
uv run --project $T python -m chess_nn.label --candidates $T/data/tactics_candidates_a.txt --out-dir $T/data/shards_tactics \
    --prefix shard_tactics_a --depth 9 --workers 10 --record $T/data/tactics_candidates_a_labels.jsonl >> $L 2>&1

# 3. Train and play dev L1-L3 (skips finished runs and levels).
uv run --project $T python -m chess_nn.sweep $T/sweeps/tact_a.json >> $T/runs/tact_a_sweep.log 2>&1

# 4. Paired dev games against the matched control.
uv run --project $T python -m chess_nn.paired games $R=$C --tag dev_100 > $T/runs/tact_a_paired_dev.json 2>> $L

# 5. Server-equivalent start-position games (measurement only).
[[ -f $R/startpos_summary.json ]] || uv run --project $T python -m chess_nn.evaluate $R/model.onnx --levels 1 2 3 \
    --openings $T/data/startpos_openings.txt > $R/startpos_eval.log 2>&1

# 6. Stockfish depth 12 with history on the candidate's L2 games it did not win (same as the control's and conv_a's).
D=$T/data/diagnosis_dev100_L2_tact-a-h1024
if [[ ! -f $D/summary.json ]]; then
  mkdir -p $D
  uv run --project $T python - $R/dev_100_L2.json $D/games.jsonl <<'EOF'
import json, sys
games = json.load(open(sys.argv[1]))["levels"][0]["games"]
with open(sys.argv[2], "w") as out:
    for g in games:
        net_white = g["color"] == "white"
        result = "1/2-1/2" if g["points"] == 0.5 else ("1-0" if (g["points"] == 1.0) == net_white else "0-1")
        out.write(json.dumps({"start_fen": g["starting_fen"], "moves": [m["uci"] for m in g["moves"]],
                              "net_white": net_white, "result": result, "end": g["reason"], "level": 2}) + "\n")
EOF
  uv run --project $T python -m chess_nn.diagnose $D/games.jsonl --out $D --depth 12 --workers 10 > $D.log 2>&1
fi
echo "tact_a pipeline: done" >> $L
