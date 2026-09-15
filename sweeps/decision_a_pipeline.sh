#!/bin/zsh
# decision_a, one bounded test of targeted data: the h1024 recipe with the decision shard added
# (sweeps/decision_a.json), two seeds, each paired with the control that differs only in data.
# Steps run one at a time so match evaluation never overlaps training.
T=/Users/nikhildsouza/workspace/chess-challenge/training
cd $T
set -e

# 1. Train and evaluate on the dev openings (L1-L3).
uv run --project $T python -m chess_nn.sweep $T/sweeps/decision_a.json > $T/runs/decision_a_sweep.log 2>&1

# 2. Fresh confirmation openings for the candidates (controls: decision_a_controls_fresh.sh).
for run in dec-a-h1024 dec-a-h1024-seed1; do
  [[ -f $T/runs/$run/fresh_100_summary.json ]] && continue
  uv run --project $T python -m chess_nn.evaluate $T/runs/$run/model.onnx --levels 1 2 3 \
      --openings $T/data/fresh_openings_100.txt > $T/runs/$run/fresh_100_eval.log 2>&1
done

# 3. Move quality on the clean fresh decisions (h1024 games with training overlap removed).
uv run --project $T python -m chess_nn.move_quality $T/data/diagnosis_fresh_h1024_clean --sample 1500 --depth 12 --workers 8 \
    --runs $T/runs/jj6-h1024-s1000-cp20 $T/runs/rank-b-h1024-w0-seed1 $T/runs/dec-a-h1024 $T/runs/dec-a-h1024-seed1 \
    > $T/runs/decision_a_move_quality.log 2>&1

# 4. Paired comparisons.
PAIRS=($T/runs/dec-a-h1024=$T/runs/jj6-h1024-s1000-cp20 $T/runs/dec-a-h1024-seed1=$T/runs/rank-b-h1024-w0-seed1)
uv run --project $T python -m chess_nn.paired games $PAIRS --tag dev_100 > $T/runs/decision_a_paired_dev.json
uv run --project $T python -m chess_nn.paired games $PAIRS --tag fresh_100 > $T/runs/decision_a_paired_fresh.json
uv run --project $T python -m chess_nn.paired decisions $PAIRS --quality diagnosis_fresh_h1024_clean > $T/runs/decision_a_paired_decisions.txt
echo "decision_a pipeline: done"
