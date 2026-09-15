#!/bin/zsh
# Matched h1024 controls on the fresh confirmation openings (fixed; 0 overlap with dev and official).
# Run once; the fresh set is not used for any tuning afterwards.
T=/Users/nikhildsouza/workspace/chess-challenge/training
cd $T
for run in jj6-h1024-s1000-cp20 rank-b-h1024-w0-seed1; do
  [[ -f $T/runs/$run/fresh_100_summary.json ]] && continue
  uv run --project $T python -m chess_nn.evaluate $T/runs/$run/model.onnx --levels 1 2 3 \
      --openings $T/data/fresh_openings_100.txt > $T/runs/$run/fresh_100_eval.log 2>&1 || exit 1
done
echo "controls on fresh openings: done"
