#!/bin/zsh
# leaf_b: the same leaf distribution as leaf_a, but with the quiet-best-move filter.
#
# leaf_a trained on every position the search evaluates, labeled with d9. Leaf accuracy improved
# more than anything else tried (MAE 3.29 -> 2.58, level r 0.347 -> 0.395) and play collapsed
# (L2 54.2 -> 33.0). The suspected reason: those targets include positions with tactics pending,
# where d9's score already accounts for the captures that the harness's quiescence plays out
# itself, so the static evaluator is trained to double-count them. The broad mix never had this
# problem because chess_nn.label keeps only positions whose best move is quiet.
#
# This arm puts the leaves through that same filter, separating distribution shift (worth
# fixing) from the double-counting confound (which leaf_a introduced).
T=/Users/nikhildsouza/workspace/chess-challenge/training
cd $T
set -e
trap 'echo $? > $T/runs/leaf_b_pipeline.status' EXIT
rm -f $T/runs/leaf_b_pipeline.status

if [[ ! -s $T/data/leaf_candidates_a.txt ]]; then
  uv run --project $T python - <<'EOF'
import json
from pathlib import Path
root = Path("/Users/nikhildsouza/workspace/chess-challenge/training")
fens = [json.loads(line)["fen"] for line in (root / "runs/teacher/leaves_train.jsonl").read_text().splitlines() if line.strip()]
(root / "data/leaf_candidates_a.txt").write_text("\n".join(fens) + "\n")
print(f"{len(fens):,} leaf candidates")
EOF
fi

uv run --project $T python -m chess_nn.label --candidates $T/data/leaf_candidates_a.txt \
    --out-dir $T/data/shards_leafq --prefix shard_leafq_a --depth 9 --workers 10 \
    >> $T/data/label_leafq.log 2>&1
tail -1 $T/data/label_leafq.log

exec /bin/zsh $T/sweeps/run_experiment.sh $T/sweeps/leaf_b.json leaf-b-h1024
