#!/bin/zsh
# Is the data-scaling curve alive? Every model so far lands at leaf MAE 3.21-3.31 against d9
# across an 8x parameter range, and fid_a showed the objective weighting does not move it
# either. Before spending hours labeling more, check whether LESS data is worse: train the same
# recipe on 25% and 50% of the mix and measure leaf error. A curve that still falls between 50%
# and 100% says more data should help; a flat one says the limit is representational, and the
# labeling effort should stop in favour of architecture work.
#
# No game evaluation here: this is a measurement, and play is judged separately.
T=/Users/nikhildsouza/workspace/chess-challenge/training
cd $T
set -e
trap 'echo $? > $T/runs/scale_probe.status' EXIT
rm -f $T/runs/scale_probe.status

uv run --project $T python - <<'EOF'
from pathlib import Path
from chess_nn.train import TrainConfig, run_training
from chess_nn.sweep import expand_shards

shards = expand_shards(["data/shards/shard_000[0-3].npz", "data/shards/shard_2017-07_000[0-1].npz"])
for name, limit in [("scale-q1-h1024", 1_090_438), ("scale-h1-h1024", 2_180_876)]:
    if (Path("runs") / name / "summary.json").exists():
        print(f"{name}: exists, skipping")
        continue
    run_training(TrainConfig(name=name, shards=shards, hidden=1024, head=[32, 32], epochs=20,
                             batch_size=16384, wdl_scale_cp=1000, cp_loss_weight=0.2, seed=0, limit=limit))
EOF

uv run --project $T python -m chess_nn.leaf_error $T/runs/teacher/leaves_d9_dev5.jsonl \
    $T/runs/scale-q1-h1024 $T/runs/scale-h1-h1024 $T/runs/jj6-h1024-s1000-cp20
echo "scale probe: done"
