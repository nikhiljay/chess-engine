#!/bin/zsh
# Does a different function class move the number that is stuck?
#
# Finding 71: in level positions every NNUE-style net here correlates ~0.35 with d9 against
# Stockfish static's 0.561, and that number does not move with data (0.337 at 25% -> 0.347 at
# 100%) or capacity (0.359 at h8192). A convolution sees a square with its neighbours and
# shares weights across the board, which is the bias tactics want.
#
# Like-for-like: the same 2.18M positions the MLP half-data probe used (scale-h1-h1024, level
# r 0.345), so any difference is the architecture. Judged on leaf accuracy first; play follows
# only if this moves.
T=/Users/nikhildsouza/workspace/chess-challenge/training
cd $T
set -e
trap 'echo $? > $T/runs/cnn_probe.status' EXIT
rm -f $T/runs/cnn_probe.status

uv run --project $T python - <<'EOF'
from pathlib import Path
from chess_nn.train import TrainConfig, run_training
from chess_nn.sweep import expand_shards

shards = expand_shards(["data/shards/shard_000[0-3].npz", "data/shards/shard_2017-07_000[0-1].npz"])
if not (Path("runs/cnn-probe-c32b6/summary.json")).exists():
    run_training(TrainConfig(name="cnn-probe-c32b6", shards=shards, conv_channels=32, conv_blocks=6,
                             epochs=10, batch_size=8192, wdl_scale_cp=1000, cp_loss_weight=0.2,
                             seed=0, limit=2_180_876))
EOF

uv run --project $T python -m chess_nn.leaf_error $T/runs/teacher/leaves_d9_dev5.jsonl \
    $T/runs/cnn-probe-c32b6 $T/runs/scale-h1-h1024 $T/runs/jj6-h1024-s1000-cp20 | grep -E '^[a-z]|level \('
echo "cnn probe: done"
