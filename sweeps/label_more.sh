#!/bin/zsh
# Label more of the already-sampled candidates, to roughly double the training mix.
#
# Every model so far plateaus at the same leaf error (MAE 3.21-3.31 against d9) across an 8x
# range of parameters, which is the signature of a data or label limit rather than a capacity
# one. June shards 0-3 and July 0-1 are labeled; this continues where labeling stopped, with
# the same plain depth 9 the broad mix used, so the only change is quantity.
#
# Labeling skips finished shards, so this is resumable. Runs at low priority (nice) and with
# 6 workers so a training or evaluation run keeps its cores.
T=/Users/nikhildsouza/workspace/chess-challenge/training
cd $T
set -e
trap 'echo $? > $T/data/label_more.status' EXIT
rm -f $T/data/label_more.status

nice -n 10 uv run --project $T python -m chess_nn.label --candidates $T/data/candidates.txt \
    --depth 9 --workers 6 >> $T/data/label_june_more.log 2>&1
echo "June labeling: done"

nice -n 10 uv run --project $T python -m chess_nn.label --candidates $T/data/candidates_2017-07.txt \
    --prefix shard_2017-07 --depth 9 --workers 6 >> $T/data/label_july_more.log 2>&1
echo "July labeling: done"
