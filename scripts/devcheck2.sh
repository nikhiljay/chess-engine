#!/bin/bash
# Dev-strength pass over any set of models, on the 100 dev openings (200 games per level).
#
#   devcheck2.sh <model-run-name> [more...]
#
# Round two asks a different question from round one. Several models were discarded because they
# lost the grader's start-position games, but those are eight deterministic games and say little
# about general strength — the wide nets in particular (h3072, h8192) were never measured properly.
# If one of them is a genuinely stronger engine, it is the better base to hunt around, since passing
# Level 4 is a constraint we can search for while strength is the thing we actually want.
export PATH="$HOME/.local/bin:$PATH"
cd ~/chess-challenge/training || exit 1
COMPETE=~/chess-challenge/target/release/compete
OPENINGS=data/dev_openings_100.txt
OUT=~/devcheck
mkdir -p $OUT

for model in "$@"; do
  [ -f runs/$model/model.onnx ] || { echo "missing runs/$model"; continue; }
  for level in 1 2 3 4; do
    [ -s $OUT/${model}_L${level}.json ] && continue   # already measured
    $COMPETE runs/$model/model.onnx --level $level --openings $OPENINGS \
        --json-output $OUT/${model}_L${level}.json > /dev/null 2>&1 &
  done
done
wait
echo "devcheck2 complete: $*"
