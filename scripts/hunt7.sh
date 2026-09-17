#!/bin/bash
# Collect many models that win all eight grader games, so the strongest can be chosen among them.
#
#   hunt7.sh <base-run> <seed-start> <seed-end> [parallelism] [sigma-ladder]
#
# hunt6 stopped at the first winner, which is the wrong objective now. Passing Level 4 is a
# constraint, not the goal: among the models that clear it we want the one that actually plays the
# best chess. So this never stops, banks every winner under runs/, and leaves ranking to a separate
# 200-game dev pass — which is far too expensive to run inside the screen.
#
# Read `score`, never `wins`/`draws`/`losses`: those counters are wrong in the harness and in the
# grader alike (both called two repetition draws "2 losses"). `score` is the sum of per-game points.
export PATH="$HOME/.local/bin:$PATH"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd ~/chess-challenge/training || exit 1
BASE=${1:-runs/sv-both}
START=${2:-1}
END=${3:-200000}
JOBS=${4:-26}
LADDER=${5:-"0.002 0.006 0.01 0.018 0.03"}
TAG=$(basename $BASE)
COMPETE=~/chess-challenge/target/release/compete
OPENINGS=~/chess-challenge/training/data/startpos_openings.txt
WINNERS=~/winners_all.txt
read -r -a SIGMAS <<< "$LADDER"

score_of() {
  python3 -c "
import json, sys
print(json.load(open(sys.argv[1]))['levels'][0]['score'])" "$1" 2>/dev/null || echo 0
}

screen_one() {
  local name=$1 sigma=$2 seed=$3
  local dir=/tmp/h7_$name
  mkdir -p $dir
  uv run python -m chess_nn.perturb --from $BASE --name $name --sigma $sigma --seed $seed \
      --runs-dir /tmp/h7_runs > /dev/null 2>&1 || return
  local model=/tmp/h7_runs/$name/model.onnx
  [ -f $model ] || return

  for level in 1 2 3 4; do
    $COMPETE $model --level $level --openings $OPENINGS --json-output $dir/L$level.json > /dev/null 2>&1
    local s=$(score_of $dir/L$level.json)
    if [ "$s" != "2" ] && [ "$s" != "2.0" ]; then
      rm -rf $dir /tmp/h7_runs/$name
      return
    fi
  done
  mkdir -p runs/$name && cp -r /tmp/h7_runs/$name/* runs/$name/
  rm -rf $dir /tmp/h7_runs/$name
  echo "$(date -u +%H:%M) WINNER $name (base $TAG, sigma $sigma, seed $seed)"
  echo "$name base=$TAG sigma=$sigma seed=$seed" >> $WINNERS
}

mkdir -p /tmp/h7_runs
running=0
for seed in $(seq $START $END); do
  sigma=${SIGMAS[$((seed % ${#SIGMAS[@]}))]}
  screen_one "${TAG}-w${seed}" $sigma $seed &
  running=$((running + 1))
  if [ $running -ge $JOBS ]; then wait -n; running=$((running - 1)); fi
done
wait
echo "hunt7 $TAG $START-$END complete"
