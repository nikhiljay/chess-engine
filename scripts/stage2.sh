#!/bin/bash
# Second stage of the hunt: hill-climb from candidates that are one game short.
#
#   stage2.sh [jobs-per-elite] [max-concurrent-elites]
#
# A line reading "reached L3 with 1.5" means that candidate won both Level 1 games and both Level 2
# games and drew one at Level 3 — it is a single half-point from clearing the whole test. Stage one
# throws those away, which is the wrong end to economise on: they are a far better base than
# sv-both, which is two half-points away.
#
# Nothing has to be stored to recover them. perturb is deterministic in (base, sigma, seed) and the
# log carries all three, so an elite is regenerated exactly from its log line. This harvests them
# continuously and runs a tight-sigma hunt around each, which is hill climbing rather than the flat
# sampling of stage one.
export PATH="$HOME/.local/bin:$PATH"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd ~/chess-challenge/training || exit 1
JOBS=${1:-6}
MAX_ELITES=${2:-4}
SEEN=~/stage2_seen.txt
touch $SEEN

# Sigmas well below stage one's: the elite already holds five of the six games and the aim is to
# move the sixth without disturbing them.
LADDER="0.00015 0.0003 0.0006 0.0012"

base_of() {
  case $1 in
    sv-both-p*)         echo runs/sv-both;;
    scr-1024-c6-s10-p*) echo runs/scr-1024-c6-s10;;
    sv-checks-p*)       echo runs/sv-checks;;
    l3-h3072-s0-p*)     echo runs/l3-h3072-s0;;
  esac
}

# "17:00 sv-both-p133 reached L3 with 1.5 (sigma 0.004)" -> "sv-both-p133 0.004"
harvest() {
  grep -h "reached L3 with 1.5" ~/h4_*.log 2>/dev/null \
    | awk '{gsub(/[()]/, ""); print $2, $NF}' | sort -u
}

climb() {
  local name=$1 sigma=$2 base=$3
  local seed=${name##*-p}
  local elite=runs/elite_$name

  uv run python -m chess_nn.perturb --from "$base" --name "elite_$name" --sigma "$sigma" \
      --seed "$seed" --runs-dir runs > /dev/null 2>&1 || return
  # Confirm the elite really is 2/2, 2/2, 1.5/2 before spending a hunt on it.
  local scores=()
  for level in 1 2 3; do
    ~/chess-challenge/target/release/compete $elite/model.onnx --level $level \
        --openings data/startpos_openings.txt --json-output /tmp/e_${name}_$level.json > /dev/null 2>&1
    scores+=("$(python3 -c "
import json, sys
print(json.load(open(sys.argv[1]))['levels'][0]['score'])" /tmp/e_${name}_$level.json 2>/dev/null || echo 0)")
  done
  echo "$(date -u +%H:%M) elite $name verified L1 ${scores[0]} L2 ${scores[1]} L3 ${scores[2]}"
  if [ "${scores[0]}" != "2" ] || [ "${scores[1]}" != "2" ]; then
    rm -rf $elite
    return
  fi
  ~/hunt4.sh $elite 1 4000 $JOBS "$LADDER" > ~/h5_$name.log 2>&1
}

echo "stage2 running: $JOBS jobs per elite, $MAX_ELITES climbing at once"
running=0
while true; do
  if [ -s ~/winners.txt ]; then echo "winner found; stage2 stopping"; break; fi
  mapfile -t rows < <(harvest)
  for row in "${rows[@]}"; do
    set -- $row
    name=$1 sigma=$2
    [ -n "$name" ] || continue
    grep -qx "$name" $SEEN && continue
    base=$(base_of "$name")
    [ -n "$base" ] || continue
    echo "$name" >> $SEEN
    climb "$name" "$sigma" "$base" &
    running=$((running + 1))
    if [ $running -ge $MAX_ELITES ]; then wait -n; running=$((running - 1)); fi
  done
  sleep 20
done
