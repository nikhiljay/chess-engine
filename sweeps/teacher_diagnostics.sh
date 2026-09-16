#!/bin/zsh
# Two bounded diagnostics following finding 67.
#
# 1. leaves — every position the student's own search evaluates across its 40 recorded dev L2
#    games, scored by the d9 teacher. Measures how far the student's values sit from the target
#    it is already trained on, on the positions its search actually uses.
#
# 2. noise — the d9 teacher with position-seeded Gaussian noise added at each leaf, sigma in
#    {0.25, 0.5, 1, 2} pawns, 40 games per arm, paired against the student on the same openings.
#    This measures sensitivity to UNSTRUCTURED perturbation. A network's errors are systematic
#    and correlated across related positions, so a sigma here is not a tolerance for our net.
#
# Steps run one at a time. Exit status lands in runs/teacher/diagnostics.status.
T=/Users/nikhildsouza/workspace/chess-challenge/training
M=$T/runs/jj6-h1024-s1000-cp20
O=$T/runs/teacher
STRIDE=5
cd $T
set -e
trap 'echo $? > $O/diagnostics.status' EXIT
mkdir -p $O
rm -f $O/diagnostics.status

# The student's own recorded games on the pilot's openings (durable copy).
if [[ ! -f $O/student_dev5_L2_games.txt ]]; then
  uv run --project $T python - $M/dev_100_L2.json $O/student_dev5_L2_games.txt $STRIDE <<'EOF'
import json, sys
from pathlib import Path
games = json.loads(Path(sys.argv[1]).read_text())["levels"][0]["games"]
subset = [g for g in games if g["opening_index"] % int(sys.argv[3]) == 0]
Path(sys.argv[2]).write_text("".join(
    f"2 {g['color']} {g['starting_fen']}|{' '.join(m['uci'] for m in g['moves'])}\n" for g in subset))
print(f"{len(subset)} recorded games")
EOF
fi

# 1. Leaves the student's search evaluates, scored by the teacher.
if [[ ! -f $O/leaves_d9_dev5.jsonl ]]; then
  $T/datagen/target/release/teacher leaves --model $M/model.onnx --teacher d9 \
      --sample 20000 --seed 0 --threads 8 \
      < $O/student_dev5_L2_games.txt > $O/leaves_d9_dev5.jsonl 2> $O/leaves_d9_dev5.log
  tail -2 $O/leaves_d9_dev5.log
fi

# 2. Noise sensitivity of the d9 teacher.
for sigma in 0.25 0.5 1 2; do
  arm=noise$sigma
  [[ -f $O/$arm/dev5_L2.json ]] && continue
  mkdir -p $O/$arm
  $T/datagen/target/release/teacher play --model $M/model.onnx --teacher d9 --noise $sigma \
      --openings $T/data/dev_openings_100.txt --stride $STRIDE --level 2 --threads 8 \
      --out $O/$arm/dev5_L2.json > $O/$arm/play.log 2>&1
  tail -1 $O/$arm/play.log
done

# Summaries for chess_nn.paired (repetition_while_winning is left out: it is defined by the
# net's own eval, which means nothing for a Stockfish arm).
for arm in noise0.25 noise0.5 noise1 noise2; do
  uv run --project $T python - $O/$arm/dev5_L2.json $O/$arm/dev5_summary.json <<'EOF'
import json, sys
from pathlib import Path
level = json.loads(Path(sys.argv[1]).read_text())["levels"][0]
games = level["games"]
points = [g["points"] for g in games]
Path(sys.argv[2]).write_text(json.dumps([{
    "level": level["level"],
    "score": sum(points),
    "score_pct": round(100 * sum(points) / len(points), 1),
    "wins": sum(p == 1.0 for p in points),
    "draws": sum(p == 0.5 for p in points),
    "losses": sum(p == 0.0 for p in points),
    "passed": False,
    "end_reasons": {reason: sum(g["reason"] == reason for g in games) for reason in {g["reason"] for g in games}},
}], indent=2))
EOF
done

# Paired against the student, same openings.
for arm in noise0.25 noise0.5 noise1 noise2; do
  echo "== $arm vs student"
  uv run --project $T python -m chess_nn.paired games $O/$arm=$O/student --tag dev5 --levels 2 | python3 -c "
import json, sys
for level, v in json.load(sys.stdin).items():
    p = v['pairs'][0]; c, k, pr = p['candidate_outcomes'], p['control_outcomes'], p['paired']
    print(level, 'noisy teacher', c['score_pct'], c['wdl'], 'rep', c['repetition_draws'],
          '| student', k['score_pct'], k['wdl'],
          '| diff', round(pr['mean_points_per_game'], 4), '+-', round(pr['se'], 4), pr['ci95'])
"
done
echo "teacher diagnostics: done"
