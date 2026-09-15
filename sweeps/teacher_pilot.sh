#!/bin/zsh
# Teacher-versus-student pilot: the harness's own move selection (verified port) with the
# evaluator swapped, on every fifth dev opening x both colours at Level 2 (40 games per arm).
#
#   d9     — Stockfish depth 9 under our training transform (clamp +-2500 cp, mate -> +-25 pawns)
#   raw    — the same search, unclamped, mate distance preserved (isolates the transform)
#   static — Stockfish's static evaluation, no lookahead
#
# The student arm is jj6-h1024-s1000-cp20's recorded dev games, subset to the same openings in
# the same order, so `chess_nn.paired games` pairs them directly. Arms run one at a time.
# Exit status lands in runs/teacher/pilot.status.
T=/Users/nikhildsouza/workspace/chess-challenge/training
M=$T/runs/jj6-h1024-s1000-cp20
O=$T/runs/teacher
STRIDE=5
cd $T
set -e
trap 'echo $? > $O/pilot.status' EXIT
mkdir -p $O
rm -f $O/pilot.status

# The student arm: the control's recorded games on the same openings, renumbered to match.
uv run --project $T python - $M/dev_100_L2.json $O/student/dev5_L2.json $STRIDE <<'EOF'
import json, sys
from pathlib import Path
source, target, stride = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
data = json.loads(source.read_text())
level = data["levels"][0]
games = [g for g in level["games"] if g["opening_index"] % stride == 0]
for g in games:
    g["opening_index"] //= stride
points = [g["points"] for g in games]
level["games"] = games
level["score"] = sum(points)
level["score_pct"] = 100.0 * sum(points) / len(games)
level["wins"] = sum(p == 1.0 for p in points)
level["draws"] = sum(p == 0.5 for p in points)
level["losses"] = sum(p == 0.0 for p in points)
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(json.dumps(data))
print(f"student arm: {len(games)} games, {level['score']}/{len(games)} ({level['score_pct']:.1f}%)")
EOF

# Teacher arms, one at a time.
for teacher in d9 raw static; do
  [[ -f $O/$teacher/dev5_L2.json ]] && continue
  mkdir -p $O/$teacher
  $T/datagen/target/release/teacher play --model $M/model.onnx --teacher $teacher \
      --openings $T/data/dev_openings_100.txt --stride $STRIDE --level 2 --threads 8 \
      --out $O/$teacher/dev5_L2.json > $O/$teacher/play.log 2>&1
  tail -1 $O/$teacher/play.log
done

# chess_nn.paired reads {tag}_summary.json next to the games. "Repetition while winning" is
# defined by the net's own eval, which means nothing for a Stockfish arm, so it is left out.
for arm in student d9 raw static; do
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

# Paired comparison of each teacher against the student on the same openings.
for teacher in d9 raw static; do
  echo "== $teacher vs student"
  uv run --project $T python -m chess_nn.paired games $O/$teacher=$O/student --tag dev5 --levels 2 | python3 -c "
import json, sys
for level, v in json.load(sys.stdin).items():
    p = v['pairs'][0]; c, k, pr = p['candidate_outcomes'], p['control_outcomes'], p['paired']
    print(level, 'teacher', c['score_pct'], c['wdl'], 'rep', c['repetition_draws'],
          '| student', k['score_pct'], k['wdl'], 'rep', k['repetition_draws'],
          '| diff', round(pr['mean_points_per_game'], 4), '+-', round(pr['se'], 4), pr['ci95'],
          'better/worse', pr['openings_better'], pr['openings_worse'])
"
done
echo "teacher pilot: done"
