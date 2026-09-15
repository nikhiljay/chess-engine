# Overnight log — 2026-09-14

Goal: beat leaderboard #1 `st-h128-s800-9M` (99,358 params, passes Level 2) —
pass Level 3, or pass Level 2 with fewer params. Iteration uses dev openings
only; official openings only to confirm a milestone.

## Findings so far

All dev numbers are `data/dev_openings_100.txt` (200 games per level) unless
noted. Full table: `uv run python -m chess_nn.report`.

1. **More data helps.** M1 (109k positions) 41% L1 on the 50-opening set;
   same net on June shard 0 (726k) 56% on that set.
2. **Shard-0 sweep (726k positions):** h256-s400 L1 61% / L2 38%;
   h256-s800 57/41; h128-s400 56/41; h128-s800 50/31; h128-none-s800 54/32.
   WDL scale 800 gave no clear gain at this data size.
3. **Repetition is the dominant leak.** ~45% of games end in threefold
   repetition. For h256-s400, 108 of 400 games were repetition draws with the
   net ahead by its own eval; Stockfish depth 14 confirms 94 above +2 and 81
   above +5 (8 forced mates). Nearly all are two-move shuffles (a piece moves
   back and forth while the opponent shuffles), across all game phases.
   Depth-1 search cannot see mate-in-2, so the eval itself must rank
   "closer to mate" higher.
4. **Mate ladder on human data: no gain.** Only 2.0% of human positions are
   mates, so targets barely change; L1 still moved 9 points between
   near-identical runs, so run-to-run noise is large.
5. **Seed noise is ~±3 points.** h256-s400 on shard 0 with seeds 0/1/2:
   L1 60.8 / 54.8 / 55.5%, L2 37.8 / 41.0 / 42.5%. Most differences in
   findings 2 and 4 are within this noise; only h128-s800 looks clearly worse.
   Conclusion: at 726k human positions, architecture/scale/ladder choices
   barely matter; the data and the repetition leak do.
6. **compete's W/D/L tally is wrong** (`0.5 as u32` counts draws as losses);
   scores and pass/fail are correct. `evaluate.py` recounts from points.

## In progress at time of writing

- June labeling (`data/candidates.txt`, 6.84M candidates → ~5M positions):
  shards land in `data/shards/shard_000N.npz`.
- Self-play positions (`data/selfplay_L2_a.txt`, 212,568 FENs from
  m2a-h256-shard0 vs Level 2 with 10% random moves), labeled with depth 9 +
  depth-14 re-search when |score| ≥ 600 → `data/shards_selfplay/`.
- Seed-variance sweep (`sweeps/shard0_seeds.json`).
- Shard0 + self-play sweep (`sweeps/shard0_selfplay.json`).
- July 2017 download (`data/lichess_2017-07.pgn.zst.part`, ~130 KiB/s).

## Resuming after an interruption

Every step is resumable: labeling skips finished shards (a partial shard is
redone), sweeps skip finished runs and levels, curl resumes with `-C -`.

```bash
cd training

# June labeling (resumes at the first missing shard)
uv run python -m chess_nn.label --candidates data/candidates.txt --depth 9

# Self-play labeling (if data/shards_selfplay/shard_selfplay_a_0000.npz is missing)
uv run python -m chess_nn.label --candidates data/selfplay_L2_a.txt \
  --out-dir data/shards_selfplay --prefix shard_selfplay_a \
  --depth 9 --deep-depth 14 --deep-threshold-cp 600 --workers 6

# Sweeps (skip finished runs)
uv run python -m chess_nn.sweep sweeps/shard0_seeds.json
uv run python -m chess_nn.sweep sweeps/shard0_selfplay.json

# July download, then sampling and labeling
(cd data && curl -C - -o lichess_2017-07.pgn.zst.part \
  https://database.lichess.org/standard/lichess_db_standard_rated_2017-07.pgn.zst \
  && mv lichess_2017-07.pgn.zst.part lichess_2017-07.pgn.zst)
uv run python -m chess_nn.sample_positions --pgn data/lichess_2017-07.pgn.zst \
  --out data/candidates_2017-07.txt --max-positions 7000000 --seed 3
uv run python -m chess_nn.label --candidates data/candidates_2017-07.txt \
  --prefix shard_2017-07 --depth 9
```
