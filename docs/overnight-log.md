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
6. **June shards 0-3 (2.9M positions):** h256-s400 L1 64.0% / L2 45.0%,
   +7 / +5 over the shard-0 seed mean, so more data does help the 215k net;
   h128-s400 56.5% / 41.8%, no gain, so width matters once data grows.
   Train/val gap nearly closed (0.0087 / 0.0092). Repetition still ~38% of
   games, with 56 (L1) and 37 (L2) of them the net ahead by its own eval.
7. **Capacity gains get absorbed by repetition.** On June shards 0-3:
   h256 seed1 L1 63.7% / L2 46.0%; h512 (429k params) 63.7 / 49.8;
   h256 40 epochs 63.2 / 47.2. Bigger or longer-trained nets cut losses
   (L1 34 → 21-25, L2 71 → 49-55) but those games become repetition draws
   (L1 76 → 95-103) with the net ahead in 64-71 of them. L1 is capped near
   64% until conversion is fixed; h512 converting its "winning" repetitions
   would score ~81%.
8. **Progress shaping: no improvement.** `chess_nn/shaping.py` adds a
   winning-side bonus to non-mate targets with |score| ≥ 300 cp: 10 cp per
   pawn rank advanced, 4 cp per point of material traded off, and 15 cp per
   step of mop-up (losing king to the edge, kings close) once the loser has
   no pawns and ≤ a rook. h256 on shards 0-3: s400 L1 62.7% / L2 43.0% with
   repetition-while-winning rising to 75 / 68; s1000 61.0 / 45.5. The two
   positions in a shuffle loop have identical progress features, so the
   bonus cannot break the tie, and 10-15 cp sits inside the net's eval error.
9. **Loop diagnosis (h512, shards 0-3).** At loop positions the net's scores
   for its top moves are nearly flat in won endgames: KR+P vs K with mate in 5,
   Rg7 +8.61 / Rf7 +8.60 / h7 +8.59 / Rb7 +8.50, while Stockfish sees mate in
   4-7 for three of them and +14 for the fourth. It also sometimes prefers moves Stockfish rates as
   draws (Re5: net +7.31, Stockfish 0). The WDL loss at scale 400 maps +8 and
   mate to nearly the same win probability, so the net gets almost no gradient
   to rank them. (Caveat: children were ranked by static eval; the harness
   uses quiescence, so capture moves are overrated in that listing.)
   Next test: `sweeps/june4_selfplay.json` — shards 0-3 plus self-play
   positions, with the mate ladder and an auxiliary Huber loss in pawns
   (`cp_loss_weight`) that keeps gradient between large advantages.
10. **Self-play data (`data/shards_selfplay/shard_selfplay_a_0000.npz`):**
    154,944 positions kept of 212,568 (depth 9, depth-14 re-search at
    |score| ≥ 600, ~50 positions/s with 6 workers). 14.3% are mate scores
    (human data: 2.0%), 44% are non-mate |score| ≥ 600, 77% are |score| ≥ 300,
    i.e. heavily weighted toward won positions, as intended.
11. **Self-play positions cut losses, not repetition.** h256-s400 on shard 0 +
    self-play (881k, 18% self-play): L1 64.0% / L2 38.2% with 22 L1 losses
    (shard-0 seeds: 35-43 losses, L1 mean 57.0, L2 mean 40.4); with the mate
    ladder 62.0 / 39.2. L1 gains +5-7 over the seed mean, beyond noise, but
    repetition-while-winning stays at 57-62 and L2 does not move. The mate
    ladder shows no effect even with 14% mates in the self-play data.
12. **Shards 0-3 + self-play (3.06M, 5% self-play), h256-s400:** plain
    L1 62.5% / L2 45.5%; + ladder 61.8 / 42.8; + ladder + cp_loss 0.02
    58.8 / 48.5. At 5% share, self-play and the ladder are not measurable
    against the j4 baselines (64.0-63.7 / 45.0-46.0). The weak pawn-space loss
    hints at better conversion on L2 (repetition 104 → 76,
    repetition-while-winning 65 → 43, wins 39 → 59) but L1 losses rose to 38,
    so it is inconclusive at 0.02; `sweeps/june4_selfplay_cp.json` tests 0.1.
13. **cp_loss 0.1: no clear conversion gain.** Shards 0-3 + self-play with
    the ladder: h256 L1 60.5% / L2 43.8%; h512 64.8 / 49.0 (h512 baseline
    63.7 / 49.8). Repetition still ~45% of games, repetition-while-winning
    59-61 (L1) and 47-51 (L2). None of WDL scale, mate ladder, progress
    shaping or the pawn-space loss fixed conversion. What measurably helped:
    more human data (+7 L1 from 726k → 2.9M) and self-play positions at an
    ~18% share (+5-7 L1 via fewer losses). Last test before the interruption:
    `sweeps/june4_selfplay5x.json` (self-play shard repeated 5× ≈ 21% share).
14. **Repeating the self-play shard 5× (~21% share) did not help.** h512 +
    ladder + cp_loss 0.1: L1 60.8% / L2 44.5% (1×: 64.8 / 49.0); h256-s400:
    64.0 / 48.0 (j4 baselines 64.0-63.7 / 45.0-46.0). Duplicated positions
    likely overfit; the shard-0 self-play gain came from fresh positions.
15. **(2026-09-15, after the reboot) July shards 0-1 add nothing.** h512 on
    June 0-3 + July 0-1 (4.36M): L1 65.2% / L2 50.0%, repetition 89 / 102,
    repetition-while-winning 59 / 59 — the same as June alone (63.7 / 49.8).
    h256: 62.3 / 49.2 (June alone 64.0 / 45.0), within seed noise.
    More human-game positions at depth 9 have plateaued; local labeling was
    stopped after July shard 1 to free the CPU.
16. **Lichess eval database as a data source.** `lichess_db_eval.jsonl.zst`
    (22 GB) holds positions with deep Stockfish analyses and best-move lines,
    so no local labeling is needed. `chess_nn.import_evals` filters them like
    `label` (not in check, quiet legal best move, no official openings). On a
    200k-line sample: 72.7% kept, median depth 30, 18.6% mate scores (human
    data 2.0%), 24% of positions with ≤10 pieces — the won endgames the net
    fails to convert.
17. **Eval-database pilot hurts.** First 10M lines at 40% sampling → 2.80M
    positions (61 s). h512 on those alone: L1 52.0% / L2 33.2% (losses
    60 / 101). Mixed with June 0-3 + July 0-1 (7.17M, 39% eval-db): L1 60.5% /
    L2 45.5%, repetition-while-winning 70 / 55 — below the human-only 65.2 /
    50.0 on both levels. Likely cause: analysis-board positions and depth-30+
    middlegame scores the depth-1 net cannot learn. Next test: only the
    eval-db endgames (≤10 pieces) mixed in at a small share
    (`sweeps/evpilot_eg.json`: 587,700 pilot positions, 50% mates, ~12% of
    the mix). Caveat: the file is not shuffled — 200k lines at line 60M have
    median depth 23 (start: 30), 10.3% mates (18.6%), 15% ≤10 pieces (24%) —
    so the pilot slice is not representative of a full-file sample.
18. **Eval-db endgames at ~12% share: no clear gain.** June 0-3 + July 0-1 +
    587,700 pilot positions with ≤10 pieces (h512): L1 64.8% / L2 49.2%
    (baseline 65.2 / 50.0); with the mate ladder 68.0 / 46.5. L1 +2.8 is
    inside seed noise and L2 fell; repetition-while-winning stays 59-65.
    Seed-1 replicate of the ladder run: 65.2 / 42.0, so the 68.0 was noise
    and L2 is worse on both seeds. Eval-db endgames are ruled out; the full
    eval-db conversion is dropped (the download is kept in `data/`).
    Loop-children generation (jj6-h512 vs L1, 8000 games): 2,280 loops
    (28.5%), 347,571 positions in 523 s.
19. **Loop children are slow to label.** `label --bench 2000` at depth 9 +
    deep 14 (≥600 cp): 103 positions/s with datagen sharing the CPU, 52.9%
    kept (most loop children are lopsided, so most get the deep re-search).
    Candidates from L1 + L2 are deduplicated, shuffled (seed 0) and labeled
    in 100k shards so partial results are usable and a restart loses at most
    one shard.
20. **Loop-children data (`data/shards_loops/shard_loops_a_*.npz`).** L2
    generation: 8000 games, 2,456 loops (30.7%), 359,188 positions. L1 + L2:
    706,614 unique candidates → 391,594 labeled positions (55% kept, ~140/s,
    ~80 min). 6.3% mates; non-mate |score| ≥ 300 cp 81.5%, ≥ 600 cp 57.0%,
    < 100 cp 8.5%; median 16 pieces, 20.9% with ≤10. In
    `sweeps/loops_a.json` (June 0-3 + July 0-1 + loops) they are 8.2% of
    positions — below the ~18% share at which fresh self-play helped (11).
21. **Low-memory kills are not a training leak.** The loops_a sweep was
    killed twice by the harness's low-memory guard, both ~2 min into
    loopa-h512 (epoch 17-18). A direct run with 5 s memory sampling finished
    all 20 epochs: process footprint 3.3 GB while loading, then flat at
    2.7 GB; free memory 49-50%, with brief wired-memory swings to 8.6-9.9 GB
    (35-38% free) from other processes after the OS update. That run is
    reused as `runs/loopa-h512-s400` (same shards, order, seed and settings).
    Sweeps now run with `PYTHONUNBUFFERED=1` so logs survive a kill.
    A third kill hit during loopa-h256's games; at that moment the ChatGPT
    app's Codex renderer held 8.6 GB, the compressor ~11 GB, and swap was 0.
22. **Loop children at 8.2%: best L1 yet, but L2 falls and conversion is not
    fixed.** June 0-3 + July 0-1 + loops_a (h512): L1 69.0% / L2 43.8%
    (baseline 65.2 / 50.0), losses 19 / 68, repetition-while-winning
    54 / 54 (59 / 59). With the mate ladder: 64.0 / 48.5, repetition-while-
    winning 65 / 64. L1 +3.8 is at the edge of seed noise; L2 −6.2 is beyond
    it. A seed-1 replicate of the plain run checks the L1 number.
    h256 + loops_a: L1 68.5% / L2 46.5% (June+July h256 62.3 / 49.2), L1
    losses 30 → 19 — the same L1 pattern as h512 (losses 25 → 19), so the L1
    gain holds across two widths. L2 again does not improve: 122 draws,
    repetition-while-winning 56 → 80.
23. **Loop parents for a ranking diagnostic.** `datagen --emit parents`
    writes the positions the net moved from in loops instead of their
    children. loopa-h512 vs L1, 100 games: 30 loops → 180 unique parents
    (6 per loop), 6 in check, 32.5 legal moves on average. Next: on one
    parent set, compare each model's harness move with Stockfish's top moves
    to test whether the net picks draw-ish moves when a winning one exists.
24. **At loop positions the net does not blunder; it shuffles among moves
    Stockfish also rates as winning.** 180 parents from note 23, Stockfish
    depth 10 (best move, and `searchmoves` for the net's move); 143 have a
    best move ≥ +300 cp. jj6-h512: agrees with Stockfish's best 32.2%,
    median loss 15 cp, mean 86 cp (capped at 1000); in winning parents its
    move stays ≥ +300 in 96.5% and drops to ≤ +100 in 0.7%. loopa-h512:
    28.9%, 15 cp, 78 cp, 97.2%, 1.4%. So per-move Stockfish scores barely
    separate progress from shuffling, and a loss that ranks siblings by
    them has little to learn from. What breaks a loop is long-horizon
    progress (mate distance, pawn advance, trades), which depth-10 cp does
    not express. The sibling-ranking loss is not pursued.
25. **Loop data replicate: the L1 gain was mostly noise and L2 is worse.**
    loopa-h512 seed 1: L1 65.5% / L2 42.8% (seed 0: 69.0 / 43.8). Loop-data
    h512 mean L1 67.3 / L2 43.3 against the human-only 65.2 / 50.0; h256
    68.5 / 46.5 against 62.3 / 49.2. Loop children at 8.2% trade a small L1
    gain for a clear L2 loss, so they are not used going forward.
26. **WDL scale at full data (June 0-3 + July 0-1, h512).** s800: L1 61.5% /
    L2 46.5%. s800 + cp_loss 0.1: 64.5 / 50.7, seed 1 65.0 / 47.0. s1000 +
    cp_loss 0.1: 63.7 / **57.2** — best L2 so far (+7.2 over s400's 50.0),
    L2 losses 49 → 37, wins 49 → 66, repetition-while-winning 59 → 66.
    Single seed; the s800 seeds already differ by 3.7 on L2.
    `sweeps/jj6_scale2.json` runs a seed-1 replicate, s1500 + cp 0.1 and
    s1000 + cp 0.2.
27. **Large WDL scale + pawn loss is a real, modest gain.** s1000 + cp 0.1
    seed 1: L1 **72.0%** / L2 50.7% (seed 0: 63.7 / 57.2); s1500 + cp 0.1:
    69.0 / 53.0; s1000 + cp 0.2: 69.2 / 53.2 (L1 losses 14). Mean of the four
    runs L1 68.5 / L2 53.5 against s400's 65.2 / 50.0 — all four beat the
    baseline on at least one level and none falls below it on both. The 72.0
    is the best of four noisy runs, not the recipe's level, so it is not
    treated as the ≥72% official-run milestone. Next:
    `sweeps/jj6_scale3.json` (s1500 + cp 0.2, s1000 + cp 0.2 seed 1, h1024).
28. **s1000 + cp 0.2 puts dev L1 at ~70%; L2 stays in the low 50s.** s1000 +
    cp 0.2 seed 1: 70.8 / 49.2 (seed 0 69.2 / 53.2 → mean 70.0 / 51.2);
    s1000 + cp 0.1 seeds mean 67.9 / 54.0. s1500 + cp 0.2: 60.5 / 50.7, so
    both knobs high at once hurts. h1024 (856k params) + s1000 + cp 0.2:
    69.8 / 54.2 with 13 L1 losses — width barely moves L2. L2
    repetition-while-winning is still 67-78: converting those games is
    worth ~15-20 L2 points, so conversion remains the path to L2.
    Next: `sweeps/jj6_shape2.json` — progress shaping (finding 8) retried at
    3-7× the bonuses on top of s1000 + cp 0.2, where the pawn loss keeps
    gradient in won positions.
29. **Strong progress shaping hurts, even with the pawn loss.** On s1000 +
    cp 0.2 (unshaped seeds mean L1 70.0 / L2 51.2): mid bonuses (pawn 40,
    trade 15, mop-up 50 cp, threshold 300) 67.2 / 48.5 and 66.5 / 46.2
    (mean 66.9 / 47.4); strong (80 / 30 / 100) 60.2 / 42.5. Repetition-
    while-winning drops only slightly (L2 67-70 → 56-59) while losses rise
    (L2 41-51 → 57-74): the bonus makes the net overvalue pawn pushes and
    trades. Shaping is ruled out.
30. **Wider head and longer training: no gain.** On s1000 (h512): head
    128×32 (530k params) + cp 0.2: L1 69.2% / L2 51.7%; + cp 0.1: 68.5 /
    46.5; 40 epochs + cp 0.2: 64.2 / 49.2. The head and epoch count are not
    the limit.
31. **King-bucketed feature transformer (`king_buckets: true`).** Each
    perspective's transformer is chosen by its own king square: queenside /
    kingside × back rank / 2nd / 3rd-4th / 5th-8th = 8 buckets
    (`model.KingBucketNet`). Training stores shared + per-bucket delta
    weights so rare buckets start from the shared transformer; export merges
    them and picks each position's bucket with ArgMax → Gather inside the
    graph, so inference costs one transformer multiply. h512 ≈ 3.19M
    harness-counted params, h256 ≈ 1.6M. Tests: zero deltas equal EvalNet,
    ONNX parity, harness validation and param count
    (`tests/test_king_buckets.py`). First sweep: `sweeps/kb1.json`.
32. **King buckets: no overall gain.** s1000 on June 0-3 + July 0-1:
    kb-h512 + cp 0.2 L1 67.8% / L2 51.2% (L2 W/D/L 66/73/61);
    kb-h256 + cp 0.2 69.5 / 52.2 (65/79/56); kb-h512 + cp 0.1 66.5 / 50.0.
    Mean 67.9 / 51.1 against ~70 / 51-54 without buckets. Buckets convert
    more L2 wins (repetition-while-winning 41-51 in two runs) but lose more
    games (56-61 L2 losses) — the losses are tactical, which king-relative
    piece weights do not address.
33. **Goal is now Level 3 (2026-09-15).** Every candidate is graded on L1-L3
    (`sweeps/grade_l3.json` grades the best existing runs). The L3 bot
    searches depth 3 with TT/PVS/null-move over a full handcrafted eval
    (tapered PSTs, mobility, king-attack count, pawn shield, passed pawns,
    structure, bishop pair, rook files). compete logs keep only per-game
    results, so losing positions need our own game records.
    Data: self-play from jj6-h1024-s1000-cp20 against L2 and L3 (12k games
    each, 10% random moves, `data/selfplay_{L2,L3}_b.txt`), labeled first,
    then the remaining human shards.
34. **Level 3 baseline: ~29%.** jj6-h1024-s1000-cp20 58/200 (29%, 235 s);
    jj6-h512-s1000-cp10-seed1 56/200 (28%). Passing needs 140/200.
35. **Diagnosis with game history (user's plan, 2026-09-15).** Finding 24
    gave Stockfish loop positions without the game, so a shuffle back into
    a repeated position looked as good as progress. `datagen --record` now
    writes whole games; `chess_nn.diagnose` replays each with
    `position fen <start> moves ...` so Stockfish scores repetitions as
    draws, and reports repetition draws the net really had won, the move
    where the win slipped (and whether it repeated a position), and each
    loss's biggest drop. Games use random training openings; the dev
    openings stay for measurement. Next: a ranking objective on those
    decisions (good continuation vs the tempting move), then capacity.
    ThreatNet (in-graph attack/threat features, `threat_hidden`) is written
    and tested but parked until the ranking signal is measured.
36. **Recorded games (h1024, random training openings, no random moves).**
    L2: 400 games, score 46.6% — 103 wins, 167 draws (all threefold
    repetition), 130 losses (all checkmate), median 69 plies. L3: 24.2% —
    35 wins, 124 repetition draws, 241 checkmate losses, median 71 plies.
    No stalemates or fifty-move draws; games run to mate. Against L3 the
    main leak is losses (60% of games), not conversion, so the ranking pairs
    keep every clearly worse move (gap ≥ 100 cp), not only repetition slips.
    Also L3: jj6-h512-s1000-cp10 57.5/200 (29%).
37. **L3 on dev openings (W/D/L).** jj6-h1024-s1000-cp20 29.0% (20/76/104);
    jj6-h512-s1000-cp10-seed1 28.0% (23/66/111); jj6-h512-s1000-cp10 28.7%
    (16/83/101); kb-h256-s1000-cp20 33.2% (31/71/98); kb-h512-s1000-cp20
    26.8% (18/71/111). Every model loses about half its L3 games; king
    buckets h256 is the best L3 score so far but within ~±3 of the rest.
38. **Stockfish diagnosis with history (h1024, depth 14, games the net did
    not win).** Net move = Stockfish best in 38.6% (L2) / 36.4% (L3).
    - L2 (297 games): 167 repetition draws, 150 (90%) genuinely won
      (median best +9.2 pawns); 143 slipped. 130 losses: 106 contain one
      blunder (median drop 6.7 pawns from +0.3, ply 32); 67 had been winning.
    - L3 (365 games): 124 repetition draws, 92 (74%) won (median +8.1);
      241 losses: 198 with one blunder (median drop 6.4 pawns from +1.4,
      ply 42); 130 (54%) had been winning first.
    - The flagged slip sits a median 2 (L2) / 10 (L3) plies before the end:
      Stockfish counts a repetition as a draw only when it would complete
      the threefold (or repeats after its root), so earlier shuffles back
      into once-seen positions still score as winning, while the win was
      first reached ~34 plies earlier. Hence `rank_pairs` adds a rule: a
      winning-side move into an already-seen position, when Stockfish's
      best is not one, is a pair regardless of the gap.
    - h1024 L2 alone: 1,380 score pairs — too few; a larger diagnosis
      (1,500 games per level from h1024 and kb-h256, depth 12) is running.
    - rank_pairs_a (h1024 L2 + L3): 3,104 pairs — 3,057 by score gap, 47 by
      the repetition rule; 2,786 train / 318 val. Of 746 net moves that
      repeated a position: 323 were not in a winning position, 232 were
      already score pairs, 80 matched Stockfish's best, 64 had a child in
      check or game over, 47 became rule pairs. So ~279 pairs cover winning
      shuffles and the rest are blunders — consistent with L3's losses.
40. **Ranking objective, first try (rank_a, h512 s1000 cp 0.2, 2,786 train
    pairs): play gets clearly worse.** Control w0 (identical to
    jj6-h512-s1000-cp20): L1 69.2 / L2 53.2 / L3 29.0, held-out pair
    accuracy 0.52 → 0.36 over training. w0.5: 57.0 / 44.2 / 18.8, accuracy
    0.46. w2: 60.0 / 42.8 / 21.8, accuracy 0.48. The loss drops 9-13 points
    at every level and held-out accuracy stays below 0.5, so it memorizes
    the pairs without generalizing, and at w >= 0.5 it outweighs the score
    losses (train loss ~0.36 vs rank ~0.7·w). The control shows the base net
    prefers the worse child in 64% of these pairs — they are real blind
    spots. Most pairs are blunders whose refutation is a tactic the static
    features do not expose, so next: many more pairs (diagnosis_b), much
    smaller weights (0.05, 0.2), and ThreatNet's attack/threat inputs, with
    and without ranking (`sweeps/threat_a.json`, `sweeps/rank_b.json`).
39. **Why the earlier conversion fixes failed, given 38.** Every one of
    them changed per-position targets from Stockfish searches that did not
    know the game: mate ladder, WDL scale, pawn-space loss (5, 12-13, 26),
    eval-db endgames (17-18) and loop-children data (22, 25) all labeled a
    shuffle's child with the same winning score as a progress move's child,
    so the gradient never separated them (finding 24's 15 cp median gap).
    Progress shaping (8, 29) added bonuses to won human positions, where
    loop positions share the same features, and distorted non-quiet
    evaluations enough to raise losses. King buckets (32) added
    king-relative capacity, but the losses are single tactical blunders
    from level or better positions (6-7 pawns at move 16-21), which
    king-relative weights do not address. What none of them supplied is a
    signal that compares the moves available in the same position — the
    ranking objective.
19. **compete's W/D/L tally is wrong** (`0.5 as u32` counts draws as losses);
   scores and pass/fail are correct. `evaluate.py` recounts from points.

## State at the interruption

- June labeling (`data/candidates.txt`, 6.84M candidates → ~5M positions) was
  stopped deliberately after `shard_0003` (2.9M positions) to give self-play
  labeling the CPU; shards 4-6 remain and resume picks up at `shard_0004`.
- Self-play positions (`data/selfplay_L2_a.txt`, 212,568 FENs from
  m2a-h256-shard0 vs Level 2 with 10% random moves) are labeled:
  `data/shards_selfplay/shard_selfplay_a_0000.npz` (154,944 positions).
- Every sweep listed under "Resuming" finished; results are in `runs/` and
  committed.
- The July 2017 download finished (`data/lichess_2017-07.pgn.zst`); July
  sampling was running when work stopped and will be cut off by the update.
  Re-run it from scratch (it overwrites `data/candidates_2017-07.txt`), then
  label July.

## Next steps after the interruption

**Updated 2026-09-15 afternoon (findings 26-29). Leaderboard #1 is not
beaten yet.** Best recipe: h512 on June 0-3 + July 0-1 (4.36M) with WDL scale
1000 and `cp_loss_weight` 0.1-0.2 — seed means L1 70.0 / L2 51.2 (cp 0.2) and
67.9 / 54.0 (cp 0.1), against 65.2 / 50.0 at scale 400. Candidate 2 below
worked modestly; candidate 3 (strong shaping) failed; width (h1024) adds
little. L2 is the gap: ~70 of 200 L2 games are still repetition draws the
net thinks it is winning. Work is paused pending candidate 1.

Earlier state (midday): `jj6-h512-s400` (June 0-3 + July 0-1, 4.36M) L1
65.2% / L2 50.0%; `j4-h512-32x32-s400` 63.7 / 49.8. No official run has been
made.

Ruled out since the interruption (findings 15-25): more human data (July),
the Lichess eval database whole or endgames-only, and loop-children
self-play data (L1 +~2, L2 −6). Earlier: WDL scale 800/1000 at 726k, mate
ladder, progress shaping at 10-15 cp, pawn-space loss 0.02/0.1, repeated
self-play. Finding 24: at loop positions the net plays moves Stockfish also
rates as winning (median loss 15 cp), so the leak is lack of progress, not
blunders, and per-move Stockfish scores cannot teach progress.

Candidate next steps, most informative first:
1. **Calibrate dev vs official (needs the user's OK).** One `--official`
   run of `jj6-h512-s400` at L1 and L2 — a measurement, not tuning. #1
   passes L2 at 70% with 99k params while our 429k net scores 50% on dev
   L2; the gap is either the opening sets or a real technique gap, and the
   next steps differ.
2. **Scale + pawn loss at full data.** WDL scale 800-1000 was only tried at
   726k; retry at 4.36M with `cp_loss_weight` 0.1, ≥2 seeds. In won
   positions the scale-400 target is saturated, leaving eval differences
   between winning moves unconstrained.
3. **Strong progress shaping in won endgames.** Earlier bonuses (10-15 cp)
   sat inside eval error; retry at 50-100 cp per step for |score| ≥ 600 with
   the pawn loss so the gradient survives.
4. **h128 at larger data** (#1's likely recipe: h128, scale 800, 9M).

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

# Sweeps (skip finished runs and levels)
uv run python -m chess_nn.sweep sweeps/shard0_seeds.json
uv run python -m chess_nn.sweep sweeps/shard0_selfplay.json
uv run python -m chess_nn.sweep sweeps/june4.json
uv run python -m chess_nn.sweep sweeps/june4b.json
uv run python -m chess_nn.sweep sweeps/june4_shaped.json
uv run python -m chess_nn.sweep sweeps/june4_selfplay.json
uv run python -m chess_nn.sweep sweeps/june4_selfplay_cp.json
uv run python -m chess_nn.sweep sweeps/june4_selfplay5x.json

# Table of every run
uv run python -m chess_nn.report

# Lichess eval database (2026-09-15): download (resumable), then convert.
(cd data && curl -C - -o lichess_db_eval.jsonl.zst.part \
  https://database.lichess.org/lichess_db_eval.jsonl.zst \
  && mv lichess_db_eval.jsonl.zst.part lichess_db_eval.jsonl.zst)
# Pilot: first 10M lines of the file (works on the partial download too)
mkfifo /tmp/pilot.fifo
(zstd -dc data/lichess_db_eval.jsonl.zst* | head -n 10000000 > /tmp/pilot.fifo &)
uv run python -m chess_nn.import_evals --evals /tmp/pilot.fifo \
  --out-dir data/shards_evaldb_pilot --sample-prob 0.4
uv run python -m chess_nn.sweep sweeps/evpilot_only.json
uv run python -m chess_nn.sweep sweeps/evpilot_mix.json
# Full: ~354M lines sampled at 3% → ~7.7M positions (seeded; skips finished shards)
uv run python -m chess_nn.import_evals --evals data/lichess_db_eval.jsonl.zst \
  --out-dir data/shards_evaldb --sample-prob 0.03
uv run python -m chess_nn.sweep sweeps/evdb_only.json
uv run python -m chess_nn.sweep sweeps/evdb_mix.json
uv run python -m chess_nn.sweep sweeps/evpilot_eg.json   # pilot endgames only

# Loop children (2026-09-15): children of repetition positions the net thought
# it was winning, from jj6-h512 vs Levels 1 and 2 (200 games → 72 loops, ~12k positions)
(cd datagen && cargo build --release)
for L in 1 2; do datagen/target/release/datagen runs/jj6-h512-s400/model.onnx \
  --out data/loops_L${L}_a.txt --games 8000 --threads 10 --level $L --random-move-prob 0 \
  --sample-prob 0 --loop-plies 16 --min-eval 3 --seed 1$L; done
# dedupe + shuffle (seed 0); rerunning this reproduces the same order, so label resumes
uv run python -c "import random; fens = list(dict.fromkeys(l.strip() for f in ['data/loops_L1_a.txt', 'data/loops_L2_a.txt'] for l in open(f) if l.strip())); random.Random(0).shuffle(fens); open('data/loops_a.txt', 'w').write('\n'.join(fens) + '\n')"
uv run python -m chess_nn.label --candidates data/loops_a.txt --out-dir data/shards_loops \
  --prefix shard_loops_a --depth 9 --deep-depth 14 --deep-threshold-cp 600 --shard-size 100000
uv run python -m chess_nn.sweep sweeps/loops_a.json

# July download, then sampling and labeling
(cd data && curl -C - -o lichess_2017-07.pgn.zst.part \
  https://database.lichess.org/standard/lichess_db_standard_rated_2017-07.pgn.zst \
  && mv lichess_2017-07.pgn.zst.part lichess_2017-07.pgn.zst)
uv run python -m chess_nn.sample_positions --pgn data/lichess_2017-07.pgn.zst \
  --out data/candidates_2017-07.txt --max-positions 7000000 --seed 3
uv run python -m chess_nn.label --candidates data/candidates_2017-07.txt \
  --prefix shard_2017-07 --depth 9
```
