# NN Eval Training Pipeline — Design

**Date:** 2026-09-14
**Status:** Approved design, pre-implementation

## Goal

Train an ONNX scalar evaluation network for the chess-challenge harness
(`../` — benedictbrady/chess-challenge). Ranking is highest level passed, then
fewest parameters, so the strategy is **climb levels first, shrink later**.

## Harness facts that shape the design

- The NN runs depth-1 search + quiescence (`engine/src/nn.rs`). Positions the
  net actually scores are mostly quiet (captures resolved) or stand-pat nodes.
- Output is a scalar, positive = good for side to move. Mate is scored as
  ±100,000, so net outputs must stay far below that.
- Input is `[N, 1540]`: two 770-float halves (STM perspective, NSTM
  perspective), each 12×64 piece planes + 2 castling flags.
- `compete` is fully deterministic: 25 fixed openings × 2 colors, both bots
  deterministic. The same model always plays the same 50 games.
- `NnEvalBot` has no game history, so it cannot see repetition. Threefold
  repetition ends the game as a draw.
- There is no per-move time limit; only parameter count matters for ranking.
- ONNX requirements: input named `board`, output `[N, 1]`, `ir_version = 8`,
  opset 17, named batch dimension on input and output, ≤ 10M parameters.

## Principles

- **No tuning against the official 50 games.** All iteration uses a separate
  dev opening set. The official run confirms milestones only.
- Training positions exclude the 25 official opening positions.
- Stockfish-labeled data is used (discussed with the user as in-spirit under
  the written rules).

## Repository layout

`training/` is gitignored by the harness and is its own git repository.

```
training/
  pyproject.toml            # uv project, Python 3.12
  src/chess_nn/
    harness.py              # paths into the harness, tool loading
    encoding.py             # vectorized feature-index encoder
    sample_positions.py     # Lichess PGN → candidate FENs
    label.py                # Stockfish worker pool → labeled shards
    shards.py               # shard read/write
    model.py                # NNUE-style network
    train.py                # training loop
    export.py               # ONNX export + validation + parity check
    dev_openings.py         # build dev opening set
    evaluate.py             # run compete, summarize, repetition monitoring
  tests/
  docs/specs/
  data/                     # gitignored: PGNs, shards
  runs/<name>/              # config, metrics, compete JSON (models gitignored)
```

## 1. Data pipeline

**Source.** One Lichess standard rated month from database.lichess.org, chosen
to be 2–5 GB compressed, streamed through zstd without decompressing to disk.

**Game filter.** Both players rated ≥ 1800; base time ≥ 180 s;
`Termination` is `Normal`.

**Position sampling.**
- Skip the first 16 plies.
- Sample up to 6 positions per game uniformly at random.
- Skip positions where the side to move is in check.
- Deduplicate on FEN without move counters.
- Drop any position whose FEN (without move counters) matches a line of
  `../data/openings.txt`.

**Labeling.**
- Stockfish 19 (`brew install stockfish`), 10 worker processes, `Threads 1`,
  small hash, `go depth D`.
- `D` defaults to 9 and is set by benchmarking 1,000 positions before the full
  run (target: 5M positions labeled in a few hours).
- Record the score from the side-to-move perspective and the best move.
- Drop positions whose best move is a capture or promotion, so the data
  matches the quiet positions quiescence search hands the net.
- Scores are clamped to ±2500 cp; mate scores map to ±2500.

**Volume.** ~5M labeled positions for v1.

**Storage.** Shards of 1M positions (`.npz`):
- `features`: int16 `[N, 68]` active-feature indices into the 1540 input,
  padded with −1 (max 32 pieces × 2 halves + 4 castling flags).
- `score_cp`: int16 `[N]`.
- `fens`: text sidecar for debugging.
Labeling is resumable: completed shards are skipped.

**Encoding correctness.** The vectorized encoder is tested for exact equality
against `../tools/encoding.py`, and against the Rust `dump-encoding` binary on a
few hundred FENs including castling, en passant, and black-to-move positions.

## 2. Model and training

**Architecture (NNUE-style, shared feature transformer).**

```
board[N,1540]
  stm  = board[:, 0:770]
  nstm = board[:, 770:1540]
  ft   = Linear(770 → H)  shared by both halves, clamp(0, 1)
  x    = concat(ft(stm), ft(nstm))        # [N, 2H]
  x    = ReLU(Linear(2H → 32))
  x    = ReLU(Linear(32 → 32))
  eval = Linear(32 → 1)                   # nn.Linear → Gemm
```

H = 256 for v1 → 214,881 parameters. (H = 128 → 108,001.)

**Output scale and loss.** The net's output is in pawns. Loss is MSE in
win-probability space: `(sigmoid(100·out / 400) − sigmoid(score_cp / 400))²`.
This down-weights large disagreements in lopsided positions and keeps output
magnitudes small relative to the harness mate score.

**Training.**
- PyTorch on MPS; AdamW; cosine LR schedule from 1e-3; batch ~16k.
- 2% of positions held out for validation.
- Dense 1540 inputs are built on-device from `features` via scatter.
- No augmentation in v1.

**Export.**
- The graph is built directly with `onnx.helper` from the trained weights
  (Split → shared Gemm ×2 → Clip → Concat → Gemm/Relu ×2 → Gemm), opset 17,
  `ir_version` 8, named `batch` dimension on `board` and `eval`. This avoids
  torch exporter default drift (torch 2.14) and the MatMul+Add pitfall. The
  Split sizes initializer adds 2 counted parameters.
- Every export runs `../tools/validate_model.py` and a parity test (ONNX
  Runtime vs. PyTorch on 1,000 positions, max abs diff < 1e-4).

**Environment.** uv project pinned to Python 3.12: torch, python-chess, numpy,
onnx, onnxruntime, zstandard, pytest.

## 3. Evaluation loop and testing

**Dev openings.** ~50 opening positions (2–6 plies) drawn from Lichess games,
excluding every official opening, written in the `data/openings.txt` format.
Iteration uses:

```
cargo run -p cli --release --bin compete -- model.onnx --level N \
  --openings training/data/dev_openings.txt --json-output runs/<name>/dev_LN.json
```

The official openings run only when a dev result suggests a milestone.

**Run tracking.** `runs/<name>/` holds `config.json`, `metrics.json` (train and
validation loss per epoch), and compete JSON for dev and official runs.

**Per-checkpoint sanity checks.**
- Validation loss.
- Start position evaluates near 0 (|eval| < 0.5 pawns).
- A position a queen up evaluates strongly positive (> +5 pawns), and the
  same material deficit evaluates strongly negative (< −5 pawns).

**Repetition monitoring.** For dev games ending in `repetition`, take the
final FEN from the compete JSON, evaluate it with the model in ONNX Runtime
from the NN's side, and count games where that eval is > +2 pawns (a winning
position thrown away). Mitigation is v2 work if this is common.

**Tests (pytest).**
- Encoder parity with `tools/encoding.py` and Rust `dump-encoding`.
- Sampling filter rules (check, ply skip, opening exclusion, dedupe).
- Label filter (capture/promotion best move dropped, mate clamping).
- Export validity and ONNX/PyTorch parity.
- End-to-end smoke: 1k positions → 1 epoch → export → one level-1 game.

## Milestones

- **M1:** Full pipeline end-to-end on 100k positions.
- **M2:** 5M positions, H = 256, pass Level 1 (dev first, then official).
- **M3:** Iterate toward Level 2.

## Out of scope for v1

- Self-play or harness-generated positions.
- Parameter shrinking, pruning, or quantization.
- Blending game results into labels.
- Changes to the harness itself.
