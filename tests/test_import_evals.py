import json

import chess
import numpy as np
import zstandard

from chess_nn.harness import OFFICIAL_OPENINGS, read_opening_fens
from chess_nn.import_evals import best_analysis, convert, row_from_record
from chess_nn.label import SCORE_LIMIT_CP
from chess_nn.sample_positions import load_opening_keys
from chess_nn.shards import load_shards

BLACK_TO_MOVE = "7r/1p3k2/p1bPR3/5p2/2B2P1p/8/PP4P1/3K4 b - -"
ROOK_ENDGAME = "8/4r3/2R2pk1/6pp/3P4/6P1/5K1P/8 b - -"
WHITE_TO_MOVE = "6k1/6p1/8/4K3/4NN2/8/8/8 w - -"
OPEN_CENTER = "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6"


def record(fen, depth=30, **pv):
    return {"fen": fen, "evals": [{"pvs": [pv], "knodes": 1000, "depth": depth}]}


def test_best_analysis_uses_deepest_eval_and_first_pv():
    rec = {
        "fen": OPEN_CENTER,
        "evals": [
            {"pvs": [{"cp": 10, "line": "g1f3 g8f6"}], "knodes": 900, "depth": 20},
            {"pvs": [{"cp": 40, "line": "b1c3 g8f6"}, {"cp": 30, "line": "g1f3"}], "knodes": 100, "depth": 31},
        ],
    }
    assert best_analysis(rec) == (31, 40, 0, "b1c3")
    assert best_analysis(record(WHITE_TO_MOVE, mate=15, line="e5e6 g8f8")) == (30, 0, 15, "e5e6")
    assert best_analysis({"fen": OPEN_CENTER, "evals": []}) is None


def test_scores_flip_to_side_to_move():
    fen, score, mate, indices = row_from_record(record(BLACK_TO_MOVE, cp=69, line="f7g7 e6e2"), 0, set())
    assert (score, mate) == (-69, 0)
    assert fen.startswith("7r/1p3k2/") and len(indices) > 0
    _, score, mate, _ = row_from_record(record(BLACK_TO_MOVE, mate=-3, line="f7g7"), 0, set())
    assert (score, mate) == (SCORE_LIMIT_CP, 3)
    _, score, mate, _ = row_from_record(record(WHITE_TO_MOVE, mate=15, line="e5e6"), 0, set())
    assert (score, mate) == (SCORE_LIMIT_CP, 15)
    _, score, _, _ = row_from_record(record(WHITE_TO_MOVE, cp=90000, line="e5e6"), 0, set())
    assert score == SCORE_LIMIT_CP


def test_filters():
    assert row_from_record(record(OPEN_CENTER, cp=20, line="g1f3"), 0, set()) is not None
    assert row_from_record(record(OPEN_CENTER, cp=20, line="e4d5"), 0, set()) is None  # capture
    assert row_from_record(record(OPEN_CENTER, cp=20, line="e2e5"), 0, set()) is None  # illegal
    assert row_from_record(record(OPEN_CENTER, depth=10, cp=20, line="g1f3"), 20, set()) is None
    assert row_from_record(record("4k3/8/8/8/8/8/8/4R2K b - -", cp=-900, line="e8d8"), 0, set()) is None  # check
    assert row_from_record({"fen": "not a fen", "evals": []}, 0, set()) is None
    too_many_pieces = "QQQQQQQQ/QQQQQQQQ/QQQQQQQQ/8/8/8/8/K6k w - -"
    assert row_from_record(record(too_many_pieces, cp=9000, line="a1b1"), 0, set()) is None


def test_official_openings_are_excluded():
    official = read_opening_fens(OFFICIAL_OPENINGS)[0]
    board = chess.Board(official)
    quiet = next(m for m in board.legal_moves if not board.is_capture(m) and not m.promotion)
    rec = record(" ".join(official.split()[:4]), cp=0, line=quiet.uci())
    assert row_from_record(rec, 0, load_opening_keys()) is None


def test_convert_writes_shards(tmp_path):
    lines = [
        record(BLACK_TO_MOVE, cp=69, line="f7g7"),
        record("4k3/8/8/8/8/8/8/4R2K b - -", cp=-900, line="e8d8"),
        record(WHITE_TO_MOVE, mate=15, line="e5e6"),
        record(ROOK_ENDGAME, cp=20, line="e7a7"),
    ]
    data = "".join(json.dumps(rec) + "\n" for rec in lines).encode()
    path = tmp_path / "evals.jsonl.zst"
    path.write_bytes(zstandard.ZstdCompressor().compress(data))

    stats = convert(path, tmp_path / "shards", shard_size=2, workers=0)
    assert (stats["sampled"], stats["kept"], stats["shards_written"]) == (4, 3, 2)
    _, scores, mates = load_shards(sorted((tmp_path / "shards").glob("shard_evaldb_*.npz")))
    assert scores.tolist() == [-69, SCORE_LIMIT_CP, -20]
    assert mates.tolist() == [0, 15, 0]

    rerun = convert(path, tmp_path / "shards", shard_size=2, workers=0)
    assert rerun["shards_written"] == 0 and rerun["kept"] == 3
    assert np.load(tmp_path / "shards" / "shard_evaldb_0001.npz")["score_cp"].tolist() == [-20]
