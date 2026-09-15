import json

import chess

from chess_nn.label import record_line, row_for

FEN = "4k3/8/8/4p3/3Q4/8/8/4K3 b - - 0 1"


def test_record_marks_positions_the_quiet_filter_drops():
    capture = ((850, 0), "e5d4")
    line = json.loads(record_line(FEN, capture, row_for(FEN, capture)))
    assert line == {"fen": FEN, "score_cp": 850, "mate": 0, "best": "e5d4", "kept": False}


def test_record_keeps_quiet_best_moves_and_unscored_positions():
    quiet = ((-20, 0), "e8d7")
    assert json.loads(record_line(FEN, quiet, row_for(FEN, quiet)))["kept"] is True
    assert json.loads(record_line(FEN, None, None)) == {"fen": FEN, "score_cp": None, "mate": None, "best": None, "kept": False}
    assert row_for(FEN, quiet)[3] is not None and chess.Board(FEN).is_valid()
