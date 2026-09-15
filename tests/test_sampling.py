import io
import random

import chess
import chess.pgn
import zstandard

from chess_nn.sample_positions import (
    iter_games,
    load_opening_keys,
    position_key,
    qualifies,
    sample_game,
)

GOOD_HEADERS = {"WhiteElo": "1900", "BlackElo": "2100", "TimeControl": "300+0", "Termination": "Normal"}

LONG_GAME = (
    "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 "
    "8. c3 O-O 9. h3 Nb8 10. d4 Nbd7 11. c4 c6 12. cxb5 axb5 13. Nc3 Bb7 14. Bg5 b4 *"
)


def _moves(pgn: str) -> list[chess.Move]:
    return list(chess.pgn.read_game(io.StringIO(pgn)).mainline_moves())


def test_qualifies_accepts_rated_normal_game():
    assert qualifies(GOOD_HEADERS)


def test_qualifies_rejects_low_rating_fast_or_abnormal_games():
    assert not qualifies({**GOOD_HEADERS, "BlackElo": "1799"})
    assert not qualifies({**GOOD_HEADERS, "WhiteElo": "?"})
    assert not qualifies({**GOOD_HEADERS, "TimeControl": "60+1"})
    assert not qualifies({**GOOD_HEADERS, "TimeControl": "-"})
    assert not qualifies({**GOOD_HEADERS, "Termination": "Time forfeit"})


def test_sample_game_skips_opening_plies_and_checks():
    boards = sample_game(_moves(LONG_GAME), random.Random(0), set(), per_game=100)
    assert boards
    assert all(len(board.move_stack) == 0 for board in boards)
    assert all(not board.is_check() for board in boards)
    # 28 plies played, first 16 skipped → at most 12 positions, none before move 9 for white.
    assert len(boards) <= 12
    assert all(board.fullmove_number >= 9 for board in boards)


def test_sample_game_caps_positions_per_game():
    assert len(sample_game(_moves(LONG_GAME), random.Random(0), set(), per_game=3)) == 3


def test_sample_game_excludes_opening_positions():
    moves = _moves("1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 *")
    everything = sample_game(moves, random.Random(0), set(), skip_plies=0, per_game=100)
    italian = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3")
    excluded = sample_game(moves, random.Random(0), load_opening_keys(), skip_plies=0, per_game=100)
    assert position_key(italian) in {position_key(b) for b in everything}
    assert position_key(italian) not in {position_key(b) for b in excluded}


def test_official_opening_keys_load():
    assert len(load_opening_keys()) == 25


def _game(white_elo: str) -> str:
    return (
        f'[WhiteElo "{white_elo}"]\n[BlackElo "2000"]\n[TimeControl "600+0"]\n'
        f'[Termination "Normal"]\n\n{LONG_GAME}\n\n'
    )


def test_iter_games_skips_non_qualifying(tmp_path):
    pgn = tmp_path / "games.pgn"
    pgn.write_text(_game("1500") + _game("2000"))
    collectors = list(iter_games(pgn))
    assert [c.skipped for c in collectors] == [True, False]
    assert len(collectors[1].moves) == 28


def test_iter_games_reads_all_zstd_frames(tmp_path):
    compressor = zstandard.ZstdCompressor()
    frames = b"".join(compressor.compress(_game(elo).encode()) for elo in ("2000", "1500", "2100"))
    for name in ("games.pgn.zst", "games.pgn.zst.part"):
        path = tmp_path / name
        path.write_bytes(frames)
        assert [c.skipped for c in iter_games(path)] == [False, True, False]
