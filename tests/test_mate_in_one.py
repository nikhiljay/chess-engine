import io

import chess
import chess.pgn

from chess_nn.mate_in_one import allows_mate, safe_moves, scan_game, searched_reply, unseen_mate

# The diagnostic game: the net (Black) plays 32...Nxe4 and allows 33. Rh8#.
DIAGNOSTIC_PGN = """[White "Human"]
[Black "jj6-h1024-s1000-cp20"]
[Result "1-0"]

1. d4 Nf6 2. c4 e6 3. Nc3 Nc6 4. Nf3 Be7 5. Bg5 O-O 6. e3 h6 7. Bh4 g5 8. Bg3 d6
9. h3 Nh5 10. Bh2 Ng7 11. Bd3 f5 12. a3 Bf6 13. Qc2 Qe8 14. O-O-O e5 15. dxe5 dxe5
16. Nd2 e4 17. Be2 Qf7 18. Nd5 Bd8 19. Nb3 Be6 20. Nc5 Bc8 21. Rd2 b6
22. Nb3 Bb7 23. Rhd1 h5 24. g4 fxg4 25. hxg4 hxg4 26. Bxg4 Kh8 27. Rh1 Nh5
28. Nd4 Nxd4 29. Rxd4 c5 30. Rxe4 Nf6 31. Be5+ Kg8 32. f3 Nxe4 33. Rh8# 1-0
"""


def diagnostic_decision() -> tuple[chess.Board, chess.Move]:
    game = chess.pgn.read_game(io.StringIO(DIAGNOSTIC_PGN))
    moves = list(game.mainline_moves())
    board = game.board()
    for move in moves[:63]:  # through 32. f3
        board.push(move)
    return board, moves[63]


def test_diagnostic_nxe4_allows_the_unseen_rh8_mate():
    board, nxe4 = diagnostic_decision()
    assert board.san(nxe4) == "Nxe4"
    assert unseen_mate(board, nxe4) == chess.Move.from_uci("h1h8")
    assert allows_mate(board, nxe4)
    assert safe_moves(board), "some Black move allows no mate in one"


def test_mates_the_search_examines_are_not_unseen():
    # 1. e4 e5 2. Bc4 Nc6 3. Qh5 and now ...Nf6?? allows Qxf7#, a capture the search follows.
    board = chess.Board()
    for san in ["e4", "e5", "Bc4", "Nc6", "Qh5"]:
        board.push_san(san)
    nf6 = board.parse_san("Nf6")
    assert allows_mate(board, nf6)
    assert unseen_mate(board, nf6) is None


def test_searched_replies_match_the_harness_capture_list():
    castling = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    assert searched_reply(castling, chess.Move.from_uci("e1g1"))
    en_passant = chess.Board("rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3")
    assert not searched_reply(en_passant, chess.Move.from_uci("e5f6"))
    assert searched_reply(chess.Board("8/P6k/8/8/8/8/8/K7 w - - 0 1"), chess.Move.from_uci("a7a8q"))
    assert not searched_reply(chess.Board(), chess.Move.from_uci("g1f3"))


def test_scan_finds_the_diagnostic_decision_in_a_game_record():
    game = chess.pgn.read_game(io.StringIO(DIAGNOSTIC_PGN))
    record = {"start_fen": chess.STARTING_FEN, "moves": [m.uci() for m in game.mainline_moves()], "net_white": False, "level": 0}
    [item] = scan_game((record, "diagnostic"))
    assert item["ply"] == 63 and item["played"] == "f6e4" and item["mate"] == "h1h8" and item["safe"]
