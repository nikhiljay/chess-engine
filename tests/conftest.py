import random

import chess
import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: end-to-end tests needing stockfish and the harness binaries")


def random_positions(rng: random.Random, count: int) -> list[chess.Board]:
    """Non-terminal, not-in-check positions reached by random play."""
    boards = []
    while len(boards) < count:
        board = chess.Board()
        for _ in range(rng.randint(4, 80)):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
        if not board.is_check() and any(board.generate_legal_moves()):
            boards.append(board.copy(stack=False))
    return boards


@pytest.fixture
def rng() -> random.Random:
    return random.Random(1234)
