import numpy as np

from pkg.env import compress, move, Direction, get_action_mask, any_moves_left


def test_compress_no_merge():
    row = np.array([2, 4, 8, 16])
    compressed, reward = compress(row)
    assert np.array_equal(compressed, row)
    assert reward == 0


def test_compress_with_merge():
    row = np.array([2, 2, 0, 0])
    compressed, reward = compress(row)
    assert np.array_equal(compressed, np.array([4, 0, 0, 0]))
    assert reward == 4


def test_move_left_merge():
    board = np.array(
        [
            [2, 2, 0, 0],
            [4, 0, 4, 0],
            [2, 2, 2, 2],
            [0, 0, 0, 0],
        ]
    )
    new_board, moved, reward = move(board, Direction.LEFT)
    expected_board = np.array(
        [
            [4, 0, 0, 0],
            [8, 0, 0, 0],
            [4, 2, 2, 0],
            [0, 0, 0, 0],
        ]
    )
    assert moved is True
    assert np.array_equal(new_board, expected_board)
    assert reward == 4 + 8 + 4  # 16


def test_move_no_change():
    board = np.array(
        [
            [2, 4, 8, 16],
            [32, 64, 128, 256],
            [512, 1024, 2, 4],
            [8, 16, 32, 64],
        ]
    )
    new_board, moved, reward = move(board, Direction.RIGHT)
    assert moved is False
    assert np.array_equal(new_board, board)
    assert reward == 0


def test_get_action_mask_some_valid():
    board = np.array(
        [
            [2, 2, 0, 0],
            [0, 0, 0, 0],
            [2, 0, 2, 0],
            [0, 0, 0, 0],
        ]
    )
    mask = get_action_mask(board)
    assert mask.dtype == bool
    assert mask.shape == (4,)
    assert mask[Direction.LEFT]
    assert mask[Direction.RIGHT]
    assert mask[Direction.UP]
    assert mask[Direction.DOWN]


def test_get_action_mask_none_valid():
    board = np.array(
        [
            [2, 4, 8, 16],
            [32, 64, 128, 256],
            [512, 1024, 2, 4],
            [8, 16, 32, 64],
        ]
    )
    mask = get_action_mask(board)
    assert mask.shape == (4,)
    assert not mask.any()


def test_any_moves_left_true():
    board = np.array(
        [
            [2, 2, 0, 0],
            [4, 0, 0, 0],
            [2, 2, 2, 2],
            [0, 0, 0, 0],
        ]
    )
    assert any_moves_left(board)


def test_any_moves_left_false():
    board = np.array(
        [
            [2, 4, 8, 16],
            [32, 64, 128, 256],
            [512, 1024, 2, 4],
            [8, 16, 32, 64],
        ]
    )
    assert not any_moves_left(board)


def test_merge_right():
    board = np.array(
        [
            [0, 0, 2, 2],
            [0, 4, 0, 4],
            [2, 2, 2, 2],
            [0, 0, 0, 0],
        ]
    )
    new_board, moved, reward = move(board, Direction.RIGHT)

    expected = np.array(
        [
            [0, 0, 0, 4],
            [0, 0, 0, 8],
            [0, 2, 2, 4],
            [0, 0, 0, 0],
        ]
    )
    assert moved is True
    assert reward == 4 + 8 + 4
    assert np.array_equal(new_board, expected)


def test_merge_up():
    board = np.array(
        [
            [2, 4, 0, 0],
            [2, 4, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
    )
    new_board, moved, reward = move(board, Direction.UP)

    expected = np.array(
        [
            [4, 8, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
    )
    assert moved is True
    assert reward == 4 + 8
    assert np.array_equal(new_board, expected)


def test_merge_down():
    board = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [2, 4, 0, 0],
            [2, 4, 0, 0],
        ]
    )
    new_board, moved, reward = move(board, Direction.DOWN)

    expected = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [4, 8, 0, 0],
        ]
    )
    assert moved is True
    assert reward == 4 + 8
    assert np.array_equal(new_board, expected)
