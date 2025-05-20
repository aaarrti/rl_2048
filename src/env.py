from typing import Any
from enum import IntEnum
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numba import njit


class Direction(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class Game2048Env(gym.Env):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = 4
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0, high=2**16, shape=(self.size * self.size,), dtype=np.int32
        )
        self.board = np.zeros((self.size, self.size), dtype=np.int32)

    def reset(self, seed: int | None = None, options=None) -> tuple[np.ndarray, dict]:  # type: ignore
        super().reset(seed=seed)
        self.board.fill(0)
        self.board = add_random_tile(add_random_tile(self.board))
        return self.board.flatten(), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:

        action = Direction(action)
        self.board, moved, reward = move(self.board, action)
        if moved:
            self.board = add_random_tile(self.board)
        done = not any_moves_left(self.board)
        return self.board.flatten(), reward, done, False, {}

    def render(self):
        raise NotImplementedError()

    def close(self):
        pass


@njit
def compress(row: np.ndarray) -> tuple[np.ndarray, int]:
    non_zero = row[row != 0]
    merged = []
    reward = 0
    skip = False
    i = 0
    while i < len(non_zero):
        if not skip and i + 1 < len(non_zero) and non_zero[i] == non_zero[i + 1]:
            val = non_zero[i] * 2
            merged.append(val)
            reward += val
            skip = True
        else:
            if not skip:
                merged.append(non_zero[i])
            skip = False
        i += 1 if not skip else 2
        skip = False

    merged += [0] * (len(row) - len(merged))
    return np.array(merged, dtype=row.dtype), reward


@njit
def move(board: np.ndarray, direction: Direction) -> tuple[np.ndarray, bool, float]:
    board = board.copy()
    reward_total = 0.0
    moved = False
    size = board.shape[0]
    for i in range(size):
        if direction == Direction.UP:
            col, reward = compress(board[:, i])
            if not np.array_equal(board[:, i], col):
                moved = True
            board[:, i] = col
        elif direction == Direction.DOWN:
            col, reward = compress(board[::-1, i])
            col = col[::-1]
            if not np.array_equal(board[:, i], col):
                moved = True
            board[:, i] = col
        elif direction == Direction.LEFT:
            row, reward = compress(board[i])
            if not np.array_equal(board[i], row):
                moved = True
            board[i] = row
        elif direction == Direction.RIGHT:
            row, reward = compress(board[i][::-1])
            row = row[::-1]
            if not np.array_equal(board[i], row):
                moved = True
            board[i] = row
        reward_total += reward  # type: ignore
    return board, moved, reward_total


@njit
def any_moves_left(board: np.ndarray) -> bool:
    return bool(np.any(get_action_mask(board)))


@njit
def get_action_mask(board: np.ndarray) -> np.ndarray:
    if np.any(board == 0):
        return np.array([True, True, True, True])

    return np.array(
        [
            move(board, Direction.UP)[1],
            move(board, Direction.DOWN)[1],
            move(board, Direction.LEFT)[1],
            move(board, Direction.RIGHT)[1],
        ],
    )


@njit
def add_random_tile(board: np.ndarray) -> np.ndarray:
    # board = board.copy()
    empty_count = 0
    size = board.shape[0]

    # First count empty cells
    for i in range(size):
        for j in range(size):
            if board[i, j] == 0:
                empty_count += 1

    if empty_count == 0:
        return board  # No space to add

    # Choose the n-th empty cell
    target_index = np.random.randint(0, empty_count)
    value = 4 if np.random.random() < 0.1 else 2

    # Find and fill that empty cell
    empty_seen = 0
    for i in range(size):
        for j in range(size):
            if board[i, j] == 0:
                if empty_seen == target_index:
                    board[i, j] = value
                    return board
                empty_seen += 1

    return board  # fallback
