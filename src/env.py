from typing import Any
from enum import IntEnum
import gymnasium as gym
from gymnasium.utils import seeding
import numpy as np
from gymnasium import spaces
from numba import njit
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QLabel, QGridLayout
from PyQt5.QtCore import QTimer

from typing import Protocol


class SelectAction(Protocol):

    def __call__(self, observation: np.ndarray, mask: np.ndarray) -> int: ...


class Direction(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class Game2048Env(gym.Env):

    metadata = {"vectorizable": True}

    def __init__(self, use_custom_reward: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = 4
        self.use_custom_reward = use_custom_reward
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0, high=2**16, shape=(self.size * self.size,), dtype=np.int32
        )
        self.board = np.zeros((self.size, self.size), dtype=np.int32)

    def reset(self, seed: int | None = None, options=None) -> tuple[np.ndarray, dict]:  # type: ignore
        super().reset(seed=seed)
        self.board.fill(0)
        self.add_random_tile()
        self.add_random_tile()
        return self.board.flatten(), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = Direction(action)
        if self.use_custom_reward:
            new_board, moved, merge_score = move(self.board, action)
            reward = custom_reward(self.board, new_board, merge_score)
            self.board = new_board
        else:
            self.board, moved, reward = move(self.board, action)
        if moved:
            self.add_random_tile()
        done = not any_moves_left(self.board)
        return self.board.flatten(), reward, done, False, {}

    def render(self):
        raise NotImplementedError()

    def close(self):
        pass

    def get_action_mask(self) -> np.ndarray:
        return get_action_mask(self.board)

    def add_random_tile(self):
        self.board = add_random_tile(self.board, self.np_random)


class Agent2048Player(QWidget):
    def __init__(self, select_action: SelectAction):
        super().__init__()
        self.select_action = select_action
        self._np_random, _ = seeding.np_random(22)
        self.setWindowTitle("2048 Agent")
        self.grid = QGridLayout()
        self.setLayout(self.grid)
        self.cells = [[QLabel("0") for _ in range(4)] for _ in range(4)]
        for i in range(4):
            for j in range(4):
                label = self.cells[i][j]
                label.setFixedSize(80, 80)
                label.setStyleSheet(
                    "background-color: lightgray; font-size: 24px; text-align: center;"
                )
                label.setAlignment(Qt.AlignCenter)  # type: ignore
                self.grid.addWidget(label, i, j)

        self.reset_game()

        self.timer = QTimer()
        self.timer.timeout.connect(self.agent_step)
        self.timer.start(300)

    def reset_game(self):
        self.board = np.zeros((4, 4), dtype=np.int32)
        self.add_random_tile()
        self.add_random_tile()
        self.update_board()

    def agent_step(self):
        mask = get_action_mask(self.board)
        if not mask.any():
            self.timer.stop()
            print("Game Over!")
            return

        action = self.select_action(self.board, mask)
        action = Direction(action)
        # print(f"{action = }")
        self.board, moved, _ = move(self.board, action)
        if moved:
            self.add_random_tile()

        self.update_board()

    def update_board(self):
        for i in range(4):
            for j in range(4):
                val = self.board[i, j]
                self.cells[i][j].setText(str(val) if val != 0 else "")
                self.cells[i][j].setStyleSheet(
                    f"""
                    background-color: {'#EEE4DA' if val else 'lightgray'};
                    font-size: 24px;
                    text-align: center;
                """
                )

    def add_random_tile(self):
        self.board = add_random_tile(self.board, self._np_random)


@njit
def custom_reward(board_before: np.ndarray, board_after: np.ndarray, merge_score: float) -> float:
    """
    Compute a shaped reward for 2048 that includes:
    - Merge reward (raw score)
    - Max tile bonus
    - Smoothness penalty
    - Movement bonus

    Args:
        board_before: The board before the move (4x4 int array).
        board_after: The board after the move.
        merge_score: The raw merge score from tile combining.

    Returns:
        A float reward value.
    """

    # Max tile bonus: encourage reaching higher values
    max_tile_after = np.max(board_after)
    max_tile_bonus = np.log2(max_tile_after) * 0.1 if max_tile_after > 0 else 0.0

    # Smoothness penalty: penalize boards with high variance
    smoothness_penalty = -np.std(board_after) / 10.0

    # Movement bonus: reward if board changed significantly
    moved = not np.array_equal(board_before, board_after)
    movement_bonus = 0.1 if moved else 0.0

    # Total reward: merge reward + tile progress - chaos + motion
    reward = merge_score + max_tile_bonus + smoothness_penalty + movement_bonus
    return reward  # type: ignore


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
    return np.array(
        [
            move(board, Direction.UP)[1],
            move(board, Direction.DOWN)[1],
            move(board, Direction.LEFT)[1],
            move(board, Direction.RIGHT)[1],
        ],
    )


@njit
def add_random_tile(board: np.ndarray, rng: np.random.Generator) -> np.ndarray:
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
    target_index = rng.integers(0, empty_count)
    value = 4 if rng.random() < 0.1 else 2

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
