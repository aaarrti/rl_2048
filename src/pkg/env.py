import random
from typing import Any
from enum import IntEnum
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QPainter, QColor, QFont
from PyQt5.QtCore import Qt
from numba import njit


class Direction(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class Game2048Env(gym.Env):
    metadata = {"render_modes": ["human, ascii"], "render_fps": 4, "is_parallelizable": True}

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
        self._add_random_tile()
        self._add_random_tile()
        return self.board.flatten(), {}

    def _add_random_tile(self):
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            i, j = random.choice(empty_cells)
            self.board[i][j] = 2 if random.random() < 0.9 else 4

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = Direction(action)
        self.board, moved, reward = move(self.board, action)
        if moved:
            self._add_random_tile()
        done = not any_moves_left(self.board)
        return self.board.flatten(), reward, done, False, {}

    def render(self):
        match self.render_mode:
            case "ascii":
                print(self.board)
            case "human":
                raise NotImplementedError()
            case mode:
                raise ValueError(f"Unknown render mode {mode}")

    def close(self):
        pass


class Game2048Renderer(QWidget):
    def __init__(self, state_getter, cell_size=100, padding=10):
        super().__init__()
        self.get_state = state_getter
        self.cell_size = cell_size
        self.padding = padding
        self.board_size = 4
        self.setWindowTitle("2048 - PyQt Renderer")

        total_size = self.board_size * self.cell_size + (self.board_size + 1) * self.padding
        self.setFixedSize(total_size, total_size)

        self.show()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        board = self.get_state()

        for row in range(self.board_size):
            for col in range(self.board_size):
                value = board[row, col]
                x = col * (self.cell_size + self.padding) + self.padding
                y = row * (self.cell_size + self.padding) + self.padding
                self.draw_cell(painter, x, y, value)

    def draw_cell(self, painter, x, y, value):
        rect_color = self.get_color(value)
        painter.setBrush(QColor(*rect_color))
        painter.setPen(Qt.NoPen)
        painter.drawRect(x, y, self.cell_size, self.cell_size)

        if value != 0:
            painter.setPen(Qt.black)
            font = QFont("Arial", 24, QFont.Bold)
            painter.setFont(font)
            text = str(value)
            text_width = painter.fontMetrics().width(text)
            text_height = painter.fontMetrics().height()
            painter.drawText(
                x + (self.cell_size - text_width) / 2,
                y + (self.cell_size + text_height) / 2 - 10,
                text,
            )

    def get_color(self, value):
        # Basic coloring; can be improved
        color_map = {
            0: (205, 193, 180),
            2: (238, 228, 218),
            4: (237, 224, 200),
            8: (242, 177, 121),
            16: (245, 149, 99),
            32: (246, 124, 95),
            64: (246, 94, 59),
            128: (237, 207, 114),
            256: (237, 204, 97),
            512: (237, 200, 80),
            1024: (237, 197, 63),
            2048: (237, 194, 46),
        }
        return color_map.get(value, (60, 58, 50))  # Default dark color for high values

    def update_board(self):
        self.repaint()


# @njit
def compress(row: np.ndarray) -> tuple[np.ndarray, float]:
    size = row.shape[0]
    new_row = row[row != 0]
    new_row = list(new_row)
    i = 0
    reward = 0
    while i < len(new_row) - 1:
        if new_row[i] == new_row[i + 1]:
            new_row[i] *= 2
            reward += new_row[i]
            del new_row[i + 1]
            new_row.append(0)
            i += 1
        i += 1

    return np.array(new_row + [0] * (size - len(new_row))), reward


# @njit
def move(board: np.ndarray, direction: Direction) -> tuple[np.ndarray, bool, float]:
    board = board.copy()
    reward_total = 0
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


# @njit
def any_moves_left(board: np.ndarray) -> bool:
    return bool(np.any(get_action_mask(board)))


# @njit
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
