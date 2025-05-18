import random

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class Game2048Env(gym.Env):
    metadata = {"render_modes": ["human, ascii"], "render_fps": 4, "is_parallelizable": True}

    def __init__(self):
        super(Game2048Env, self).__init__()
        self.size = 4
        self.action_space = spaces.Discrete(4)  # 0=up, 1=down, 2=left, 3=right
        self.observation_space = spaces.Box(
            low=0, high=2**16, shape=(self.size * self.size,), dtype=np.int32
        )
        self.board = np.zeros((self.size, self.size), dtype=np.int32)

    def reset(self, seed=None, options=None) -> tuple[np.ndarray, dict]:  # type: ignore
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

    def _move(self, board, direction):
        def compress(row):
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
            return np.array(new_row + [0] * (self.size - len(new_row))), reward

        reward_total = 0
        moved = False
        for i in range(self.size):
            if direction == 0:  # up
                col, reward = compress(board[:, i])
                if not np.array_equal(board[:, i], col):
                    moved = True
                board[:, i] = col
            elif direction == 1:  # down
                col, reward = compress(board[::-1, i])
                col = col[::-1]
                if not np.array_equal(board[:, i], col):
                    moved = True
                board[:, i] = col
            elif direction == 2:  # left
                row, reward = compress(board[i])
                if not np.array_equal(board[i], row):
                    moved = True
                board[i] = row
            elif direction == 3:  # right
                row, reward = compress(board[i][::-1])
                row = row[::-1]
                if not np.array_equal(board[i], row):
                    moved = True
                board[i] = row
            reward_total += reward  # type: ignore
        return moved, reward_total

    def step(self, action):
        old_board = self.board.copy()
        moved, reward = self._move(self.board, action)

        if moved:
            self._add_random_tile()

        done = not self._any_moves_left()
        return self.board.flatten(), reward, done, False, {}

    def _any_moves_left(self):
        if np.any(self.board == 0):
            return True
        for i in range(self.size):
            for j in range(self.size - 1):
                if (
                    self.board[i][j] == self.board[i][j + 1]
                    or self.board[j][i] == self.board[j + 1][i]
                ):
                    return True
        return False

    def render(self):
        print(self.board)

    def close(self):
        pass
