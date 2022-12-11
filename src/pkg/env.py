from __future__ import annotations

import numpy as np
import gym
import gym.spaces as spaces
from gym.utils import seeding

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Optional, Tuple, Dict


class Base2048Env(gym.Env):
    metadata = {
        "render.modes": ["human"],
    }
    LEFT = 0
    UP = 1
    RIGHT = 2
    DOWN = 3

    ACTION_STRING = {
        LEFT: "left",
        UP: "up",
        RIGHT: "right",
        DOWN: "down",
    }

    board: np.ndarray

    def __init__(self, width=4, height=4, seed: Optional[int] = 0):
        self.width = width
        self.height = height

        self.observation_space = spaces.Box(
            low=2, high=2**32, shape=(self.width, self.height), dtype=np.int64
        )
        self.action_space = spaces.Discrete(4)

        # Internal Variables
        self.np_random, _ = seeding.np_random(seed)
        self.reset()

    def step(self, action: int) -> Tuple[np.ndarray, np.float, bool, bool, Dict]:
        # Align board action with left action
        rotated_obs = np.rot90(self.board, k=action)
        reward, updated_obs = self._slide_left_and_merge(rotated_obs)
        self.board = np.rot90(updated_obs, k=4 - action)
        # Place one random tile on empty location
        self._place_random_tiles(self.board, count=1)
        done = self.is_done()
        return self.board, reward, done, False, {}

    def is_done(self) -> bool:
        copy_board = self.board.copy()
        if not copy_board.all():
            return False
        for action in [0, 1, 2, 3]:
            rotated_obs = np.rot90(copy_board, k=action)
            _, updated_obs = self._slide_left_and_merge(rotated_obs)
            if not updated_obs.all():
                return False
        return True

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> np.ndarray:
        """Place 2 tiles on empty board."""
        self.board = np.zeros((self.width, self.height), dtype=np.int64)
        self.board = self._place_random_tiles(self.board.copy(), count=2)
        return self.board

    def _sample_tiles(self, count=1) -> np.ndarray:
        """Sample tile 2 or 4."""

        choices = [2, 4]
        probs = [0.9, 0.1]

        tiles = self.np_random.choice(choices, size=count, p=probs)
        return tiles

    def _sample_tile_locations(self, board: np.ndarray, count: int = 1) -> np.ndarray:
        """Sample grid locations with no tile."""

        zero_locs = np.argwhere(board == 0)
        zero_indices = self.np_random.choice(len(zero_locs), size=count)

        zero_pos = zero_locs[zero_indices]
        zero_pos = list(zip(*zero_pos))
        zero_pos = np.asarray(zero_pos)
        if count == 1:
            zero_pos = np.expand_dims(zero_pos, 0)
        return zero_pos

    def _place_random_tiles(self, board: np.ndarray, count=1) -> np.ndarray:
        if board.all():
            return board

        tiles = self._sample_tiles(count)
        tile_locs = self._sample_tile_locations(board, count)
        for i in range(count):
            board[tile_locs[i][0], tile_locs[i][1]] = tiles[i]
        return board

    def _slide_left_and_merge(self, board: np.ndarray) -> Tuple[float, np.ndarray]:
        """Slide tiles on a grid to the left and merge."""

        result = []

        score = 0
        for row in board:
            row = np.extract(row > 0, row)
            score_, result_row = self._try_merge(row)
            score += score_
            row = np.pad(
                np.array(result_row),
                (0, self.width - len(result_row)),
                "constant",
                constant_values=(0,),
            )
            result.append(row)

        return score, np.array(result, dtype=np.int64)

    @staticmethod
    def _try_merge(row: np.ndarray) -> Tuple[int, np.ndarray]:
        score = 0
        result_row = []

        i = 1
        while i < len(row):
            if row[i] == row[i - 1]:
                score += row[i] + row[i - 1]
                result_row.append(row[i] + row[i - 1])
                i += 2
            else:
                result_row.append(row[i - 1])
                i += 1

        if i == len(row):
            result_row.append(row[i - 1])

        return score, np.asarray(result_row)

    def render(self, mode: str = "human"):
        print()
        for i in self.board:
            print("-" * 29)
            print("|", end="")
            for j in i:
                if j == 0:
                    print(f"{' ':<6}|", end="")
                else:
                    print(f"{j:<6}|", end="")
            print()
        print("-" * 29)

    def get_action_mask(self) -> np.ndarray:
        # TODO
        return np.asarray([1, 1, 1, 1])
