import sys
from numba import njit
import numpy as np
from PyQt5.QtWidgets import QApplication
import numpy as np
from env import Agent2048Player, Direction, get_action_mask, move


MAX_DEPTH = 10


@njit
def greedy_dfs_reward(observation: np.ndarray, mask: np.ndarray, current_depth: int = 0):

    if current_depth == MAX_DEPTH:
        return np.max(observation)

    actions = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.DOWN]

    rewards = np.zeros(shape=(4,), dtype=np.float32)

    for action in actions:

        action_id = action.value

        if mask[action_id] == 1:
            new_board, _, _ = move(observation, action)
            new_mask = get_action_mask(new_board)
            rewards[action_id] = greedy_dfs_reward(new_board, new_mask, current_depth + 1)

    return np.max(rewards)


@njit
def select_action(observation: np.ndarray, mask: np.ndarray):

    actions = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.DOWN]
    rewards = np.zeros(shape=(4,), dtype=np.float32)

    for action in actions:

        action_id = action.value

        if mask[action_id] == 1:

            new_board, _, _ = move(observation, action)
            new_mask = get_action_mask(new_board)

            rewards[action_id] = greedy_dfs_reward(new_board, new_mask, 1)

    selected_action = int(np.argmax(rewards))
    return selected_action


def main():

    app = QApplication(sys.argv)
    window = Agent2048Player(select_action)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
