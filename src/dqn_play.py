import sys
import numpy as np
import jax
from flax.traverse_util import unflatten_dict
from PyQt5.QtWidgets import QApplication

from dqn_model import DuelingDQN
from env import Agent2048Player


def main():
    model = DuelingDQN()

    params = np.load("models/dqn.npz")
    params = {i: params[i] for i in params.keys()}
    params = unflatten_dict(params, sep="/")

    @jax.jit
    def pred_fn(x):
        return model.apply(params, x)

    def select_action(observation: np.ndarray, mask: np.ndarray) -> int:
        q_values = pred_fn(observation.reshape(1, 16))
        q_values = np.array(q_values)[0]

        q_values[~mask] = -1e9
        return int(np.argmax(q_values))

    app = QApplication(sys.argv)
    window = Agent2048Player(select_action)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
