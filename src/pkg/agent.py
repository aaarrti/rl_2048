# Copyright 2022 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Agent utilities, incl. choosing the move and running in a separate process."""

from __future__ import annotations
import collections
import functools
import multiprocessing

import flax
import jax
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, Callable
from .env_utils import create_env


@functools.partial(jax.jit, static_argnums=0)
def policy_action(
    apply_fn: Callable[..., Any],
    params: flax.core.frozen_dict.FrozenDict,
    state: np.ndarray,
):
    """Forward pass of the network."""
    out = apply_fn({"params": params}, state)
    return out


ExpTuple = collections.namedtuple(
    "ExpTuple", ["state", "action", "reward", "value", "log_prob", "done"]
)


class RemoteSimulator:
    """Wrap functionality for an agent emulating game in a separate process."""

    def __init__(self, max_steps: int):
        """Start the remote process and create Pipe() to communicate with it."""
        parent_conn, child_conn = multiprocessing.Pipe()
        self.proc = multiprocessing.Process(
            target=rcv_action_send_exp, args=(child_conn, max_steps)
        )
        self.proc.daemon = True
        self.conn = parent_conn
        self.proc.start()


def rcv_action_send_exp(conn: multiprocessing.Pipe, max_steps: int):
    """Run the remote agents."""
    env = create_env(True, max_steps)

    while True:
        obs = env.reset()
        done = False
        # Observations fetched from Atari env need additional batch dimension.
        state = obs[None, ...]
        while not done:
            conn.send(state)
            action = conn.recv()

            next_state = obs[None, ...] if not done else None
            obs, reward, done, truncated, _ = env.step(action)

            experience = (state, action, reward, done or truncated)
            conn.send(experience)
            if done or truncated:
                break
            state = next_state
