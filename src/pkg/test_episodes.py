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

"""Test policy by playing a full game."""
from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Callable
    from flax.core.frozen_dict import FrozenDict
import numpy as np

from . import env_utils, agent


def policy_test(
    n_episodes: int, apply_fn: Callable, params: FrozenDict, max_steps: int
) -> float:
    """Perform a test of the policy in Game environment."""
    total_reward = 0.0
    test_env = env_utils.create_env(False, max_steps)
    for _ in range(n_episodes):
        obs = test_env.reset()
        state = obs[None, ...]  # add batch dimension
        total_reward = 0.0
        for _ in itertools.count():
            log_probs, _ = agent.policy_action(apply_fn, params, state)
            probs = np.exp(np.array(log_probs, dtype=np.float32))
            probabilities = probs[0] / probs[0].sum()
            action = np.random.choice(probs.shape[1], p=probabilities)
            obs, reward, done, truncated, _ = test_env.step(action)
            stopped = done or truncated
            total_reward += reward
            next_state = obs[None, ...] if not stopped else None
            state = next_state
            if stopped:
                break
    return total_reward
