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

"""Class and functions to define and initialize the actor-critic model."""

from flax import linen as nn
import jax.numpy as jnp


class ActorCritic(nn.Module):
    """Class defining the actor-critic model."""

    num_outputs: int

    @nn.compact
    def __call__(self, x):
        dtype = jnp.float32
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(16)(x)
        x = nn.Dense(32)(x)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        logits = nn.Dense(self.num_outputs)(x)
        policy_log_probabilities = nn.log_softmax(logits)
        value = nn.Dense(features=1, name="value", dtype=dtype)(x)
        return policy_log_probabilities, value
