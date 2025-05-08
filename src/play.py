from __future__ import annotations

import logging as base_logger
import time
from typing import TYPE_CHECKING

import jax
import numpy as np
from absl import app, flags, logging
from flax.linen import jit
from flax.serialization import from_state_dict, to_state_dict
from flax.traverse_util import flatten_dict, unflatten_dict
from ml_collections.config_dict import ConfigDict
from ml_collections.config_flags import config_flags
from safetensors.flax import load_file, save_file

from pkg.config import ConfigProto
from pkg.models import ActorCritic
from pkg.util import get_initial_params
from pkg.env_utils import create_env
from pkg.agent import policy_action


CONFIG = config_flags.DEFINE_config_file("config", default="src/config.py")


def main(config: ConfigProto):
    state_dict = load_file("models/params.safetensors")
    state_dict = unflatten_dict(state_dict, sep="/")

    model = jit(ActorCritic)(num_outputs=config.num_actions)
    # fmt: off
    init_params = get_initial_params(jax.random.PRNGKey(config.seed), model)  # noqa
    # fmt: on
    params = from_state_dict(init_params, state_dict)
    del init_params
    game = create_env(False, 100)

    observation, _ = game.reset()

    while True:
        time.sleep(0.5)
        observation = np.expand_dims(observation, 0)
        game.render()
        log_probs, values = policy_action(model.apply, params, observation)
        log_probs, values = jax.device_get((log_probs, values))
        probs = np.exp(np.array(log_probs))
        action = np.random.choice(probs.shape[1], p=probs[0])
        observation, _, done, truncated, _ = game.step(action)
        if done or truncated:
            break


def entrypoint(*args):
    config = CONFIG.value
    main(config)


if __name__ == "__main__":
    app.run(entrypoint)
