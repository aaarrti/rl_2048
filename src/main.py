from __future__ import annotations

import time
import numpy as np
from absl import app, flags, logging
from ml_collections.config_flags import config_flags
from typing import TYPE_CHECKING
import jax
from flax.serialization import to_state_dict, from_state_dict
from flax.linen import jit
import logging as base_logger
from flax.traverse_util import flatten_dict, unflatten_dict
from safetensors.flax import save_file, load_file

if TYPE_CHECKING:
    from ml_collections.config_dict import ConfigDict

from pkg import ppo_lib, models, env_utils, agent

base_logger.getLogger().setLevel(logging.DEBUG)
del base_logger
logging.set_verbosity("debug")
jax.config.update("jax_log_compiles", True)
jax.config.update("jax_debug_nans", True)


FLAGS = flags.FLAGS
flags.DEFINE_enum(
    name="task",
    enum_values=["train", "play"],
    required=True,
    default=None,
    help="Choose which task to execute",
)
CONFIG = config_flags.DEFINE_config_file("config", default="src/config.py")


def train(config: ConfigDict):
    num_actions = CONFIG.value.num_actions
    logging.debug(f"Playing {2048} with {num_actions} actions")
    model = jit(models.ActorCritic)(num_outputs=num_actions)
    state = ppo_lib.train(model, config)
    state_dict = to_state_dict(state.params)
    state_dict = flatten_dict(state_dict, sep="/")
    save_file(state_dict, "models/params.safetensors")


def play(config: ConfigDict):
    state_dict = load_file("models/params.safetensors")
    state_dict = unflatten_dict(state_dict, sep="/")

    model = jit(models.ActorCritic)(num_outputs=config.num_actions)
    # fmt: off
    init_params = ppo_lib.get_initial_params(jax.random.PRNGKey(config.seed), model)  # noqa
    # fmt: on
    params = from_state_dict(init_params, state_dict)
    del init_params
    game = env_utils.create_env(False, 100)

    observation = game.reset()

    while True:
        time.sleep(0.5)
        observation = np.expand_dims(observation, 0)
        game.render()
        log_probs, values = agent.policy_action(model.apply, params, observation)
        log_probs, values = jax.device_get((log_probs, values))
        probs = np.exp(np.array(log_probs))
        action = np.random.choice(probs.shape[1], p=probs[0])
        observation, _, done, truncated, _ = game.step(action)
        if done or truncated:
            break


def main(*args):

    config = CONFIG.value
    logging.debug("-" * 100)
    logging.debug(f"{jax.devices() = }")
    logging.debug(f"{config = }")
    logging.debug("-" * 100)

    if FLAGS.task == "train":
        train(config)

    if FLAGS.task == "play":
        play(config)


if __name__ == "__main__":
    app.run(main)
