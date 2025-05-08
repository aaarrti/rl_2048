from __future__ import annotations

import itertools
from collections.abc import Callable
from functools import partial
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np
import optax
from absl import app, logging
from flax.serialization import to_state_dict
from flax.training.train_state import TrainState
from flax.traverse_util import flatten_dict
from ml_collections.config_flags import config_flags
from safetensors.flax import save_file

from pkg.agent import ExpTuple, RemoteSimulator, policy_action
from pkg.config import ConfigProto
from pkg.env_utils import create_env
from pkg.models import ActorCritic, get_initial_params

CONFIG = config_flags.DEFINE_config_file("config", default="src/config.py")

type TODO = Any


def policy_test(
    n_episodes: int, apply_fn: Callable, params: Mapping[str, Any], max_steps: int
) -> float:
    """Perform a test of the policy in Game environment."""
    total_reward = 0.0
    test_env = create_env(False, max_steps)
    for _ in range(n_episodes):
        obs, _ = test_env.reset()
        state = obs[None, ...]  # add batch dimension
        total_reward = 0.0
        for _ in itertools.count():
            log_probs, _ = policy_action(apply_fn, params, state)
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


@jax.jit
@partial(jax.vmap, in_axes=(1, 1, 1, None, None), out_axes=1)
def gae_advantages(
    rewards: np.ndarray,
    terminal_masks: np.ndarray,
    values: np.ndarray,
    discount: float,
    gae_param: float,
) -> jax.Array:
    """Use Generalized Advantage Estimation (GAE) to compute advantages.

    As defined by eqs. (11-12) in PPO paper arXiv: 1707.06347. Implementation uses
    key observation that A_{t} = delta_t + gamma*lambda*A_{t+1}.

    Args:
      rewards: array shaped (actor_steps, num_agents), rewards from the game
      terminal_masks: array shaped (actor_steps, num_agents), zeros for terminal
                      and ones for non-terminal states
      values: array shaped (actor_steps, num_agents), values estimated by critic
      discount: RL discount usually denoted with gamma
      gae_param: GAE parameter usually denoted with lambda

    Returns:
      advantages: calculated advantages shaped (actor_steps, num_agents)
    """
    assert rewards.shape[0] + 1 == values.shape[0], (
        "One more value needed; Eq. " "(12) in PPO paper requires " "V(s_{t+1}) for delta_t"
    )
    advantages = []
    gae = 0.0
    for t in reversed(range(len(rewards))):
        # Masks used to set next state value to 0 for terminal states.
        value_diff = discount * values[t + 1] * terminal_masks[t] - values[t]
        delta = rewards[t] + value_diff
        # Masks[t] used to ensure that values before and after a terminal state
        # are independent of each other.
        gae = delta + discount * gae_param * terminal_masks[t] * gae
        advantages.append(gae)
    advantages = advantages[::-1]
    return jnp.array(advantages)


@partial(jax.jit, static_argnums=(2,))
def train_step(
    state: TrainState,
    trajectories: tuple,
    batch_size: int,
    *,
    clip_param: float,
    vf_coeff: float,
    entropy_coeff: float,
) -> tuple[TrainState, float]:
    """Compilable train step.

    Runs an entire epoch of training (i.e. the loop over minibatches within
    an epoch is included here for performance reasons).

    Args:
      state: the train state
      trajectories: Tuple of the following five elements forming the experience:
                    states: shape (steps_per_agent*num_agents, 84, 84, 4)
                    actions: shape (steps_per_agent*num_agents, 84, 84, 4)
                    old_log_probs: shape (steps_per_agent*num_agents, )
                    returns: shape (steps_per_agent*num_agents, )
                    advantages: (steps_per_agent*num_agents, )
      batch_size: the minibatch size, static argument
      clip_param: the PPO clipping parameter used to clamp ratios in loss function
      vf_coeff: weighs value function loss in total loss
      entropy_coeff: weighs entropy bonus in the total loss

    Returns:
      optimizer: new optimizer after the parameters update
      loss: loss summed over training steps
    """
    iterations = trajectories[0].shape[0] // batch_size
    trajectories = jax.tree_util.tree_map(
        lambda x: x.reshape((iterations, batch_size) + x.shape[1:]), trajectories
    )
    loss = 0.0
    for batch in zip(*trajectories):
        grad_fn = jax.value_and_grad(loss_fn)
        l, grads = grad_fn(state.params, state.apply_fn, batch, clip_param, vf_coeff, entropy_coeff)
        loss += l
        state = state.apply_gradients(grads=grads)
    return state, loss


def get_experience(
    state: TrainState,
    simulators: list[RemoteSimulator],
    steps_per_actor: int,
) -> list[list[ExpTuple]]:
    """Collect experience from agents.

    Runs `steps_per_actor` time steps of the game for each of the `simulators`.
    """
    all_experience = []
    # Range up to steps_per_actor + 1 to get one more value needed for GAE.
    for _ in range(steps_per_actor + 1):
        sim_states = []
        for sim in simulators:
            sim_state = sim.conn.recv()
            sim_states.append(sim_state)
        sim_states = np.concatenate(sim_states, axis=0)
        log_probs, values = policy_action(state.apply_fn, state.params, sim_states)
        log_probs, values = jax.device_get((log_probs, values))
        probs = np.exp(np.array(log_probs))
        for i, sim in enumerate(simulators):
            probabilities = probs[i]
            action = np.random.choice(probs.shape[1], p=probabilities)
            sim.conn.send(action)
        experiences = []
        for i, sim in enumerate(simulators):
            sim_state, action, reward, done = sim.conn.recv()
            value = values[i, 0]
            log_prob = log_probs[i][action]
            sample = ExpTuple(sim_state, action, reward, value, log_prob, done)
            experiences.append(sample)
        all_experience.append(experiences)
    return all_experience


def process_experience(
    experience: list[list[ExpTuple]],
    actor_steps: int,
    num_agents: int,
    gamma: float,
    lambda_: float,
):
    """Process experience for training, including advantage estimation.

    Args:
      experience: collected from agents in the form of nested lists/namedtuple
      actor_steps: number of steps each agent has completed
      num_agents: number of agents that collected experience
      gamma: dicount parameter
      lambda_: GAE parameter

    Returns:
      trajectories: trajectories readily accessible for `train_step()` function
    """
    obs_shape = (16,)
    exp_dims = (actor_steps, num_agents)
    values_dims = (actor_steps + 1, num_agents)
    states = np.zeros(exp_dims + obs_shape, dtype=np.float32)
    actions = np.zeros(exp_dims, dtype=np.int32)
    rewards = np.zeros(exp_dims, dtype=np.float32)
    values = np.zeros(values_dims, dtype=np.float32)
    log_probs = np.zeros(exp_dims, dtype=np.float32)
    dones = np.zeros(exp_dims, dtype=np.float32)

    for t in range(len(experience) - 1):  # experience[-1] only for next_values
        for agent_id, exp_agent in enumerate(experience[t]):
            states[t, agent_id, ...] = exp_agent.state
            actions[t, agent_id] = exp_agent.action
            rewards[t, agent_id] = exp_agent.reward
            values[t, agent_id] = exp_agent.value
            log_probs[t, agent_id] = exp_agent.log_prob
            # Dones need to be 0 for terminal states.
            dones[t, agent_id] = float(not exp_agent.done)
    for a in range(num_agents):
        values[-1, a] = experience[-1][a].value
    advantages = gae_advantages(rewards, dones, values, gamma, lambda_)
    returns = advantages + values[:-1, :]
    # After preprocessing, concatenate data from all agents.
    trajectories = (states, actions, log_probs, returns, advantages)
    trajectory_len = num_agents * actor_steps
    trajectories = tuple(
        map(lambda x: np.reshape(x, (trajectory_len,) + x.shape[2:]), trajectories)
    )
    return trajectories


def create_train_state(
    params: Mapping[str, Any], model: ActorCritic, config: ConfigProto, train_steps: int
) -> TrainState:
    if config.decaying_lr_and_clip_param:
        lr = optax.linear_schedule(
            init_value=config.learning_rate, end_value=0.0, transition_steps=train_steps
        )
    else:
        lr = config.learning_rate
    tx = optax.adam(lr)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    return state


def loss_fn(
    params: Mapping[str, Any],
    apply_fn: TODO,
    minibatch: TODO,
    clip_param: float,
    vf_coeff: float,
    entropy_coeff: float,
) -> float:
    """Evaluate the loss function.

    Compute loss as a sum of three components: the negative of the PPO clipped
    surrogate objective, the value function loss and the negative of the entropy
    bonus.

    Args:
      params: the parameters of the actor-critic model
      apply_fn: the actor-critic model's apply function
      minibatch: Tuple of five elements forming one experience batch:
                 states: shape (batch_size, 84, 84, 4)
                 actions: shape (batch_size, 84, 84, 4)
                 old_log_probs: shape (batch_size,)
                 returns: shape (batch_size,)
                 advantages: shape (batch_size,)
      clip_param: the PPO clipping parameter used to clamp ratios in loss function
      vf_coeff: weighs value function loss in total loss
      entropy_coeff: weighs entropy bonus in the total loss

    Returns:
      loss: the PPO loss, scalar quantity
    """
    states, actions, old_log_probs, returns, advantages = minibatch
    log_probs, values = policy_action(apply_fn, params, states)
    values = values[:, 0]  # Convert shapes: (batch, 1) to (batch, ).
    probs = jnp.exp(log_probs)

    value_loss = jnp.mean(jnp.square(returns - values), axis=0)

    entropy = jnp.sum(-probs * log_probs, axis=1).mean()

    log_probs_act_taken = jax.vmap(lambda lp, a: lp[a])(log_probs, actions)
    ratios = jnp.exp(log_probs_act_taken - old_log_probs)
    # Advantage normalization (following the OpenAI baselines).
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    pg_loss = ratios * advantages
    clipped_loss = advantages * jax.lax.clamp(1.0 - clip_param, ratios, 1.0 + clip_param)
    ppo_loss = -jnp.mean(jnp.minimum(pg_loss, clipped_loss), axis=0)

    return ppo_loss + vf_coeff * value_loss - entropy_coeff * entropy  # type: ignore


def train_loop(model: ActorCritic, config: ConfigProto) -> TrainState:

    simulators = [RemoteSimulator(config.max_steps) for _ in range(config.num_agents)]

    loop_steps = config.total_frames // (config.num_agents * config.actor_steps)
    log_frequency = config.log_frequency
    # train_step does multiple steps per call for better performance
    # compute number of steps per call here to convert between the number of
    # train steps and the inner number of optimizer steps
    iterations_per_step = config.num_agents * config.actor_steps // config.batch_size

    initial_params = get_initial_params(jax.random.PRNGKey(config.seed), model)
    state = create_train_state(
        initial_params,
        model,
        config,
        loop_steps * config.num_epochs * iterations_per_step,
    )
    del initial_params
    # number of train iterations done by each train_step

    start_step = int(state.step) // config.num_epochs // iterations_per_step
    logging.info("Start training from step: %s", start_step)

    for step in range(start_step, loop_steps):
        # Bookkeeping and testing.
        if step % log_frequency == 0:
            score = policy_test(1, state.apply_fn, state.params, config.max_steps)
            frames = step * config.num_agents * config.actor_steps
            logging.info(
                "Step %s/%s:\nframes seen %s\nscore %s\n\n", step, loop_steps, frames, score
            )

        # Core training code.
        alpha = 1.0 - step / loop_steps if config.decaying_lr_and_clip_param else 1.0
        all_experiences = get_experience(state, simulators, config.actor_steps)
        trajectories = process_experience(
            all_experiences,
            config.actor_steps,
            config.num_agents,
            config.gamma,
            config.lambda_,
        )
        clip_param = config.clip_param * alpha
        for _ in range(config.num_epochs):
            permutation = np.random.permutation(config.num_agents * config.actor_steps)
            trajectories = tuple(x[permutation] for x in trajectories)
            state, _ = train_step(
                state,
                trajectories,
                config.batch_size,
                clip_param=clip_param,
                vf_coeff=config.vf_coeff,
                entropy_coeff=config.entropy_coeff,
            )
    return state


def main(config: ConfigProto):
    model = ActorCritic(num_outputs=config.num_actions)
    state = train_loop(model, config)
    state_dict = to_state_dict(state.params)
    state_dict = flatten_dict(state_dict, sep="/")
    save_file(state_dict, "models/params.safetensors")  # type: ignore


def entrypoint(*args):
    config = CONFIG.value
    logging.debug("-" * 100)
    logging.debug(f"{jax.devices() = }")
    logging.debug(f"{config = }")
    logging.debug("-" * 100)
    logging.debug(f"Playing 2048 with {config.num_actions} actions")
    main(config)


if __name__ == "__main__":
    app.run(entrypoint)
