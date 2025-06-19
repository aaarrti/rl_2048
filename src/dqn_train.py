import jax
import jax.numpy as jnp
import numpy as np
import optax
from collections import deque
from typing import NamedTuple
from ml_collections import config_flags
from absl import app, logging
from flax.traverse_util import flatten_dict
import random

from env import Game2048Env
from dqn_config import ConfigProto
from dqn_model import DuelingDQN


_CONFIG = config_flags.DEFINE_config_file("config", default="src/dqn_config.py")

logging.set_verbosity(logging.INFO)
random.seed(22)


class Transition(NamedTuple):
    state: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    next_state: jnp.ndarray
    done: jnp.ndarray


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*map(jnp.asarray, zip(*batch)))

    def __len__(self):
        return len(self.buffer)


def main(config: ConfigProto):

    @jax.jit
    def loss_fn(params, target_params, batch: Transition, gamma: float):
        nonlocal model
        q_values = model.apply(params, batch.state)
        q_action = jax.vmap(lambda q, a: q[a])(q_values, batch.action)

        next_q = model.apply(target_params, batch.next_state)
        max_next_q = jnp.max(next_q, axis=1)  # type: ignore
        target = batch.reward + gamma * max_next_q * (1.0 - batch.done)

        return jnp.mean((q_action - target) ** 2)

    def select_action(
        params,
        state: jnp.ndarray | np.ndarray,
        epsilon: float,
        mask: jnp.ndarray | np.ndarray | None = None,
        n_actions: int = 4,
    ):
        if random.random() < epsilon:
            if mask is not None:
                valid_actions = np.where(mask)[0]
                return int(random.choice(valid_actions))
            else:
                return random.randint(0, n_actions - 1)
        else:
            q_values = model.apply(params, state)
            if mask is not None:
                q_values = jnp.where(mask, q_values, -jnp.inf)  # type: ignore
            return int(jnp.argmax(q_values))  # type: ignore

    env = Game2048Env(use_custom_reward=config.use_custom_reward)
    obs_shape = env.observation_space.shape[0]  # type: ignore
    model = DuelingDQN()
    rng = jax.random.PRNGKey(config.prng_seed)
    dummy_input = jnp.ones((1, obs_shape))
    params = model.init(rng, dummy_input)
    del rng
    target_params = params

    optimizer = optax.adam(config.learning_rate)
    opt_state = optimizer.init(params)

    buffer = ReplayBuffer(config.replay_buffer_capacity)

    step = 0
    episode_rewards = []

    for episode in range(1, config.num_episodes + 1):
        state, _ = env.reset()
        state = jnp.array(state).flatten()
        done = False
        total_reward = 0
        epsilon = config.epsilon

        while not done:
            mask = env.get_action_mask()
            action = select_action(params, state[None, :], epsilon, mask=mask)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = jnp.array(next_state).flatten()

            buffer.push(state, action, reward, next_state, float(done))
            state = next_state
            total_reward += reward

            if len(buffer) >= config.batch_size:
                batch = buffer.sample(config.batch_size)

                grad_fn = jax.value_and_grad(loss_fn)
                _, grads = grad_fn(params, target_params, batch, config.gamma)
                updates, opt_state = optimizer.update(grads, opt_state)
                params = optax.apply_updates(params, updates)

                if step % config.update_target_every == 0:
                    target_params = params

                epsilon = max(config.min_epsilon, epsilon * config.epsilon_decay)
                step += 1

        episode_rewards.append(total_reward)
        if episode == 1 or episode % 10 == 0 or episode == config.num_episodes:
            avg = np.mean(episode_rewards[-10:])
            logging.info(f"Episode {episode}, Avg Reward: {avg:.2f}, Epsilon: {epsilon:.2f}")

    flat_params = flatten_dict(params, sep="/")
    np.savez("models/dqn.npz", **flat_params)  # type: ignore


def entrypoint(_):
    main(_CONFIG.value)


if __name__ == "__main__":
    app.run(entrypoint)
