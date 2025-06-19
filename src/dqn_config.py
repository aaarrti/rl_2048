from ml_collections import ConfigDict
from typing import Protocol


class ConfigProto(Protocol):
    num_episodes: int
    batch_size: int
    update_target_every: int
    use_custom_reward: bool
    replay_buffer_capacity: int
    gamma: float
    epsilon_decay: float
    min_epsilon: float
    epsilon: float
    prng_seed: int
    learning_rate: float


def get_config():
    config = ConfigDict()
    config.num_episodes = 10_000
    config.batch_size = 1024
    config.update_target_every = 2_000
    config.use_custom_reward = False
    config.replay_buffer_capacity = 100_000
    config.gamma = 0.99
    config.epsilon = 1.0
    config.epsilon_decay = 0.995
    config.min_epsilon = 0.1
    config.prng_seed = 22
    config.learning_rate = 1e-3
    return config
