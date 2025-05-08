from ml_collections.config_dict import ConfigDict


def get_config() -> ConfigDict:
    config = ConfigDict()
    # Number available actions in environment
    config.num_actions = 4
    config.momentum = 0.9
    # Total number of frames seen during training.
    config.total_frames = 400_000
    # The learning rate for the Adam optimizer.
    config.learning_rate = 2.5e-4
    # Batch size used in training.
    config.batch_size = 32
    # Number of agents playing in parallel.
    config.num_agents = 8
    # Number of steps each agent performs in one policy unroll.
    config.actor_steps = 8
    # Number of training epochs per each unroll of the policy.
    config.num_epochs = 40
    # RL discount parameter.
    config.gamma = 0.99
    # Generalized Advantage Estimation parameter.
    config.lambda_ = 0.95
    # The PPO clipping parameter used to clamp ratios in loss function.
    config.clip_param = 0.1
    # Weight of value function loss in the total loss.
    config.vf_coeff = 0.5
    # Weight of entropy bonus in the total loss.
    config.entropy_coeff = 0.01
    # Linearly decay learning rate and clipping parameter to zero during the training.
    config.decaying_lr_and_clip_param = True
    # Log metric every n epoch.
    config.log_frequency = 40
    # Limit of steps for the environment.
    config.max_steps = 1000
    # Seed for the PRNG
    config.seed = 69
    return config
