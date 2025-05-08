from __future__ import annotations

from typing import Protocol


class ConfigProto(Protocol):

    num_actions: int
    momentum: float
    total_frames: int
    learning_rate: float
    batch_size: int
    num_agents: int
    actor_steps: int
    num_epochs: int
    gamma: int
    lambda_: float
    clip_param: float
    vf_coeff: float
    entropy_coeff: float
    decaying_lr_and_clip_param: bool
    log_frequency: int
    max_steps: int
    seed: int
