import numpy as np
import torch.nn as nn
import torch


class Actor(nn.Module):
    def __init__(self, obs_dim: int = 16, action_dim: int = 4):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, action_dim), std=0.01),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class Critic(nn.Module):
    def __init__(self, obs_dim: int = 4):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 1), std=1.0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    torch.nn.init.orthogonal_(layer.weight, std)  # type: ignore
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer
