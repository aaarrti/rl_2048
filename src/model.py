import numpy as np
import torch.nn as nn
import torch


class DuelingDQN(nn.Module):
    """
    Dueling Deep Q-Network with shared base, value, and advantage streams.

    Based on:
        Wang et al. (2016), "Dueling Network Architectures for Deep Reinforcement Learning"
        https://arxiv.org/abs/1511.06581

    """

    def __init__(self, input_dim: int = 16, n_actions: int = 4, hidden_dim: int = 64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.advantage = nn.Linear(hidden_dim, n_actions)
        self.value = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)  # Flatten 4x4 board → (B, 16)
        features = self.shared(x)
        adv = self.advantage(features)
        val = self.value(features)
        q = val + (adv - adv.mean(dim=1, keepdim=True))
        return q


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


def make_activation(activation: str | None) -> nn.Module:

    match activation:
        case "relu":
            return nn.ReLU()
        case "selu":
            return nn.SELU()
        case "gelu":
            return nn.GELU()
        case None:
            return nn.Identity()
        case _:
            raise ValueError(f"Unknow activation {activation}")
