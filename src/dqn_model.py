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
