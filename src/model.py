from collections.abc import Sequence
from typing import Literal
import torch.nn as nn
import torch


class DuelingDQN(nn.Module):
    """
    Dueling Deep Q-Network with shared base, value, and advantage streams.

    Based on:
        Wang et al. (2016), "Dueling Network Architectures for Deep Reinforcement Learning"
        https://arxiv.org/abs/1511.06581

    Args:
        input_dim (int): Dimensionality of input features (e.g., 16 for flattened 2048 board)
        hidden_dims (list of int): Sizes of hidden layers in the shared trunk
        output_dim (int): Number of discrete actions (e.g., 4 for 2048: up/down/left/right)
        dropout_prob (float): Dropout probability after each activation
        activation (nn.Module): PyTorch activation class (default: GELU)
    """

    def __init__(
        self,
        input_dim: int = 16,
        hidden_dims: Sequence[int] = [64, 64],
        output_dim: int = 4,
        dropout_prob: float = 0.1,
        activation: Literal["relu", "gelu", "selu"] = "gelu",
    ):
        super().__init__()
        self.activation = make_activation(activation)

        # Shared feature extractor
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(self.activation)
            layers.append(nn.Dropout(dropout_prob))
            in_dim = h_dim
        self.shared = nn.Sequential(*layers)

        # Value stream (V(s))
        self.value_head = nn.Sequential(nn.Linear(in_dim, 128), self.activation, nn.Linear(128, 1))

        # Advantage stream (A(s,a))
        self.advantage_head = nn.Sequential(
            nn.Linear(in_dim, 128), self.activation, nn.Linear(128, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.shared(x)  # shape: [batch, feat]
        value = self.value_head(features)  # shape: [batch, 1]
        advantage = self.advantage_head(features)  # shape: [batch, actions]

        # Combine streams into Q(s, a)
        q = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q


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
