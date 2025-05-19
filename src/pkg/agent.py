import random
import numpy as np
from collections import deque
import torch
import torch.nn.functional as F

from pkg.model import DuelingDQN


class ReplayBuffer:
    def __init__(self, capacity: int = 100_000):
        self.buffer = deque(maxlen=capacity)

    def add(self, s: np.ndarray, a: int, r: float, s2: np.ndarray, d: bool, mask: np.ndarray):
        """
        Store a single transition in the replay buffer.

        Args:
            s (np.ndarray): The current observation (state), shape = [obs_dim]
            a (int): The action taken in this state.
            r (float): The reward received after taking the action.
            s2 (np.ndarray): The next observation (next state), shape = [obs_dim]
            d (bool): Whether the episode ended after this transition (done flag).
            mask (np.ndarray): Boolean array of valid actions for the next state, shape = [num_actions]
        """
        self.buffer.append((s, a, r, s2, d, mask))

    def sample(
        self, batch_size: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, d, mask = map(np.stack, zip(*batch, strict=True))
        return s, a, r, s2, d, mask

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    Deep Q-Network (DQN) Agent for discrete action spaces.

    This agent implements the DQN algorithm introduced in:

        Mnih et al. (2013), "Playing Atari with Deep Reinforcement Learning"
        https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

    Key features:
    - ε-greedy exploration
    - Experience replay buffer
    - Target network with periodic updates
    - Support for action masking (for environments like 2048)

    Args:
        obs_dim (int): Dimensionality of input observations.
        n_actions (int): Number of discrete actions.
        device (str): Device identifier, e.g., "cpu" or "cuda".

    Attributes:
        q_net (nn.Module): The online Q-network.
        target_net (nn.Module): The target Q-network, updated periodically.
        optimizer (torch.optim.Optimizer): Optimizer for q_net.
        buffer (ReplayBuffer): Stores past transitions for training.
        gamma (float): Discount factor for future rewards.
        epsilon (float): Current exploration rate.
        epsilon_min (float): Minimum value for ε-greedy policy.
        epsilon_decay (float): Multiplicative decay factor for ε.
        update_target_every (int): Frequency (in steps) for syncing target_net.
        learn_every (int): Frequency (in steps) for training updates.
        step_count (int): Tracks the number of environment steps.
    """

    def __init__(
        self,
        obs_dim: int = 16,
        n_actions: int = 4,
        device: str | torch.device = "cpu",
        batch_size: int = 32,
    ):
        self.device = device
        self.q_net = DuelingDQN(obs_dim).to(device)
        self.target_net = DuelingDQN(obs_dim).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=1e-3)
        self.buffer = ReplayBuffer()
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        # self.update_target_every = 500
        self.learn_every = 4
        self.step_count = 0
        self.n_actions = n_actions
        self.batch_size = 32

    def select_action(self, state: np.ndarray, mask: np.ndarray) -> int:
        if not mask.any():
            return 0  # fallback dummy action if all invalid (should only happen at terminal)

        if np.random.rand() < self.epsilon:
            return np.random.choice(np.flatnonzero(mask))
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            q_values = self.q_net(state_tensor)[0].detach().cpu().numpy()
            q_values[~mask] = -1e9
            return int(np.argmax(q_values))

    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return

        s, a, r, s2, d, mask = self.buffer.sample(self.batch_size)

        s = torch.tensor(s, dtype=torch.float32).to(self.device)
        a = torch.tensor(a, dtype=torch.int64).unsqueeze(1).to(self.device)
        r = torch.tensor(r, dtype=torch.float32).unsqueeze(1).to(self.device)
        s2 = torch.tensor(s2, dtype=torch.float32).to(self.device)
        d = torch.tensor(d, dtype=torch.float32).unsqueeze(1).to(self.device)
        mask = torch.tensor(mask, dtype=torch.bool).to(self.device)

        q = self.q_net(s).gather(1, a)

        # Double DQN target
        next_q_online = self.q_net(s2)
        next_q_online[~mask] = -1e9
        best_actions = torch.argmax(next_q_online, dim=1, keepdim=True)

        next_q_target = self.target_net(s2)
        target = r + self.gamma * (1 - d) * next_q_target.gather(1, best_actions)

        loss = F.mse_loss(q, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step_count += 1
        # if self.step_count % self.update_target_every == 0:
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
