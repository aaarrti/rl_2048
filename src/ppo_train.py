import argparse
import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
from torch.distributions.categorical import Categorical
import itertools
from collections.abc import Sequence
from rich import print


from env import Game2048Env
from ppo_model import Actor, Critic

np.random.seed(22)
torch.manual_seed(22)
# UserWarning: TensorFloat32 tensor cores for float32 matrix multiplication available but not enabled.
# Consider setting `torch.set_float32_matmul_precision('high')` for better performance.
torch.set_float32_matmul_precision("high")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RolloutBuffer:
    def __init__(
        self,
        num_rollout_steps: int,
        num_envs: int = 4,
        obs_shape: Sequence[int] = (16,),
        action_shape: Sequence[int] = (4,),
        lambda_gae: float = 0.95,
        gamma: float = 0.99,
    ):
        """_summary_

        Args:
            num_rollout_steps (int): _description_
            num_envs (int, optional): _description_. Defaults to 4.
            obs_shape (Sequence[int], optional): _description_. Defaults to (16,).
            action_shape (Sequence[int], optional): _description_. Defaults to (4,).
            lambda_gae (float, optional): _description_. Defaults to 0.95.
            gamma (float, optional): Disacount factor. Defaults to 0.99.
        """
        self.num_rollout_steps = num_rollout_steps
        self.num_envs = num_envs
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.gamma = gamma
        self.lambda_gae = lambda_gae

        self.observations = np.zeros((num_rollout_steps, num_envs, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((num_rollout_steps, num_envs, *action_shape), dtype=np.int64)
        self.log_probs = np.zeros((num_rollout_steps, num_envs), dtype=np.float32)
        self.rewards = np.zeros((num_rollout_steps, num_envs), dtype=np.float32)
        self.dones = np.zeros((num_rollout_steps, num_envs), dtype=np.bool_)
        self.values = np.zeros((num_rollout_steps, num_envs), dtype=np.float32)
        self.action_masks = np.zeros((num_rollout_steps, num_envs, *action_shape), dtype=np.bool_)
        self.advantages = np.zeros((num_rollout_steps, num_envs), dtype=np.float32)
        self.returns = np.zeros((num_rollout_steps, num_envs), dtype=np.float32)
        self.ptr = 0

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        log_prob: np.ndarray,
        reward: np.ndarray,
        done: bool,
        value: np.ndarray,
        action_mask: np.ndarray,
    ):

        assert self.ptr < self.num_rollout_steps

        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.log_probs[self.ptr] = log_prob
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.values[self.ptr] = value
        self.action_masks[self.ptr] = action_mask  # Store action_mask
        self.ptr += 1

    def compute_returns_and_advantages(
        self, next_value_bootstrap: torch.Tensor, next_done_bootstrap: torch.Tensor
    ):
        last_gae_lam = 0
        for t in reversed(range(self.num_rollout_steps)):
            if t == self.num_rollout_steps - 1:
                next_non_terminal = 1.0 - next_done_bootstrap
                next_values = next_value_bootstrap
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_values = self.values[t + 1]

            delta = self.rewards[t] + self.gamma * next_values * next_non_terminal - self.values[t]
            self.advantages[t] = last_gae_lam = (
                delta + self.gamma * self.lambda_gae * next_non_terminal * last_gae_lam
            )

        self.returns = self.advantages + self.values

    def get_batch(self):
        if self.ptr != self.num_rollout_steps:
            print(
                f"Warning: RolloutBuffer not full. Current ptr: {self.ptr}, Expected: {self.num_rollout_steps}"
            )

        num_total_samples = self.num_rollout_steps * self.num_envs

        b_obs = self.observations.reshape((num_total_samples, *self.obs_shape))
        b_actions = self.actions.reshape((num_total_samples, *self.action_shape)).squeeze()
        b_log_probs = self.log_probs.reshape(num_total_samples)
        b_advantages = self.advantages.reshape(num_total_samples)
        b_returns = self.returns.reshape(num_total_samples)
        b_values = self.values.reshape(num_total_samples)
        b_action_masks = self.action_masks.reshape((num_total_samples, *self.action_shape))

        b_obs_tensor = torch.tensor(b_obs, dtype=torch.float32)
        b_actions_tensor = torch.tensor(b_actions, dtype=torch.long).to(device)
        b_log_probs_tensor = torch.tensor(b_log_probs, dtype=torch.float32).to(device)
        b_advantages_tensor = torch.tensor(b_advantages, dtype=torch.float32).to(device)
        b_returns_tensor = torch.tensor(b_returns, dtype=torch.float32).to(device)
        b_values_tensor = torch.tensor(b_values, dtype=torch.float32).to(device)
        b_action_masks_tensor = torch.tensor(b_action_masks, dtype=torch.bool).to(device)

        return (
            b_obs_tensor,
            b_actions_tensor,
            b_log_probs_tensor,
            b_advantages_tensor,
            b_returns_tensor,
            b_values_tensor,
            b_action_masks_tensor,
        )

    def reset(self):
        self.ptr = 0
        self.observations.fill(0)
        self.actions.fill(0)
        self.log_probs.fill(0)
        self.rewards.fill(0)
        self.dones.fill(0)
        self.action_masks.fill(0)
        self.advantages.fill(0)
        self.returns.fill(0)


# --- Neural Networks ---


class PPOAgent:

    def __init__(
        self,
        obs_dim: int = 16,
        action_dim: int = 4,
        learning_rate: float = 3e-4,
        num_total_samples: int = 1_000_000,
        num_epochs: int = 1,
        batch_size: int = 64,
        clip_epsilon: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        max_grad_norm: float = 0.5,
    ):
        """_summary_

        Args:
            obs_dim (int, optional): _description_. Defaults to 16.
            action_dim (int, optional): _description_. Defaults to 4.
            learning_rate (float, optional): _description_. Defaults to 3e-4.
            num_total_samples (int, optional): _description_. Defaults to 1_000_000.
            num_epochs (int, optional): _description_. Defaults to 1.
            batch_size (int, optional): _description_. Defaults to 64.
            clip_epsilon (float, optional): PPO clipping parameter. Defaults to 0.2.
            vf_coef (float, optional): Value function loss coefficient. Defaults to 0.5.
            ent_coef (float, optional): Entropy bonus coefficient. Defaults to 0.01.
        """
        self.actor = Actor(obs_dim, action_dim).to(device)
        self.critic = Critic(obs_dim).to(device)
        # self.actor = torch.compile(self.actor_uncompiled, fullgraph=True, mode="max-autotune")
        # self.critic = torch.compile(self.critic_uncompiled, fullgraph=True, mode="max-autotune")
        all_params = itertools.chain(self.actor.parameters(), self.critic.parameters())  # type: ignore
        self.optimizer = optim.Adam(all_params, lr=learning_rate, eps=1e-5)
        self.num_total_samples = num_total_samples
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.clip_epsilon = clip_epsilon
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm

    def _preprocess_obs(self, obs_np_batch):
        if not isinstance(obs_np_batch, torch.Tensor):
            obs_np_batch = np.array(obs_np_batch, dtype=np.float32)
        else:
            obs_np_batch = obs_np_batch.cpu().numpy().astype(np.float32)

        obs_log2 = np.log2(obs_np_batch + 1e-9)
        obs_log2[obs_np_batch == 0] = 0
        flat_obs = obs_log2.reshape(obs_np_batch.shape[0], -1)
        return torch.tensor(flat_obs, dtype=torch.float32).to(device)

    def get_action_value_and_log_prob_for_step(
        self, obs_tensor_processed: torch.Tensor, action_masks_tensor: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        obs_tensor_processed: (num_envs, flat_obs_dim)
        action_masks_tensor: (num_envs, action_dim), boolean
        """
        logits = self.actor(obs_tensor_processed)

        # Apply action masking
        # Set logits of invalid actions to negative infinity
        masked_logits = torch.where(
            action_masks_tensor, logits, torch.tensor(float("-inf")).to(device)
        )

        distribution = Categorical(logits=masked_logits)
        action = distribution.sample()
        log_prob = distribution.log_prob(action)
        value = self.critic(obs_tensor_processed).squeeze(-1)
        return action, value, log_prob

    def evaluate_actions(
        self,
        obs_tensor_processed: torch.Tensor,
        actions_tensor: torch.Tensor,
        action_masks_tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        obs_tensor_processed: (batch_size, flat_obs_dim)
        actions_tensor: (batch_size,)
        action_masks_tensor: (batch_size, action_dim), boolean
        """
        logits = self.actor(obs_tensor_processed)

        # Apply action masking
        masked_logits = torch.where(
            action_masks_tensor, logits, torch.tensor(float("-inf")).to(device)
        )

        distribution = Categorical(logits=masked_logits)
        new_log_probs = distribution.log_prob(actions_tensor)
        entropy = distribution.entropy()
        new_values = self.critic(obs_tensor_processed).squeeze(-1)
        return new_log_probs, entropy, new_values

    def update(self, rollout_buffer: RolloutBuffer):
        (
            b_obs_raw,
            b_actions,
            b_log_probs_old,
            b_advantages,
            b_returns,
            b_values_old,
            b_action_masks,
        ) = rollout_buffer.get_batch()

        num_total_samples = self.num_total_samples
        indices = np.arange(num_total_samples)

        clip_epsilon = self.clip_epsilon

        for _ in range(self.num_epochs):
            np.random.shuffle(indices)
            for start in range(0, num_total_samples, self.batch_size):

                # torch.compiler.cudagraph_mark_step_begin()
                end = start + self.batch_size
                mb_indices = indices[start:end]

                mb_obs_raw = b_obs_raw[mb_indices]
                mb_actions = b_actions[mb_indices]
                mb_log_probs_old = b_log_probs_old[mb_indices]
                mb_advantages = b_advantages[mb_indices]
                mb_returns = b_returns[mb_indices]
                mb_values_old_ref = b_values_old[mb_indices]
                mb_action_masks = b_action_masks[mb_indices]  # Get minibatch action masks

                mb_obs_processed = self._preprocess_obs(mb_obs_raw)

                # Pass action masks to evaluate_actions
                new_log_probs, entropy, new_values = self.evaluate_actions(
                    mb_obs_processed, mb_actions, mb_action_masks
                )

                logratio = new_log_probs - mb_log_probs_old
                ratio = torch.exp(logratio)

                norm_mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                    mb_advantages.std() + 1e-8
                )

                pg_loss1 = -norm_mb_advantages * ratio
                pg_loss2 = -norm_mb_advantages * torch.clamp(
                    ratio, 1 - clip_epsilon, 1 + clip_epsilon
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss_unclipped = (new_values - mb_returns) ** 2
                v_clipped = mb_values_old_ref + torch.clamp(
                    new_values - mb_values_old_ref, -clip_epsilon, clip_epsilon
                )
                v_loss_clipped = (v_clipped - mb_returns) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                vf_loss = 0.5 * v_loss_max.mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - self.ent_coef * entropy_loss + self.vf_coef * vf_loss

                self.optimizer.zero_grad()
                loss.backward()
                all_params = itertools.chain(self.actor.parameters(), self.critic.parameters())  # type: ignore
                torch.nn.utils.clip_grad_norm_(all_params, self.max_grad_norm)
                self.optimizer.step()

    def save(self):
        torch.save(self.actor.state_dict(), "models/ppo.actor.pt")
        torch.save(self.critic.state_dict(), "models/ppo.critic.pt")


def make_env(rank: int, seed: int = 22):

    def _init():
        env = Game2048Env()
        env.reset(seed=seed + rank)
        return env

    return _init


def main(num_envs: int, total_timesteps: int, batch_size: int, epochs: int):
    num_rollout_steps = 2 - 48 // num_envs

    envs = gym.vector.SyncVectorEnv([make_env(i) for i in range(num_envs)])
    agent = PPOAgent(num_epochs=epochs, batch_size=batch_size, num_total_samples=total_timesteps)
    rollout_buffer = RolloutBuffer(num_rollout_steps, num_envs=num_envs)

    obs, _ = envs.reset()

    num_updates = total_timesteps // num_envs

    for update_idx in range(1, num_updates + 1):
        action_masks_list = envs.call("get_action_mask")
        action_masks_np = np.stack(action_masks_list, axis=0)
        action_masks_tensor = torch.tensor(action_masks_np, dtype=torch.bool).to(device)

        obs_processed = agent._preprocess_obs(obs)

        with torch.no_grad():
            # Pass action masks to the agent for sampling
            action, value, log_prob = agent.get_action_value_and_log_prob_for_step(
                obs_processed, action_masks_tensor
            )

        next_obs, reward, terminated, truncated, _ = envs.step(action.cpu().numpy())
        done = terminated | truncated

        rollout_buffer.add(
            obs,
            action.cpu().numpy(),
            log_prob.cpu().numpy(),
            reward,
            done,
            value.cpu().numpy(),
            action_masks_np,
        )
        obs = next_obs

        with torch.no_grad():
            next_obs_processed = agent._preprocess_obs(next_obs)  # type: ignore
            # For bootstrap value, we don't strictly need the mask of next_obs,
            # as we are just estimating V(s_T+1).
            # However, if the critic's evaluation somehow depended on valid actions (not typical),
            # we might fetch it. For standard PPO, critic is independent of action masks.
            next_value_bootstrap = agent.critic(next_obs_processed).squeeze(-1).cpu().numpy()
            last_step_dones = rollout_buffer.dones[num_rollout_steps - 1]
            rollout_buffer.compute_returns_and_advantages(next_value_bootstrap, last_step_dones)

        agent.update(rollout_buffer)
        rollout_buffer.reset()

        if update_idx % 10 == 0:  # type: ignore
            print(f"Completed PPO Update {update_idx}/{num_updates}")  # type: ignore

    envs.close()
    print("Training finished.")
    agent.save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    main(**vars(parser.parse_args()))
