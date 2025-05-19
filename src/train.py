from rich import print
import rich.traceback
import numpy as np
import torch


from pkg.env import Game2048Env, get_action_mask
from pkg.agent import DQNAgent

NUM_EPISODES = 2000
BATCH_SIZE = 32

rich.traceback.install(show_locals=True)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = Game2048Env()
    obs, _ = env.reset()

    agent = DQNAgent(device=device)

    for episode in range(NUM_EPISODES):
        obs, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            flat_obs = obs.flatten().astype(np.float32)
            mask = get_action_mask(obs.reshape((4, 4)))
            action = agent.select_action(flat_obs, mask)

            next_obs, reward, done, _, _ = env.step(action)
            next_flat = next_obs.flatten().astype(np.float32)
            next_mask = get_action_mask(next_obs.reshape((4, 4)))

            agent.buffer.add(flat_obs, action, reward, next_flat, done, next_mask)
            agent.train_step()

            obs = next_obs
            total_reward += reward

        print(f"Episode {episode:4d} — Reward: {total_reward:4d} — ε: {agent.epsilon:.3f}")


if __name__ == "__main__":
    main()
