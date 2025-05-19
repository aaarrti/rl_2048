import argparse
from rich import print
import rich.traceback
import numpy as np
import torch


from pkg.env import Game2048Env, get_action_mask
from pkg.agent import DQNAgent


rich.traceback.install(show_locals=True)

# UserWarning: TensorFloat32 tensor cores for float32 matrix multiplication available but not enabled.
# Consider setting `torch.set_float32_matmul_precision('high')` for better performance.
torch.set_float32_matmul_precision("high")


def main(num_episodes: int, batch_size: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = Game2048Env()
    obs, _ = env.reset()

    agent = DQNAgent(device=device, batch_size=batch_size)

    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            mask = get_action_mask(obs.reshape([4, 4]))
            action = agent.select_action(obs.astype(np.float32), mask)

            next_obs, reward, done, _, _ = env.step(action)
            next_mask = get_action_mask(next_obs.reshape([4, 4]))

            agent.buffer.add(obs, action, reward, next_obs, done, next_mask)
            agent.train_step()

            obs = next_obs
            total_reward += reward

        total_reward = int(total_reward)
        print(f"Episode {episode:4d} — Reward: {total_reward:4d} — ε: {agent.epsilon:.3f}")

    # scripted = torch.jit.trace(agent.q_net_uncompiled, torch.randn(1, 16, device=device))
    # scripted.save("model_s/cripted.pt")  # type: ignore


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-episodes", default=2_000, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    main(**vars(parser.parse_args()))
