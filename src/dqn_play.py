import sys
import numpy as np
import torch


from PyQt5.QtWidgets import QApplication
from dqn_model import DuelingDQN
from env import Agent2048Player

# UserWarning: TensorFloat32 tensor cores for float32 matrix multiplication available but not enabled.
# Consider setting `torch.set_float32_matmul_precision('high')` for better performance.
torch.set_float32_matmul_precision("high")


def main():
    device = torch.device("cuda")
    model = DuelingDQN().to(device)
    model.load_state_dict(torch.load("models/dqn.pt", map_location=device))
    model.eval()

    def select_action(observation: np.ndarray, mask: np.ndarray) -> int:
        with torch.inference_mode():
            board = torch.tensor(
                observation.flatten(), device=device, dtype=torch.float32
            ).unsqueeze(0)
            q_values = model(board)[0].cpu().numpy()
            q_values[~mask] = -1e9
            return int(np.argmax(q_values))

    app = QApplication(sys.argv)
    window = Agent2048Player(select_action)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
