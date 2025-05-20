import sys
import numpy as np
import torch
from rich import print

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QGridLayout
from PyQt5.QtCore import QTimer

from model import DuelingDQN
from env import move, get_action_mask, add_random_tile, Direction


# UserWarning: TensorFloat32 tensor cores for float32 matrix multiplication available but not enabled.
# Consider setting `torch.set_float32_matmul_precision('high')` for better performance.
torch.set_float32_matmul_precision("high")


class DQN2048Player(QWidget):
    def __init__(self, model_path: str, device: torch.device | str = "cpu"):
        super().__init__()

        self.model = DuelingDQN().to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        self.model = torch.compile(self.model, fullgraph=True, mode="max-autotune")

        self.setWindowTitle("2048 Agent")
        self.grid = QGridLayout()
        self.setLayout(self.grid)
        self.cells = [[QLabel("0") for _ in range(4)] for _ in range(4)]
        for i in range(4):
            for j in range(4):
                label = self.cells[i][j]
                label.setFixedSize(80, 80)
                label.setStyleSheet(
                    "background-color: lightgray; font-size: 24px; text-align: center;"
                )
                label.setAlignment(Qt.AlignCenter)  # type: ignore
                self.grid.addWidget(label, i, j)

        self.reset_game()

        self.timer = QTimer()
        self.timer.timeout.connect(self.agent_step)
        self.timer.start(300)
        self.device = device

    def reset_game(self):
        self.board = np.zeros((4, 4), dtype=np.int32)
        self.board = add_random_tile(self.board)
        self.board = add_random_tile(self.board)
        self.update_board()

    def board_to_tensor(self) -> torch.Tensor:
        return torch.tensor(
            self.board.flatten(), dtype=torch.float32, device=self.device
        ).unsqueeze(0)

    def select_action(self) -> Direction | None:
        mask = get_action_mask(self.board)
        if not mask.any():
            return None
        with torch.inference_mode():
            q_values = self.model(self.board_to_tensor())[0].cpu().numpy()
            q_values[~mask] = -1e9
            action = int(np.argmax(q_values))
            return Direction(action)

    def agent_step(self):
        action = self.select_action()
        if action is None:
            self.timer.stop()
            print("Game Over!")
            return

        # print(f"{action = }")
        new_board, moved, _ = move(self.board, action)
        if moved:
            new_board = add_random_tile(new_board)

        self.board = new_board
        self.update_board()

    def update_board(self):
        for i in range(4):
            for j in range(4):
                val = self.board[i, j]
                self.cells[i][j].setText(str(val) if val != 0 else "")
                self.cells[i][j].setStyleSheet(
                    f"""
                    background-color: {'#EEE4DA' if val else 'lightgray'};
                    font-size: 24px;
                    text-align: center;
                """
                )


def main():
    device = torch.device("cuda")
    app = QApplication(sys.argv)
    window = DQN2048Player(model_path="models/weights.pt", device=device)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
