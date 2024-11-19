import random

import numpy as np


class TicTacToe:
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.done = False
        self.winner = None
        self.current_player = 1  # Player 1 is 1 (X), Player 2 is -1 (O)

    def reset(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.done = False
        self.winner = None
        self.current_player = 1
        return self.board.flatten()

    def available_actions(self):
        return [i for i in range(9) if self.board.flatten()[i] == 0]

    def get_reward(self):
        reward = 0
        if self.winner == self.current_player:
            reward = 1
        elif self.winner == 0:
            reward = -1
        return reward

    def step(self, action):
        if self.board.flatten()[action] == 0:
            row, col = divmod(action, 3)
            self.board[row, col] = self.current_player

            if self.check_winner():
                self.done = True
                self.winner = self.current_player
                reward = 1 if self.current_player == 1 else -1
            elif len(self.available_actions()) == 0:
                self.done = True
                self.winner = 0  # Tie
                reward = 0
            else:
                self.current_player *= -1
                reward = 0
            return self.board.flatten(), reward, self.done
        else:
            raise ValueError("Invalid action")

    def check_winner(self):
        for i in range(3):
            if abs(sum(self.board[i, :])) == 3 or abs(sum(self.board[:, i])) == 3:
                return True
        if (
            abs(sum([self.board[i, i] for i in range(3)])) == 3
            or abs(sum([self.board[i, 2 - i] for i in range(3)])) == 3
        ):
            return True
        return False

    def render(self):
        board_symbols = np.where(
            self.board == 1, "X", np.where(self.board == -1, "O", " ")
        )
        print("\nBoard:")
        print("\n".join([" | ".join(row) for row in board_symbols]))
        print()
