import numpy as np
import random

class TicTacToe:
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.done = False
        self.winner = None
        self.current_player = 1

    def reset(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.done = False
        self.winner = None
        self.current_player = 1
        return self.board.flatten()

    def available_actions(self):
        return [i for i in range(9) if self.board.flatten()[i] == 0]

    def step(self, action):
        available_actions = self.available_actions()  # Récupérer les actions disponibles

        if action not in available_actions:
            print(f"Invalid action: {action}, available actions: {available_actions}")
            raise ValueError("Invalid action selected")

        # Si l'action est valide, on continue le jeu
        row, col = divmod(action, 3)
        self.board[row, col] = self.current_player

        if self.check_winner():
            self.done = True
            self.winner = self.current_player
            reward = 1 if self.current_player == 1 else -1
        elif len(self.available_actions()) == 0:
            self.done = True
            self.winner = 0  # Match nul
            reward = 0
        else:
            self.current_player *= -1
            reward = 0

        return self.board.flatten(), reward, self.done

    def check_winner(self):
        for i in range(3):
            if abs(sum(self.board[i, :])) == 3 or abs(sum(self.board[:, i])) == 3:
                return True
        if abs(sum([self.board[i, i] for i in range(3)])) == 3 or abs(
                sum([self.board[i, 2 - i] for i in range(3)])) == 3:
            return True
        return False

    def render(self):
        print(self.board)

