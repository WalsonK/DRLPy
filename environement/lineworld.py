import numpy as np


class LineWorld:
    def __init__(self, length: int, is_random: bool = False, start_position: int = 1):
        self.current_player = 1
        self.length = length
        self.agent_position = (
            np.random.randint(1, length - 1) if is_random else start_position
        )
        self.terminal_positions = [0, length - 1]
        self.winner = None
        self.done = False

    def available_actions(self):
        if not self.is_game_over():
            return [0, 1]  # 0: gauche, 1: droite
        return []  # Pas d'actions possibles dans un état terminal

    def reset(self):
        self.agent_position = np.random.randint(1, self.length - 1)
        self.winner = None
        self.done = False  # Réinitialiser l'état 'done'
        return self.state()

    def state(self):
        state_vector = np.zeros(self.length)
        state_vector[self.agent_position] = 1
        return state_vector

    def step(self, action: int):
        assert action in self.available_actions(), "Action invalide"

        if action == 0:
            self.agent_position -= 1  # Aller à gauche
        elif action == 1:
            self.agent_position += 1  # Aller à droite

        reward = self.score()
        self.done = self.is_game_over()
        if reward == 1.0:
            self.winner = self.current_player

        # Retourner l'état suivant, la récompense, et si la partie est terminée
        return self.state(), reward, self.done

    def score(self):
        if self.agent_position == self.terminal_positions[0]:
            return -1.0
        elif self.agent_position == self.terminal_positions[1]:
            return 1.0
        return 0.0

    def is_game_over(self):
        return self.agent_position in self.terminal_positions

    def render(self):
        for i in range(self.length):
            print("X" if self.agent_position == i else "_", end="")
        print()
