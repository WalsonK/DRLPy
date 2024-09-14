import numpy as np


class LineWorld:
    def __init__(self, length: int, is_random: bool, start_position: int):
        if is_random:
            self.agent_position = np.random.randint(1, length - 1)
        else:
            self.agent_position = start_position
        self.all_positions = list(range(0, length + 1))
        self.terminal_positions = [0, length]

    def available_actions(self):
        if self.agent_position in self.all_positions[1:-1]:
            return [0, 1]  # 0: left, 1: right
        return []

    def is_game_over(self):
        return True if self.agent_position in self.terminal_positions else False

    def state_id(self):
        return self.agent_position

    def step(self, action: int):
        assert (not self.is_game_over())
        assert action in self.available_actions()

        if action == 0:
            self.agent_position -= 1
        else:
            self.agent_position += 1

    def score(self):
        if self.agent_position == self.terminal_positions[0]:
            return -1.0
        if self.agent_position == self.terminal_positions[1]:
            return 1.0
        return 0.0

    def display(self):
        for i in range(len(self.all_positions)):
            print('X' if self.agent_position == i else '_', end='')
        print()

    def reset(self):
        self.agent_position = 2
