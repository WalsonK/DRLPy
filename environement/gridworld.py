import numpy as np
import random

class GridWorld:
    def __init__(self, size=5, obstacles=None, start=(0, 0), goal=(4, 4)):
        self.size = size
        self.start = start
        self.goal = goal
        self.agent_pos = list(start)
        self.obstacles = obstacles if obstacles else [(1, 1), (2, 2), (3, 3)]
        self.done = False
        self.state_size = size * size
        self.num_actions = 4  # Haut, Bas, Gauche, Droite

    def reset(self):
        self.agent_pos = list(self.start)
        self.done = False
        return self._get_state()

    def step(self, action):
        if self.done:
            raise ValueError("Game is over. Please reset the environment.")

        moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        move = moves[action]
        new_pos = [self.agent_pos[0] + move[0], self.agent_pos[1] + move[1]]

        if 0 <= new_pos[0] < self.size and 0 <= new_pos[1] < self.size:
            self.agent_pos = new_pos

        if tuple(self.agent_pos) == self.goal:
            self.done = True
            return self._get_state(), 10, self.done  # Reward for reaching the goal

        if tuple(self.agent_pos) in self.obstacles:
            self.done = True
            return self._get_state(), -10, self.done  # Penalty for hitting an obstacle

        return self._get_state(), -1, self.done  # Small penalty for each step

    def _get_state(self):
        state = np.zeros((self.size, self.size))
        state[tuple(self.agent_pos)] = 1  # Position de l'agent
        return state.flatten()

    def render(self):
        grid = np.full((self.size, self.size), '.')
        grid[tuple(self.agent_pos)] = 'A'
        grid[self.goal] = 'G'
        for obstacle in self.obstacles:
            grid[obstacle] = 'X'
        print("\n".join(" ".join(row) for row in grid))
        print("\n")
