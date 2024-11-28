import numpy as np


class GridWorld:
    def __init__(self, size=5):
        """
        Classe GridWorldLikeLineWorld : Un environnement de grille avec des conditions de victoire et de défaite.

        :param size: Taille de la grille (size x size).
        """
        self.size = size
        self.current_player = 1
        self.winner = None
        self.terminal_positions = [
            (0, 0),
            (size - 1, size - 1),
        ]  # Coins comme positions terminales
        self.agent_pos = (
            1,
            1,
        )  # self._get_random_position()  # Position initiale aléatoire
        self.done = False

    def _get_random_position(self):
        while True:
            pos = (np.random.randint(0, self.size), np.random.randint(0, self.size))
            if pos not in self.terminal_positions:
                return pos

    def available_actions(self):
        if not self.is_game_over():
            actions = []

            # Vérifier chaque direction et s'assurer qu'elle ne sort pas de la grille
            if self.agent_pos[0] > 0:  # Peut aller vers le haut
                actions.append(0)
            if self.agent_pos[0] < self.size - 1:  # Peut aller vers le bas
                actions.append(1)
            if self.agent_pos[1] > 0:  # Peut aller à gauche
                actions.append(2)
            if self.agent_pos[1] < self.size - 1:  # Peut aller à droite
                actions.append(3)
            return actions

        return []

    def reset(self):
        self.agent_pos = (1, 1)  # self._get_random_position()
        self.done = False
        self.winner = None
        return self.state()

    def state(self):
        state_vector = np.zeros((self.size, self.size))
        state_vector[tuple(self.agent_pos)] = 1
        return state_vector.flatten()

    def step(self, action: int):
        assert action in self.available_actions(), "Action invalide"

        # Mouvements : 0: haut, 1: bas, 2: gauche, 3: droite
        moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        move = moves[action]
        new_pos = [self.agent_pos[0] + move[0], self.agent_pos[1] + move[1]]

        # Valider si le mouvement est dans les limites de la grille
        if 0 <= new_pos[0] < self.size and 0 <= new_pos[1] < self.size:
            self.agent_pos = new_pos

        reward = self.score()
        self.done = self.is_game_over()
        if reward == 1.0:
            self.winner = self.current_player

        # Retourner l'état suivant, la récompense, et si la partie est terminée
        return self.state(), reward, self.done

    def score(self):
        if tuple(self.agent_pos) == self.terminal_positions[0]:
            return -1.0
        elif tuple(self.agent_pos) == self.terminal_positions[1]:
            return 1.0
        return 0.0

    def get_reward(self):
        return self.score()

    def is_game_over(self):
        return tuple(self.agent_pos) in self.terminal_positions

    def render(self):
        grid = np.full((self.size, self.size), "_")
        grid[tuple(self.agent_pos)] = "A"  # Position de l'agent
        grid[self.terminal_positions[0]] = "L"  # Position terminale gauche (lose)
        grid[self.terminal_positions[1]] = "R"  # Position terminale droite (win)
        print("\n".join(" ".join(row) for row in grid))
        print()
