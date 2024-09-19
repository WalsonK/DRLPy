import numpy as np


class LineWorld:
    def __init__(self, length: int, is_random: bool = False, start_position: int = 1):
        """
        Classe LineWorld : Un environnement simple où un agent peut se déplacer sur une ligne.

        :param length: Longueur de la ligne (nombre de positions).
        :param is_random: Si True, la position initiale de l'agent sera aléatoire.
        :param start_position: Position de départ de l'agent (si is_random est False).
        """
        self.length = length
        self.agent_position = np.random.randint(1, length - 1) if is_random else start_position
        self.terminal_positions = [0, length - 1]
        self.done = False

    def available_actions(self):
        """
        Renvoie les actions possibles à la position actuelle de l'agent.
        :return: [0, 1] où 0 signifie 'gauche' et 1 signifie 'droite'
        """
        if not self.is_game_over():
            return [0, 1]  # 0: gauche, 1: droite
        return []  # Pas d'actions possibles dans un état terminal

    def reset(self):
        """
        Réinitialise l'environnement en repositionnant l'agent à une position aléatoire (non terminale).
        :return: L'état initial (position de l'agent sous forme d'un vecteur)
        """
        self.agent_position = np.random.randint(1, self.length - 1)
        self.done = False  # Réinitialiser l'état 'done'
        return self.state()

    def state(self):
        """
        Retourne l'état actuel sous forme de vecteur (one-hot encoding).
        :return: Un vecteur d'état de taille 'length' où 1 correspond à la position de l'agent.
        """
        state_vector = np.zeros(self.length)
        state_vector[self.agent_position] = 1
        return state_vector

    def step(self, action: int):
        """
        Fait avancer l'agent en fonction de l'action (gauche ou droite).

        :param action: 0 pour 'gauche', 1 pour 'droite'
        :return: Un tuple (next_state, reward, done) après l'action
        """
        assert action in self.available_actions(), "Action invalide"

        if action == 0:
            self.agent_position -= 1  # Aller à gauche
        elif action == 1:
            self.agent_position += 1  # Aller à droite

        reward = self.score()
        self.done = self.is_game_over()

        # Retourner l'état suivant, la récompense, et si la partie est terminée
        return self.state(), reward, self.done

    def score(self):
        """
        Retourne le score de la partie en fonction de la position de l'agent.
        :return: -1 si l'agent atteint la position terminale gauche, 1 pour la droite, 0 sinon.
        """
        if self.agent_position == self.terminal_positions[0]:
            return -1.0
        elif self.agent_position == self.terminal_positions[1]:
            return 1.0
        return 0.0

    def is_game_over(self):
        """
        Vérifie si la partie est terminée (l'agent est dans une position terminale).
        :return: True si la partie est terminée, sinon False.
        """
        return self.agent_position in self.terminal_positions

    def render(self):
        """
        Affiche une représentation de la ligne et de la position actuelle de l'agent.
        """
        for i in range(self.length):
            print('X' if self.agent_position == i else '_', end='')
        print()
