import random


class Farkle:
    def __init__(self, winning_score=10000):
        self.winning_score = winning_score
        self.scores = [0, 0]  # Scores des deux joueurs
        self.current_player = 0  # Indique quel joueur est en train de jouer (0 ou 1)
        self.current_turn_score = 0  # Score temporaire pour le tour en cours
        self.remaining_dice = 6  # Nombre de dés restants à lancer
        self.done = False  # Indique si la partie est terminée
        self.winner = None  # Indique le gagnant (0 ou 1)

    def reset(self):
        self.scores = [0, 0]
        self.current_player = 0
        self.current_turn_score = 0
        self.remaining_dice = 6
        self.done = False
        self.winner = None
        return self.get_state()

    def available_actions(self):
        # Ici on peut inclure la logique pour déterminer les actions possibles
        # On suppose qu'un joueur peut toujours "lancer" (roll) ou "banquer" (bank)
        return ['roll', 'bank']

    def roll_dice(self):
        # Simule le lancer de dés restants
        dice = [random.randint(1, 6) for _ in range(self.remaining_dice)]
        return dice

    def step(self, action):
        if action == 'roll':
            dice = self.roll_dice()
            # Calcul du score du lancer de dés (on simplifie ici)
            score = sum(dice)  # On ajoute simplement la somme des dés pour cet exemple
            self.current_turn_score += score
            self.remaining_dice = len(dice)

            if self.remaining_dice == 0:  # Si tous les dés sont utilisés, tour terminé
                self.scores[self.current_player] += self.current_turn_score
                self.current_turn_score = 0
                self.remaining_dice = 6
                self.switch_player()

        elif action == 'bank':
            # Le joueur décide de garder son score et termine son tour
            self.scores[self.current_player] += self.current_turn_score
            self.current_turn_score = 0
            self.remaining_dice = 6
            self.switch_player()

        # Vérifier si la partie est terminée
        if self.scores[self.current_player] >= self.winning_score:
            self.done = True
            self.winner = self.current_player  # Le joueur actuel gagne

        return self.get_state(), self.get_reward(), self.done

    def get_state(self):
        # Retourne une représentation de l'état actuel
        return {
            'scores': self.scores,
            'current_turn_score': self.current_turn_score,
            'remaining_dice': self.remaining_dice,
            'current_player': self.current_player
        }

    def get_reward(self):
        # Le reward peut être défini en fonction des scores ou d'autres critères
        if self.done:
            if self.winner == self.current_player:
                return 1  # Victoire du joueur actuel
            else:
                return -1  # Défaite du joueur actuel
        return 0  # Pas de reward si la partie n'est pas finie

    def switch_player(self):
        # Alterner entre les deux joueurs
        self.current_player = 1 - self.current_player

    def render(self):
        # Affiche l'état du jeu
        print(f"Scores: Joueur 1 = {self.scores[0]}, Joueur 2 = {self.scores[1]}")
        print(f"Score du tour: {self.current_turn_score}, Dés restants: {self.remaining_dice}")
        print(f"Current Player: {'Joueur 1' if self.current_player == 0 else 'Joueur 2'}")
