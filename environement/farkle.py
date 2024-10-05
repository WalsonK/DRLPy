import random
import time


def convert_input_list(array):
    res = []
    for element in array:
        try:
            res.append(int(element))
        except ValueError:
            continue
    return res


class Farkle:
    def __init__(self, winning_score=10000):
        self.winning_score = winning_score
        self.scores = [0, 0]  # Scores des deux joueurs
        self.current_player = 0  # Indique quel joueur est en train de jouer (0 ou 1)
        self.current_bank = []  # La banque du joueur
        self.current_turn_score = 0  # Score temporaire pour le tour en cours
        self.dice_list = []
        self.switch_turn = False    # True to switch
        self.remaining_dice = 6  # Nombre de dés restants à lancer
        self.done = False  # Indique si la partie est terminée
        self.winner = None  # Indique le gagnant (0 ou 1)
        self.dice_art = {
            1: ("┌─────────┐",
                "│         │",
                "│    ●    │",
                "│         │",
                "└─────────┘",
                ),
            2: ("┌─────────┐",
                "│  ●      │",
                "│         │",
                "│      ●  │",
                "└─────────┘",
                ),
            3: ("┌─────────┐",
                "│  ●      │",
                "│    ●    │",
                "│      ●  │",
                "└─────────┘",
                ),
            4: ("┌─────────┐",
                "│  ●   ●  │",
                "│         │",
                "│  ●   ●  │",
                "└─────────┘",
                ),
            5: ("┌─────────┐",
                "│  ●   ●  │",
                "│    ●    │",
                "│  ●   ●  │",
                "└─────────┘",
                ),
            6: ("┌─────────┐",
                "│  ●   ●  │",
                "│  ●   ●  │",
                "│  ●   ●  │",
                "└─────────┘",
                )
        }

    def reset(self):
        self.scores = [0, 0]
        self.current_player = 0
        self.current_turn_score = 0
        self.remaining_dice = 6
        self.done = False
        self.winner = None

    def available_actions(self):
        # Ici on peut inclure la logique pour déterminer les actions possibles
        # On suppose qu'un joueur peut toujours "lancer" (roll) ou "banquer" (bank)
        return ['roll', 'bank']

    def roll_dice(self):
        # Simule le lancer de dés restants
        self.dice_list = [random.randint(1, 6) for _ in range(self.remaining_dice)]
        self.print_dice(self.dice_list)

    def print_dice(self, dlist):
        # Display dices
        for line in range(5):
            for dice in dlist:
                print(self.dice_art[dice][line], end=" ")
            print()

    def get_triplets(self):
        occurrences = {}

        # Compter les occurrences de chaque dé
        for die in self.dice_list:
            if die in occurrences:
                occurrences[die] += 1
            else:
                occurrences[die] = 1

        triplets = []
        new_dice_list = []

        # Parcourir les occurrences pour séparer les triplets et les dés restants
        for die, count in occurrences.items():
            # Ajouter les triplets
            while count >= 3:
                triplets += [die] * 3  # Ajouter trois dés pour chaque triplet
                count -= 3

            # Si des dés restants ne forment pas un triplet, les ajouter à la nouvelle liste
            for _ in range(count):
                new_dice_list.append(die)

        # Print triplets
        if len(triplets) > 0:
            self.current_bank += triplets.copy()
            print(f"Player {self.current_player} Bank:")
            self.print_dice(self.current_bank)

            # new dice list
            print(f"new dice list:")
            self.dice_list = new_dice_list
            self.print_dice(self.dice_list)
            self.remaining_dice = len(self.dice_list)

    def bank_dice(self, index_list):
        for index in sorted(index_list, reverse=True):
            if 1 <= index <= len(self.dice_list):
                self.current_bank.append(self.dice_list.pop(index - 1))
                self.remaining_dice -= 1

    def calculate_score(self):
        occurrences = {}

        # Calc occurrences
        for die in self.current_bank:
            if die in occurrences:
                occurrences[die] += 1
            else:
                occurrences[die] = 1

        for die, count in occurrences.items():
            while count >= 3:
                if die == 1:
                    self.current_turn_score += 1000
                else:
                    self.current_turn_score += die * 100
                count -= 3

            for _ in range(count):
                if die == 1:
                    self.current_turn_score += 100
                else:
                    self.current_turn_score += die * 10

        self.scores[self.current_player] += self.current_turn_score
        print(f"Score of player {self.current_player}: {self.scores[self.current_player]}")
        self.current_turn_score = 0

    def switch_player(self):
        self.current_player = 1 if self.current_player == 0 else 0
        self.remaining_dice = 6
        self.current_bank = []
        self.current_turn_score = 0
        print(f"Game Score : {env.scores} ")

    def step(self, action, banked=None):
        if action == 'r':
            if self.remaining_dice == 1:
                self.roll_dice()
                # verifie combo sinon score du round = 0 (FARKLE) turn over
                if self.dice_list[0] not in self.current_bank:
                    self.current_bank = []
                    # calc score
                    self.calculate_score()
                    print(f"Player {self.current_player} Bank: {self.current_bank}")
                    print("Turn over")
                    self.switch_player()
                else:
                    self.current_bank.append(self.dice_list[0])
                    # calc score
                    self.calculate_score()
                    print(f"Player {self.current_player} Bank: {self.current_bank}")
                    print("Turn over")
                    self.switch_player()
            else:
                self.roll_dice()
                self.get_triplets()
                if len(self.dice_list) == 0:
                    self.calculate_score()
                    self.switch_player()

        if action == 'b':
            if self.remaining_dice == 1:
                self.current_bank.append(self.dice_list[0])
                # Calc score
                self.calculate_score()
                print(f"Player {self.current_player} Bank: {self.current_bank}")
                print("Turn over")
                self.switch_player()
            else:
                if banked is None:
                    d_to_bank = input("Write index of dice, separate by 1 space\n>")
                    d_to_bank = list(d_to_bank)
                    d_to_bank = convert_input_list(d_to_bank)
                    self.bank_dice(d_to_bank)
                else:
                    self.bank_dice(banked)

                print(f"Player {self.current_player} Bank:")
                self.print_dice(self.current_bank)
                if len(self.dice_list) > 0:
                    print(f"new dice list:")
                    self.print_dice(self.dice_list)
                else:
                    self.calculate_score()
                    self.switch_player()
        if action == 's':
            print(self.get_state())

    def bot_turn(self, botPlayer=1):
        print(f"Player {self.current_player} turn:")
        self.roll_dice()
        self.get_triplets()
        if len(self.dice_list) > 0:
            while self.current_player == botPlayer:
                action_rand = random.randint(0, 1)
                if action_rand == 0:
                    print(f"Action of Player {self.current_player} : Roll")
                    self.step('r')
                if action_rand == 1:
                    rand_bank = [random.randint(1, len(self.dice_list))]
                    print(f"Action of Player {self.current_player} : Bank dice-{rand_bank}")
                    self.step('b', banked=rand_bank)
        else:
            self.calculate_score()
            self.switch_player()

    def play_game(self, isBotGame=False):
        def solo_round(isb):
            if self.current_player == 0:
                if not isb:
                    choice = input("Would you like to roll or bank? (r/b)\n>")
                    self.step(choice)
                    if self.remaining_dice == 0:
                        self.calculate_score()
                else:
                    self.bot_turn(botPlayer=0)
            else:
                self.bot_turn()

        print(f"Game Score : {env.scores} ")
        while all(s <= 1000 for s in self.scores):
            solo_round(isBotGame)
        if any(s <= 1000 for s in self.scores):
            solo_round(isBotGame)
            self.winner = self.scores.index(max(s for s in self.scores if s <= 1000))
            self.done = True

        self.reset()

    def get_state(self):
        p1_bank = self.current_bank if(self.current_player == 0) else [0, 0, 0, 0, 0, 0]
        p2_bank = self.current_bank if(self.current_player == 1) else [0, 0, 0, 0, 0, 0]
        dice_list = self.dice_list[:6] + [0] * (6 - len(self.dice_list))
        return [self.current_player] + p1_bank + p2_bank + dice_list

    def get_reward(self):
        if self.done:
            if self.winner == self.current_player:
                return 1  # Victoire du joueur actuel
            else:
                return -1  # Défaite du joueur actuel
        return 0  # Pas de reward si la partie n'est pas finie



env = Farkle()
env.play_game()

# Game / sec
#start_time = time.time()
#total = 0
#while time.time() - start_time < 30:
#    env.play_game(isBotGame=True)
 #   total += 1

#print(f"\n{total} Game in 60 seconds")
#print(f"Average call : {total / 30:.2f} /s")

# IDEE ENCODAGE
# State :
# [main player 0] [main player 1] [des libre]
#                                      -> des libre [_, _, _, _, _] : 1 à 6 les vals et 0 pour vide
# ACTION :
# roll or bank [0, 1] -> 1 ou 0  [index]
#                                  -> index [_, _, _, _, _] : 1 à 6 les vals et 0 pour vide

