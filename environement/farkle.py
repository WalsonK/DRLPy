import random


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
        return self.get_state()

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

        print(f"Current Score: {self.current_turn_score}")
        self.scores[self.current_player] += self.current_turn_score
        print(f"Score of player {self.current_player}: {self.scores[self.current_player]}")
        self.current_turn_score = 0

    def step(self, action):
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
                    self.current_player = 1
                else:
                    self.current_bank.append(self.dice_list[0])
                    # calc score
                    self.calculate_score()
                    print(f"Player {self.current_player} Bank: {self.current_bank}")
                    print("Turn over")
                    self.current_player = 1
            else:
                self.roll_dice()
                self.get_triplets()

        if action == 'b':
            if self.remaining_dice == 1:
                self.current_bank.append(self.dice_list[0])
                # Calc score
                self.calculate_score()
                print(f"Player {self.current_player} Bank: {self.current_bank}")
                print("Turn over")
                self.current_player = 1
            else:
                d_to_bank = input("Write index of dice, separate by 1 space\n>")
                d_to_bank = list(d_to_bank)
                d_to_bank = convert_input_list(d_to_bank)
                self.bank_dice(d_to_bank)
                print(f"Player {self.current_player} Bank:")
                self.print_dice(self.current_bank)
                print(f"new dice list:")
                self.print_dice(self.dice_list)


env = Farkle()
print(f"Player {env.current_player} turn ")
env.roll_dice()
env.get_triplets()
while env.current_player == 0:
    choice = input("Would you like to roll or bank? (r/b)\n>")
    env.step(choice)
    if env.remaining_dice == 0:
        env.calculate_score()

    if env.remaining_dice == 0:
        env.current_player = 1
