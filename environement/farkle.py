import random
import time
from tqdm import tqdm
from itertools import chain, combinations
import numpy as np
from collections import Counter


def convert_input_list(array):
    res = []
    for element in array:
        try:
            res.append(int(element))
        except ValueError:
            continue
    return res


def powerset(iterable):
    xs = list(iterable)
    return chain.from_iterable(combinations(xs, n) for n in range(len(xs) + 1))


class Farkle:
    def __init__(self, winning_score=10000, printing=True):
        self.printify = printing
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

    def available_actions(self):
        score, original_vec = self.calculate_score()
        # Calc all vecs possible :
        index = [i for i, x in enumerate(original_vec) if x == 1]

        av_actions = {}
        for comb in powerset(index):
            new_list = [0] * len(original_vec)
            for id in comb:
                new_list[id] = 1

            binary_int = int("".join(map(str, new_list)), 2)
            av_actions[binary_int] = new_list

        av_actions[0] = [score]

        return av_actions
        # return ['roll', 'bank']

    def roll_dice(self):
        self.dice_list = [random.randint(1, 6) for _ in range(self.remaining_dice)]
        if self.printify:
            self.print_dice(self.dice_list)

    def calculate_score(self) -> (int, list):
        """
        Calculate the maximum possible score based on the given dice rolls.

        This function evaluates the score from a list of dice, checking for special combinations
        like a straight (1-6), three pairs, or multiple identical dice. It then computes the
        score based on the rules and returns the total score.

        Returns:
            int: The total score calculated based on the combinations found.
        """
        total_score = 0

        # Check for specific combinations
        if self.check_straight():
            return 1500  # A straight gives the maximum score immediately

        if self.check_three_pairs():
            return 1000  # Three pairs give a score of 1000

        # Handle multiples (3, 4, 5, or 6 of the same number)
        multiples_score, used_dice, binary_multiple = self.check_multiples()
        total_score += multiples_score

        # Handle individual dice (1s and 5s) that are not part of other combinations
        score, binary_solo = self.check_individual_scores(used_dice)
        total_score += score

        # Mix binary selectable dice
        binary_selectable = [max(m, s) for m, s in zip(binary_multiple, binary_solo)]

        return total_score, binary_selectable

    def check_straight(self) -> bool:
        """
        Check if the given dice result in a straight (1-6).

        A straight is a combination where all numbers from 1 to 6 are rolled.

        Returns:
            bool: True if the dice form a straight (1-6), False otherwise.
        """
        return sorted(self.dice_list) == [1, 2, 3, 4, 5, 6]

    def check_three_pairs(self) -> bool:
        """
        Check if the given dice contain three pairs.

        A combination of three pairs is when there are exactly three different numbers,
        each occurring twice.

        Returns:
            bool: True if the dice contain three pairs, False otherwise.
        """
        counts = Counter(self.dice_list)
        pairs = sum(1 for count in counts.values() if count == 2)
        return pairs == 3

    def check_multiples(self) -> tuple:
        """
        Check for multiples (3, 4, 5x, or 6 of the same number) in the dice.

        This function calculates the score for any multiples found and returns the score
        along with a list of dice that have been used for these combinations, and a binary list
        indicating which dice were used (1 for used, 0 for not used).

        Returns:
            tuple: A tuple containing:
                - score (int): The score based on the multiples found.
                - used_dice (list): A list of integers representing the dice used in the multiples.
                - binary_used_dice (list): A list of integers (1 or 0) indicating which dice were used.
        """
        counts = Counter(self.dice_list)
        score = 0
        used_dice = []

        binary_used_dice = [0] * len(self.dice_list)

        for num, count in counts.items():
            if count >= 3:
                base_score = 1000 if num == 1 else num * 100  # 1000 for three 1s, otherwise num * 100

                score += base_score

                if count == 4:
                    score += base_score  # x2 for 4 dice
                elif count == 5:
                    score += base_score * 3  # x4 for 5 dice
                elif count == 6:
                    score += base_score * 7  # x8 for 6 dice

                used_dice.extend([num] * count)

                used_count = 0
                for i, d in enumerate(self.dice_list):
                    if d == num and used_count < count:
                        binary_used_dice[i] = 1
                        used_count += 1

        return score, used_dice, binary_used_dice

    def check_individual_scores(self, used_dice: list) -> (int, list):
        """
        Calculate points for individual dice (1s and 5s) that are not part of any combinations.

        This function handles scoring for any remaining 1s or 5s that have not been used
        in other combinations like multiples.

        Args:
            used_dice (list): A list of integers representing dice already used in other combinations.

        Returns:
            int: The score based on the remaining individual 1s and 5s.
        """
        remaining_counts = Counter(self.dice_list) - Counter(used_dice)
        score = 0
        binary_used_dice = [0] * len(self.dice_list)

        score += remaining_counts[1] * 100  # Each remaining 1 is worth 100 points
        score += remaining_counts[5] * 50  # Each remaining 5 is worth 50 points

        used_count_1s = 0
        used_count_5s = 0
        for i, d in enumerate(self.dice_list):
            if d == 1 and used_count_1s < remaining_counts[1]:
                binary_used_dice[i] = 1
                used_count_1s += 1
            elif d == 5 and used_count_5s < remaining_counts[5]:
                binary_used_dice[i] = 1
                used_count_5s += 1

        return score, binary_used_dice
    """
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
            if self.printify:
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
        # Use the new scoring system
        self.current_turn_score = calculate_score(self.current_bank)

        # Add to the current player's total score
        self.scores[self.current_player] += self.current_turn_score

        if self.printify:
            print(f"Score of player {self.current_player}: {self.scores[self.current_player]}")

        # Reset current turn score for the next round
        self.current_turn_score = 0
    """

    def switch_player(self):
        self.current_player = 1 if self.current_player == 0 else 0
        self.remaining_dice = 6
        self.current_bank = []
        self.current_turn_score = 0
        if self.printify:
            print(f"Game Score : {self.scores} ")

    def step(self, action, banked=None):
        if action == 0:
            score, _ = self.calculate_score()
            self.scores[self.current_player] += score
            self.switch_player()

        if action == 'r':
            if self.remaining_dice == 1:
                self.roll_dice()
                # verifie combo sinon score du round = 0 (FARKLE) turn over
                if self.dice_list[0] not in self.current_bank:
                    self.current_bank = []
                    # calc score
                    self.calculate_score()
                    if self.printify:
                        print(f"Player {self.current_player} Bank: {self.current_bank}")
                        print("Turn over")
                    self.switch_player()
                else:
                    self.current_bank.append(self.dice_list[0])
                    # calc score
                    self.calculate_score()
                    if self.printify:
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
                if self.printify:
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

                if self.printify:
                    print(f"Player {self.current_player} Bank:")
                self.print_dice(self.current_bank)
                if len(self.dice_list) > 0:
                    if self.printify:
                        print(f"new dice list:")
                    self.print_dice(self.dice_list)
                else:
                    self.calculate_score()
                    self.switch_player()
        if action == 's':
            print(self.get_state())

    def bot_turn(self, botPlayer=1):
        if self.printify:
            print(f"Player {self.current_player} turn:")
        self.roll_dice()
        if len(self.dice_list) > 0:
            while self.current_player == botPlayer:
                action_rand = random.randint(0, 1)
                if action_rand == 0:
                    if self.printify:
                        print(f"Action of Player {self.current_player} : Roll")
                    self.step('r')
                if action_rand == 1:
                    rand_bank = [random.randint(1, len(self.dice_list))]
                    if self.printify:
                        print(f"Action of Player {self.current_player} : Bank dice-{rand_bank}")
                    self.step('b', banked=rand_bank)
        else:
            self.calculate_score()
            self.switch_player()

    def play_game(self, isBotGame=False):
        def solo_round(isb):
            print(f"player to play : {self.current_player}")
            if self.current_player == 0:
                if not isb:
                    print("Would you like to :")
                    av_actions = self.available_actions()
                    for action, vector in av_actions.items():
                        human_indice = [i +1 for i, x in enumerate(vector) if x == 1]
                        if human_indice:
                            if len(human_indice) > 1:
                                print(f"{action} : Select les dés {human_indice} et relancer les autres")
                            else:
                                print(f"{action} : Select le {human_indice} dé et relancer les autres")
                        else:
                            print(f"{action} : Bank {vector[0]} score et terminer tour !")
                    choice = int(input("> "))
                    self.step(choice)

                    if self.remaining_dice == 0:
                        self.calculate_score()
                else:
                    self.bot_turn(botPlayer=0)
            else:
                self.bot_turn()

        if self.printify:
            print(f"Game Score : {env.scores} ")
        self.roll_dice()
        while all(s <= 10000 for s in self.scores):
            solo_round(isBotGame)
        if any(s <= 10000 for s in self.scores):
            solo_round(isBotGame)
            # self.winner = self.scores.index(max(s for s in self.scores if s <= 1000))
            # self.done = True

        self.reset()

    def print_dice(self, dlist):
        # Display dices
        if self.printify:
            for line in range(5):
                for dice in dlist:
                    print(self.dice_art[dice][line], end=" ")
                print()


env = Farkle(printing=True)
game_mode = input("Would you like to play ? (y/n)\n>")
if game_mode == 'y':
    env.play_game()
elif game_mode == 't':
    print(env.available_actions())
elif game_mode == 'n':
    # Game / sec
    duration = 30
    start_time = time.time()
    total = 0

    with tqdm(total=duration, desc="Playing game", unit="s", bar_format='{l_bar}{bar} {n_fmt}/{total_fmt} s') as pbar:
        while time.time() - start_time < duration:
            env.play_game(isBotGame=True)
            total += 1

            elapsed_time = time.time() - start_time
            progress = min(elapsed_time, duration)
            pbar.n = round(progress, 2)
            pbar.refresh()

    print(f"\n{total} Game in 30 seconds")
    game_per_second = total / 30
    print(f"{game_per_second :.2f} Games/s")
