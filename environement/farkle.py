import random
import time
from collections import Counter
from itertools import chain, combinations

import numpy as np
from tqdm import tqdm


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
        self.current_player = 1  # Indique quel joueur est en train de jouer (1 ou 2)
        self.current_turn_score = 0  # Score temporaire pour le tour en cours
        self.switch_turn = False  # True to switch
        self.remaining_dice = 6  # Nombre de dés restants à lancer
        self.done = False  # Indique si la partie est terminée
        self.winner = None  # Indique le gagnant (0 ou 1)
        self.dice_list = []
        self.state_size = len(self.get_state())
        self.dice_list = [1, 1, 1, 1, 1, 1]
        self.actions_size = len(self.available_actions())
        self.dice_list = []
        self.dice_art = {
            1: (
                "┌─────────┐",
                "│         │",
                "│    ●    │",
                "│         │",
                "└─────────┘",
            ),
            2: (
                "┌─────────┐",
                "│  ●      │",
                "│         │",
                "│      ●  │",
                "└─────────┘",
            ),
            3: (
                "┌─────────┐",
                "│  ●      │",
                "│    ●    │",
                "│      ●  │",
                "└─────────┘",
            ),
            4: (
                "┌─────────┐",
                "│  ●   ●  │",
                "│         │",
                "│  ●   ●  │",
                "└─────────┘",
            ),
            5: (
                "┌─────────┐",
                "│  ●   ●  │",
                "│    ●    │",
                "│  ●   ●  │",
                "└─────────┘",
            ),
            6: (
                "┌─────────┐",
                "│  ●   ●  │",
                "│  ●   ●  │",
                "│  ●   ●  │",
                "└─────────┘",
            ),
        }

    def reset(self):
        """Reset the game state and return the starting state."""
        self.scores = [0, 0]
        self.current_player = 1
        self.current_turn_score = 0
        self.remaining_dice = 6
        self.done = False
        self.winner = None
        return self.get_state()

    def get_state(self):
        """Get the current state of the game."""
        dice_list = self.dice_list[:6] + [0] * (6 - len(self.dice_list))
        return np.array(
            [self.current_player] + dice_list + [self.current_turn_score] + self.scores
        )

    def get_reward(self):
        """Get the current reward of the game."""
        if self.done:
            if self.winner == 0:
                return 1  # Victoire du joueur actuel
            else:
                return -1  # Défaite du joueur actuel
        return 0  # Pas de reward si la partie n'est pas finie

    def available_actions(self):
        score, original_vec = self.calculate_score(self.dice_list)
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

    def roll_dice(self):
        self.dice_list = [random.randint(1, 6) for _ in range(self.remaining_dice)]
        if self.printify:
            self.print_dice(self.dice_list)
        if self.remaining_dice == 6 and (
            self.check_straight(self.dice_list)
            or self.check_three_pairs(self.dice_list)
        ):
            score, _ = self.calculate_score(self.dice_list)
            self.current_turn_score += score
            if self.printify:
                if self.check_straight(self.dice_list):
                    print(f"Et c'est un straight donc + {score} points !")
                elif self.check_three_pairs(self.dice_list):
                    print(f"Trois pairs donc + {score} points !")
            self.roll_dice()

    def calculate_score(self, dices: list) -> (int, list):
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
        if self.check_straight(dices):
            return 1500, [0] * len(
                dices
            )  # A straight gives the maximum score immediately

        if self.check_three_pairs(dices):
            return 1000, [0] * len(dices)  # Three pairs give a score of 1000

        # Check for "Six no scoring dice" rule only on the first roll (when remaining_dice == 6)
        if self.remaining_dice == 6 and self.check_six_no_scoring_dice(dices):
            return 500, [0] * len(dices)

        # Handle multiples (3, 4, 5, or 6 of the same number)
        multiples_score, used_dice, binary_multiple = self.check_multiples(dices)
        total_score += multiples_score

        # Handle individual dice (1s and 5s) that are not part of other combinations
        score, binary_solo = self.check_individual_scores(dices, used_dice)
        total_score += score

        # Mix binary selectable dice
        binary_selectable = [max(m, s) for m, s in zip(binary_multiple, binary_solo)]

        return total_score, binary_selectable

    def check_straight(self, dices: list) -> bool:
        """
        Check if the given dice result in a straight (1-6).

        A straight is a combination where all numbers from 1 to 6 are rolled.

        Returns:
            bool: True if the dice form a straight (1-6), False otherwise.
        """
        return sorted(dices) == [1, 2, 3, 4, 5, 6]

    def check_three_pairs(self, dices: list) -> bool:
        """
        Check if the given dice contain three pairs.

        A combination of three pairs is when there are exactly three different numbers,
        each occurring twice.

        Returns:
            bool: True if the dice contain three pairs, False otherwise.
        """
        counts = Counter(dices)
        pairs = sum(1 for count in counts.values() if count == 2)
        return pairs == 3

    def check_multiples(self, dices: list) -> tuple:
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
        counts = Counter(dices)
        score = 0
        used_dice = []

        binary_used_dice = [0] * len(dices)

        for num, count in counts.items():
            if count >= 3:
                base_score = (
                    1000 if num == 1 else num * 100
                )  # 1000 for three 1s, otherwise num * 100

                score += base_score

                if count == 4:
                    score += base_score  # x2 for 4 dice
                elif count == 5:
                    score += base_score * 3  # x4 for 5 dice
                elif count == 6:
                    score += base_score * 7  # x8 for 6 dice

                used_dice.extend([num] * count)

                used_count = 0
                for i, d in enumerate(dices):
                    if d == num and used_count < count:
                        binary_used_dice[i] = 1
                        used_count += 1

        return score, used_dice, binary_used_dice

    def check_six_no_scoring_dice(self, dice: list) -> bool:
        """
        Check if none of the six dice result in any scoring combinations.

        This function determines whether the current set of six dice contains
        no scoring combinations (such as a straight, three pairs, multiples,
        or individual 1s and 5s). If no scoring combination is detected, the
        function returns a score of 500 for the "Six no scoring dice" rule.

        Args:
            dice (list): A list of integers representing the dice rolled (values between 1 and 6).

        Returns:
            int: Returns 500 if none of the dice result in a scoring combination, otherwise returns 0.
        """
        # Vérifier s'il y a une suite (1-6)
        if self.check_straight(dice):
            return False

        # Vérifier s'il y a trois paires
        if self.check_three_pairs(dice):
            return False

        # Vérifier les multiples (3 dés identiques ou plus)
        score_multiples, _, _ = self.check_multiples(dice)
        if score_multiples > 0:
            return False

        # Vérifier s'il y a des 1 ou des 5 individuels
        score_individuals, _ = self.check_individual_scores(dice, [])
        if score_individuals > 0:
            return False

        # Si aucun score n'a été détecté
        return True

    def check_individual_scores(self, dices: list, used_dice: list) -> (int, list):
        """
        Calculate points for individual dice (1s and 5s) that are not part of any combinations.

        This function handles scoring for any remaining 1s or 5s that have not been used
        in other combinations like multiples.

        Args:
            used_dice (list): A list of integers representing dice already used in other combinations.

        Returns:
            int: The score based on the remaining individual 1s and 5s.
        """
        remaining_counts = Counter(dices) - Counter(used_dice)
        score = 0
        binary_used_dice = [0] * len(dices)

        score += remaining_counts[1] * 100  # Each remaining 1 is worth 100 points
        score += remaining_counts[5] * 50  # Each remaining 5 is worth 50 points

        used_count_1s = 0
        used_count_5s = 0
        for i, d in enumerate(dices):
            if d == 1 and used_count_1s < remaining_counts[1]:
                binary_used_dice[i] = 1
                used_count_1s += 1
            elif d == 5 and used_count_5s < remaining_counts[5]:
                binary_used_dice[i] = 1
                used_count_5s += 1

        return score, binary_used_dice

    def switch_player(self):
        self.current_player = 2 if self.current_player == 1 else 1
        self.remaining_dice = 6
        self.current_turn_score = 0
        if self.printify:
            print(f"Game Score : {self.scores} ")
            print(f"Player {self.current_player} Turn :")
        self.roll_dice()

    def step(self, dice_selected: list):
        if len(dice_selected) == 1:
            # Action 0
            score, _ = self.calculate_score(self.dice_list)
            self.current_turn_score += score
            self.scores[self.current_player - 1] += self.current_turn_score
            self.switch_player()
        else:
            # Other Action
            selected_dice = [
                self.dice_list[i] for i, x in enumerate(dice_selected) if x == 1
            ]
            score, _ = self.calculate_score(selected_dice)
            self.current_turn_score += score
            count_dice_used = dice_selected.count(1)
            self.remaining_dice -= count_dice_used
            if self.remaining_dice == 0:
                # Select all dices
                self.scores[self.current_player - 1] += self.current_turn_score
                self.switch_player()
            else:
                self.roll_dice()
                # Check Farkle case
                _, b = self.calculate_score(self.dice_list)
                count_no_selectable = b.count(0)
                if len(self.dice_list) == count_no_selectable:
                    if self.printify:
                        print("FARKLE !!!!")
                    self.switch_player()
        # Check if winner
        if any(s >= self.winning_score for s in self.scores):
            self.winner = 0 if self.scores[0] > self.scores[1] else 1
            self.done = True
        return self.get_state(), self.get_reward(), self.done

    def bot_turn(self, agentPlayer=1, agent=None):
        if self.current_turn_score > 0 and self.printify:
            print(f"Current turn score : {self.current_turn_score}")
        av_actions = self.available_actions()
        keys = list(av_actions.keys())
        action = (
            agent.choose_action(self.get_state(), keys)
            if agent
            else random.choice(keys)
        )
        if self.printify:
            self.print_available_actions(av_actions)
            print(f"Player {self.current_player} choose action {action}")
        self.step(av_actions[action])

    def play_game(
        self, isBotGame=False, show=False, agentPlayer=None, agentOpponent=None
    ):
        def solo_round(isb):
            if self.current_player == 1:
                if self.current_turn_score > 0 and self.printify:
                    print(f"Current turn score : {self.current_turn_score}")
                if not isb:
                    print("Would you like to :")
                    av_actions = self.available_actions()
                    if self.printify:
                        self.print_available_actions(av_actions)
                    choice = int(input("> "))
                    self.step(av_actions[choice])
                else:
                    if agentPlayer is not None:
                        self.bot_turn(agent=agentPlayer)
                    else:
                        self.bot_turn()
            else:
                if agentOpponent is not None:
                    self.bot_turn(agent=agentOpponent)
                else:
                    self.bot_turn()

        if show:
            self.printify = True

        if self.printify:
            print(f"Game Score : {self.scores} ")
        self.roll_dice()
        while all(s <= self.winning_score for s in self.scores):
            solo_round(isBotGame)
        if any(s <= self.winning_score for s in self.scores):
            solo_round(isBotGame)

        self.done = True
        self.winner = 0 if self.scores[0] > self.scores[1] else 1
        winner = self.winner
        if self.printify:
            print(f"Game Score : {self.scores} ")
            print(f"Player {self.winner} won :)")
        self.reset()
        return winner

    def print_dice(self, dlist):
        # Display dices
        if self.printify:
            for line in range(5):
                for dice in dlist:
                    print(self.dice_art[dice][line], end=" ")
                print()

    def print_available_actions(self, available_actions):
        for action, vector in available_actions.items():
            human_indice = [i + 1 for i, x in enumerate(vector) if x == 1]
            if human_indice:
                if len(human_indice) > 1:
                    print(
                        f"{action} : Select les dés {human_indice} et relancer les autres"
                    )
                else:
                    print(
                        f"{action} : Select le {human_indice} dé et relancer les autres"
                    )
            else:
                msg = f"{action} : Bank {vector[0]}"
                if self.current_turn_score > 0:
                    msg += f" + {self.current_turn_score}"
                msg += f" score et terminer tour !"
                print(msg)


"""env = Farkle(printing=False)
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
"""
