from collections import Counter
def calculate_score(dice: list) -> int:
    """
    Calculate the maximum possible score based on the given dice rolls.

    This function evaluates the score from a list of dice, checking for special combinations
    like a straight (1-6), three pairs, or multiple identical dice. It then computes the
    score based on the rules and returns the total score.

    Args:
        dice (list): A list of integers representing the dice rolled.

    Returns:
        int: The total score calculated based on the combinations found.
    """
    total_score = 0

    # Check for specific combinations
    if check_straight(dice):
        return 1500  # A straight gives the maximum score immediately

    if check_three_pairs(dice):
        return 1000  # Three pairs give a score of 1000

    # Handle multiples (3, 4, 5, or 6 of the same number)
    multiples_score, used_dice = check_multiples(dice)
    total_score += multiples_score

    # Handle individual dice (1s and 5s) that are not part of other combinations
    total_score += check_individual_scores(dice, used_dice)

    return total_score


def check_straight(dice: list) -> bool:
    """
    Check if the given dice result in a straight (1-6).

    A straight is a combination where all numbers from 1 to 6 are rolled.

    Args:
        dice (list): A list of integers representing the dice rolled.

    Returns:
        bool: True if the dice form a straight (1-6), False otherwise.
    """
    return sorted(dice) == [1, 2, 3, 4, 5, 6]


def check_three_pairs(dice: list) -> bool:
    """
    Check if the given dice contain three pairs.

    A combination of three pairs is when there are exactly three different numbers,
    each occurring twice.

    Args:
        dice (list): A list of integers representing the dice rolled.

    Returns:
        bool: True if the dice contain three pairs, False otherwise.
    """
    counts = Counter(dice)
    pairs = sum(1 for count in counts.values() if count == 2)
    return pairs == 3


def check_multiples(dice: list) -> tuple:
    """
    Check for multiples (3, 4, 5, or 6 of the same number) in the dice.

    This function calculates the score for any multiples found and returns the score
    along with a list of dice that have been used for these combinations.

    Args:
        dice (list): A list of integers representing the dice rolled.

    Returns:
        tuple: A tuple containing:
            - score (int): The score based on the multiples found.
            - used_dice (list): A list of integers representing the dice used in the multiples.
    """
    counts = Counter(dice)
    score = 0
    used_dice = []

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

    return score, used_dice


def check_individual_scores(dice: list, used_dice: list) -> int:
    """
    Calculate points for individual dice (1s and 5s) that are not part of any combinations.

    This function handles scoring for any remaining 1s or 5s that have not been used
    in other combinations like multiples.

    Args:
        dice (list): A list of integers representing the dice rolled.
        used_dice (list): A list of integers representing dice already used in other combinations.

    Returns:
        int: The score based on the remaining individual 1s and 5s.
    """
    remaining_counts = Counter(dice) - Counter(used_dice)
    score = 0

    score += remaining_counts[1] * 100  # Each remaining 1 is worth 100 points
    score += remaining_counts[5] * 50   # Each remaining 5 is worth 50 points

    return score

#dice = [1, 1, 1, 1,1,1]
#print(calculate_score(dice))
