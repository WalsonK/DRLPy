from collections import Counter

def calculate_score(dice):
    """ Calcule le score maximum possible en fonction des dés donnés """
    total_score = 0

    # Vérification des différentes combinaisons
    if check_straight(dice):
        return 1500  # La suite donne le score maximum directement

    if check_three_pairs(dice):
        return 1000  # Trois paires

    # Gestion des multiples (3, 4, 5, 6 d'un même nombre)
    multiples_score, used_dice = check_multiples(dice)
    total_score += multiples_score

    # Gestion des dés individuels (1 et 5) restants
    total_score += check_individual_scores(dice, used_dice)

    return total_score

def check_straight(dice):
    """ Vérifie si les dés forment une suite de 1 à 6 """
    return sorted(dice) == [1, 2, 3, 4, 5, 6]

def check_three_pairs(dice):
    """ Vérifie si les dés forment trois paires """
    counts = Counter(dice)
    pairs = sum(1 for count in counts.values() if count == 2)
    return pairs == 3

def check_multiples(dice):
    """ Vérifie les combinaisons multiples : 3, 4, 5, ou 6 d'un même nombre et calcule le score """
    counts = Counter(dice)
    score = 0

    # Liste pour stocker les dés qui font partie des combinaisons multiples, donc ne doivent pas être comptés comme individuels
    used_dice = []

    for num, count in counts.items():
        if count >= 3:
            # Calcul du score de base pour trois dés identiques
            base_score = 1000 if num == 1 else num * 100  # 1000 pour trois 1, sinon num * 100 (200, 300, etc.)

            # Le score du triplet est ajouté une seule fois
            score += base_score

            # Multiplier le score en fonction du nombre de dés au-delà du triplet
            if count == 4:
                score += base_score  # x2 pour 4 dés
            elif count == 5:
                score += base_score * 3  # x4 pour 5 dés
            elif count == 6:
                score += base_score * 7  # x8 pour 6 dés

            # Ajouter ces dés utilisés à la liste des dés à ne pas compter comme individuels
            used_dice.extend([num] * count)

    return score, used_dice

def check_individual_scores(dice, used_dice):
    """ Gère les points pour les dés individuels (1 et 5) qui ne font pas partie d'autres combinaisons """
    counts = Counter(dice)
    score = 0

    # Recalculer le nombre de dés non utilisés dans les combinaisons multiples
    remaining_counts = Counter(dice) - Counter(used_dice)

    # Ajouter 100 points pour chaque 1 non utilisé dans les combinaisons multiples
    score += remaining_counts[1] * 100

    # Ajouter 50 points pour chaque 5 non utilisé dans les combinaisons multiples
    score += remaining_counts[5] * 50

    return score

# Exemple d'utilisation
dice = [1, 1, 1, 1,1,1]  # Un exemple de liste de dés
print(calculate_score(dice))  # Cela devrait donner 2000 points (1000 pour 4x1 et 1000 pour les 5)
