import unittest
from environement.tools import *


# On doit importer les fonctions du fichier du jeu que tu as créé, mais je vais les définir ici pour que les tests fonctionnent

class TestFarkleScoring(unittest.TestCase):

    def test_check_straight(self):
        # Cas où c'est une suite
        self.assertTrue(check_straight([1, 2, 3, 4, 5, 6]))
        # Cas où ce n'est pas une suite
        self.assertFalse(check_straight([1, 2, 3, 4, 5, 5]))
        self.assertFalse(check_straight([1, 2, 3, 4, 6]))

    def test_check_three_pairs(self):
        # Cas où il y a trois paires
        self.assertTrue(check_three_pairs([1, 1, 2, 2, 3, 3]))
        # Cas où il n'y a pas trois paires
        self.assertFalse(check_three_pairs([1, 1, 2, 2, 3, 4]))
        self.assertFalse(check_three_pairs([1, 2, 3, 4, 5, 6]))

    def test_check_multiples(self):
        # Cas pour 3 dés identiques
        self.assertEqual(check_multiples([1, 1, 1]), (1000, [1, 1, 1]))
        self.assertEqual(check_multiples([2, 2, 2]), (200, [2, 2, 2]))
        self.assertEqual(check_multiples([5, 5, 5]), (500, [5, 5, 5]))
        self.assertEqual(check_multiples([6, 6, 6]), (600, [6, 6, 6]))

        # Cas pour 4 dés identiques (multiplie par 2)
        self.assertEqual(check_multiples([1, 1, 1, 1]), (2000, [1, 1, 1, 1]))
        self.assertEqual(check_multiples([3, 3, 3, 3]), (600, [3, 3, 3, 3]))

        # Cas pour 5 dés identiques (multiplie par 4)
        self.assertEqual(check_multiples([1, 1, 1, 1, 1]), (4000, [1, 1, 1, 1, 1]))
        self.assertEqual(check_multiples([4, 4, 4, 4, 4]), (1600, [4, 4, 4, 4, 4]))

        # Cas pour 6 dés identiques (multiplie par 8)
        self.assertEqual(check_multiples([1, 1, 1, 1, 1, 1]), (8000, [1, 1, 1, 1, 1, 1]))
        self.assertEqual(check_multiples([2, 2, 2, 2, 2, 2]), (1600, [2, 2, 2, 2, 2, 2]))

    def test_check_individual_scores(self):
        # Cas pour des 1 non multiples
        self.assertEqual(check_individual_scores([1], []), 100)
        self.assertEqual(check_individual_scores([1, 1], []), 200)

        # Cas pour des 5 non multiples
        self.assertEqual(check_individual_scores([5], []), 50)
        self.assertEqual(check_individual_scores([5, 5], []), 100)

        # Cas pour des 1 et des 5 mélangés
        self.assertEqual(check_individual_scores([1, 5], []), 150)
        self.assertEqual(check_individual_scores([1, 1, 5, 5], []), 300)

        # Cas où tous les dés sont utilisés dans des multiples
        self.assertEqual(check_individual_scores([1, 1, 1], [1, 1, 1]), 0)
        self.assertEqual(check_individual_scores([5, 5, 5], [5, 5, 5]), 0)

    def test_calculate_score(self):
        # Cas où il y a une suite
        self.assertEqual(calculate_score([1, 2, 3, 4, 5, 6]), 1500)

        # Cas où il y a trois paires
        self.assertEqual(calculate_score([1, 1, 2, 2, 3, 3]), 1000)

        # Cas avec trois 1 et trois 5
        self.assertEqual(calculate_score([1, 1, 1, 5, 5, 5]), 1500)

        # Cas avec quatre 1
        self.assertEqual(calculate_score([1, 1, 1, 1]), 2000)

        # Cas avec six dés identiques
        self.assertEqual(calculate_score([2, 2, 2, 2, 2, 2]), 1600)


if __name__ == '__main__':
    unittest.main()
