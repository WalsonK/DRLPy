import pytest
import numpy as np

def test_find_one(vector = np.array([1, 3, 2, 1, 6, 4])):
    index = np.where(vector == 1)[0]
    assert (index == np.array([0, 3])).all()

def test_find_two(vector = np.array([4, 2, 6, 2, 5, 3])):
    index = np.where(vector == 2)[0]
    assert (index == np.array([1, 3])).all()

def test_find_three(vector = np.array([3, 1, 5, 3, 6, 4])):
    index = np.where(vector == 3)[0]
    assert (index == np.array([0, 3])).all()

def test_find_four(vector = np.array([4, 4, 1, 5, 6, 2])):
    index = np.where(vector == 4)[0]
    assert (index == np.array([0, 1])).all()

def test_find_five(vector = np.array([5, 6, 5, 3, 2, 1])):
    index = np.where(vector == 5)[0]
    assert (index == np.array([0, 2])).all()

def test_find_six(vector = np.array([6, 4, 2, 6, 1, 3])):
    index = np.where(vector == 6)[0]
    assert (index == np.array([0, 3])).all()
