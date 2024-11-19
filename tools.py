import random
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np


def random_player(env):
    """Chooses a random action from available actions."""
    available_actions = env.available_actions()
    return random.choice(available_actions)





def print_metrics(episodes, scores, episode_times, actions, losses):
    # scores metrics
    plt.plot(episodes, scores)
    plt.xlabel("Episodes")
    plt.ylabel("Score")
    plt.show()
    # Episode times metrics
    plt.plot(episodes, episode_times)
    plt.xlabel("Episodes")
    plt.ylabel("Time (s)")
    plt.show()
    # Actions choose distribution
    counts = Counter(actions)
    plt.figure(figsize=(14, 10))
    plt.barh(list(counts.keys()), list(counts.values()))
    plt.xlabel("Counts")
    plt.ylabel("Action")
    plt.yticks(ticks=list(counts.keys()))
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()
    # Losses metrics
    plt.plot(losses)
    plt.xlabel("Episodes")
    plt.ylabel("Losses")
    plt.show()
