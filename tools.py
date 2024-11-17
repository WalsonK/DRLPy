import random
import matplotlib.pyplot as plt
from collections import Counter


def random_player(env):
    """Chooses a random action from available actions."""
    available_actions = env.available_actions()
    return random.choice(available_actions)


def print_metrics(episodes, scores=None, episode_times=None, action_times=None, actions=None, steps_per_game=None,
                  losses=None):
    # scores metrics
    if scores:
        plt.plot(episodes, scores)
        plt.title("Evolution of scores per episode")
        plt.xlabel("Episodes")
        plt.ylabel("Score")
        plt.show()
        plt.close()

    # Episode times metrics
    if episode_times:
        plt.plot(episodes, episode_times)
        plt.title("Evolution of episode time")
        plt.xlabel("Episodes")
        plt.ylabel("Time (s)")
        plt.show()
        plt.close()

    # Average Action time
    if action_times:
        plt.plot(episodes, action_times)
        plt.title("Evolution of action mean time per episode")
        plt.xlabel("Episodes")
        plt.ylabel("Average action time (s)")
        plt.show()
        plt.close()

    # Actions choose distribution
    if actions:
        counts = Counter(actions)
        plt.figure(figsize=(14, 10))
        plt.barh(list(counts.keys()), list(counts.values()))
        plt.title("Distribution of action take by the agent")
        plt.xlabel("Counts")
        plt.ylabel("Action")
        plt.yticks(ticks=list(counts.keys()))
        plt.grid(axis="x", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.show()
        plt.close()

    # Agent step per game
    if steps_per_game:
        plt.plot(episodes, steps_per_game)
        plt.title("Evolution of agent step per episode")
        plt.xlabel("Episodes")
        plt.ylabel("Nombre de step")
        plt.show()
        plt.close()

    # Losses metrics
    if losses:
        plt.plot(losses)
        plt.title("Losses evolution per episode")
        plt.xlabel("Episodes")
        plt.ylabel("Losses")
        plt.show()
        plt.close()
