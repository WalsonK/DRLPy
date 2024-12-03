import random
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np


def random_player(env):
    """Chooses a random action from available actions."""
    available_actions = env.available_actions()
    return random.choice(available_actions)


def print_metrics(
    episodes,
    scores=None,
    episode_times=None,
    action_times=None,
    actions=None,
    steps_per_game=None,
    losses=None,
    is_training: bool = True,
    algo_name="",
    env_name="",
    metric_for="",
):
    # scores metrics
    if scores:
        plt.plot(episodes, scores)
        plt.title(
            f"Evolution of scores per episode in Training of {env_name} \nfor {algo_name} in {episodes.stop} episodes"
            if is_training
            else f"Evolution of scores per episode in Test of {env_name} \nfor {algo_name} in {episodes.stop} episodes "
            f"for {metric_for}"
        )
        plt.xlabel("Episodes")
        plt.ylabel("Score")
        plt.show()
        plt.close()

    # Episode times metrics
    if episode_times:
        plt.plot(episodes, episode_times)
        plt.title(
            f"Evolution of episode time in Training of {env_name} \nfor {algo_name} in {episodes.stop} episodes"
            if is_training
            else f"Evolution of episode time in Test of {env_name} \nfor {algo_name} in {episodes.stop} episodes "
            f"for {metric_for}"
        )
        plt.xlabel("Episodes")
        plt.ylabel("Time (s)")
        plt.show()
        plt.close()

    # Average Action time
    if action_times:
        plt.plot(episodes, action_times)
        plt.title(
            f"Evolution of action mean time per episode in Training of {env_name} \nfor {algo_name} in {episodes.stop} "
            f"episodes"
            if is_training
            else f"Evolution of action mean time per episode in Test of {env_name} \nfor {algo_name} in {episodes.stop} "
            f"episodes for {metric_for}"
        )
        plt.xlabel("Episodes")
        plt.ylabel("Average action time (s)")
        plt.show()
        plt.close()

    # Actions choose distribution
    if actions:
        counts = Counter(actions)
        plt.figure(figsize=(14, 10))
        plt.barh(list(counts.keys()), list(counts.values()))
        plt.title(
            f"Distribution of action take by the agent in Training of {env_name} \nfor {algo_name} in {episodes.stop} "
            f"episodes"
            if is_training
            else f"Distribution of action take by the agent in Test of {env_name} \nfor {algo_name} in {episodes.stop} "
            f"episodes for {metric_for}"
        )
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
        plt.title(
            f"Evolution of agent step per episode in Training of {env_name} \nfor {algo_name} in {episodes.stop} "
            f"episodes"
            if is_training
            else f"Evolution of agent step per episode in Test of {env_name} \nfor {algo_name} in {episodes.stop} "
            f"episodes for {metric_for}"
        )
        plt.xlabel("Episodes")
        plt.ylabel("Nombre de step")
        plt.show()
        plt.close()

    # Losses metrics
    if losses:
        if isinstance(losses, list) and all(
            isinstance(sublist, list) for sublist in losses
        ):
            plt.plot(episodes, losses[0], label="Policy loss")
            plt.plot(episodes, losses[1], label="Baseline loss")
        else:
            plt.plot(losses)
        plt.title(
            f"Losses evolution per episode in Training of {env_name} \nfor {algo_name} in {episodes.stop} episodes "
            if is_training
            else f"Losses evolution per episode in Test of {env_name} \nfor {algo_name} in {episodes.stop} episodes "
            f"for {metric_for}"
        )
        plt.xlabel("Episodes")
        plt.ylabel("Losses")
        plt.show()
        plt.close()
