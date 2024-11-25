import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from environement.farkle import Farkle
import numpy as np
import random
from tqdm import tqdm, trange
import time
from tools import print_metrics
import os
import pickle


class ReinforceActorCritic:
    def __init__(self, state_size, action_size, learning_rate=0.1, gamma=0.9):
        self.learning_rate = learning_rate
        self.theta = self.build_model(state_size, action_size)
        self.baseline = self.build_model(state_size, 1)
        self.gamma = gamma

    def build_model(self, state_size, action_size):
        model = Sequential()
        model.add(Dense(64, input_dim=state_size, activation="relu"))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(action_size, activation="linear"))
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="mse",
        )
        return model

    def choose_action(self, state, available_actions):
        prediction = self.theta.predict(state.reshape(1, -1), verbose=0)
        action_index = np.argmax([prediction[0][i] for i in available_actions])
        return available_actions[action_index]

    def train(self, environment, episodes, max_steps, test_intervals=[1000, 10_000, 100_000, 1_000_000]):
        scores_list = []
        episode_times = []
        action_times = []
        actions_list = []
        steps_per_game = []
        losses_per_episode = []
        policy_losses_per_episode = []
        baseline_losses_per_episode = []

        with open(
            f"report/training_results_{self.__class__.__name__}_{environment.__class__.__name__}_{episodes}episodes.txt",
            "a",
        ) as file:
            file.write("Training Started\n")
            file.write(f"Training with {episodes} episodes and max steps {max_steps}\n")

            for episode in range(episodes):
                state = environment.reset()
                state = np.expand_dims(state, axis=0)
                step_count = 0
                importance = 1
                baseline = self.baseline.predict(state, verbose=0)[0][0]

                pbar = tqdm(
                    total=max_steps,
                    desc=f"Episode {episode + 1}/ {episodes}",
                    unit="Step",
                    bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
                    postfix=f"total reward: 0, agent Step: {step_count}, Average Action Time: 0",
                    dynamic_ncols=True,
                )
                agent_action_times = []
                episode_policy_loss = 0
                episode_baseline_loss = 0
                episode_start_time = time.time()

                if isinstance(environment, Farkle):
                    environment.roll_dice()
                while not environment.done and step_count < max_steps:
                    available_actions = environment.available_actions()
                    keys = (
                        list(available_actions.keys())
                        if isinstance(environment, Farkle)
                        else available_actions
                    )

                    if (
                        hasattr(environment, "current_player")
                        and environment.current_player == 1
                    ):
                        action_start_time = time.time()
                        action = self.choose_action(state, keys)
                        action_end_time = time.time()
                        agent_action_times.append(action_end_time - action_start_time)
                        actions_list.append(action)
                        step_count += 1
                    else:
                        action = random.choice(keys)

                    next_state, reward, done = (
                        environment.step(available_actions[action])
                        if isinstance(environment, Farkle)
                        else environment.step(action)
                    )
                    next_state = np.expand_dims(next_state, axis=0)

                    next_baseline = (
                        self.baseline.predict(next_state, verbose=0)[0][0]
                        if not done
                        else 0
                    )
                    delta = reward + (self.gamma * next_baseline) - baseline

                    baseline_loss = self.update_baseline(state, delta)
                    policy_loss = self.update_policy(state, action, delta, importance)

                    episode_policy_loss += policy_loss
                    episode_baseline_loss += baseline_loss

                    importance = importance * self.gamma
                    state = next_state
                    baseline = next_baseline

                    pbar.update(1)
                    pbar.set_postfix(
                        {
                            "Reward": reward,
                            "Agent Step": step_count,
                            "Policy loss": policy_loss,
                            "Baseline loss": baseline_loss,
                            "Average Action Time": np.mean(agent_action_times)
                            if len(agent_action_times) > 0
                            else 0,
                        }
                    )

                    if done:
                        episode_end_time = time.time()
                        episode_times.append(episode_end_time - episode_start_time)
                        scores_list.append(reward)
                        action_times.append(np.mean(agent_action_times))
                        steps_per_game.append(step_count)
                        policy_losses_per_episode.append(episode_policy_loss)
                        baseline_losses_per_episode.append(episode_baseline_loss)

                pbar.close()

                if (episode + 1) in test_intervals:
                    win_rate, avg_reward = self.test(
                        environment,
                        10,
                        max_steps,
                        model_name=environment.__class__.__name__ + "_" + str(episode + 1)
                    )

            file.write("\nTraining Complete\n")
            file.write(
                f"Final Mean Score after {episodes} episodes: {np.mean(scores_list)}\n"
            )
            file.write(f"Total training time: {np.sum(episode_times)} seconds\n")

        losses_per_episode.append(policy_losses_per_episode)
        losses_per_episode.append(baseline_losses_per_episode)

        # Print metrics
        print_metrics(
            range(episodes),
            scores_list,
            episode_times,
            action_times,
            actions_list,
            steps_per_game,
            losses_per_episode,
            algo_name=self.__class__.__name__,
            env_name=environment.__class__.__name__,
        )
        return np.mean(scores_list)

    def test(self, environment, episodes, max_steps, model_name=None):
        scores_list = []
        episode_times = []
        action_times = []
        actions_list = []
        win_games = 0
        for e in trange(episodes, desc="Testing", unit="episode"):
            episode_start_time = time.time()
            episode_action_times = []
            state = environment.reset()
            done = False
            step_count = 0

            if isinstance(environment, Farkle):
                winner, reward, a_list, a_times = environment.play_game(
                    isBotGame=True, show=False, agentPlayer=self
                )
                if winner == 0:
                    win_games += 1
                episode_end_time = time.time()
                actions_list = a_list
                episode_action_times = a_times
                scores_list.append(reward)

            else:
                while not environment.done and step_count < max_steps:
                    available_actions = environment.available_actions()

                    if (
                        hasattr(environment, "current_player")
                        and environment.current_player == 1
                    ):
                        action_start_time = time.time()
                        action = self.choose_action(state, available_actions)
                        action_end_time = time.time()
                        episode_action_times.append(action_end_time - action_start_time)
                        actions_list.append(action)
                        step_count += 1
                    else:
                        action = random.choice(available_actions)

                    next_state, reward, done = environment.step(action)
                    state = next_state

                    if environment.done and environment.winner == 1.0:
                        win_games += 1
                        break

                episode_end_time = time.time()
                if not done and step_count >= max_steps:
                    scores_list.append(environment.get_score())
                    print(f"Episode {e + 1}/{episodes} reached max steps ({max_steps})")
            action_times.append(np.mean(episode_action_times))
            episode_times.append(episode_end_time - episode_start_time)
        print(
            f"Winrate:\n- {win_games} game wined\n- {episodes} game played\n- Accuracy : {(win_games / episodes) * 100:.2f}%"
        )
        # Print metrics
        print_metrics(
            episodes=range(episodes),
            scores=scores_list,
            episode_times=episode_times,
            action_times=action_times,
            actions=actions_list,
            is_training=False,
            algo_name=self.__class__.__name__,
            env_name=environment.__class__.__name__,
        )

        model_name = environment.__class__.__name__ + "_" + str(episodes) if model_name is None else model_name
        self.save_model(model_name)

        return (win_games / episodes) * 100, np.mean(scores_list)

    def update_baseline(self, state, delta):
        with tf.GradientTape() as tape:
            baseline_values = self.baseline(state, training=True)
            baseline_loss = tf.reduce_mean(
                (baseline_values - (baseline_values + delta)) ** 2
            )

        grads = tape.gradient(baseline_loss, self.baseline.trainable_variables)

        for i, var in enumerate(self.baseline.trainable_variables):
            var.assign_sub(self.learning_rate * grads[i])

        return baseline_loss.numpy()

    def update_policy(self, state, action, delta, i):
        with tf.GradientTape() as tape:
            action_probs = self.theta(state, training=True)
            log_probs = tf.math.log(action_probs[0, action])
            loss = -i * delta * log_probs

        grads = tape.gradient(loss, self.theta.trainable_variables)

        for i, var in enumerate(self.theta.trainable_variables):
            var.assign_add(self.learning_rate * grads[i])

        return loss.numpy()

    def save_model(self, game_name):
        agent_data = {
            "learning_rate": self.learning_rate,
            "theta_weights": self.theta.get_weights(),
            "baseline_weights": self.baseline.get_weights(),
            "gamma": self.gamma,
        }
        os.makedirs("agents", exist_ok=True)
        with open(f"agents/{self.__class__.__name__}_{game_name}.pkl", "wb") as f:
            pickle.dump(agent_data, f)
        print(f"Agent {self.__class__.__name__} pour le jeu {game_name} sauvegardé.")

    def load_model(self, game_name):
        agent_path = f"agents/{self.__class__.__name__}_{game_name}.pkl"
        if os.path.exists(agent_path):
            with open(agent_path, "rb") as f:
                agent_data = pickle.load(f)
            self.learning_rate = agent_data["learning_rate"]
            self.theta.set_weights(agent_data["theta_weights"])
            self.baseline.set_weights(agent_data["baseline_weights"])
            self.gamma = agent_data["gamma"]
            print(f"Agent {self.__class__.__name__} pour le jeu {game_name} chargé.")
        else:
            print(f"Aucun agent sauvegardé pour le jeu {game_name} trouvé.")


# _env = Farkle(printing=False)
# _model = ReinforceActorCritic(_env.state_size, _env.actions_size, 0.01, 0.001)

# _model.train(environment=_env, episodes=10, max_steps=300)
# _model.test(environment=_env, episodes=4, max_steps=300)
# _model.save_model("farkle_10")

# model_test = ReinforceActorCritic(_env.state_size, _env.actions_size, 1, 1)
# model_test.load_model("farkle_10")
# model_test.test(environment=_env, episodes=4, max_steps=300)
