import os
import pickle
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tqdm import tqdm

from tools import print_metrics


class DQL:
    def __init__(
        self,
        state_size,
        action_size,
        learning_rate=0.001,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation="relu"))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss=tf.keras.losses.MeanSquaredError(),
        )
        return model

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def choose_action(self, state, available_actions):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(available_actions)
        q_values = self.model.predict(state.reshape(1, -1), verbose=0)
        valid_q_values = np.full(len(q_values[0]), -np.inf)
        for action in available_actions:
            valid_q_values[action] = q_values[0][action]
        return np.argmax(valid_q_values)

    def learn(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            next_q_values = self.model.predict(next_state.reshape(1, -1), verbose=0)
            target += self.gamma * np.max(next_q_values[0])

        current_q_values = self.model.predict(state.reshape(1, -1), verbose=0)
        current_q_values[0][action] = target

        history = self.model.fit(
            state.reshape(1, -1), current_q_values, epochs=1, verbose=0
        )
        loss = history.history["loss"][0]
        return loss

    def train(self, env, episodes=200, max_steps=500, test_intervals=[1000, 10_000, 100_000, 1000000]):
        scores_list = []
        losses_per_episode = []
        episode_times = []
        agent_action_times = []
        action_list = []

        with open(f"report/training_results_{self.__class__.__name__}_{env.__class__.__name__}_{episodes}episodes.txt", "a") as file:
            file.write("Training Started\n")
            file.write(f"Training with {episodes} episodes and max steps {max_steps}\n")

            for e in range(episodes):
                start_time = time.time()
                state = env.reset()
                total_reward = 0
                step_count = 0
                episode_losses = []

                pbar = tqdm(
                    total=max_steps,
                    desc=f"Episode {e + 1}/{episodes}",
                    unit="Step",
                    bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
                    postfix=f"total reward: {total_reward}, Epsilon : {self.epsilon:.4f}, agent Step: {step_count}, "
                    f"Average Action Time: 0",
                    dynamic_ncols=True,
                )

                if hasattr(env, "roll_dice"):
                    env.roll_dice()

                while not env.done and step_count < max_steps:
                    available_actions = env.available_actions()
                    keys = (
                        list(available_actions.keys())
                        if hasattr(env, "available_actions")
                        and isinstance(env.available_actions(), dict)
                        else available_actions
                    )

                    if hasattr(env, "current_player") and env.current_player == 1:
                        start_time_action = time.time()
                        action = self.choose_action(state, keys)
                        end_time_action = time.time()
                        agent_action_times.append(end_time_action - start_time_action)
                        step_count += 1
                    else:
                        action = np.random.choice(keys)

                    action_list.append(action)

                    pbar.update(1)

                    if hasattr(env, "available_actions") and isinstance(
                        env.available_actions(), dict
                    ):
                        next_state, reward, done = env.step(available_actions[action])
                    else:
                        next_state, reward, done = env.step(action)

                    loss = self.learn(state, action, reward, next_state, done)
                    episode_losses.append(loss)

                    state = next_state
                    total_reward += reward
                    pbar.set_postfix(
                        {
                            "Total Reward": total_reward,
                            "Epsilon": self.epsilon,
                            "Agent Step": step_count,
                            "Average Action Time": np.mean(agent_action_times)
                            if len(agent_action_times) > 0
                            else 0,
                        }
                    )

                    if env.done:
                        scores_list.append(total_reward)
                        losses_per_episode.append(
                            np.mean(episode_losses) if episode_losses else 0
                        )
                        break

                if not env.done and step_count >= max_steps:
                    print(f"Episode {e + 1}/{episodes} reached max steps ({max_steps})")
                    scores_list.append(total_reward)
                    losses_per_episode.append(
                        np.mean(episode_losses) if episode_losses else 0
                    )

                end_time = time.time()
                episode_times.append(end_time - start_time)

                self.update_epsilon()
                pbar.close()
                if (e + 1) in test_intervals:
                    win_rate, avg_reward = self.test(env, episodes=200, max_steps=max_steps)
                    file.write(f"Test after {e + 1} episodes: Average score: {avg_reward}, Win rate: {win_rate}\n")

            file.write("\nTraining Complete\n")
            file.write(f"Final Mean Score after {episodes} episodes: {np.mean(scores_list)}\n")
            file.write(f"Total training time: {np.sum(episode_times)} seconds\n")

        print_metrics(
            episodes=range(episodes),
            scores=scores_list,
            episode_times=episode_times,
            losses=losses_per_episode,
            actions=action_list,
        )

        return np.mean(scores_list)

    def test(self, env, episodes=200, max_steps=10):
        win_game = 0
        total_reward = 0
        for e in tqdm(range(episodes), desc="Testing"):
            state = env.reset()
            done = False
            step_count = 0
            episode_reward = 0

            if hasattr(env, "play_game"):  # Pour Farkle
                winner = env.play_game(isBotGame=True, show=False, agentPlayer=self)
                if winner == 0:
                    win_game += 1
            else:
                while not env.done and step_count < max_steps:
                    available_actions = env.available_actions()

                    if hasattr(env, "current_player") and env.current_player == 1:
                        action = self.choose_action(state, available_actions)
                        step_count += 1
                    else:
                        action = np.random.choice(available_actions)

                    next_state, reward, done = env.step(action)
                    episode_reward += reward
                    state = next_state

                    if env.done and hasattr(env, "winner") and env.winner == 1.0:
                        win_game += 1
                        break

                total_reward += episode_reward

                if not done and step_count >= max_steps:
                    print(f"Episode {e + 1}/{episodes} reached max steps ({max_steps})")

        avg_reward = total_reward / episodes
        print(
            f"Test Results:\n"
            f"- Games won: {win_game}/{episodes}\n"
            f"- Win rate: {(win_game / episodes) * 100:.2f}%\n"
            f"- Average reward per episode: {avg_reward:.2f}"
        )
        win_rate = win_game / episodes
        return win_rate, avg_reward

    def save_model(self, model_name):
        """Save model and parameters"""
        os.makedirs("saved_models", exist_ok=True)
        model_path = f"saved_models/{model_name}.h5"
        self.model.save(model_path)

        # Save additional parameters
        params = {
            "state_size": self.state_size,
            "action_size": self.action_size,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
        }

        with open(f"saved_models/{model_name}_params.pkl", "wb") as f:
            pickle.dump(params, f)

        print(f"Model and parameters saved as '{model_name}'.")

    def load_model(self, model_name):
        """Load model and parameters"""
        model_path = f"saved_models/{model_name}.h5"
        params_path = f"saved_models/{model_name}_params.pkl"

        if os.path.exists(model_path) and os.path.exists(params_path):
            self.model = tf.keras.models.load_model(model_path)

            with open(params_path, "rb") as f:
                params = pickle.load(f)

            self.state_size = params["state_size"]
            self.action_size = params["action_size"]
            self.learning_rate = params["learning_rate"]
            self.gamma = params["gamma"]
            self.epsilon = params["epsilon"]
            self.epsilon_min = params["epsilon_min"]
            self.epsilon_decay = params["epsilon_decay"]

            print(f"Model and parameters loaded from '{model_name}'.")
        else:
            print(f"No saved model found with the name '{model_name}'.")
