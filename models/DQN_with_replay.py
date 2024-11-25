import os
import pickle
import time
from collections import deque

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tqdm import tqdm, trange

from environement.farkle import Farkle
from environement.gridworld import GridWorld
from environement.lineworld import LineWorld
from tools import *


class DQN_with_replay:
    def __init__(
        self,
        state_size,
        action_size,
        learning_rate=0.01,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        batch_size=64,
        memory_size=2000,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
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

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return 0
        minibatch = random.sample(self.memory, self.batch_size)

        states = np.array([transition[0] for transition in minibatch])
        next_states = np.array([transition[3] for transition in minibatch])

        q_values = self.model.predict(states, verbose=0)
        next_q_values = self.model.predict(next_states, verbose=0)

        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            target = reward
            if not done:
                target += self.gamma * np.amax(next_q_values[i])
            q_values[i][action] = target

        history = self.model.fit(states, q_values, epochs=1, verbose=0)
        loss = history.history["loss"][0]
        return loss

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def choose_action(self, state, available_actions):
        if np.random.rand() <= self.epsilon:
            return random.choice(available_actions)
        q_values = self.model.predict(state.reshape(1, -1), verbose=0)
        valid_q_values = np.full(len(q_values[0]), -np.inf)
        for action in available_actions:
            valid_q_values[action] = q_values[0][action]
        return np.argmax(valid_q_values)

    def train(
        self,
        env,
        episodes=200,
        max_steps=500,
        test_intervals=[1000, 10_000, 100_000, 1000000],
    ):
        scores_list = []
        losses_per_episode = []
        episode_times = []
        agent_action_times = []
        actions_list = []
        step_by_episode = []

        with open(
            f"report/training_results_{self.__class__.__name__}_{env.__class__.__name__}_{episodes}episodes.txt",
            "a",
        ) as file:
            file.write("Training Started\n")
            file.write(f"Training with {episodes} episodes and max steps {max_steps}\n")

            for e in range(episodes):
                start_time = time.time()
                state = env.reset()
                total_reward = 0
                step_count = 0

                pbar = tqdm(
                    total=max_steps,
                    desc=f"Episode {e + 1}/{episodes}",
                    unit="Step",
                    bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
                    postfix=f"total reward: {total_reward}, Epsilon : {self.epsilon:.4f}, agent Step: {step_count}, "
                    f"Average Action Time: 0",
                    dynamic_ncols=True,
                )

                if isinstance(env, Farkle):
                    env.roll_dice()

                while not env.done and step_count < max_steps:
                    available_actions = env.available_actions()
                    keys = (
                        list(available_actions.keys())
                        if isinstance(env, Farkle)
                        else available_actions
                    )

                    if hasattr(env, "current_player") and env.current_player == 1:
                        action_start_time = time.time()
                        action = self.choose_action(state, keys)
                        action_end_time = time.time()
                        agent_action_times.append(action_end_time - action_start_time)
                        actions_list.append(action)
                        step_count += 1
                    else:
                        action = random.choice(keys)

                    pbar.update(1)
                    next_state, reward, done = (
                        env.step(available_actions[action])
                        if isinstance(env, Farkle)
                        else env.step(action)
                    )

                    self.remember(state, action, reward, next_state, done)
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
                        break

                episode_duration = time.time() - start_time
                episode_times.append(episode_duration)
                step_by_episode.append(step_count)
                episode_loss = self.replay()
                losses_per_episode.append(episode_loss)

                if not env.done and step_count >= max_steps:
                    print(f"Episode {e + 1}/{episodes} reached max steps ({max_steps})")
                    scores_list.append(total_reward)
                    losses_per_episode.append(0)

                self.update_epsilon()
                pbar.close()
                if (e + 1) in test_intervals:
                    win_rate, avg_reward = self.test(
                        env,
                        episodes=200,
                        max_steps=max_steps,
                        model_name=env.__class__.__name__ + "_" + str(e + 1)
                    )
                    file.write(
                        f"Test after {e + 1} episodes: Average score: {avg_reward}, Win rate: {win_rate}\n"
                    )

            file.write("\nTraining Complete\n")
            file.write(
                f"Final Mean Score after {episodes} episodes: {np.mean(scores_list)}\n"
            )
            file.write(f"Total training time: {np.sum(episode_times)} seconds\n")

        print_metrics(
            episodes=range(episodes),
            scores=scores_list,
            episode_times=episode_times,
            steps_per_game=step_by_episode,
            losses=losses_per_episode,
            actions=actions_list,
            algo_name=self.__class__.__name__,
            env_name=env.__class__.__name__,
        )

        return np.mean(scores_list)

    def test(self, env, episodes=200, max_steps=10, model_name=None):
        scores_list = []
        episode_times = []
        action_times = []
        actions_list = []
        step_by_episode = []
        win_game = 0
        total_reward = 0
        for e in trange(episodes, desc=f"Test"):
            episode_start_time = time.time()
            episode_action_times = []

            state = env.reset()
            done = False
            step_count = 0
            episode_reward = 0

            if isinstance(env, Farkle):
                winner, reward, a_list, a_times = env.play_game(
                    isBotGame=True, show=False, agentPlayer=self
                )
                if winner == 0:
                    win_game += 1
                episode_end_time = time.time()
                actions_list = a_list
                episode_action_times = a_times
                scores_list.append(reward)

            else:
                while not env.done and step_count < max_steps:
                    available_actions = env.available_actions()

                    if hasattr(env, "current_player") and env.current_player == 1:
                        action_start_time = time.time()
                        action = self.choose_action(state, available_actions)
                        action_end_time = time.time()
                        episode_action_times.append(action_end_time - action_start_time)
                        actions_list.append(action)
                        step_count += 1
                    else:
                        action = random.choice(available_actions)

                    next_state, reward, done = env.step(action)
                    episode_reward += reward
                    state = next_state

                    if env.done and env.winner == 1.0:
                        win_game += 1
                        break

                total_reward += episode_reward
                episode_end_time = time.time()

                if not done and step_count >= max_steps:
                    print(f"Episode {e + 1}/{episodes} reached max steps ({max_steps})")

            action_times.append(np.mean(episode_action_times))
            episode_times.append(episode_end_time - episode_start_time)
            step_by_episode.append(step_count)

        avg_reward = total_reward / episodes
        print(
            f"Winrate:\n- {win_game} game wined\n- {episodes} game played\n- Accuracy : {(win_game / episodes) * 100:.2f}%"
        )
        win_rate = win_game / episodes
        # Print metrics
        print_metrics(
            episodes=range(episodes),
            scores=scores_list,
            episode_times=episode_times,
            steps_per_game=step_by_episode,
            actions=actions_list,
            algo_name=self.__class__.__name__,
            env_name=env.__class__.__name__,
        )

        model_name = env.__class__.__name__ + "_" + str(episodes) if model_name is None else model_name
        self.save_model(model_name)

        return win_rate, avg_reward

    def save_model(self, game_name):
        """Save model and parameters"""
        os.makedirs("agents", exist_ok=True)
        model_path = f"agents/{self.__class__.__name__}_{game_name}.h5"
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
            "batch_size": self.batch_size,
        }

        with open(
            f"agents/{self.__class__.__name__}_{game_name}_params.pkl", "wb"
        ) as f:
            pickle.dump(params, f)

        print(f"Model and parameters saved as '{game_name}'.")

    def load_model(self, game_name):
        """Load model and parameters"""
        model_path = f"agents/{self.__class__.__name__}_{game_name}.h5"
        params_path = f"agents/{self.__class__.__name__}_{game_name}_params.pkl"

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

            print(f"Model and parameters loaded for '{game_name}'.")
        else:
            print(f"No saved model found with the game '{game_name}'.")
