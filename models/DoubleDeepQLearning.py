import os
import pickle
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential, clone_model, load_model
from tensorflow.python.keras.utils.vis_utils import is_wrapped_model
from tqdm import tqdm

from tools import print_metrics


class DDQL:
    def __init__(
        self,
        state_size,
        action_size,
        learning_rate=0.01,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        target_update_frequency=100,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.target_update_frequency = target_update_frequency
        self.training_step = 0

        self.main_model = self.build_model()
        self.target_model = clone_model(self.main_model)
        self.target_model.set_weights(self.main_model.get_weights())

    def build_model(self):
        model = Sequential(
            [
                Dense(64, input_dim=self.state_size, activation="relu"),
                Dense(64, activation="relu"),
                Dense(self.action_size, activation="linear"),
            ]
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="mse",
        )
        return model

    def update_target_model(self):
        """Mise à jour du réseau cible avec les poids du réseau principal"""
        self.target_model.set_weights(self.main_model.get_weights())

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def choose_action(self, state, available_actions):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(available_actions)

        q_values = self.main_model.predict(state.reshape(1, -1), verbose=0)
        valid_q_values = np.full(len(q_values[0]), -np.inf)
        for action in available_actions:
            valid_q_values[action] = q_values[0][action]
        return np.argmax(valid_q_values)

    def learn(self, state, action, reward, next_state, done):
        self.training_step += 1

        # Calcul de la Q-valeur cible en utilisant les deux réseaux
        if not done:
            # Le réseau principal sélectionne l'action
            next_q_values_main = self.main_model.predict(
                next_state.reshape(1, -1), verbose=0
            )
            best_action = np.argmax(next_q_values_main[0])

            # Le réseau cible évalue cette action
            next_q_values_target = self.target_model.predict(
                next_state.reshape(1, -1), verbose=0
            )
            target = reward + self.gamma * next_q_values_target[0][best_action]
        else:
            target = reward

        current_q_values = self.main_model.predict(state.reshape(1, -1), verbose=0)
        current_q_values[0][action] = target
        history = self.main_model.fit(
            state.reshape(1, -1), current_q_values, epochs=1, verbose=0
        )

        loss = history.history["loss"][0]

        # Mise à jour périodique du réseau cible
        if self.training_step % self.target_update_frequency == 0:
            self.update_target_model()

        return loss

    def train(
        self,
        env,
        episodes=200,
        max_steps=500,
        test_intervals=[1000, 10_000, 100_000, 1000000],
    ):
        """
        Entraîne l'agent et effectue des tests à des intervalles spécifiques, en enregistrant les résultats dans un fichier.
        Args:
            env: L'environnement dans lequel l'agent évolue.
            episodes: Le nombre total d'épisodes d'entraînement.
            max_steps: Le nombre maximal de steps par épisode.
            test_intervals: Liste des intervalles d'épisodes pour effectuer les tests.
        """
        scores_list = []
        losses_per_episode = []
        episode_times = []
        agent_action_times = []
        action_list = []
        step_by_episode = []

        # Ouvrir le fichier pour écrire les résultats
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
                episode_losses = []

                pbar = tqdm(
                    total=max_steps,
                    desc=f"Episode {e + 1}/{episodes}",
                    unit="Step",
                    bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
                    postfix=f"Total reward: {total_reward}, Epsilon : {self.epsilon:.4f}, Agent Step: {step_count}, "
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
                        break

                if not env.done and step_count >= max_steps:
                    print(f"Episode {e + 1}/{episodes} reached max steps ({max_steps})")

                end_time = time.time()
                episode_times.append(end_time - start_time)
                scores_list.append(total_reward)
                losses_per_episode.append(np.mean(episode_losses))
                step_by_episode.append(step_count)
                print(step_by_episode)
                episode_losses.append(loss)

                self.update_epsilon()
                pbar.close()

                if test_intervals is not None and (e + 1) in test_intervals:
                    win_rate, avg_score = self.test(
                        env,
                        episodes=10,
                        max_steps=max_steps,
                        model_name=env.__class__.__name__ + "_" + str(e + 1),
                        is_saving_after_train=True,
                    )
                    file.write(
                        f"Test after {e + 1} episodes: Average score: {avg_score}, Win rate: {win_rate}\n"
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
                losses=losses_per_episode,
                steps_per_game=step_by_episode,
                actions=action_list,
                algo_name=self.__class__.__name__,
                env_name=env.__class__.__name__,
            )

        return np.mean(scores_list)

    def test(
        self,
        env,
        episodes=200,
        max_steps=10,
        model_name=None,
        is_saving_after_train=False,
    ):
        scores_list = []
        episode_times = []
        action_times = []
        actions_list = []
        step_by_episode = []
        win_game = 0
        total_reward = 0

        for e in tqdm(range(episodes), desc="Testing"):
            episode_start_time = time.time()
            episode_action_times = []
            state = env.reset()
            episode_reward = 0
            step_count = 0

            if hasattr(env, "play_game"):
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
                        action = np.random.choice(available_actions)

                    next_state, reward, done = env.step(action)
                    episode_reward += reward
                    state = next_state

                    if env.done and hasattr(env, "winner") and env.winner == 1.0:
                        win_game += 1
                        break

                episode_end_time = time.time()
                total_reward += episode_reward

            action_times.append(np.mean(episode_action_times))
            episode_times.append(episode_end_time - episode_start_time)
            step_by_episode.append(step_count)

        avg_reward = total_reward / episodes
        print(
            f"Test Results:\n"
            f"- Games won: {win_game}/{episodes}\n"
            f"- Win rate: {(win_game / episodes) * 100:.2f}%\n"
            f"- Average reward per episode: {avg_reward:.2f}"
        )
        # Print metrics
        print_metrics(
            episodes=range(episodes),
            scores=scores_list,
            episode_times=episode_times,
            steps_per_game=step_by_episode,
            actions=actions_list,
            algo_name=self.__class__.__name__,
            env_name=env.__class__.__name__,
            metric_for=str(model_name.split("_")[-1].split(".")[0])
            + " episodes trained"
            if is_saving_after_train
            else "",
        )
        win_rate = win_game / episodes

        if is_saving_after_train:
            model_name = (
                env.__class__.__name__ + "_" + str(episodes)
                if model_name is None
                else model_name
            )
            self.save_model(model_name)

        return win_rate, avg_reward

    def save_model(self, game_name):
        agent_data = {
            "main_model": self.main_model,
            "target_model": self.target_model,
            "state_size": self.state_size,
            "action_size": self.action_size,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "target_update_frequency": self.target_update_frequency,
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
            self.main_model = agent_data["main_model"]
            self.target_model = agent_data["target_model"]
            self.state_size = agent_data["state_size"]
            self.action_size = agent_data["action_size"]
            self.learning_rate = agent_data["learning_rate"]
            self.gamma = agent_data["gamma"]
            self.epsilon = agent_data["epsilon"]
            self.epsilon_min = agent_data["epsilon_min"]
            self.epsilon_decay = agent_data["epsilon_decay"]
            self.target_update_frequency = agent_data["target_update_frequency"]
            print(f"Agent {self.__class__.__name__} pour le jeu {game_name} chargé.")
        else:
            print(f"Aucun agent sauvegardé pour le jeu {game_name} trouvé.")
