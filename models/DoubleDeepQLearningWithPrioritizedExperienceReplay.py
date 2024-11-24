import os
import pickle
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential, clone_model, load_model
from tqdm import tqdm

from tools import print_metrics


class SumTree:
    """Structure de données pour le Prioritized Experience Replay"""

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0
        self.write = 0

    def _propagate(self, idx, change):
        """Propage la mise à jour des priorités"""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        """Trouve l'index de l'échantillon basé sur la priorité"""
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        """Retourne la somme totale des priorités"""
        return self.tree[0]

    def add(self, p, data):
        """Ajoute une nouvelle expérience avec sa priorité"""
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx, p):
        """Met à jour la priorité d'une expérience"""
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        """Récupère une expérience basée sur une valeur de priorité"""
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[dataIdx]


class PrioritizedReplayBuffer:
    """Gestion du buffer d'expérience avec priorités"""

    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = 1e-6
        self.tree = SumTree(capacity)

    def add(self, error, sample):
        """Ajoute une nouvelle expérience avec sa priorité"""
        p = (np.abs(error) + self.epsilon) ** self.alpha
        self.tree.add(p, sample)

    def sample(self, batch_size):
        """Échantillonne un batch d'expériences basé sur les priorités"""
        batch = []
        idxs = []
        segment = self.tree.total() / batch_size
        priorities = []

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)
            idx, p, data = self.tree.get(s)

            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        self.beta = np.min([1.0, self.beta + self.beta_increment])
        return batch, idxs, is_weight

    def update(self, idx, error):
        """Met à jour la priorité d'une expérience"""
        p = (np.abs(error) + self.epsilon) ** self.alpha
        self.tree.update(idx, p)


class DDQLWithPER:
    def __init__(
        self,
        state_size,
        action_size,
        learning_rate=0.01,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        memory_size=10000,
        batch_size=64,
        target_update_frequency=100,
        alpha=0.6,
        beta=0.4,
        beta_increment=0.001,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency
        self.training_step = 0

        self.main_model = self.build_model()
        self.target_model = clone_model(self.main_model)
        self.target_model.set_weights(self.main_model.get_weights())

        self.memory = PrioritizedReplayBuffer(
            memory_size, alpha=alpha, beta=beta, beta_increment=beta_increment
        )

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
            loss=tf.keras.losses.MeanSquaredError(),
        )
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.main_model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        # Calcul de l'erreur TD initiale pour la priorité
        target = reward
        if not done:
            next_q_values = self.target_model.predict(
                next_state.reshape(1, -1), verbose=0
            )
            target += self.gamma * np.max(next_q_values[0])

        current_q_values = self.main_model.predict(state.reshape(1, -1), verbose=0)
        error = abs(target - current_q_values[0][action])

        self.memory.add(error, (state, action, reward, next_state, done))

    def choose_action(self, state, available_actions):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(available_actions)

        q_values = self.main_model.predict(state.reshape(1, -1), verbose=0)
        valid_q_values = np.full(len(q_values[0]), -np.inf)
        for action in available_actions:
            valid_q_values[action] = q_values[0][action]
        return np.argmax(valid_q_values)

    def replay(self):
        if self.memory.tree.n_entries < self.batch_size:
            return

        # Échantillonnage du batch avec priorités
        batch, indices, is_weights = self.memory.sample(self.batch_size)
        states = np.array([exp[0] for exp in batch])
        actions = np.array([exp[1] for exp in batch])
        rewards = np.array([exp[2] for exp in batch])
        next_states = np.array([exp[3] for exp in batch])
        dones = np.array([exp[4] for exp in batch])

        # Calcul des Q-valeurs cibles avec Double DQN
        current_q_values = self.main_model.predict(states, verbose=0)
        next_q_values_main = self.main_model.predict(next_states, verbose=0)
        next_q_values_target = self.target_model.predict(next_states, verbose=0)

        max_actions = np.argmax(next_q_values_main, axis=1)

        for i in range(self.batch_size):
            if dones[i]:
                target = rewards[i]
            else:
                target = (
                    rewards[i] + self.gamma * next_q_values_target[i][max_actions[i]]
                )

            # Calcul de l'erreur TD pour la mise à jour des priorités
            error = abs(target - current_q_values[i][actions[i]])
            self.memory.update(indices[i], error)

            # Application des poids d'importance sampling
            current_q_values[i][actions[i]] = current_q_values[i][
                actions[i]
            ] + is_weights[i] * (target - current_q_values[i][actions[i]])

        self.main_model.fit(states, current_q_values, epochs=1, verbose=0)

        self.training_step += 1
        if self.training_step % self.target_update_frequency == 0:
            self.update_target_model()

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

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
        action_list = []
        best_score = float("-inf")
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
                episode_losses = []

                pbar = tqdm(
                    total=max_steps,
                    desc=f"Episode {e + 1}/{episodes}",
                    unit="Step",
                    bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
                    postfix=f"total reward: {total_reward}, Epsilon: {self.epsilon:.4f}, agent Step: {step_count}, Average Action Time: 0",
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
                        start_action_time = time.time()
                        action = self.choose_action(state, keys)
                        end_action_time = time.time()
                        agent_action_times.append(end_action_time - start_action_time)
                        step_count += 1
                    else:
                        action = np.random.choice(keys)

                    action_list.append(action)  # Enregistrer l'action choisie

                    pbar.update(1)

                    if hasattr(env, "available_actions") and isinstance(
                        env.available_actions(), dict
                    ):
                        next_state, reward, done = env.step(available_actions[action])
                    else:
                        next_state, reward, done = env.step(action)

                    self.remember(state, action, reward, next_state, done)

                    if len(self.memory.tree.data) >= self.batch_size:
                        batch, indices, is_weights = self.memory.sample(self.batch_size)
                        states = np.array([exp[0] for exp in batch])
                        actions = np.array([exp[1] for exp in batch])
                        rewards = np.array([exp[2] for exp in batch])
                        next_states = np.array([exp[3] for exp in batch])
                        dones = np.array([exp[4] for exp in batch])

                        current_q_values = self.main_model.predict(states, verbose=0)
                        next_q_values_main = self.main_model.predict(
                            next_states, verbose=0
                        )
                        next_q_values_target = self.target_model.predict(
                            next_states, verbose=0
                        )

                        target_q_values = current_q_values.copy()
                        for i in range(self.batch_size):
                            if dones[i]:
                                target = rewards[i]
                            else:
                                best_action = np.argmax(next_q_values_main[i])
                                target = (
                                    rewards[i]
                                    + self.gamma * next_q_values_target[i][best_action]
                                )
                            target_q_values[i][actions[i]] = target

                        loss = tf.reduce_mean(
                            tf.multiply(
                                is_weights,
                                tf.reduce_mean(
                                    tf.square(target_q_values - current_q_values),
                                    axis=1,
                                ),
                            )
                        ).numpy()

                        episode_losses.append(loss)

                    self.replay()
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
                            "Loss": np.mean(episode_losses) if episode_losses else 0,
                        }
                    )

                    if env.done:
                        if total_reward > best_score:
                            best_score = total_reward
                        break

                pbar.close()

                scores_list.append(total_reward)
                losses_per_episode.append(
                    np.mean(episode_losses) if episode_losses else 0
                )
                end_time = time.time()
                episode_times.append(end_time - start_time)
                step_by_episode.append(step_count)

                self.update_epsilon()

                if (e + 1) in test_intervals:
                    win_rate, avg_reward = self.test(
                        env, episodes=200, max_steps=max_steps
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
        test_intervals=[1000, 10_000, 100_000, 1000000],
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

                total_reward += episode_reward
                episode_end_time = time.time()

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
        self.save_model(env.__class__.__name__)
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
