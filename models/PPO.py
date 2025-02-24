import os
import pickle
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input, LayerNormalization
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

from tools import print_metrics


class PPO:
    def __init__(
        self,
        state_size: int,
        action_size: int,
        learning_rate: float = 0.1,
        gamma: float = 0.99,
        lam: float = 0.95,
        epsilon_clip: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        epochs: int = 10,
        mini_batch_size: int = 64,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.lam = lam
        self.epsilon_clip = epsilon_clip
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.epochs = epochs
        self.mini_batch_size = mini_batch_size

        self.hidden_sizes = [256, 128]
        self.max_grad_norm = 0.5
        self.target_kl = 0.015

        self.policy_network, self.value_network = self._build_networks()
        self.old_policy_network = self._clone_policy_network()

        self.policy_optimizer = Adam(learning_rate=self.learning_rate)
        self.value_optimizer = Adam(learning_rate=self.learning_rate)

    def _build_networks(self):
        def create_base_network(input_layer):
            x = input_layer
            for size in self.hidden_sizes:
                x = Dense(size, activation="relu")(x)
                x = LayerNormalization()(x)
            return x

        state_input = Input(shape=(self.state_size,))
        shared_policy = create_base_network(state_input)
        policy_output = Dense(self.action_size, activation="softmax")(shared_policy)
        policy_network = Model(inputs=state_input, outputs=policy_output)

        shared_value = create_base_network(state_input)
        value_output = Dense(1)(shared_value)
        value_network = Model(inputs=state_input, outputs=value_output)

        return policy_network, value_network

    def _clone_policy_network(self):
        return tf.keras.models.clone_model(self.policy_network)

    def choose_action(self, state: np.ndarray, available_actions=None):
        state = np.array(
            state, dtype=np.float32
        )  # Assurez-vous que l'état est un tableau NumPy
        policy = self.policy_network.predict(state.reshape(1, -1), verbose=0)[
            0
        ]  # Prédiction des probabilités d'action

        if available_actions is not None:
            # Récupération des indices disponibles
            if isinstance(available_actions, dict):
                available_indices = list(available_actions.keys())
            else:
                available_indices = available_actions

            # Application du masque
            mask = np.zeros_like(policy)
            mask[available_indices] = 1
            policy = policy * mask

            # Gestion des cas où aucune action n'est possible
            if policy.sum() == 0:
                # Distribue uniformément les probabilités sur les actions disponibles
                policy[available_indices] = 1 / len(available_indices)
            else:
                policy = policy / policy.sum()

        return np.random.choice(len(policy), p=policy)

    def train(
        self,
        env,
        episodes=1000,
        max_steps=500,
        test_intervals=[1000, 10_000, 100_000, 1000000],
    ):
        scores_list = []
        episode_times = []
        agent_action_times = []
        action_list = []
        step_by_episode = []

        with open(
            f"report/training_results_{self.__class__.__name__}_{env.__class__.__name__}_{episodes}episodes.txt",
            "a",
        ) as file:
            file.write("Training Started\n")
            file.write(f"Training with {episodes} episodes and max steps {max_steps}\n")

            pbar = tqdm(
                range(episodes),
                total=episodes,
                unit="episodes",
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
                postfix=f"total reward: 0, agent Step: 0, Average Action Time: 0",
            )

            for e in pbar:
                start_time = time.time()
                state = env.reset()
                total_reward = 0
                step_count = 0
                episode_losses = []

                states, actions, rewards, dones, values, log_probs = (
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                )

                if hasattr(env, "roll_dice"):
                    env.roll_dice()

                while not env.done and step_count < max_steps:
                    available_actions = env.available_actions()
                    start_time = time.time()
                    action = self.choose_action(state, available_actions)
                    end_time = time.time()

                    value = self.value_network.predict(state.reshape(1, -1), verbose=0)[
                        0
                    ][0]
                    policy = self.policy_network.predict(
                        state.reshape(1, -1), verbose=0
                    )[0]
                    log_prob = np.log(policy[action] + 1e-8)

                    if isinstance(available_actions, dict):
                        next_state, reward, done = env.step(available_actions[action])
                    else:
                        next_state, reward, done = env.step(action)

                    states.append(state)
                    actions.append(action)
                    action_list.append(action)
                    rewards.append(reward)
                    dones.append(done)
                    values.append(value)
                    log_probs.append(log_prob)
                    agent_action_times.append(end_time - start_time)

                    step_count += 1
                    total_reward += reward

                    state = next_state

                    if env.done:
                        scores_list.append(total_reward)
                        break

                if not env.done and step_count >= max_steps:
                    scores_list.append(total_reward)

                # Compute GAE and returns
                values.append(
                    self.value_network.predict(state.reshape(1, -1), verbose=0)[0][0]
                )
                advantages, returns = self._compute_gae(rewards, values, dones)

                end_time = time.time()
                episode_times.append(end_time - start_time)
                step_by_episode.append(step_count)

                self._update_networks(states, actions, advantages, returns, log_probs)

                pbar.update(1)
                pbar.set_postfix(
                    {
                        "Total Reward": total_reward,
                        "Agent Step": step_count,
                        "Average Action Time": np.mean(agent_action_times)
                        if agent_action_times
                        else 0,
                    }
                )

                if test_intervals is not None and (e + 1) in test_intervals:
                    win_rate, avg_reward = self.test(
                        env,
                        episodes=10,
                        max_steps=max_steps,
                        model_name=env.__class__.__name__ + "_" + str(e + 1),
                        is_saving_after_train=True,
                    )
                    file.write(
                        f"Test after {e + 1} episodes: Average score: {avg_reward}, Win rate: {win_rate}\n"
                    )

            pbar.close()
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
            actions=action_list,
            algo_name=self.__class__.__name__,
            env_name=env.__class__.__name__,
        )

        return np.mean(scores_list)

    def _compute_gae(self, rewards, values, dones):
        advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = 0

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values[:-1]
        return advantages, returns

    def _update_networks(self, states, actions, advantages, returns, old_log_probs):
        states = np.array(states)
        actions = np.array(actions)
        advantages = np.array(advantages)
        returns = np.array(returns)
        old_log_probs = np.array(old_log_probs)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        indices = np.arange(len(states))

        for _ in range(self.epochs):
            np.random.shuffle(indices)

            for start_idx in range(0, len(states), self.mini_batch_size):
                batch_indices = indices[start_idx : start_idx + self.mini_batch_size]

                with tf.GradientTape() as policy_tape:
                    batch_states = tf.convert_to_tensor(states[batch_indices])
                    policy_dist = self.policy_network(batch_states)
                    batch_actions = actions[batch_indices]
                    batch_advantages = advantages[batch_indices]

                    # Compute new log probs
                    action_masks = tf.one_hot(batch_actions, self.action_size)
                    new_log_probs = tf.reduce_sum(
                        tf.math.log(policy_dist + 1e-8) * action_masks, axis=1
                    )

                    # Compute ratios and surrogate losses
                    ratios = tf.exp(new_log_probs - old_log_probs[batch_indices])
                    surrogate1 = ratios * batch_advantages
                    surrogate2 = (
                        tf.clip_by_value(
                            ratios, 1 - self.epsilon_clip, 1 + self.epsilon_clip
                        )
                        * batch_advantages
                    )

                    # Compute policy loss with entropy bonus
                    entropy = -tf.reduce_mean(
                        tf.reduce_sum(
                            policy_dist * tf.math.log(policy_dist + 1e-8), axis=1
                        )
                    )
                    policy_loss = (
                        -tf.reduce_mean(tf.minimum(surrogate1, surrogate2))
                        - self.entropy_coef * entropy
                    )

                # Update policy network
                policy_grads = policy_tape.gradient(
                    policy_loss, self.policy_network.trainable_variables
                )
                policy_grads, _ = tf.clip_by_global_norm(
                    policy_grads, self.max_grad_norm
                )
                self.policy_optimizer.apply_gradients(
                    zip(policy_grads, self.policy_network.trainable_variables)
                )

                with tf.GradientTape() as value_tape:
                    value_pred = self.value_network(batch_states)
                    value_loss = tf.reduce_mean(
                        tf.square(returns[batch_indices] - value_pred)
                    )

                value_grads = value_tape.gradient(
                    value_loss, self.value_network.trainable_variables
                )
                value_grads, _ = tf.clip_by_global_norm(value_grads, self.max_grad_norm)
                self.value_optimizer.apply_gradients(
                    zip(value_grads, self.value_network.trainable_variables)
                )

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
                scores_list.append(total_reward)
                step_by_episode.append(step_count)

            action_times.append(np.mean(episode_action_times))
            episode_times.append(episode_end_time - episode_start_time)

        win_rate = win_game / episodes
        avg_reward = total_reward / episodes
        print(
            f"Test Results:\n"
            f"- Games won: {win_game}/{episodes}\n"
            f"- Win rate: {(win_game / episodes) * 100:.2f}%\n"
            f"- Average reward per episode: {avg_reward:.2f}"
        )

        if is_saving_after_train:
            model_name = (
                env.__class__.__name__ + "_" + str(episodes)
                if model_name is None
                else model_name
            )
            self.save_model(model_name)

        # Print metrics
        print_metrics(
            episodes=range(episodes),
            scores=scores_list,
            episode_times=episode_times,
            action_times=action_times,
            steps_per_game=step_by_episode,
            actions=actions_list,
            is_training=False,
            algo_name=self.__class__.__name__,
            env_name=env.__class__.__name__,
            metric_for=str(model_name.split("_")[-1].split(".")[0])
            + " episodes trained"
            if is_saving_after_train
            else "",
        )
        return win_rate, avg_reward

    def save_model(self, game_name):
        try:
            os.makedirs("agents", exist_ok=True)
            self.policy_network.save(
                f"agents/{self.__class__.__name__}_{game_name}_policy.h5"
            )
            self.value_network.save(
                f"agents/{self.__class__.__name__}_{game_name}_value.h5"
            )

            params = {
                "state_size": self.state_size,
                "action_size": self.action_size,
                "learning_rate": self.learning_rate,
                "gamma": self.gamma,
                "epsilon_clip": self.epsilon_clip,
                "value_loss_coef": self.value_loss_coef,
                "entropy_coef": self.entropy_coef,
                "epochs": self.epochs,
                "mini_batch_size": self.mini_batch_size,
            }

            with open(
                f"agents/{self.__class__.__name__}_{game_name}_params.pkl", "wb"
            ) as f:
                pickle.dump(params, f)

        except Exception as e:
            print(f"Error saving model: {e}")

    def load_model(self, game_name):
        """Load the model from disk"""
        try:
            policy_path = f"agents/{self.__class__.__name__}_{game_name}_policy.h5"
            value_path = f"agents/{self.__class__.__name__}_{game_name}_value.h5"
            params_path = f"agents/{self.__class__.__name__}_{game_name}_params.pkl"

            if not (
                os.path.exists(policy_path)
                and os.path.exists(value_path)
                and os.path.exists(params_path)
            ):
                raise FileNotFoundError("One or more model files are missing")

            self.policy_network = tf.keras.models.load_model(policy_path)
            self.value_network = tf.keras.models.load_model(value_path)
            self.old_policy_network = self._clone_policy_network()

            with open(params_path, "rb") as f:
                params = pickle.load(f)

            if not isinstance(params, dict):
                raise ValueError("Parameters file is corrupted or invalid")

            self.state_size = params.get("state_size", self.state_size)
            self.action_size = params.get("action_size", self.action_size)
            self.learning_rate = params.get("learning_rate", self.learning_rate)
            self.gamma = params.get("gamma", self.gamma)
            self.epsilon_clip = params.get("epsilon_clip", self.epsilon_clip)
            self.value_loss_coef = params.get("value_loss_coef", self.value_loss_coef)
            self.entropy_coef = params.get("entropy_coef", self.entropy_coef)
            self.epochs = params.get("epochs", self.epochs)
            self.mini_batch_size = params.get("mini_batch_size", self.mini_batch_size)

        except FileNotFoundError as e:
            print(f"Model file not found: {e}")
        except ValueError as e:
            print(f"Error in parameters: {e}")
        except Exception as e:
            print(f"Unexpected error loading model: {e}")
