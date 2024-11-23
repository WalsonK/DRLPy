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
        learning_rate: float = 0.01,
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
        losses_per_episode = []
        episode_times = []
        agent_action_times = []
        action_list = []

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
                    postfix=f"Total Reward: {total_reward}, Agent Step: {step_count}, Average Action Time: 0",
                    dynamic_ncols=True,
                )

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

                    state = next_state

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

                pbar.close()

                # Compute GAE and returns
                values.append(
                    self.value_network.predict(state.reshape(1, -1), verbose=0)[0][0]
                )
                advantages, returns = self._compute_gae(rewards, values, dones)

                end_time = time.time()
                episode_times.append(end_time - start_time)

                self._update_networks(states, actions, advantages, returns, log_probs)

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
            actions=action_list,
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

    def test(self, env, episodes=200, max_steps=500):
        win_games = 0
        total_reward = 0

        for episode in tqdm(range(episodes), desc="Testing"):
            if hasattr(env, "play_game"):
                winner = env.play_game(isBotGame=True, show=False, agentPlayer=self)
                if winner == 0:
                    win_games += 1
            else:
                state = env.reset()
                done = False
                episode_reward = 0
                step_count = 0

                while not done and step_count < max_steps:
                    available_actions = env.available_actions()

                    if hasattr(env, "current_player") and env.current_player == 1:
                        action = self.choose_action(state, available_actions)
                    else:
                        if isinstance(available_actions, dict):
                            action = np.random.choice(list(available_actions.keys()))
                        else:
                            action = np.random.choice(available_actions)

                    next_state, reward, done = env.step(action)
                    episode_reward += reward
                    state = next_state
                    step_count += 1

                    if done and hasattr(env, "winner"):
                        if env.winner == 1:
                            win_games += 1

                total_reward += episode_reward

        win_rate = win_games / episodes
        avg_reward = total_reward / episodes
        print(f"\nTest Results:")
        print(f"Win Rate: {win_rate:.2%}")
        print(f"Average Reward: {avg_reward:.2f}")

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
            self.policy_network = tf.keras.models.load_model(
                f"agents/{self.__class__.__name__}_{game_name}_policy.h5"
            )
            self.value_network = tf.keras.models.load_model(
                f"agents/{self.__class__.__name__}_{game_name}_value.h5"
            )
            self.old_policy_network = self._clone_policy_network()

            with open(
                f"agents/{self.__class__.__name__}_{game_name}_params.pkl", "rb"
            ) as f:
                params = pickle.load(f)
                self.__dict__.update(params)

        except Exception as e:
            print(f"Error loading model: {e}")
