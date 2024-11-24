import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from environement.farkle import Farkle
import random
import time
import numpy as np
from tqdm import tqdm, trange
from tools import print_metrics
import os
import pickle


class ReinforceBaseline:
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

    def train(self, environment, episodes, max_steps):
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
                # generate episode
                start_time = time.time()
                states, actions, rewards, agent_action_times = self.generate_episode(
                    environment, max_steps
                )
                end_time = time.time()
                episode_times.append(end_time - start_time)
                action_times.append(np.mean(agent_action_times))
                steps_per_game.append(len(states))

                # Metrics
                t = 0
                pbar = tqdm(
                    total=len(states),
                    desc=f"Episode {episode + 1}/ {episodes}",
                    unit="Step",
                    bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
                    postfix=f"total reward: 0, agent Step: {t}, Average Action Time: 0",
                    dynamic_ncols=True,
                )

                # Calc cumulative reward
                G = self.calculate_reward(rewards)

                episode_policy_loss = 0
                episode_baseline_loss = 0
                for t in range(len(states)):
                    state = np.expand_dims(states[t], axis=0)
                    action = actions[t]
                    G_t = G[t]
                    actions_list.append(action)

                    baseline = self.baseline.predict(state, verbose=0)[0][0]
                    advantage = G_t - baseline

                    policy_loss = self.update_policy(state, action, advantage, t)
                    baseline_loss = self.update_baseline(state, G_t, t)

                    episode_policy_loss += policy_loss
                    episode_baseline_loss += baseline_loss

                    pbar.update(1)
                    pbar.set_postfix(
                        {
                            "Total Reward": G_t,
                            "Agent Step": t,
                            "Policy loss": policy_loss,
                            "Baseline loss": baseline_loss,
                            "Average Action Time": np.mean(agent_action_times)
                            if len(agent_action_times) > 0
                            else 0,
                        }
                    )

                pbar.close()
                scores_list.append(G[-1])
                policy_losses_per_episode.append(episode_policy_loss)
                baseline_losses_per_episode.append(episode_baseline_loss)

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

    def test(self, environment, episodes, max_steps):
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

    def generate_episode(self, environment, max_step):
        states, actions, rewards, agent_action_times = [], [], [], []
        step_count = 0
        state = environment.reset()

        if isinstance(environment, Farkle):
            environment.roll_dice()
        while not environment.done and step_count < max_step:
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
                start_time = time.time()
                action = self.choose_action(state, keys)
                end_time = time.time()
                agent_action_times.append(end_time - start_time)
            else:
                action = random.choice(keys)

            next_state, reward, done = (
                environment.step(available_actions[action])
                if isinstance(environment, Farkle)
                else environment.step(action)
            )

            if (
                hasattr(environment, "current_player")
                and environment.current_player == 1
            ):
                states.append(state)
                actions.append(action)
                step_count += 1

            state = next_state
            rewards.append(reward)

        if not environment.done and step_count >= max_step:
            print(f"reached max steps ({max_step})")

        return states, actions, rewards, agent_action_times

    def choose_action(self, state, available_actions):
        prediction = self.theta.predict(state.reshape(1, -1), verbose=0)
        action_index = np.argmax([prediction[0][i] for i in available_actions])
        return available_actions[action_index]

    def calculate_reward(self, rewards):
        gt = 0
        returns = []
        for reward in reversed(rewards):
            gt = self.gamma * gt + reward
            returns.insert(0, gt)
        return returns

    def update_policy(self, state, action, advantage, t):
        with tf.GradientTape() as tape:
            action_probs = self.theta(state, training=True)
            log_prob = tf.math.log(action_probs[0, action])
            loss = -log_prob * advantage

        # Update policy with gradient
        grads = tape.gradient(loss, self.theta.trainable_variables)

        # Mise à jour manuelle de chaque variable de theta en appliquant le taux d'apprentissage
        for i, var in enumerate(self.theta.trainable_variables):
            var.assign_add(self.learning_rate * (self.gamma**t) * advantage * grads[i])

        return loss.numpy()

    def update_baseline(self, state, G_t, t):
        with tf.GradientTape() as tape:
            baseline_values = self.baseline(state, training=True)
            baseline_loss = tf.reduce_mean((G_t - baseline_values) ** 2)

        grads = tape.gradient(baseline_loss, self.baseline.trainable_variables)

        for i, var in enumerate(self.baseline.trainable_variables):
            var.assign_sub(
                self.learning_rate
                * (self.gamma**t)
                * (G_t - baseline_values).numpy()[0, 0]
                * grads[i]
            )

        return baseline_loss.numpy()

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
# _model = ReinforceBaseline(_env.state_size, _env.actions_size, learning_rate=0.01, gamma=0.9)

# _model.train(environment=_env, episodes=10, max_steps=300)
# _model.test(environment=_env, episodes=4, max_steps=200)
# _model.save_model("reinforce_baseline_farkle_10")

# model_test = ReinforceBaseline(_env.state_size, _env.actions_size, 1, 1)
# model_test.load_model("reinforce_baseline_farkle_10")
# model_test.test(environment=_env, episodes=4, max_steps=200)
