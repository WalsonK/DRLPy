import numpy as np
import os
import pickle
import time
import random
from tqdm import tqdm

from environement.farkle import Farkle
from tools import print_metrics

from environement.lineworld import LineWorld

class TabularQLearning:
    def __init__(
        self,
        state_size,
        action_size,
        learning_rate=0.01,
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

        # Initialize Q-table with zeros
        self.q_table = np.zeros((state_size, action_size))

        # Map state vectors to indices
        self.state_to_index = {}

    def get_state_index2(self,state):
        if str(state) not in self.state_to_index:
            new_index = int("".join(map(str, state)),2)
            self.state_to_index[str(state)] = new_index
            return self.state_to_index[str(state)]

    def get_state_index(self, state):
        """
        Converts a state vector into a unique index for the Q-table.
        """
        state_tuple = tuple(state)  # Convert the state vector into a tuple (hashable)
        if state_tuple not in self.state_to_index:
            # Assign a new index if this state hasn't been seen before
            new_index = len(self.state_to_index)
            self.state_to_index[state_tuple] = new_index

            if new_index >= self.q_table.shape[0]:
                self.q_table = np.vstack([self.q_table, np.zeros((1, self.action_size))])

        return self.state_to_index[state_tuple]

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def choose_action(self, state, available_actions):
        state_index = self.get_state_index(state)
        if np.random.rand() <= self.epsilon:
            return random.choice(available_actions)
        # Filter Q-values for available actions only
        valid_q_values = np.full(self.action_size, -np.inf)
        for action in available_actions:
            valid_q_values[action] = self.q_table[state_index, action]
        return np.argmax(valid_q_values)

    def learn(self, state, action, reward, next_state, done):
        state_index = self.get_state_index(state)
        next_state_index = self.get_state_index(next_state)

        current_q_value = self.q_table[state_index, action]
        max_next_q_value = np.max(self.q_table[next_state_index]) if not done else 0
        target = reward + self.gamma * max_next_q_value

        # Update Q-value using the Q-learning formula
        self.q_table[state_index, action] += self.learning_rate * (target - current_q_value)

    def train(self, env, episodes=200, max_steps=500, test_intervals=[1000, 10_000, 100_000, 1000000]):
        scores_list = []
        episode_times = []
        actions_list = []
        agent_action_times = []

        with open(f"report/TabularQLearning_{env.__class__.__name__}_{episodes}episodes.txt", "a") as file:
            file.write("Training Started\n")
            file.write(f"Training with {episodes} episodes and max steps {max_steps}\n")

            pbar = tqdm(
                range(episodes),
                total=episodes,
                unit="episodes",
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
                postfix=f"total reward: 0, agent Step: 0, Epsilon: {self.epsilon:.4f}, Average Action Time: 0",
            )
            for e in range(episodes):
                start_time = time.time()
                state = env.reset()
                total_reward = 0
                step_count = 0

                if isinstance(env, Farkle):
                    env.roll_dice()

                while not env.done and step_count < max_steps:
                    available_actions = env.available_actions()
                    keys = (
                        list(available_actions.keys())
                        if isinstance(env, Farkle)
                        else available_actions
                    )

                    action_time_start = time.time()
                    action = self.choose_action(state, keys)
                    action_time_end = time.time()

                    agent_action_times.append(action_time_end - action_time_start)

                    actions_list.append(action)

                    next_state, reward, done = (
                        env.step(available_actions[action])
                        if isinstance(env, Farkle)
                        else env.step(action)
                    )

                    self.learn(state, action, reward, next_state, done)

                    state = next_state
                    total_reward += reward
                    step_count += 1

                    if env.done:
                        break

                end_time = time.time()
                episode_times.append(end_time - start_time)
                scores_list.append(total_reward)

                self.update_epsilon()

                pbar.update(1)
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

                if test_intervals is not None and (e + 1) in test_intervals:
                    win_rate, avg_reward = self.test(
                        env,
                        episodes=10,
                        max_steps=max_steps,
                        model_name=env.__class__.__name__ + "_" + str(e + 1),
                        is_saving_after_train=True
                    )
                    file.write(f"Test after {e + 1} episodes: Average score: {avg_reward}, Win rate: {win_rate}\n")

            pbar.close()
            file.write("\nTraining Complete\n")
            file.write(f"Final Mean Score after {episodes} episodes: {np.mean(scores_list)}\n")
            file.write(f"Total training time: {np.sum(episode_times)} seconds\n")

            print_metrics(
                episodes=range(episodes),
                scores=scores_list,
                episode_times=episode_times ,
                actions=actions_list,
                algo_name=self.__class__.__name__,
                env_name=env.__class__.__name__
        )

        return np.mean(scores_list)

    def test(self, env, episodes=10, max_steps=10, model_name=None, is_saving_after_train=False):
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
            done = False
            step_count = 0
            episode_reward = 0

            if hasattr(env, "play_game"):  # Pour Farkle
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

                if not done and step_count >= max_steps:
                    print(f"Episode {e + 1}/{episodes} reached max steps ({max_steps})")

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
            action_times=action_times,
            is_training=False,
            algo_name=self.__class__.__name__,
            env_name=env.__class__.__name__,
            metric_for=str(model_name.split("_")[-1].split(".")[0]) + " episodes trained" if is_saving_after_train
            else ""
        )
        win_rate = win_game / episodes

        if is_saving_after_train:

            model_name = env.__class__.__name__ + "_" + str(episodes) if model_name is None else model_name
            self.save_model(model_name)
        return win_rate, avg_reward

    def save_model(self, game_name):
        os.makedirs("agents", exist_ok=True)
        with open(f"agents/{self.__class__.__name__}_{game_name}.pkl", "wb") as f:
            params = {
                "state_size" : self.state_size,
                "action_size" : self.action_size,
                "learning_rate" : self.learning_rate,
                "gamma" : self.gamma,
                "epsilon" : self.epsilon,
                "epsilon_min" : self.epsilon_min,
                "epsilon_decay" : self.epsilon_decay,
                "q_table" : self.q_table,
                "state_to_index" : self.state_to_index,
            }
            pickle.dump(params, f)
        print(f"Q-table saved as '{self.__class__.__name__}_{game_name}'.")

    def load_model(self, game_name):
        model_path = f"agents/{self.__class__.__name__}_{game_name}.pkl"
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                params = pickle.load(f)
                self.state_size = params["state_size"]
                self.action_size = params["action_size"]
                self.learning_rate = params["learning_rate"]
                self.gamma = params["gamma"]
                self.epsilon = params["epsilon"]
                self.epsilon_min = params["epsilon_min"]
                self.epsilon_decay = params["epsilon_decay"]
                self.q_table = params["q_table"]
                self.state_to_index = params["state_to_index"]

            print(f"Q-table loaded from '{game_name}'.")
        else:
            print(f"No saved model found with the name '{game_name}'.")


#enve = LineWorld(5)

#model = TabularQLearning(5, 2)

#model.train(env=enve, episodes=1500, max_steps=300)
#model.test(env=enve, episodes=100, max_steps=200)
#name = "farkle_test_save_load"
# _model.save_model(name)
# model_test = Reinforce(4, 4, learning_rate=10)
# model_test.load_model(name)
# model_test.test(environment=_env, episodes=10, max_steps=200)