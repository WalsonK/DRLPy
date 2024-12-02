import os
import pickle
import numpy as np
import time
from tqdm import tqdm
from tools import print_metrics
import copy


class RandomRollout:
    def __init__(
        self,
        state_size,
        action_size,
        num_rollouts=10,
        rollout_depth=5,
        exploration_factor=1.0,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.num_rollouts = num_rollouts
        self.rollout_depth = rollout_depth
        self.exploration_factor = exploration_factor
        self.env = None  # Store environment instance

    def set_environment(self, env):
        self.env = env

    def simulate_rollout(self, initial_state, initial_action):
        if self.env is None:
            raise ValueError("Environment not set. Call set_environment() first.")

        sim_env = copy.deepcopy(self.env)
        total_reward = 0

        if hasattr(sim_env, "available_actions") and isinstance(
            sim_env.available_actions(), dict
        ):
            available_actions = sim_env.available_actions()
            _, reward, done = sim_env.step(available_actions[initial_action])
        else:
            _, reward, done = sim_env.step(initial_action)

        total_reward += reward

        depth = 0
        while not done and depth < self.rollout_depth:
            available_actions = sim_env.available_actions()
            if isinstance(available_actions, dict):
                action = np.random.choice(list(available_actions.keys()))
                _, reward, done = sim_env.step(available_actions[action])
            else:
                action = np.random.choice(available_actions)
                _, reward, done = sim_env.step(action)

            total_reward += reward
            depth += 1

        return total_reward

    def choose_action(self, state, available_actions):
        action_rewards = {}

        for action in available_actions:
            rewards = []
            for _ in range(self.num_rollouts):
                reward = self.simulate_rollout(state, action)
                rewards.append(reward)

            mean_reward = np.mean(rewards)
            std_reward = np.std(rewards)

            # Score UCB (Upper Confidence Bound)
            action_rewards[action] = mean_reward + self.exploration_factor * std_reward

        return max(action_rewards.items(), key=lambda x: x[1])[0]

    def train(
        self,
        env,
        episodes=200,
        max_steps=500,
        test_intervals=[1000, 10_000, 100_000, 1000000],
    ):
        self.set_environment(env)
        scores_list = []
        episode_times = []
        agent_action_times = []
        action_list = []
        step_by_game = []

        with open(
            f"report/training_results_{self.__class__.__name__}_{env.__class__.__name__}_{episodes}episodes.txt",
            "a",
        ) as file:
            file.write("Training Started\n")

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
                    postfix=f"Total reward: {total_reward}, Agent Step: {step_count}, Average Action Time: 0",
                )

                if hasattr(env, "roll_dice"):
                    env.roll_dice()

                while not env.done and step_count < max_steps:
                    available_actions = env.available_actions()
                    keys = (
                        list(available_actions.keys())
                        if isinstance(available_actions, dict)
                        else available_actions
                    )

                    if hasattr(env, "current_player") and env.current_player == 1:
                        start_time_action = time.time()
                        action = self.choose_action(state, keys)
                        end_time_action = time.time()
                        step_count += 1
                    else:
                        action = np.random.choice(keys)

                    action_list.append(action)

                    if isinstance(available_actions, dict):
                        next_state, reward, done = env.step(available_actions[action])
                    else:
                        next_state, reward, done = env.step(action)

                    state = next_state
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

                    if env.done:
                        break

                end_time = time.time()
                episode_times.append(end_time - start_time)
                scores_list.append(total_reward)
                step_by_game.append(step_count)
                agent_action_times.append(end_time_action - start_time_action)

                pbar.close()

                if test_intervals is not None and (e + 1) in test_intervals:
                    win_rate, avg_reward = self.test(
                        env,
                        episodes=10,
                        max_steps=max_steps,
                        model_name=env.__class__.__name__ + "_" + str(e + 1),
                        is_saving_after_train=True
                    )
                    file.write(
                        f"Test after {e + 1} episodes: Average score: {avg_reward} | Win rate: {win_rate}\n"
                    )

            file.write("\nTraining Complete\n")
            file.write(
                f"Final Mean Score after {episodes} episodes: {np.mean(scores_list)}\n"
            )

            print_metrics(
                episodes=range(episodes),
                scores=scores_list,
                episode_times=episode_times,
                actions=action_list,
                steps_per_game=step_by_game,
                algo_name=self.__class__.__name__,
                env_name=env.__class__.__name__,
            )

        return np.mean(scores_list)

    def test(self, env, episodes=200, max_steps=10, model_name=None,  is_saving_after_train=False):
        scores_list = []
        episode_times = []
        action_times = []
        actions_list = []
        step_by_game = []
        win_game = 0
        total_reward = 0

        for e in tqdm(range(episodes), desc="Testing"):
            episode_start_time = time.time()
            episode_action_times = []
            state = env.reset()
            done = False
            step_count = 0
            episode_reward = 0

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

                if not done and step_count >= max_steps:
                    print(f"Episode {e + 1}/{episodes} reached max steps ({max_steps})")

            step_by_game.append(step_count)
            action_times.append(np.mean(episode_action_times))
            episode_times.append(episode_end_time - episode_start_time)

        avg_reward = total_reward / episodes
        print(
            f"Test Results:\n"
            f"- Games won: {win_game}/{episodes}\n"
            f"- Win rate: {(win_game / episodes) * 100:.2f}%\n"
            f"- Average reward per episode: {avg_reward:.2f}"
        )

        print_metrics(
            episodes=range(episodes),
            scores=scores_list,
            episode_times=episode_times,
            action_times=action_times,
            actions=actions_list,
            steps_per_game=step_by_game,
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

        params = {
            "state_size": self.state_size,
            "action_size": self.action_size,
            "num_rollouts": self.num_rollouts,
            "rollout_depth": self.rollout_depth,
            "exploration_factor": self.exploration_factor,
        }

        params_path = f"agents/{self.__class__.__name__}_{game_name}_params.pkl"
        with open(params_path, "wb") as f:
            pickle.dump(params, f)

        print(f"Agent {self.__class__.__name__} pour le jeu {game_name} sauvegardé.")

    def load_model(self, game_name):
        params_path = f"agents/{self.__class__.__name__}_{game_name}_params.pkl"

        if os.path.exists(params_path):
            with open(params_path, "rb") as f:
                params = pickle.load(f)

            self.state_size = params["state_size"]
            self.action_size = params["action_size"]
            self.num_rollouts = params["num_rollouts"]
            self.rollout_depth = params["rollout_depth"]
            self.exploration_factor = params["exploration_factor"]

            print(f"Agent {self.__class__.__name__} pour le jeu {game_name} chargé.")
        else:
            print(f"Aucun agent sauvegardé pour le jeu {game_name} trouvé.")
