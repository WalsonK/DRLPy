from environement.farkle import Farkle
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tqdm import tqdm, trange
import time


class Reinforce:
    def __init__(self, state_size, action_size, learning_rate):
        self.learning_rate = learning_rate
        self.theta = self.build_model(state_size, action_size)
        self.gamma = 0.9

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

    def train(self, num_episodes, environment, max_step):
        for episode in range(num_episodes):
            # generate episode
            states, actions, rewards, agent_action_times = self.generate_episode(environment, max_step)
            print(states[-1])

            # metrics
            t = 0
            pbar = tqdm(
                total=len(states), desc=f"Episode {episode + 1}/ {num_episodes}", unit="Step",
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
                postfix=f"total reward: 0, agent Step: {t}, Average Action Time: 0",
                dynamic_ncols=True
            )

            # Calc cumulative reward
            G = self.calculate_reward(rewards)

            # Update policy
            for t in range(len(states)):
                state = np.expand_dims(states[t], axis=0)
                action = actions[t]
                G_t = G[t]

                # Calc gradient
                with tf.GradientTape() as tape:
                    action_probs = self.theta(state, training=True)
                    log_prob = tf.math.log(action_probs[0, action])
                    loss = -log_prob * G_t  # On maximise donc on minimise -G_t * log π(A_t|S_t)

                # Update policy with gradient
                grads = tape.gradient(loss, self.theta.trainable_variables)

                # Mise à jour manuelle de chaque variable de theta en appliquant le taux d'apprentissage
                for i, var in enumerate(self.theta.trainable_variables):
                    var.assign_add(self.learning_rate * (self.gamma ** t) * G_t * grads[i])

                pbar.update(1)
                pbar.set_postfix({
                    "Total Reward": G_t,
                    "Agent Step": t,
                    "Average Action Time": np.mean(agent_action_times) if len(agent_action_times) > 0 else 0 # np.mean(agent_action_times) if len(agent_action_times) > 0 else 0,
                })
            pbar.close()

    def test(self, environment, episodes, max_step):
        win_games = 0
        for e in trange(episodes, desc="Testing", unit="episode"):
            state = environment.reset()
            done = False
            step_count = 0

            if isinstance(environment, Farkle):
                winner = environment.play_game(isBotGame=True, show=False, agentPlayer=self)
                if winner == 0:
                    win_games += 1

            else:
                while not environment.done and step_count < max_step:
                    available_actions = environment.available_actions()

                    if hasattr(environment, "current_player") and environment.current_player == 1:
                        action = self.choose_action(state, available_actions)
                        step_count += 1
                    else:
                        action = random.choice(available_actions)

                    next_state, reward, done = environment.step(action)
                    state = next_state

                    if environment.done and environment.winner == 1.0:
                        win_games += 1
                        break

                if not done and step_count >= max_step:
                    print(f"Episode {e + 1}/{episodes} reached max steps ({max_step})")
        print(
            f"Winrate:\n- {win_games} game wined\n- {episodes} game played\n- Accuracy : {(win_games / episodes) * 100:.2f}%"
        )

    def generate_episode(self, environment, max_step):
        states, actions, rewards, agent_action_times = [], [], [], []
        step_count = 0
        environment.reset()

        if isinstance(environment, Farkle):
            environment.roll_dice()
        state = environment.get_state()
        while not environment.done and step_count < max_step:
            available_actions = environment.available_actions()
            keys = (
                list(available_actions.keys())
                if isinstance(environment, Farkle)
                else available_actions
            )

            if hasattr(environment, "current_player") and environment.current_player == 1:
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

            if hasattr(environment, "current_player") and environment.current_player == 1:
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                step_count += 1

            state = next_state

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


_env = Farkle(printing=False)
_model = Reinforce(_env.state_size, _env.actions_size, learning_rate=0.1)

_model.train(num_episodes=2, environment=_env, max_step=300)
_model.test(environment=_env, episodes=10, max_step=200)
