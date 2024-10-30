import random
from collections import deque

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tqdm import tqdm

from environement.farkle import Farkle
from tools import *


class DQN_with_replay:
    def __init__(
        self,
        state_size,
        action_size,
        learning_rate=0.001,
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
            loss="mse",
        )
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)

        states = np.array([transition[0] for transition in minibatch])
        next_states = np.array([transition[3] for transition in minibatch])

        q_values = self.model.predict(states, verbose=0)
        next_q_values = self.model.predict(next_states, verbose=0)

        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            target = reward
            if not done:
                target += self.gamma * np.amax(
                    next_q_values[i]
                )  # Utilise next_q_values pré-calculé
            q_values[i][action] = (
                target  # Mettre à jour l'action cible avec le nouveau calcul
            )

        # Mise à jour du modèle avec un seul appel fit sur tout le batch
        self.model.fit(states, q_values, epochs=1, verbose=0)

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def choose_action(self, state, available_actions):
        if np.random.rand() <= self.epsilon:
            return random.choice(available_actions)
        q_values = self.model.predict(state.reshape(1, -1))
        valid_q_values = np.full(len(q_values[0]), -np.inf)
        for action in available_actions:
            valid_q_values[action] = q_values[0][action]
        return np.argmax(valid_q_values)

    def train(self, env, episodes=200, max_steps=500):
        win_game = 0

        for e in range(episodes):
            state = env.reset()
            total_reward = 0
            done = False
            step_count = 0

            pbar = tqdm(total=max_steps, desc=f"Episode {e + 1}/{episodes}")

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
                    action = self.choose_action(state, keys)
                    step_count += 1
                    pbar.update(1)
                else:
                    action = self.choose_action(state, keys)

                next_state, reward, done = (
                    env.step(available_actions[action])
                    if isinstance(env, Farkle)
                    else env.step(action)
                )
                self.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

                if isinstance(env, Farkle) and any(
                    s >= env.winning_score for s in env.scores
                ):
                    if env.scores[0] > env.scores[1]:
                        win_game += 1
                if env.done:
                    print(
                        f"Episode {e + 1}/{episodes}, Total Reward: {total_reward}, Epsilon: {self.epsilon:.4f}, Steps: {step_count}"
                    )
                    break

            if not done and step_count >= max_steps:
                print(f"Episode {e + 1}/{episodes} reached max steps ({max_steps})")

            self.replay()  # Replay and update DQN model

            self.update_epsilon()
        print(f"Winrate: {win_game} / {episodes} = {win_game / episodes}")
