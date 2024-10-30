import random
from collections import deque

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

learning_rate = 0.001
gamma = 0.95
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
batch_size = 64
memory = deque(maxlen=2000)

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    print("GPUs detected:", gpus)
else:
    print("No GPUs detected.")


def build_model(state_size, action_size):
    model = Sequential()
    model.add(Dense(64, input_dim=state_size, activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(action_size, activation="linear"))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mse"
    )
    return model


def remember(state, action, reward, next_state, done):
    memory.append((state, action, reward, next_state, done))


def replay(model, action_size):
    if len(memory) < batch_size:
        return
    minibatch = random.sample(memory, batch_size)
    for state, action, reward, next_state, done in minibatch:
        target = reward
        if not done:
            target += gamma * np.amax(model.predict(next_state.reshape(1, -1), verbose=0))
        target_f = model.predict(state.reshape(1, -1), verbose=0)
        target_f[0][action] = target
        model.fit(state.reshape(1, -1), target_f, epochs=1, verbose=0)


def choose_action(state, model, epsilon, available_actions):
    if np.random.rand() <= epsilon:
        return random.choice(available_actions)
    q_values = model.predict(state.reshape(1, -1), verbose=0)
    valid_q_values = np.full(
        len(q_values[0]), -np.inf
    )  # Initialiser toutes les valeurs à -inf
    for action in available_actions:
        valid_q_values[action] = q_values[0][action]

    return np.argmax(valid_q_values)


def update_epsilon(epsilon):
    return max(epsilon_min, epsilon * epsilon_decay)
