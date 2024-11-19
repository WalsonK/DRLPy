import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from environement.farkle import Farkle
import numpy as np
import random


class ReinforceActorCritic:
    def __init__(self, state_size, action_size, learning_rate, gamma):
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

    def choose_action(self, state, available_actions):
        prediction = self.theta.predict(state.reshape(1, -1), verbose=0)
        action_index = np.argmax([prediction[0][i] for i in available_actions])
        return available_actions[action_index]

    def train(self, environment, episodes, max_steps):
        for episode in range(episodes):
            state = environment.reset()
            state = np.expand_dims(state, axis=0)
            step_count = 0
            importance = 1

            baseline = self.baseline.predict(state, verbose=0)[0][0]

            if isinstance(environment, Farkle):
                environment.roll_dice()
            while not environment.done and step_count < max_steps:
                available_actions = environment.available_actions()
                keys = (
                    list(available_actions.keys())
                    if isinstance(environment, Farkle)
                    else available_actions
                )

                if hasattr(environment, "current_player") and environment.current_player == 1:
                    action = self.choose_action(state, keys)
                    step_count += 1
                else:
                    action = random.choice(keys)

                next_state, reward, done = (
                    environment.step(available_actions[action])
                    if isinstance(environment, Farkle)
                    else environment.step(action)
                )
                next_state = np.expand_dims(next_state, axis=0)

                next_baseline = self.baseline.predict(next_state, verbose=0)[0][0] if not done else 0
                delta = reward + (self.gamma * next_baseline) - baseline

                baseline_loss = self.update_baseline(state, delta)
                policy_loss = self.update_policy(state, action, delta, importance)

                importance = importance * self.gamma
                state = next_state
                baseline = next_baseline

            print(f"Episode: {episode +1}/{episodes}, Steps: {step_count}, score : {environment.get_reward()}")

    def update_baseline(self, state, delta):
        with tf.GradientTape() as tape:
            baseline_values = self.baseline(state, training=True)
            baseline_loss = tf.reduce_mean((baseline_values - (baseline_values + delta)) ** 2)

        grads = tape.gradient(baseline_loss, self.baseline.trainable_variables)

        for i, var in enumerate(self.baseline.trainable_variables):
            var.assign_sub(self.learning_rate * grads[i])

        return baseline_loss.numpy()

    def update_policy(self, state, action, delta, i):
        with tf.GradientTape() as tape:
            action_probs = self.theta(state, training=True)
            log_probs = tf.math.log(action_probs[0, action])
            loss = -i * delta * log_probs

        grads = tape.gradient(loss, self.theta.trainable_variables)

        for i, var in enumerate(self.theta.trainable_variables):
            var.assign_add(self.learning_rate * grads[i])

        return loss.numpy()


_env = Farkle(printing=False)
_model = ReinforceActorCritic(_env.state_size, _env.actions_size, 0.01, 0.001)

_model.train(environment=_env, episodes=10, max_steps=300)
