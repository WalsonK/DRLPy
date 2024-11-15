import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential, clone_model, load_model
from tensorflow.python.keras.utils.vis_utils import is_wrapped_model
from tqdm import tqdm
import os
import pickle


class DDQL:
    def __init__(
            self,
            state_size,
            action_size,
            learning_rate=0.01,
            gamma=0.95,
            epsilon=1.0,
            epsilon_min=0.01,
            epsilon_decay=0.995,
            target_update_frequency=100
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.target_update_frequency = target_update_frequency
        self.training_step = 0

        # Création des deux réseaux
        self.main_model = self.build_model()
        self.target_model = clone_model(self.main_model)
        self.target_model.set_weights(self.main_model.get_weights())

    def build_model(self):
        model = Sequential([
            Dense(64, input_dim=self.state_size, activation="relu"),
            Dense(64, activation="relu"),
            Dense(self.action_size, activation="linear")
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="mse"
        )
        return model

    def update_target_model(self):
        """Mise à jour du réseau cible avec les poids du réseau principal"""
        self.target_model.set_weights(self.main_model.get_weights())

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def choose_action(self, state, available_actions):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(available_actions)

        q_values = self.main_model.predict(state.reshape(1, -1), verbose=0)
        valid_q_values = np.full(len(q_values[0]), -np.inf)
        for action in available_actions:
            valid_q_values[action] = q_values[0][action]
        return np.argmax(valid_q_values)

    def learn(self, state, action, reward, next_state, done):
        self.training_step += 1

        # Calcul de la Q-valeur cible en utilisant les deux réseaux
        if not done:
            # Le réseau principal sélectionne l'action
            next_q_values_main = self.main_model.predict(next_state.reshape(1, -1), verbose=0)
            best_action = np.argmax(next_q_values_main[0])

            # Le réseau cible évalue cette action
            next_q_values_target = self.target_model.predict(next_state.reshape(1, -1), verbose=0)
            target = reward + self.gamma * next_q_values_target[0][best_action]
        else:
            target = reward

        # Mise à jour du réseau principal
        current_q_values = self.main_model.predict(state.reshape(1, -1), verbose=0)
        current_q_values[0][action] = target
        self.main_model.fit(state.reshape(1, -1), current_q_values, epochs=1, verbose=0)

        # Mise à jour périodique du réseau cible
        if self.training_step % self.target_update_frequency == 0:
            self.update_target_model()

    def train(self, env, episodes=200, max_steps=500):
        scores_list = []
        best_score = float('-inf')

        for e in range(episodes):
            state = env.reset()
            total_reward = 0
            step_count = 0

            pbar = tqdm(total=max_steps, desc=f"Episode {e + 1}/{episodes}")

            if hasattr(env, 'roll_dice'):
                env.roll_dice()

            while not env.done and step_count < max_steps:
                available_actions = env.available_actions()
                keys = (list(available_actions.keys())
                        if hasattr(env, 'available_actions') and isinstance(env.available_actions(), dict)
                        else available_actions)

                if hasattr(env, "current_player") and env.current_player == 1:
                    action = self.choose_action(state, keys)
                    step_count += 1
                else:
                    action = np.random.choice(keys)

                pbar.update(1)

                # Gestion des différents types d'environnements
                if hasattr(env, 'available_actions') and isinstance(env.available_actions(), dict):
                    next_state, reward, done = env.step(available_actions[action])
                else:
                    next_state, reward, done = env.step(action)

                # Apprentissage avec DDQN
                self.learn(state, action, reward, next_state, done)

                state = next_state
                total_reward += reward

                if env.done:
                    scores_list.append(total_reward)
                    if total_reward > best_score:
                        best_score = total_reward

                    print(
                        f"Episode {e + 1}/{episodes}, "
                        f"Total Reward: {total_reward}, "
                        f"Best Score: {best_score}, "
                        f"Epsilon: {self.epsilon:.4f}, "
                        f"Steps: {step_count}"
                    )
                    break

            if not done and step_count >= max_steps:
                print(f"Episode {e + 1}/{episodes} reached max steps ({max_steps})")

            self.update_epsilon()
            pbar.close()

        return np.mean(scores_list)

    def test(self, env, episodes=200, max_steps=10):
        win_game = 0
        total_reward = 0

        for e in tqdm(range(episodes), desc="Testing"):
            state = env.reset()
            episode_reward = 0
            step_count = 0

            if hasattr(env, 'play_game'):
                winner = env.play_game(isBotGame=True, show=False, agentPlayer=self)
                if winner == 0:
                    win_game += 1
            else:
                while not env.done and step_count < max_steps:
                    available_actions = env.available_actions()

                    if hasattr(env, "current_player") and env.current_player == 1:
                        action = self.choose_action(state, available_actions)
                        step_count += 1
                    else:
                        action = np.random.choice(available_actions)

                    next_state, reward, done = env.step(action)
                    episode_reward += reward
                    state = next_state

                    if env.done and hasattr(env, 'winner') and env.winner == 1.0:
                        win_game += 1
                        break

                total_reward += episode_reward

        avg_reward = total_reward / episodes
        print(
            f"Test Results:\n"
            f"- Games won: {win_game}/{episodes}\n"
            f"- Win rate: {(win_game / episodes) * 100:.2f}%\n"
            f"- Average reward per episode: {avg_reward:.2f}"
        )

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
            "target_update_frequency": self.target_update_frequency
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