from environement.tictactoe import TicTacToe
from environement.gridworld import GridWorld
from environement.farkle import Farkle
from environement.lineworld import LineWorld
from DQL.DQL import build_model, remember, replay, choose_action, update_epsilon
import random

# Choix du jeu
def select_game(game_name):
    if game_name == 'tictactoe':
        game = TicTacToe()
        state_size = 9
        action_size = 9
    elif game_name == 'gridworld':
        game = GridWorld(size=5, start=(0, 0), goal=(4, 4), obstacles=[(1, 1), (2, 2), (3, 3)])
        state_size = 25
        action_size = 4
    elif game_name == 'lineworld':
        game = LineWorld(length=5, is_random=True, start_position=2)
        state_size = 5
        action_size = 2
    elif game_name == 'farkle':
        game = Farkle()
        state_size = 2 + 1 + 1  # Deux scores (joueurs), score du tour, dés restants
        action_size = 2  # Deux actions possibles : "roll" et "bank"
    else:
        raise ValueError("Unknown game")
    return game, state_size, action_size

def random_player(game):
    """Choisit une action aléatoire parmi les actions disponibles."""
    available_actions = game.available_actions()
    return random.choice(available_actions)

def simulate_game(game, model, epsilon=0.0):
    """
    Simule une partie complète entre l'agent DQN et le jeu (ou un joueur aléatoire pour certains jeux).
    """
    state = game.reset()
    game.render()
    print()

    # Boucle principale pour jouer le jeu jusqu'à la fin
    while not game.done:
        if hasattr(game, 'current_player') and game.current_player == 0:  # L'agent DQN joue (Player 1)
            print("Agent DQN's turn.")
            available_actions = game.available_actions()
            action = choose_action(state, model, epsilon, available_actions)  # DQN choisit l'action
        else:  # Le joueur aléatoire joue, ou DQN joue seul si pas d'adversaire
            if hasattr(game, 'current_player'):
                print("Random player's turn.")
                action = random_player(game)
            else:
                # Dans le cas de LineWorld, l'agent joue seul
                available_actions = game.available_actions()
                action = choose_action(state, model, epsilon, available_actions)

        # Appliquer l'action et obtenir le nouvel état du jeu
        next_state, reward, done = game.step(action)
        state = next_state
        game.render()  # Afficher l'état du jeu après chaque coup
        print()

        if done:
            if hasattr(game, 'winner'):
                if game.winner == 1:
                    print("Agent DQN wins!")
                elif game.winner == -1:
                    print("Random player wins!")
                else:
                    print("It's a draw!")
            else:
                if reward == 1.0:
                    print("Agent DQN wins!")
                elif reward == -1.0:
                    print("Agent DQN loses!")
            break

# Entraînement DQN
def train_dqn(game, model, state_size, action_size, episodes=6):
    epsilon = 1.0
    for e in range(episodes):
        state = game.reset()
        total_reward = 0
        done = False
        while not done:
            available_actions = game.available_actions()  # Obtenir les actions disponibles
            action = choose_action(state, model, epsilon, available_actions)  # Choisir une action valide
            next_state, reward, done = game.step(action)
            remember(state, action, reward, next_state, done)  # Sauvegarder l'expérience
            state = next_state
            total_reward += reward

            if done:
                print(f"Episode {e + 1}/{episodes}, Total Reward: {total_reward}, Epsilon: {epsilon:.4f}")
                break

        replay(model, action_size)  # Entraînement avec replay
        epsilon = update_epsilon(epsilon)  # Réduction de epsilon pour moins d'exploration


if __name__ == "__main__":
    game_name = input("Enter the game you want to play (tictactoe/gridworld/farkle/lineworld): ").strip().lower()
    game, state_size, action_size = select_game(game_name)

    model = build_model(state_size, action_size)
    train_dqn(game, model, state_size, action_size)

    print("\n--- Simulation d'une partie après entraînement ---")
    simulate_game(game, model)
