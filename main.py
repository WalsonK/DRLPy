from environement.tictactoe import TicTacToe
from environement.gridworld import GridWorld
from environement.lineworld import LineWorld
from DQL.DQL import build_model, remember, replay, choose_action, update_epsilon
import random


# Game selection logic
def select_game(game_name):
    if game_name == "tictactoe":
        game = TicTacToe()
        state_size = 9
        action_size = 9
    elif game_name == "gridworld":
        game = GridWorld(size=5)
        state_size = 25  # 5x5 grid size
        action_size = 4  # 4 possible actions: up, down, left, right
    elif game_name == "lineworld":
        game = LineWorld(length=5, is_random=True, start_position=2)
        state_size = 5
        action_size = 2  # Move left or right
    else:
        raise ValueError("Unknown game")
    return game, state_size, action_size


def random_player(game):
    """Chooses a random action from available actions."""
    available_actions = game.available_actions()
    return random.choice(available_actions)


def simulate_game(game, model, epsilon=0.0, manual=False):
    replay_game = True
    while replay_game:
        state = game.reset()
        game.render()
        print()

        while not game.done:
            if manual and isinstance(
                game, (LineWorld, GridWorld)
            ):  # Manually playing LineWorld/GridWorld
                print("Your turn (Player).")
                available_actions = game.available_actions()
                action = manual_player(available_actions)
            else:
                if (
                    hasattr(game, "current_player") and game.current_player == 0
                ):  # DQN agent plays (Player 1)
                    print("Agent DQN's turn.")
                    available_actions = game.available_actions()
                    action = choose_action(state, model, epsilon, available_actions)
                else:
                    if hasattr(game, "current_player"):
                        if (
                            manual and game.current_player == -1
                        ):  # User plays (Player 2)
                            print("Your turn (Player).")
                            available_actions = game.available_actions()
                            action = manual_player(available_actions)
                        else:
                            print("Random player's turn.")
                            action = random_player(game)
                    else:
                        available_actions = game.available_actions()
                        action = choose_action(state, model, epsilon, available_actions)

            next_state, reward, done = game.step(action)
            state = next_state
            game.render()
            print()

            if done:
                if hasattr(game, "winner"):
                    if game.winner == 1:
                        print("Agent DQN wins!")
                    elif game.winner == -1:
                        print("You win!" if manual else "Random player wins!")
                    else:
                        print("It's a draw!")
                else:
                    if reward == 1.0:
                        print("Agent DQN wins!")
                    elif reward == -1.0:
                        print("Agent DQN loses!")
                break

        replay_choice = input("Do you want to play again? (y/n): ").strip().lower()
        replay_game = True if replay_choice == "y" else False


def manual_player(available_actions):
    """
    Allows the user to choose an action from the available ones.
    """
    while True:
        try:
            print(f"Available actions: {available_actions}")
            action = int(input(f"Choose your action from {available_actions}: "))
            if action in available_actions:
                return action
            else:
                print("Invalid action. Please choose a valid action.")
        except ValueError:
            print("Invalid input. Please enter a number.")


def train_dqn(game, model, state_size, action_size, episodes=200, opponent="random"):
    epsilon = 0.8
    model_opponent = None
    if opponent == "model":
        model_opponent = build_model(state_size, action_size)

    for e in range(episodes):
        state = game.reset()
        total_reward = 0
        done = False
        while not done:
            available_actions = game.available_actions()

            if (
                hasattr(game, "current_player") and game.current_player == 0
            ):  # DQN model plays
                action = choose_action(state, model, epsilon, available_actions)
            else:
                if opponent == "random":
                    action = random_player(game)
                else:
                    action = choose_action(
                        state, model_opponent, epsilon, available_actions
                    )

            next_state, reward, done = game.step(action)
            remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if done:
                print(
                    f"Episode {e + 1}/{episodes}, Total Reward: {total_reward}, Epsilon: {epsilon:.4f}"
                )
                break

        replay(model, action_size)  # Replay and update DQN model
        if opponent == "model":
            replay(model_opponent, action_size)

        epsilon = update_epsilon(epsilon)


if __name__ == "__main__":
    game_name = (
        input(
            "Enter the game you want to play (tictactoe/gridworld/farkle/lineworld): "
        )
        .strip()
        .lower()
    )
    game, state_size, action_size = select_game(game_name)

    if game_name == "tictactoe":
        opponent_type = (
            input(
                "Do you want to train model vs random or model vs model? (random/model): "
            )
            .strip()
            .lower()
        )
        assert opponent_type in ["random", "model"], "Invalid opponent type selected."

    mode = input("Do you want to play manually? (y/n): ").strip().lower()
    manual = True if mode == "y" else False

    if game_name in ["lineworld", "gridworld"] and manual:
        print(f"\n--- Manual Game in {game_name.title()} ---")
        simulate_game(game, model=None, manual=True)
    else:
        model = build_model(state_size, action_size)

        if game_name == "tictactoe":
            train_dqn(game, model, state_size, action_size, opponent=opponent_type)
        else:
            train_dqn(game, model, state_size, action_size)

        print("\n--- Simulating a game after training ---")
        simulate_game(game, model, manual=manual)
