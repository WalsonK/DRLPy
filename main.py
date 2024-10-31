import random

from tqdm import tqdm

from environement.farkle import Farkle
from environement.gridworld import GridWorld
from environement.lineworld import LineWorld
from environement.tictactoe import TicTacToe
from models import DQN_with_replay


# Game selection logic
def select_game(name):
    """Set the environment from the user input"""
    if name == "tictactoe":
        env = TicTacToe()
        s_size = 9
        a_size = 9
    elif name == "gridworld":
        env = GridWorld(size=5)
        s_size = 25
        a_size = 4
    elif name == "lineworld":
        env = LineWorld(length=5, is_random=True, start_position=2)
        s_size = 5
        a_size = 2  # Move left or right
    elif name == "farkle":
        env = Farkle(printing=False)
        s_size = env.state_size
        a_size = env.actions_size
    else:
        raise ValueError("Unknown game")
    return env, s_size, a_size


def random_player(env):
    """Chooses a random action from available actions."""
    available_actions = env.available_actions()
    return random.choice(available_actions)


def simulate_game(game, model=None, manual=False):
    replay_game = True
    while replay_game:
        state = game.reset()

        if not isinstance(game, Farkle):
            game.render()
            print()

            while not game.done:
                if hasattr(game, "current_player") and game.current_player == 1:
                    if isinstance(model, DQN_with_replay.DQN_with_replay) and not manual:
                        # Model vs Random (model is agent, user not playing)
                        print("Agent model's turn.")
                        available_actions = game.available_actions()
                        action = model.choose_action(state, available_actions)
                    elif model is None:
                        # User vs Random (opponent is random)
                        print("Random opponent's turn.")
                        available_actions = game.available_actions()
                        action = random.choice(available_actions)
                    else:
                        # User vs Model (agent's turn)
                        print("Agent model's turn.")
                        available_actions = game.available_actions()
                        action = model.choose_action(state, available_actions)
                else:
                    if manual:
                        # User's manual turn
                        print("Your turn (Player).")
                        available_actions = game.available_actions()
                        action = manual_player(available_actions)
                    else:
                        # Random opponent's turn in Model vs Random
                        print("Random opponent's turn.")
                        available_actions = game.available_actions()
                        action = random.choice(available_actions)

                next_state, reward, done = game.step(action)
                state = next_state
                game.render()
                print()

                if done:
                    if hasattr(game, "winner"):
                        if game.winner == 1:
                            print(
                                "Agent model wins!"
                                if model
                                else "Random opponent wins!"
                            )
                        elif game.winner == -1:
                            print("You win!" if manual else "Random opponent wins!")
                        else:
                            print("It's a draw!")
                    else:
                        if reward == 1.0:
                            print(
                                "Agent model wins!"
                                if model
                                else "Random opponent wins!"
                            )
                        elif reward == -1.0:
                            print("You win!" if manual else "Agent model loses!")
                    break
        else:
            game.play_game(show=True, agentOpponent=model)

        # Replay prompt
        replay_choice = input("Do you want to play again? (y/n): ").strip().lower()
        replay_game = replay_choice == "y"


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


if __name__ == "__main__":
    game_name = (
        input(
            "Enter the game you want to play (tictactoe/gridworld/farkle/lineworld): "
        )
        .strip()
        .lower()
    )
    game, state_size, action_size = select_game(game_name)

    mode = input("Do you want to play or train? (play/train): ").strip().lower()
    manual = True if mode == "play" else False

    agent = DQN_with_replay.DQN_with_replay(state_size, action_size)

    if game_name in ["lineworld", "gridworld", "farkle", "tictactoe"] and manual:
        print(f"\n--- Manual Game in {game_name.title()} ---")
        simulate_game(game, model=None, manual=True)
    else:
        if mode == "train":
            # Train
            score = agent.train(game, episodes=5)
            print(f"Trained Mean score: {score}")
            # Test
            agent.test(game, episodes=10)
            print("\n--- Simulating a game after training ---")
            simulate_game(game, model=agent, manual=manual)
        else:
            print("Play mode is not supported for automated DQN training.")
