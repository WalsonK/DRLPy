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
                # If it's player 1's turn
                if hasattr(game, "current_player") and game.current_player == 1:
                    if manual:
                        # **User's Manual Turn** (User vs Random)
                        print("Your turn (Player).")
                        available_actions = game.available_actions()
                        action = manual_player(available_actions)
                    elif isinstance(model, DQN_with_replay.DQN_with_replay):
                        # **Model's Turn** (Model vs Random)
                        print("Agent model's turn.")
                        available_actions = game.available_actions()
                        action = model.choose_action(state, available_actions)
                    else:
                        # **Random Opponent's Turn**
                        print("Random opponent's turn.")
                        available_actions = game.available_actions()
                        action = random.choice(available_actions)
                else:
                    print("Opponent's turn.")
                    available_actions = game.available_actions()
                    action = random.choice(available_actions)

                next_state, reward, done = game.step(action)
                state = next_state

                game.render()
                print()

                if done:
                    if hasattr(game, "winner"):
                        if game.winner == 1:
                            print("You win!" if manual else "Agent model wins!")
                        elif game.winner != 1:
                            print("You lose!" if manual else "Agent model loses!")

                    break
        else:
            game.play_game(show=True, agentOpponent=model)

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

    agent = DQN_with_replay.DQN_with_replay(state_size, action_size, learning_rate=0.01)

    if game_name in ["lineworld", "gridworld", "farkle", "tictactoe"] and manual:
        print(f"\n--- Manual Game in {game_name.title()} ---")
        simulate_game(game, model=None, manual=True)
    else:
        if mode == "train":
            if game_name in ["lineworld", "gridworld"]:
                max_step = 10
            else:
                max_step = 300
            # Train
            score = agent.train(game, episodes=150, max_steps=max_step)
            print(f"Trained Mean score: {score}")
            # Test
            agent.test(game, episodes=10, max_steps=max_step)
            print("\n--- Simulating a game after training ---")
            simulate_game(game, model=agent, manual=manual)
        else:
            print("Play mode is not supported for automated DQN training.")
