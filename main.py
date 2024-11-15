import random

from tensorflow.python.distribute.values_util import aggregation_error_msg
from tensorflow.python.keras.models import load_model
from tqdm import tqdm

from environement.farkle import Farkle
from environement.gridworld import GridWorld
from environement.lineworld import LineWorld
from environement.tictactoe import TicTacToe
from models import DQN_with_replay , DeepQLearning , DoubleDeepQLearning, DoubleDeepQLearningWithPrioritizedExperienceReplay, DQN_with_replay


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
                    elif isinstance(model, DeepQLearning.DQL):
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

    mode = input("Do you want to play, train, or test? (play/train/test): ").strip().lower()
    manual = mode == "play"

    agent = DQN_with_replay.DQN_with_replay(
        state_size=state_size,
        action_size=action_size)

    if game_name == "gridworld":
        max_step = 10
    elif game_name == "lineworld":
        max_step = 5
    else:
        max_step = 300

    episode = 600

    if mode == "train":
        score = agent.train(game, episodes=episode, max_steps=max_step)
        print(f"Trained Mean score: {score}")
        agent.save_model(game_name)
        agent.test(game, episodes=episode, max_steps=max_step)
    elif mode == "test":
        agent.load_model(game_name)
        agent.test(game, episodes=episode, max_steps=max_step)
        simulate_game(game, model=None, manual=True)
    elif game_name in ["lineworld", "gridworld", "farkle", "tictactoe"] and manual:
        print(f"\n--- Manual Game in {game_name.title()} ---")
        simulate_game(game, model=None, manual=True)
    else:
        print("Invalid mode. Please choose 'play', 'train', or 'test'.")
