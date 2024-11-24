import random
from tqdm import tqdm

from environement.farkle import Farkle
from environement.gridworld import GridWorld
from environement.lineworld import LineWorld
from environement.tictactoe import TicTacToe
import models


# Game selection logic
def select_game():
    """Set the environment from the user input"""
    name = (
        input(
            "Enter the game you want to play (tictactoe/gridworld/farkle/lineworld): \n> "
        )
        .strip()
        .lower()
    )
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
    return env, name, s_size, a_size


def select_agent():
    """Set the agent from the user input"""
    agent_name = (
        int(
            input(
                "Enter the index of the agent:\n"
                "1 - Deep QLearning\n"
                "2 - Double Deep QLearning\n"
                "3 - Double Deep QLearning WithPrioritized Experience Replay\n"
                "4 - DQN With Replay\n"
                "5 - Reinforce\n"
                "6 - Reinforce with baseline\n"
                "7 - Reinforce with actor critic\n"
                "8 - PPO\n"
                "10- TabularQLearning\n"
                "> "
            )
        )
    )
    if agent_name == 1:
        model = models.DQL(state_size, action_size)
    elif agent_name == 2:
        model = models.DDQL(state_size, action_size)
    elif agent_name == 3:
        model = models.DDQLWithPER(state_size, action_size)
    elif agent_name == 4:
        model = models.DQN_with_replay(state_size, action_size)
    elif agent_name == 5:
        model = models.Reinforce(state_size, action_size)
    elif agent_name == 6:
        model = models.ReinforceBaseline(state_size, action_size)
    elif agent_name == 7:
        model = models.ReinforceActorCritic(state_size, action_size)
    elif agent_name == 8:
        model = models.PPO(state_size, action_size)
    elif agent_name == 10:
        model = models.TabularQLearning(state_size, action_size)
    else:
        model = models.DQL(state_size, action_size)
    return model


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


def train_agent(model, env, name, max_steps, episodes):
    """Train the agent"""
    print(f"Starting training with {model.__class__.__name__}")
    r = model.train(env, episodes=episodes, max_steps=max_steps)
    print(f"Trained Mean score: {r}")
    model.test(env, episodes=episodes, max_steps=max_steps)

    if input("Do you want to save the model? (y/n): \n> ").strip().lower() == "y":
        model.save_model(name)

    if input(f"Do you want to play against the {model.__class__.__name__}? (y/n): \n> ").strip().lower() == "y":
        simulate_game(env, model=model, manual=True)


def test_agent(model, env, name, max_steps, episodes):
    """Test the agent"""
    agent.load_model(name)
    agent.test(env, episodes=episodes, max_steps=max_steps)
    if input(f"Do you want to play against the {model.__class__.__name__}? (y/n): \n> ").strip().lower() == "y":
        simulate_game(env, model=model, manual=True)


if __name__ == "__main__":
    game, game_name, state_size, action_size = select_game()

    mode = input("Do you want to play, train, or test? (play/train/test): \n> ").strip().lower()
    manual = mode == "play"

    episode = 2
    if game_name == "gridworld":
        max_step = 10
    elif game_name == "lineworld":
        max_step = 5
    else:
        max_step = 300

    if mode == "train":
        episode = int(input("How many episodes you want to train?: \n> "))
        agent = select_agent()
        train_agent(agent, game, game_name, max_step, episode)
    elif mode == "test":
        episode = int(input("How many episodes you want to test?: \n> "))
        agent = select_agent()
        test_agent(agent, game, game_name, max_step, episode)
    elif game_name in ["lineworld", "gridworld", "farkle", "tictactoe"] and manual:
        print(f"\n--- Manual Game in {game_name.title()} ---")
        simulate_game(game, model=None, manual=True)
    else:
        print("Invalid mode. Please choose 'play', 'train', or 'test'.")
