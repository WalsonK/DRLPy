import random

from DQL.DQL import build_model, choose_action, remember, replay, update_epsilon
from environement.gridworld import GridWorld
from environement.lineworld import LineWorld
from environement.tictactoe import TicTacToe
from environement.farkle import Farkle


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


def simulate_game(game, model, epsilon=0.0, manual=False):
    replay_game = True
    while replay_game:
        state = game.reset()

        if not isinstance(game, Farkle):
            game.render()
            print()
            while not game.done:
                # Manually playing LineWorld/GridWorld
                if manual and isinstance(game, (LineWorld, GridWorld)):
                    print("Your turn (Player).")
                    available_actions = game.available_actions()
                    action = manual_player(available_actions)
                else:
                    if (
                        hasattr(game, "current_player") and game.current_player == 1
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
        else:
            game.play_game(show=True, agent=model)

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


def train_dqn(env, model, state_size, action_size, episodes=5, opponent="random", max_steps=500):
    win_game = 0
    epsilon = 0.8
    model_opponent = None
    if opponent == "model":
        model_opponent = build_model(state_size, action_size)

    for e in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        step_count = 0

        if isinstance(env, Farkle):
            env.roll_dice()
        while not env.done and step_count < max_steps:
            available_actions = env.available_actions()
            keys = []
            if isinstance(env, Farkle):
                keys = list(available_actions.keys())

            if hasattr(env, "current_player") and env.current_player == 1:
                action = (choose_action(state, model, epsilon, keys)
                          if isinstance(env, Farkle)
                          else choose_action(state, model, epsilon, available_actions))
                step_count += 1
            else:
                if opponent == "random":
                    action = (random.choice(keys) if isinstance(env, Farkle)
                              else random_player(env))
                else:
                    action = (choose_action(state, model_opponent, epsilon, keys) if isinstance(env, Farkle)
                              else choose_action(state, model_opponent, epsilon, available_actions))

            next_state, reward, done = (env.step(available_actions[action]) if isinstance(env, Farkle)
                                        else env.step(action))
            remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if isinstance(env, Farkle):
                if any(s >= env.winning_score for s in env.scores):
                    if env.scores[0] > env.scores[1]:
                        win_game += 1
            if env.done:
                print(f"Episode {e + 1}/{episodes}, Total Reward: {total_reward}, Epsilon: {epsilon:.4f}, Steps: {step_count}")
                break

        if not done and step_count >= max_steps:
            print(f"Episode {e + 1}/{episodes} reached max steps ({max_steps})")

        replay(model, action_size)  # Replay and update DQN model
        if opponent == "model":
            replay(model_opponent, action_size)

        epsilon = update_epsilon(epsilon)
    print(f"Winrate: {win_game} / {episodes} = {win_game / episodes}")


if __name__ == "__main__":
    game_name = input("Enter the game you want to play (tictactoe/gridworld/farkle/lineworld): ").strip().lower()
    game, states_size, actions_size = select_game(game_name)

    if game_name == "tictactoe":
        opponent_type = input("Do you want to train model vs random or model vs model? (random/model): ").strip().lower()
        assert opponent_type in ["random", "model"], "Invalid opponent type selected."

    mode = input("Do you want to play or train? (play/train): ").strip().lower()
    manual = True if mode == "play" else False

    if game_name in ["lineworld", "gridworld", "farkle"] and manual:
        print(f"\n--- Manual Game in {game_name.title()} ---")
        simulate_game(game, model=None, manual=True)
    else:
        agent = build_model(states_size, actions_size)

        if game_name == "tictactoe":
            train_dqn(game, agent, states_size, actions_size, opponent=opponent_type)
        else:
            train_dqn(game, agent, states_size, actions_size)

        print("\n--- Simulating a game after training ---")
        simulate_game(game, agent, manual=manual)
