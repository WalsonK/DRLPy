from env.lineworld import LineWorld


def print_games():
    env = LineWorld()
    print("Welcome to LineWorld!")
    print(f"Agent on position : {env.agent_position}")


if __name__ == '__main__':
    print_games()
