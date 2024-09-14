from environement.lineworld import LineWorld


def play_games():
    env = LineWorld(5, False, 1)
    print("Welcome to LineWorld!")
    env.display()

    while not env.is_game_over():
        print(f"Les actions disponible sont : {env.available_actions()}")
        ac = int(input("Choisissez une action : "))
        env.step(ac)

        env.display()

    if env.is_game_over():
        score = env.score()
        if score == -1:
            print("Game Over")
        else:
            print(f"Niveau Terminé avec un Score de : {score}")


if __name__ == '__main__':
    play_games()
