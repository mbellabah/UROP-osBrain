import argparse
from bot import Main
from libs.config.helper import timeit


@timeit
def run(rounds):
    main_class = Main(num_bots=3)
    main_class.run(rounds=rounds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('round', type=int, help='number of iterations')
    args = parser.parse_args()

    rounds = args.round
    run(rounds=rounds)

