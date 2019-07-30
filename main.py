import argparse
from bot import Main
from libs.config.helper import timeit


@timeit
def run(rounds):
    main_class.run(rounds=rounds)


def run_diagnostics(historical_trail='y', feasibility=False, consistency=False):
    main_class.run_diagnostics(historical_trail=historical_trail, feasibility=feasibility, consistency=consistency)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('round', type=int, help='number of iterations')
    parser.add_argument('feasibility', type=bool, default=False, help='whether to plot the feasibility')
    parser.add_argument('consistency', type=bool, default=False, help='whether to plot the consistency')
    parser.add_argument('historical_trail', type=str, default='y', help='show the trail of either y or nu')
    args = parser.parse_args()

    rounds = args.round
    feasibility: bool = args.feasibility
    consistency: bool = args.consistency
    historical_trail: str = args.historical_trail

    main_class = Main(num_bots=3)

    run(rounds=rounds)
    if feasibility or consistency or historical_trail:
        run_diagnostics(historical_trail=historical_trail, feasibility=feasibility, consistency=consistency)
    main_class.ns.shutdown()

