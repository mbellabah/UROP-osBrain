import numpy as np
import argparse
from bot import Main
from libs.config.helper import timeit


@timeit
def run(rounds, grid_type, adaptive):
    main_class.run(rounds=rounds, grid=grid_type, adaptive=adaptive)


def run_diagnostics(historical_trail='y', feasibility=False, consistency=False):
    main_class.run_diagnostics(historical_trail=historical_trail, feasibility=feasibility, consistency=consistency)


if __name__ == '__main__':
    np.set_printoptions(precision=5)

    parser = argparse.ArgumentParser()
    parser.add_argument('round', type=int, help='number of iterations')
    parser.add_argument('-grid', default=3, type=int, help='grid number')
    parser.add_argument('-f', action='store_true', dest='feasibility', help='whether to plot the feasibility')
    parser.add_argument('-c', action='store_true', dest='consistency', help='whether to plot the consistency')
    parser.add_argument('-historical_trail', type=str, help='show the trail of either y or nu')
    parser.add_argument('-a', action='store_true', dest='adaptive', help='turn on adaptive learning')
    args = parser.parse_args()

    rounds = args.round
    grid_type: int = args.grid
    feasibility: bool = args.feasibility
    consistency: bool = args.consistency
    historical_trail: str = args.historical_trail
    adaptive: bool = args.adaptive

    main_class = Main(num_bots=grid_type)

    run(rounds=rounds, grid_type=grid_type, adaptive=adaptive)
    if feasibility or consistency or historical_trail:
        run_diagnostics(historical_trail=historical_trail, feasibility=feasibility, consistency=consistency)
    main_class.ns.shutdown()

