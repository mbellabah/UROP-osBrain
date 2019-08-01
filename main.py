import numpy as np
import argparse
from bot import Main
from libs.config.helper import timeit


@timeit
def run(rounds, grid_type):
    main_class.run(rounds=rounds, grid=grid_type)


def run_diagnostics(historical_trail='y', feasibility=False, consistency=False):
    main_class.run_diagnostics(historical_trail=historical_trail, feasibility=feasibility, consistency=consistency)


if __name__ == '__main__':
    np.set_printoptions(precision=5)

    parser = argparse.ArgumentParser()
    parser.add_argument('round', type=int, help='number of iterations')
    parser.add_argument('-grid', type=int, help='grid number')
    parser.add_argument('-f', action='store_true', dest='feasibility', help='whether to plot the feasibility')
    parser.add_argument('-c', action='store_true', dest='consistency', help='whether to plot the consistency')
    parser.add_argument('-historical_trail', type=str, help='show the trail of either y or nu')
    args = parser.parse_args()

    rounds = args.round
    grid_type: int = args.grid
    feasibility: bool = args.feasibility
    consistency: bool = args.consistency
    historical_trail: str = args.historical_trail

    main_class = Main(num_bots=grid_type)

    run(rounds=rounds, grid_type=grid_type)
    if feasibility or consistency or historical_trail:
        run_diagnostics(historical_trail=historical_trail, feasibility=feasibility, consistency=consistency)
    main_class.ns.shutdown()

