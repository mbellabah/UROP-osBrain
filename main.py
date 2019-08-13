import matplotlib.pyplot as plt

import numpy as np
import argparse
from bot import Main

from libs.config.helper import timeit
from libs.config.helper import time_list


@timeit
def run(rounds):
    main_class.run(rounds=rounds)


def run_diagnostics(historical_trail='y', feasibility=False, consistency=False):
    main_class.run_diagnostics(historical_trail=historical_trail, feasibility=feasibility, consistency=consistency)


if __name__ == '__main__':
    np.set_printoptions(precision=5)

    parser = argparse.ArgumentParser()
    parser.add_argument('-round', default=10, type=int, help='number of iterations')
    parser.add_argument('-grid', default=3, type=int, help='grid number')
    parser.add_argument('-f', action='store_true', dest='feasibility', help='whether to plot the feasibility')
    parser.add_argument('-c', action='store_true', dest='consistency', help='whether to plot the consistency')
    parser.add_argument('-historical_trail', type=str, help='show the trail of either y or nu')
    parser.add_argument('-a', action='store_true', dest='adaptive', help='turn on adaptive learning')
    parser.add_argument('-plot', action='store_true', dest='plot_rounds', help='to determine whether things are linear')
    parser.add_argument('-cim', action='store_true', default=False, dest='cim', help='use cim network, otherwise use data')
    args = parser.parse_args()

    rounds: int = args.round
    grid_type: int = args.grid
    feasibility: bool = args.feasibility
    consistency: bool = args.consistency
    historical_trail: str = args.historical_trail
    adaptive: bool = args.adaptive
    plot: bool = args.plot_rounds
    cim: bool = args.cim

    main_class = Main(num_bots=grid_type, grid=grid_type, adaptive=adaptive, cim=cim)

    if plot:
        for r in range(1, rounds, 50):
            run(rounds=r)
    else:
        run(rounds=rounds)

    if feasibility or consistency or historical_trail:
        run_diagnostics(historical_trail=historical_trail, feasibility=feasibility, consistency=consistency)
    main_class.ns.shutdown()

    if plot:
        plt.plot(time_list)
        plt.title('Rounds plotted against runtime')
        plt.show()

