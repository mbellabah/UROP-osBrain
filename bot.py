import time
import sys
import numpy as np
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt

import osbrain
from osbrain import run_agent
from osbrain import run_nameserver
from osbrain import Agent

from libs.network import GridTopology3Node, GridTopology10Node, GridTopology26Node
from libs.atom import Atom

from libs.config.helper import col_print, print_final


# MARK: Channels
COORDINATOR_CHANNEL = 'coordinator'

osbrain.config['TRANSPORT'] = 'ipc'


# MARK: Classes
class Bot(Agent):
    def on_init(self):
        self.bind('SUB', alias=COORDINATOR_CHANNEL, handler='receive_setup')
        self.bind('REP', alias=self.name, handler='reply_to_request')

        self.bot_id: int = int(self.name[4:])
        self.atom: Atom = None

        self.round_y: int = 0
        self.round_nu_bar: int = 0

        self.neighbors: List = [None]
        self.neighbor_round: Dict[str, Tuple[int, int]] = {}

        self.is_init_y: bool = False
        self.is_init_nu_bar: bool = False

        self.feasibility: List = []

        self.historical_trail_y = []
        self.historical_trail_nu = []

        self._DEBUG = False

    # MARK: Communication
    def reply_to_request(self, request: dict) -> dict:      # When asked something, provide data on self
        """
        :param request: {is_request: bool, data_type: str, sender: str, round: tuple}
        """
        if request['is_request']:
            request_data_type = request['data_type']
            reply = self.compose_data_package(data_type=request_data_type)
            return reply

    def process_reply(self, response: dict):     # After request, how to process the subsequent reply
        """
        :param response: {sender: str, data_type: str, round: tuple, payload: dict}
        """
        self.set_package_params(**response)

    def send_request(self, request: dict, recipient: str):
        """
        :param request: {is_request: bool, data_type: str, sender: str}
        :param recipient: str
        :return:
        """
        self.send(
            address=recipient,
            message=request
        )

    def request_all(self, data_type: str):
        """
        Requests data_type from all neighbors
        """
        request: dict = {'is_request': True, 'data_type': data_type, 'sender': self.name, 'round': (self.round_y, self.round_nu_bar)}
        for j in self.neighbors:
            bot_name: str = f'Bot-{j}'
            self.send_request(request=request, recipient=bot_name)
            # all subsequent replies are handled by the process_reply handler
            reply = self.recv(bot_name)
            self.process_reply(reply)

    def receive_setup(self, data_package: dict):
        if 'is_setup' in data_package and data_package['recipient'] == self.name:       # FIXME: Really weird bug where bot-1 receives bot-10's info as well
            self.log_info(f'Received setup information, recipient: {data_package["recipient"]}')
            self.atom = Atom(atom_id=self.bot_id, node_data_package=data_package['data'])
            self.atom.adaptive_learning = data_package['adaptive']

            self.neighbors = list(set([*range(1, self.atom._global_num_nodes+1)]) - {self.bot_id})   # FIXME: list(self.atom_neighbor.keys())
            self.historical_trail_y.append(self.atom.get_y())

    # MARK: Utility
    def compose_data_package(self, data_type: str) -> dict:
        data_package = {
            'sender': self.name,
            'data_type': data_type,
            'bot_round': (self.round_y, self.round_nu_bar),
            'payload': self.atom.get_y() if data_type == 'y' else self.atom.get_nu_bar()
        }
        return data_package

    def set_package_params(self, sender: str, data_type: str, bot_round: tuple, payload: dict):
        sender_id: int = int(sender[4:])
        self.neighbor_round[sender] = bot_round

        if data_type == 'y':
            self.atom.global_atom_y[sender_id] = payload
        elif data_type == 'nu_bar':
            self.atom.global_atom_nu_bar[sender_id] = payload

    def init_pac_y(self):
        self.request_all(data_type='y')
        self.atom.init_dual_vars()
        self.historical_trail_nu.append(self.atom.nu)

    def init_pac_nu_bar(self):
        self.request_all(data_type='nu_bar')

    def run_pac_y(self):
        # Perform the updates
        self.feasibility.append(self.atom._Gj @ self.atom.get_y())

        self.request_all(data_type='nu_bar')
        self.atom.update_y_and_mu()
        self.historical_trail_y.append(self.atom.get_y())

    def run_pac_nu_bar(self):
        self.request_all(data_type='y')
        self.atom.update_nu()
        self.historical_trail_nu.append(self.atom.nu)

        self.atom.round += 1


class Coordinator(Agent):
    def on_init(self):
        self.bind('PUB', alias=COORDINATOR_CHANNEL)
        self.grid = None
        self.adaptive = False

    def init_environment(self):
        # self.grid = GridTopology10Node  # GridTopology3Node()     # FIXME
        for j in range(1, self.grid._N+1):
            atom_data_package = self.grid.graph.node(data=True)[j]
            data_package = {'is_setup': True, 'adaptive': self.adaptive, 'data': atom_data_package, 'recipient': f'Bot-{j}'}
            self.setup_bot(data_package=data_package, recipient=f'Bot-{j}')

    def setup_bot(self, data_package: dict, recipient: str):
        self.send(
            COORDINATOR_CHANNEL,
            message=data_package,
            topic=recipient
        )


class Main:
    def __init__(self, num_bots: int, grid: int, adaptive: bool):
        self.bot_dict = {}
        self.rounds = 0

        # System deployment
        self.ns = run_nameserver()
        self.coordinator = run_agent('Coordinator', base=Coordinator)
        for i in range(1, num_bots+1):
            self.bot_dict[f'Bot-{i}'] = run_agent(f'Bot-{i}', base=Bot)

        self.setup_atoms(grid, adaptive)

    def setup_atoms(self, grid: int = 3, adaptive: bool = False):
        # Connect the bots to the coordinator, then to each other
        if grid == 3:
            self.coordinator.set_attr(**{'grid': GridTopology3Node()})
        elif grid == 10:
            self.coordinator.set_attr(**{'grid': GridTopology10Node()})
        elif grid == 26:
            self.coordinator.set_attr(**{'grid': GridTopology26Node()})
        else:
            raise Exception("Must define a network topology!")

        for bot_name_a, bot_a in self.bot_dict.items():
            coordinator_addr = self.coordinator.addr(COORDINATOR_CHANNEL)
            bot_a.connect(coordinator_addr, handler={bot_name_a: 'receive_setup'})

            for bot_name_b, bot_b in self.bot_dict.items():
                if bot_name_a != bot_name_b:
                    bot_b_addr = bot_b.addr(alias=bot_name_b)
                    bot_a.connect(bot_b_addr, alias=bot_name_b)

        # Setup the environment via the coordinator
        self.coordinator.set_attr(**{'adaptive': adaptive})
        self.coordinator.init_environment()

        cached_methods = [bot.init_pac_y for bot in self.bot_dict.values()]
        cached_methods += [bot.init_pac_nu_bar for bot in self.bot_dict.values()]

        for cached in cached_methods:
            cached()

    def run(self, rounds: int = 10):
        self.rounds = rounds

        # Virtually synchronous execution -- all messages sent in a round, are received within the same round
        #       note that the req-rep patterns also enforces message-passing be done in lockstep
        cached_methods = [bot.run_pac_y for bot in self.bot_dict.values()]
        cached_methods += [bot.run_pac_nu_bar for bot in self.bot_dict.values()]

        for i in range(self.rounds):
            sys.stdout.write("Round progress: %d/%i   \r" % (i, self.rounds))
            sys.stdout.flush()

            # (1) Slightly faster than (2)
            for cached in cached_methods:
                cached()

            # (2)
            # for bot in self.bot_dict.values():
            #     bot.run_pac_y()
            #     bot.set_attr(**{'round_y': bot.get_attr('round_y') + 1})
            #
            # for bot in self.bot_dict.values():
            #     bot.run_pac_nu_bar()
            #     bot.set_attr(**{'round_nu_bar': bot.get_attr('round_nu_bar') + 1})

    def run_diagnostics(self, historical_trail='na', feasibility=False, consistency=False):
        # Print the final values to screen (stdout)
        # Some pretty printing stuff
        print_final(self.rounds)

        for bot in self.bot_dict.values():
            bot_name = f'Bot-{bot.get_attr("bot_id")}'
            final_vector = bot.get_attr('atom').get_y()
            col_print(bot_name, final_vector)
        print('#'*31)

        self.diagnostics(historical_trail=historical_trail, feasibility=feasibility, consistency=consistency)

    def diagnostics(self, historical_trail='na', feasibility=False, consistency=False):
        de_granular: int = 1    # 3 if self.rounds > 100 else 1      # will plot every granular <int>

        if feasibility:
            # constraints feasibility
            bot_feasibility: dict = {}
            for bot_name, bot in self.bot_dict.items():
                bot_feasibility[bot_name] = bot.get_attr('feasibility')

            x = []
            feasibility_error: List[float] = []
            for i in range(self.rounds):
                if i % de_granular == 0:
                    stacked_vector_tuple = ()
                    for bot_name, feasibility_vec in bot_feasibility.items():
                        stacked_vector_tuple += (feasibility_vec[i],)

                    stacked_vector: np.array = np.vstack(stacked_vector_tuple)
                    error: float = np.linalg.norm(stacked_vector)
                    feasibility_error.append(error)

                    x.append(i)

            plt.subplot(1, 2, 1)
            plt.xlabel('round number')
            plt.title("Distance to Feasibility")
            plt.ylim([0, 0.125])
            plt.plot(x, feasibility_error)

        if consistency:
            # consistency
            x = []
            consistency_error: List[float] = []
            for i in range(self.rounds):
                if i % de_granular == 0:
                    stacked_y_vector_tuple = ()
                    for bot_name, bot in self.bot_dict.items():
                        stacked_y_vector_tuple += (bot.get_attr('historical_trail_y')[i],)

                    stacked_y_vector: np.array = np.vstack(stacked_y_vector_tuple)
                    A = self.coordinator.get_attr('grid').A
                    error: float = np.linalg.norm(A@stacked_y_vector)
                    consistency_error.append(error)

                    x.append(i)

            plt.subplot(1, 2, 2)
            plt.xlabel('round number')
            plt.title("Distance to Consistency")
            plt.ylim([0, 0.125])
            plt.plot(consistency_error)

        if historical_trail in ['y', 'nu']:
            # Print the trail
            for k in range(self.rounds+1):
                if k % de_granular == 0:
                    bot_y_k = ()
                    bot_nu_k = ()

                    for bot in self.bot_dict.values():
                        bot_y_k += (bot.get_attr("historical_trail_y")[k],)
                        bot_nu_k += (bot.get_attr("historical_trail_nu")[k],)

                    bot_y_k = np.vstack(bot_y_k)
                    bot_nu_k = np.vstack(bot_nu_k)

                    print('Round', k)

                    if historical_trail == 'y':
                        print(bot_y_k, "\n")
                    elif historical_trail == 'nu':
                        print(bot_nu_k, "\n")

                    print('-'*50)

        plt.show()
        

# TODO: (1) Think about the B-j (=Qmj) and how to make it "private"
# TODO: (2) Metrics sourced in a distributed fashion {'feasibility' and 'consistency'} -- fixed num of iterations
# TODO: Implement half the penalty of every edge (incoming and outgoing)
# TODO: Implement the multiperiod stuff (Underlying architecture doesn't change)
