import time
import logging
import threading
import numpy as np
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt

import osbrain
from osbrain import run_agent
from osbrain import run_nameserver
from osbrain import Agent

from libs.network import GridTopology3Node
from libs.atom import Atom


# MARK: Channels
COORDINATOR_CHANNEL = 'coordinator'

logger = logging.getLogger('bot')
logging.basicConfig(level=logging.DEBUG, filename="logfile", filemode="w+", format="%(asctime)-15s %(message)s")

osbrain.config['TRANSPORT'] = 'ipc'


# MARK: Classes
class Bot(Agent):
    def on_init(self):
        self.bind('PUB', alias=COORDINATOR_CHANNEL)
        self.bind('REP', alias=self.name, handler='reply_to_request')

        self.bot_id: int = int(self.name[4:])
        self.atom: Atom = None

        self.round_y: int = 0
        self.round_nu_bar: int = 0

        self.neighbors: list = list({1, 2, 3} - {self.bot_id})      # FIXME: list(self.atom_neighbor.keys())
        self.neighbor_round: Dict[str, Tuple[int, int]] = {}

        self.is_init_y: bool = False
        self.is_init_nu_bar: bool = False

        self.feasibility: List[float] = []
        self.historical_trail = []
        self.toggle = True
        self._DEBUG = False

    # MARK: Communication
    def reply_to_request(self, request: dict) -> dict:      # When asked something, provide data on self
        """
        :param request: {is_request: bool, data_type: str, sender: str}
        """
        if request['is_request']:
            request_data_type = request['data_type']
            self.log_debug(f'Being asked for {request_data_type} by {request["sender"]}')
            reply = self.compose_data_package(data_type=request_data_type)
            logger.info('Bot-{} was asked for {} by {}\ngiving: {}'.format(self.bot_id, request['data_type'], request['sender'], reply['payload']))
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
        request: dict = {'is_request': True, 'data_type': data_type, 'sender': self.name}
        for j in self.neighbors:
            bot_name: str = f'Bot-{j}'
            self.log_debug(f'Requesting {data_type} from {bot_name}')
            self.send_request(request=request, recipient=bot_name)
            # all subsequent replies are handled by the process_reply handler
            reply = self.recv(bot_name)
            self.process_reply(reply)

    def receive_setup(self, data_package: dict):
        if 'is_setup' in data_package:
            self.log_info('Received setup information')
            self.atom = Atom(atom_id=self.bot_id, node_data_package=data_package['data'])

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

    def init_pac_nu_bar(self):
        self.request_all(data_type='nu_bar')

    def run_pac(self, data_type: str = 'y'):
        # Perform the updates
        self.log_info(f'Round: {self.round_y}, {self.round_nu_bar}')
        self.feasibility.append(self.atom._Gj @ self.atom.get_y())

        y_bool, nu_bar_bool = self.is_synchronized()

        # Check if y-round is synchronized
        if data_type == 'y':
            self.round_y += 1
            self.atom.update_y_and_mu()
            self.historical_trail.append(self.atom.get_y())
            self.request_all(data_type='y')

        else:
            self.round_nu_bar += 1
            self.atom.update_nu()
            self.request_all(data_type='nu_bar')

            # self.request_all(data_type='nu_bar')

    def periodic_pac(self, delta_t: float = 1):
        self.each(delta_t, 'run_pac', alias='periodic_pac')

    def is_synchronized(self) -> Tuple[bool, bool]:
        y_bool: bool = True
        nu_bar_bool: bool = True
        for round_y, _ in self.neighbor_round.values():
            if self.round_y != round_y:
                y_bool = False
                break
        for _, round_nu_bar in self.neighbor_round.values():
            if self.round_nu_bar != round_nu_bar:
                nu_bar_bool = False
                break
        return y_bool, nu_bar_bool


class Coordinator(Agent):
    def on_init(self):
        self.bind('PUB', alias=COORDINATOR_CHANNEL)
        self.grid = None

    def init_environment(self):
        self.grid = GridTopology3Node()
        for j in range(1, self.grid.num_nodes()+1):
            atom_data_package = self.grid.graph.node(data=True)[j]
            data_package = {'is_setup': True, 'data': atom_data_package}

            self.setup_bot(data_package=data_package, recipient=f'Bot-{j}')

    def setup_bot(self, data_package: dict, recipient: str):
        self.send(
            COORDINATOR_CHANNEL,
            message=data_package,
            topic=recipient
        )


class Main:
    def __init__(self, num_bots: int):
        self.bot_dict = {}

        # System deployment
        self.ns = run_nameserver()
        self.coordinator = run_agent('Coordinator', base=Coordinator)
        for i in range(1, num_bots+1):
            self.bot_dict[f'Bot-{i}'] = run_agent(f'Bot-{i}', base=Bot)

    def setup_atoms(self):
        # Connect the bots to the coordinator, then to each other
        for bot_name_a, bot_a in self.bot_dict.items():
            coordinator_addr = self.coordinator.addr(COORDINATOR_CHANNEL)
            bot_a.connect(coordinator_addr, handler={bot_name_a: 'receive_setup'})

            for bot_name_b, bot_b in self.bot_dict.items():
                if bot_name_a != bot_name_b:
                    bot_b_addr = bot_b.addr(alias=bot_name_b)
                    bot_a.connect(bot_b_addr, alias=bot_name_b)

        # Setup the environment via the coordinator
        self.coordinator.init_environment()

        for bot in self.bot_dict.values():
            bot.init_pac_y()
        for bot in self.bot_dict.values():
            bot.init_pac_nu_bar()

    def run(self, rounds: int = 10):
        self.setup_atoms()

        # Aperiodic...
        for _ in range(rounds):
            for bot in self.bot_dict.values():
                bot.run_pac(data_type='y')
            for bot in self.bot_dict.values():
                bot.run_pac(data_type='nu_bar')

        flag = True
        while flag:
            for bot in self.bot_dict.values():
                if bot.get_attr('round_y') == rounds and bot.get_attr('round_nu_bar') == rounds:
                    flag = False
                    break

        self.diagnostics()
        self.ns.shutdown()

    def diagnostics(self):
        # TODO: Fix bug here
        # constraints feasibility
        logging.info('-'*40)
        logging.info('-'*40)

        bot_feasibility: dict = {}
        for bot_name, bot in self.bot_dict.items():
            bot_feasibility[bot_name] = bot.get_attr('feasibility')

        feasibility: List[float] = []
        for i in range(len(bot_feasibility['Bot-1'])):
            stacked_vector_tuple = ()
            for bot_name, feasibility_vec in bot_feasibility.items():
                stacked_vector_tuple += (feasibility_vec[i],)

            stacked_vector: np.array = np.vstack(stacked_vector_tuple)
            error: float = np.linalg.norm(stacked_vector)
            feasibility.append(error)

        plt.plot(feasibility)
        plt.xlabel('round number')

        logging.info('-'*40)
        logging.info('-'*40)

        for bot in self.bot_dict.values():
            print(bot.get_attr('atom').get_y(), "\n")

        for _ in range(5):
            logging.info('-'*40)

        for bot in self.bot_dict.values():
            logging.info(f'{bot.get_attr("historical_trail")}\n')

        plt.show()


# TODO: (1) Think about the B-j (=Qmj) and how to make it "private"
# TODO: (2) Metrics sourced in a distributed fashion {'feasibility' and 'consistency'} -- fixed num of iterations
# TODO: Implement half the penalty of every edge (incoming and outgoing)
# TODO: Implement the multiperiod stuff (Underlying architecture doesn't change)
