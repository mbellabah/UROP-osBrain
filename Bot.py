import time
import numpy as np
from typing import Dict, Tuple

from osbrain import run_agent
from osbrain import run_nameserver
from osbrain import Agent

from libs.Network import GridTopology3Node
from libs.Atom import Atom


# MARK: Channels
COORDINATOR_CHANNEL = 'coordinator'


# MARK: Classes
class Bot(Agent):
    def on_init(self):
        self.bind('PUB', alias=COORDINATOR_CHANNEL)
        self.bind('ASYNC_REP', alias=self.name, handler='reply_to_request')

        self.bot_id: int = int(self.name[4:])
        self.atom: Atom = None

        self.round_y: int = 0
        self.round_nu_bar: int = 0

        self.neighbors: list = list({1, 2, 3} - {self.bot_id})      # FIXME: list(self.atom_neighbor.keys())
        self.neighbor_round: Dict[str, Tuple[int, int]] = {}
        self.pending: int = 0

        self.is_init_y: bool = False
        self.is_init_nu_bar: bool = False

    # MARK: Communication
    def reply_to_request(self, request: dict) -> dict:      # When asked something, provide data on self
        """
        :param request: {is_request: bool, data_type: str}
        """
        if request['is_request']:
            request_data_type = request['data_type']
            return self.compose_data_package(data_type=request_data_type)

    def process_reply(self, response: dict):     # After request, how to process the subsequent reply
        """
        :param response: {sender: str, data_type: str, round: tuple, payload: dict}
        """
        self.pending -= 1
        self.set_package_params(**response)

        # If initializing the variables
        if self.pending == 0 and self.is_init_y:
            self.atom.init_dual_vars()
            self.init_pac_nu_bar()
            self.round_y += 1
            self.is_init_y = False      # so as to not run again

    def send_request(self, request: dict, recipient: str):
        """
        :param request: {is_request: bool, data_type: str}
        :param recipient: str
        :return:
        """
        self.pending += 1
        self.send(
            recipient,
            request
        )

    def request_all(self, data_type: str):
        """
        Requests data_type from all neighbors
        """
        request: dict = {'is_request': True, 'data_type': data_type}
        for j in self.neighbors:
            bot_name: str = f'Bot-{j}'
            self.send_request(request=request, recipient=bot_name)
            # all subsequent replies are handled by the process_reply handler

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
        if not self.is_init_y:
            self.request_all(data_type='y')     # self.pending = num of neighbors
            self.is_init_y = True

    def init_pac_nu_bar(self):
        if not self.is_init_nu_bar:
            self.request_all(data_type='nu_bar')
            self.is_init_nu_bar = True

    def run_pac(self):
        # Perform the updates
        if self.pending == 0:
            if self.is_synchronized():
                self.atom.update_y_and_mu()
                self.round_y += 1
                self.request_all(data_type='y')
        else:
            self.log_info('Waiting {}'.format(self.pending))
            self.idle()

        self.log_info('round {}, pending {}, and {}'.format(self.round_y, self.pending, self.atom.mu_bar))

    def periodic_pac(self, delta_t: float = 2):
        self.each(delta_t, 'run_pac')       # FIXME: May have to change the delta t here

    def is_synchronized(self):
        for round_y, round_nu_bar in self.neighbor_round.values():
            if self.round_y != round_y:  # or self.round_nu_bar != round_nu_bar:
                return False
        return True


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
                    bot_a.connect(bot_b_addr, alias=bot_name_b, handler='process_reply')

        # Setup the environment via the coordinator
        self.coordinator.init_environment()

        # init pac
        for bot in self.bot_dict.values():
            bot.init_pac_y()

    def run(self, runtime: int):
        self.setup_atoms()

        # # Periodic...
        for bot in self.bot_dict.values():
            bot.periodic_pac()
        t_end = time.time() + runtime
        while time.time() < t_end:
            pass

        # # Aperiodic...
        # for bot in self.bot_dict.values():
        #     bot.run_pac()
        # time.sleep(runtime)

        for bot in self.bot_dict.values():
            bot.log_info(bot.get_attr('atom').mu_bar)

        self.ns.shutdown()
