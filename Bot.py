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

    # MARK: Communication
    def reply_to_request(self, request: dict) -> dict:      # When asked something, provide data on self
        """
        :param request: {is_request: bool, data_type: str}
        """
        if request['is_request']:
            request_data_type = request['data_type']
            return self.compose_data_package(data_type=request_data_type)

    def process_reply(self, response: dict):     # After request, how to process that request
        """
        :param response: {sender: str, data_type: str, round: tuple, payload: dict}
        """
        self.log_info(f'Processing the reply from {response["sender"]}')
        self.set_package_params(**response)

    def send_request(self, request: dict, recipient: str):
        """
        :param request: {is_request: bool, data_type: str}
        :param recipient: str
        :return:
        """
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

    def receive_setup(self, data_package: dict):
        if 'is_setup' in data_package:
            self.log_info('Received setup information')
            self.atom = Atom(atom_id=self.bot_id, node_data_package=data_package['data'])

    # MARK: Utility
    def compose_data_package(self, data_type: str) -> dict:
        data_package = {
            'sender': self.name,
            'data_type': data_type,
            'round': (self.round_y, self.round_nu_bar),
            'payload': self.atom.get_y() if data_type == 'y' else self.atom.get_nu_bar()
        }
        return data_package

    def set_package_params(self, sender: str, data_type: str, round: tuple, payload: dict):
        sender_id: int = int(sender[4:])
        self.neighbor_round[sender] = round

        if data_type == 'y':
            self.atom.global_atom_y[sender_id] = payload
        elif data_type == 'nu_bar':
            self.atom.global_atom_nu_bar[sender_id] = payload

    def run_pac(self):
        if self.round_y == 0 and self.round_nu_bar == 0:
            # Initial round

            self.round_y += 1
            self.round_nu_bar += 1


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

    def run(self):
        self.setup_atoms()

        bot_a = 'Bot-1'
        bot_b = 'Bot-3'

        self.bot_dict[bot_a].send_request('Hello Bot-3!', recipient=bot_b)
        self.bot_dict[bot_a].log_info('Waiting for Alice to reply')

        time.sleep(3)
        self.ns.shutdown()
