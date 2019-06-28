import time
import numpy as np
from typing import Dict, Any

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
        self.atom = None

    def reply_to_request(self, msg: dict):
        self.log_info(f'reply to a request: {msg}')

    def process_reply(self, msg: dict):
        self.log_info('Processing the reply')

    def broadcast(self, msg: dict, recipient: str):
        self.send(
            recipient,
            msg
        )

    def receive_setup(self, data_package: dict):
        if 'is_setup' in data_package:
            self.log_info('Received setup information')
            self.atom = Atom(atom_id=self.bot_id, node_data_package=data_package['data'])


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
        self.ns.shutdown()
