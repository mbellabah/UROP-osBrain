from osbrain import run_agent
from osbrain import run_nameserver
from osbrain import Agent

from libs.Network import GridTopology3Node
from libs.Atom import Atom

from typing import Dict, Any
import numpy as np
import time

# MARK: Signals
MAIN_CHANNEL = 'main'


# MARK: Agent Classes
class Node(Agent):
    def on_init(self):
        self.bind('PUB', alias=MAIN_CHANNEL)

        self.global_atom_y: Dict[int, np.array] = {}
        self.global_atom_nu_bar: Dict[int, np.array] = {}
        self.atom = None
        self.neighbors = list(set([1, 2, 3]) - {self.atom_id})     # list(self.atom._neighbors.keys()) FIXME: Technically, the latter

        self.round_y = 1
        self.round_nu_bar = 1
        self.neighbors_round = {neighbor: (0, 0) for neighbor in self.neighbors}        # (round_y, round_nu_bar)

    def initial_broadcast_y(self):
        # assumes self.atom exists
        data_package = {'round': (self.round_y, self.round_nu_bar), 'sender': self.atom_id, 'msg_type': 'y', 'payload': self.atom.get_y()}
        for j in self.neighbors:
            self.broadcast(msg=data_package, recipient=f'Node-{j}')

    def initial_broadcast_nu_bar(self):
        data_package = {'round': (self.round_y, self.round_nu_bar), 'sender': self.atom_id, 'msg_type': 'nu_bar', 'payload': self.atom.get_nu_bar()}
        for j in self.neighbors:
            self.broadcast(msg=data_package, recipient=f'Node-{j}')

    def init_dual_vars(self):
        self.atom.init_dual_vars()

    # MARK: Communications
    def broadcast(self, msg: Dict[str, Any], recipient) -> None:
        self.send(
            MAIN_CHANNEL,
            message=msg,
            topic=recipient
        )

    def broadcast_all(self, msg: Dict[str, Any]) -> None:
        for node_id in self.neighbors:
            self.broadcast(msg=msg, recipient=f'Node-{node_id}')

    def receive(self, data_package: dict):
        if 'is_setup' in data_package:      # receives from coordinator; {'is_setup': bool, 'data': dict}
            self.atom = Atom(atom_id=self.atom_id, node_data_package=data_package['data'])
        else:
            sender_round_y, sender_round_nu_bar = data_package['round']
            sender: int = data_package['sender']
            msg_type = data_package['msg_type']
            payload: np.array = data_package['payload']

            self.neighbors_round[sender] = (sender_round_y, sender_round_nu_bar)
            self.log_info(f'Received: ({sender_round_y}, {sender_round_nu_bar}) from {sender}')
            self.set_values(msg_type, sender, payload)

    def set_values(self, msg_type: str, sender: int, payload: dict):
        if msg_type == 'y':
            self.global_atom_y[sender] = payload
            self.atom.global_atom_y[sender] = payload
        elif msg_type == 'nu_bar':
            self.global_atom_nu_bar[sender] = payload
            self.atom.global_atom_nu_bar[sender] = payload

    def compose_package(self, msg_type) -> dict:
        data_package = {'round': (self.round_y, self.round_nu_bar), 'sender': self.atom_id, 'msg_type': msg_type}
        if msg_type == 'y':
            data_package['payload'] = self.atom.get_y()
        elif msg_type == 'nu_bar':
            data_package['payload'] = self.atom.get_nu_bar()

        return data_package

    def run_pac(self):
        if not self.is_synchronized(msg_type='y'):       # wait
            self.log_info(f'waiting on y : {self.neighbors_round}')
            pass
        else:
            self.atom.update_y_and_mu()
            self.round_y += 1

            self.broadcast_all(msg=self.compose_package('y'))

        if not self.is_synchronized(msg_type='nu_bar'):
            self.log_info(f'waiting on nu_bar : {self.neighbors_round}')
            pass
        else:
            self.atom.update_nu()
            self.round_nu_bar += 1

            self.broadcast_all(msg=self.compose_package('nu_bar'))

    def is_synchronized(self, msg_type: str) -> bool:
        if msg_type == 'y':
            for other_agent_round_y, _ in self.neighbors_round.values():
                if self.round_y != other_agent_round_y:
                    return False
            return True
        elif msg_type == 'nu_bar':
            for _, other_agent_round_nu_bar in self.neighbors_round.values():
                if self.round_nu_bar != other_agent_round_nu_bar:
                    return False
            return True


class Coordinator(Agent):
    def on_init(self):
        self.bind('PUB', alias=MAIN_CHANNEL)

        self.grid = None

    # MARK: Utility
    def init_environment(self):
        self.grid = GridTopology3Node()
        for j in range(1, self.grid.num_nodes()+1):
            atom_data_package = self.grid.graph.node(data=True)[j]
            data_package = {'is_setup': True, 'data': atom_data_package}

            self.setup_node(data_package=data_package, recipient=f'Node-{j}')

    # MARK: Communication
    def setup_node(self, data_package: dict, recipient: str):
        self.send(
            MAIN_CHANNEL,
            message=data_package,
            topic=recipient
        )


class Main:
    def __init__(self, n_nodes: int) -> None:
        self.node_dict = {}

        # System deployment
        self.ns = run_nameserver()

        for i in range(1, n_nodes+1):
            self.node_dict[f'Node-{i}'] = run_agent(f'Node-{i}', base=Node, attributes=dict(atom_id=i))
        self.coordinator = run_agent('Coordinator', base=Coordinator)

    def setup_atoms(self) -> None:
        # Connect all atoms together, and the coordinator
        for node_id, node_agent_a in self.node_dict.items():
            node_agent_a.connect(self.coordinator.addr(MAIN_CHANNEL), handler={node_id: 'receive'})
            for node_agent_b in self.node_dict.values():
                node_agent_a.connect(node_agent_b.addr(MAIN_CHANNEL), handler={node_id: 'receive'})

        # Setup the environment via the coordinator
        self.coordinator.init_environment()

        # Broadcast initial
        for node in self.node_dict.values():
            node.initial_broadcast_y()
        for node in self.node_dict.values():
            node.init_dual_vars()
        for node in self.node_dict.values():
            node.initial_broadcast_nu_bar()


    def run_PAC(self, T):

        # # for t in range(1, T+1):
        # #     print(f'Round: {t}/{T}')
        # for node in self.node_dict.values():
        #     node.periodic()
        #
        # time.sleep(T)
        #
        # for node in self.node_dict.values():
        #     print(node.get_attr('atom')._y)

        for node in self.node_dict.values():
            node.run_pac()

        time.sleep(10)

        # Terminate
        self.ns.shutdown()

