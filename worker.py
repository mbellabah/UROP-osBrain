from osbrain import run_agent
from osbrain import run_nameserver
from osbrain import Agent

from libs.network import GridTopology3Node
from libs.atom import Atom

from typing import Dict, Any
import numpy as np
import time

# MARK: Signals
MAIN_CHANNEL = 'main'


# MARK: Agent Classes
class Node(Agent):
    def on_init(self):
        self.bind('PUB', alias=MAIN_CHANNEL)
        self.bind('REP', alias=f'{self.name}_rep', handler='reply_to_node')
        self.bind('REQ', alias=f'{self.name}_req')

        self.global_atom_y: Dict[int, np.array] = {}
        self.global_atom_nu_bar: Dict[int, np.array] = {}
        self.atom = None
        self.neighbors = list(set([1, 2, 3]) - {self.atom_id})     # list(self.atom._neighbors.keys()) FIXME

        self.round = 0
        self.neighbor_rounds = {}

    # MARK: Initializing
    def initial_broadcast_y(self):
        data_package = {'sender': f'Node-{self.atom_id}', 'msg_type': 'y', 'payload': self.atom.get_y(), 'round': self.round}
        for j in self.neighbors:
            self.broadcast(msg=data_package, recipient=f'Node-{j}')
            node_reply = self.recv(f'Node-{j}')
            self.set_values(**node_reply)

    def initial_broadcast_nu_bar(self):
        data_package = {'sender': f'Node-{self.atom_id}', 'msg_type': 'nu_bar', 'payload': self.atom.get_nu_bar(), 'round': self.round}
        for j in self.neighbors:
            self.broadcast(msg=data_package, recipient=f'Node-{j}')
            node_reply = self.recv(f'Node-{j}')
            self.set_values(**node_reply)

    def init_dual_vars(self):
        self.atom.init_dual_vars()

    # MARK: Communications
    def broadcast(self, msg: Dict[str, Any], recipient: str) -> None:
        self.send(
            recipient,
            msg
        )

    def reply_to_node(self, data_package: dict):

        # Receive and set
        self.set_values(**data_package)

        # Reply
        data_package = self.compose_package(msg_type=data_package['msg_type'])
        return data_package

    def broadcast_all(self, msg: Dict[str, Any]) -> None:
        for node_id in self.neighbors:
            self.broadcast(msg=msg, recipient=f'Node-{node_id}')
            node_reply = self.recv(f'Node-{node_id}')
            # self.log_info(f'received {node_reply["msg_type"]} from {node_reply["sender"]} ')
            self.set_values(**node_reply)

        # Synchrony achieved
        if msg['msg_type'] == 'y':
            self.atom.update_y_and_mu()
        elif msg['msg_type'] == 'nu_bar':
            self.atom.update_nu()

        # self.log_info(self.atom.get_y())
        self.round += 1

    def receive_setup(self, data_package: dict):
        if 'is_setup' in data_package:      # receives from coordinator; {'is_setup': bool, 'data': dict}
            self.log_info('received setup info from coordinator')
            self.atom = Atom(atom_id=self.atom_id, node_data_package=data_package['data'])

    # MARK: Utility
    def set_values(self, msg_type: str, sender: str, payload: dict, round: int):
        formatted_sender: int = int(sender[5:])
        self.neighbor_rounds[sender] = round
        if msg_type == 'y':
            self.global_atom_y[sender] = payload
            self.atom.global_atom_y[formatted_sender] = payload
        elif msg_type == 'nu_bar':
            self.global_atom_nu_bar[sender] = payload
            self.atom.global_atom_nu_bar[formatted_sender] = payload

    def compose_package(self, msg_type) -> dict:
        data_package = {'sender': f'Node-{self.atom_id}', 'msg_type': msg_type, 'round': self.round}
        if msg_type == 'y':
            data_package['payload'] = self.atom.get_y()
        elif msg_type == 'nu_bar':
            data_package['payload'] = self.atom.get_nu_bar()

        return data_package

    def is_synchronized(self):
        for round in self.neighbor_rounds.values():
            if self.round != round:
                return False
        return True

    def update_pac(self):
        # Broadcast and handle the response for each
        self.log_info('HIT')
        self.broadcast_all(msg=self.compose_package(msg_type='y'))
        self.broadcast_all(msg=self.compose_package(msg_type='nu_bar'))


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
            node_agent_a.connect(self.coordinator.addr(MAIN_CHANNEL), handler={node_id: 'receive_setup'})

        for node_id_a, node_agent_a in self.node_dict.items():
            for node_id_b, node_agent_b in self.node_dict.items():
                node_agent_a.connect(node_agent_b.addr(node_id_b+'_rep'), alias=node_id_b)
                node_agent_a.connect(node_agent_b.addr(node_id_b+'_rep'), alias=node_id_b)

        # Setup the environment via the coordinator
        self.coordinator.init_environment()

        # Test 1
        # data_package = {'sender': 'Node-3', 'msg_type': 'y', 'payload': np.ones(10)}
        # self.node_dict[f'Node-3'].broadcast(msg=data_package, recipient='Node-1')
        # reply = self.node_dict['Node-3'].recv('Node-1')
        # print('reply', reply)

        # Broadcast initial
        for node in self.node_dict.values():
            node.initial_broadcast_y()
        for node in self.node_dict.values():
            node.init_dual_vars()
        for node in self.node_dict.values():
            node.initial_broadcast_nu_bar()

    def run_PAC(self, T: float):

        # for t in range(1, T+1):
        #     print(f'Round {t}/{T}')
        #     for node in self.node_dict.values():
        #         node.update_pac()

        for name, node in self.node_dict.items():
            node.each(0., 'update_pac')

        time.sleep(T)

        # for node_name, node in self.node_dict.items():
        #     print(node_name, node.get_attr('atom').get_y())

        # Terminate
        self.ns.shutdown()

