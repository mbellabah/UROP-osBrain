import os
import numpy as np
import networkx as nx
from scipy.linalg import block_diag
from scipy.io import loadmat
from typing import List, Dict, Tuple


class Network(object):
    def __init__(self, nodes=None, edges=None, graph: nx.Graph or None = None, graph_type='normal', normalize=False):
        self.graph_type = graph_type
        self.normalize: bool = normalize

        if self.normalize:
            self.V_base = 24.9e3
            self.S_base = 1000e3
            self.V_base = 24.9e3
            self.Z_base = self.V_base ** 2 / self.S_base
            self.thermal_limit = 10e6 / self.S_base
            self.define_QL = 1e3 / self.S_base

        if graph:
            self.graph = graph
        else:
            self._nodes = nodes
            self._edges = edges
            self.graph = self.construct_graph()

    def construct_graph(self) -> nx.Graph:
        if self._nodes is not None:

            if self.normalize:
                self.normalization()

            if self.graph_type == 'digraph':
                G = nx.DiGraph()
            else:
                G = nx.Graph()

            G.add_nodes_from(self._nodes)
            G.add_edges_from(self._edges)

            return G

    def normalization(self):
        for _, data_dict in self._nodes:
            data_dict['real_load'] *= 1e3/self.S_base
            data_dict['reactive_gen'] *= 1e3/self.S_base
            data_dict['real_gen'] *= 1e3/self.S_base

        for _, _, data_dict in self._edges:
            data_dict['resistance'] /= self.Z_base
            data_dict['reactance'] /= self.Z_base

    # MARK: Utility
    def render(self):
        import matplotlib.pyplot as plt
        nx.draw_kamada_kawai(self.graph, with_labels=True, font_weight='bold')
        plt.show()

    def num_nodes(self):
        return self.graph.number_of_nodes()

    def num_edges(self):
        return self.graph.number_of_edges()

    def remove_nodes(self, nodes_to_remove: List[int]):
        self.graph.remove_nodes_from(nodes_to_remove)

    def remove_edges(self, edges_to_remove: List[Tuple[int, int]]):
        self.graph.remove_edges_from(edges_to_remove)

    # MARK: Getters
    def get_nodes_attribute(self, attribute) -> Dict:
        return nx.get_node_attributes(self.graph, attribute)

    def get_numpy_nodes_attribute(self, attribute) -> np.array:
        attribute_dict: Dict[int, float] = self.get_nodes_attribute(attribute)
        return np.array(list(attribute_dict.values())).reshape((self.num_nodes(), 1))

    def get_edges_attribute(self, attribute) -> Dict:
        return nx.get_edge_attributes(self.graph, attribute)

    def get_numpy_edges_attribute(self, attribute) -> np.array:
        attribute_dict: Dict[tuple, Dict] = self.get_edges_attribute(attribute)
        return np.array(list(attribute_dict.values())).reshape((self.num_edges(), 1))

    def get_adjacency_matrix(self, edge_attribute: str) -> np.array:
        return nx.to_numpy_matrix(self.graph, weight=edge_attribute)

    # MARK: Setters
    def set_node_attribute(self, node, attribute, value):
        self.graph.node[node][attribute] = value


class GridTopologyBase(Network):
    def __init__(self, load_mat: str = '', verbose=False):
        self._N = 1

        if load_mat:
            self.mat_data: Dict = loadmat(load_mat)
            self.init_mat(verbose=verbose)
            # self.setup()

        super(GridTopologyBase, self).__init__(self._nodes, self._edges)
        self.assign_to_nodes()

    def init_mat(self, verbose=False):
        if verbose:
            print('loaded mat', self.mat_data.keys())
        for variable_key in self.mat_data:
            setattr(self, '_'+variable_key, self.mat_data[variable_key])

        # Cleanup because some of the formats from load mat are terrible

    def setup(self):
        _nodes: Dict[Tuple[int, dict]] = {
            i: (i, {
                'bus_type': 'bus', 'real_load': 0.0, 'reactive_load': 0.0, 'real_gen': 0.0, 'reactive_gen': 0.0, 'beta_pl': 1.0, 'beta_pg': 1.0, 'beta_ql': 1.0, 'beta_qg': 1.0
            }) for i in range(1, self._N+1)
        }

        # Note that node 1 is a feeder
        _nodes[1][1]['bus_type'] = 'feeder'

        for row in self._netLoads:
            node_name, real_load, reactive_load = row[0], row[1], row[2]
            _nodes[node_name][1]['real_load'] = real_load
            _nodes[node_name][1]['reactive_load'] = reactive_load

        for row in self._netPGs:
            node_name, real_gen = tuple(row)
            _nodes[node_name][1]['real_gen'] = real_gen

        for row in self._netQGs:
            node_name, reactive_gen = tuple(row)
            _nodes[node_name][1]['reactive_gen'] = reactive_gen

        # Turn _nodes into list
        self._nodes = list(_nodes.values())

        _edges: List = []
        for row in self._netEdges:
            u, v, resistance, reactance = tuple(row)
            _edges.append(
                (u, v, {'resistance': resistance, 'reactance': reactance, 'thermal_limit': 1e7/self.S_base})
            )

        self._edges = _edges

    def assign_to_nodes(self):
        """ Assign the individual constraints to each node """

        # try:
        #     self._y0 = self._y0.tolist()
        # except AttributeError as e:
        #     print("Please provide the initial conditions y0:", repr(e))

        # Dole out to the atoms or agents
        for j in range(1, self._N+1):
            i = j - 1
            pass


class GridTopology13Node(GridTopologyBase):
    def __init__(self, riaps: bool = True, verbose: bool = False):
        self.xi: float = 1.0
        self.feeder_cost = 1.0
        self._N = 13
        endpoint = '/home/riaps/Desktop/UROP-osBrain/libs/config/data/cim_GridTopo13Node.mat'

        if riaps:
            load_mat: str = f'{os.getcwd()}/libs/{endpoint}'
        else:
            load_mat: str = endpoint

        super(GridTopology13Node, self).__init__(load_mat=load_mat, verbose=verbose)


if __name__ == '__main__':
    grid = GridTopology13Node(riaps=False, verbose=True)

