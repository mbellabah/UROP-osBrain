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
        nx.draw(self.graph, with_labels=True, font_weight='bold')
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
        self.S_base = 1000e3
        self.eta_P: float = 0.1
        self.eta_Q: float = 0.1
        self._nodes = None
        self._edges = None

        if load_mat:
            self.mat_data: Dict = loadmat(load_mat)
            self.init_mat(verbose=verbose)
            self.setup()

        super(GridTopologyBase, self).__init__(self._nodes, self._edges)

        # MARK: Broad network attributes
        self.Radj: np.array = self.get_adjacency_matrix(edge_attribute='resistance')
        self.XAdj: np.array = self.get_adjacency_matrix(edge_attribute='reactance')
        self.SAdj: np.array = self.get_adjacency_matrix(edge_attribute='thermal_limit')

        self.PLUpp = self.get_numpy_nodes_attribute(attribute='real_load')
        self.PLLow = self.PLUpp*self.eta_P
        self.PGUpp = self.get_numpy_nodes_attribute(attribute='real_gen')
        self.PGLow = self.PGUpp*0
        self.QLUpp = self.get_numpy_nodes_attribute(attribute='reactive_load')
        self.QLLow = self.QLUpp*self.eta_Q
        self.QGUpp = self.get_numpy_nodes_attribute(attribute='reactive_gen')
        self.QGLow = self.QGUpp*0

        self.VLow = np.ones((self.num_nodes(), 1))*self._Vlb
        self.VLow[0] = self._V0
        self.VUpp = np.ones((self.num_nodes(), 1))*self._Vub
        self.VUpp[0] = self._V0

        # Respective Constraints, note, self.G and self.B already due to the .mat file
        self.c: np.array = np.zeros((3 * (self._N - 1) + 2, 1))
        b_tuple: tuple = (self.PLUpp, -self.PLLow, self.PGUpp, -self.PGLow, self.QLUpp, -self.QLLow, self.QGUpp, -self.QGLow, self.VUpp, -self.VLow)
        self.b: np.array = np.vstack(b_tuple)
        self.A = None

        self.assign_to_nodes()

    def init_mat(self, verbose=False):
        if verbose:
            print('loaded mat', self.mat_data.keys())
        for variable_key in self.mat_data:
            setattr(self, '_'+variable_key, self.mat_data[variable_key])

        # Cleanup because some of the formats from load mat are terrible
        self._N = np.int(self._N)

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
        self._fcnCA = self._fcnCA[0]
        Bj_mats: List[np.array] = self._fcnCA[6][0]
        Aj_mats: List[np.array] = self._fcnCA[3][0]
        b_vecs: List[np.array] = self._fcnCA[12][0]
        Gj_mats: List[np.array] = self._fcnCA[5][0]
        Qmj_mats: List[np.array] = self._fcnCA[4][0]

        # Compute the PAC parameters
        Phi = 1.8744
        alpha = 1.0
        L = 100

        GHat = (np.asarray(Gj_mats[j-1]) for j in range(1, self._N+1))
        GHat = block_diag(*GHat)
        A = tuple(np.asarray(Aj_mats[j-1]) for j in range(1, self._N+1))
        A = np.vstack(A)
        self.A = A
        total = GHat.T@GHat + A.T@A
        eigs_PAC = np.linalg.eigvals(total).real
        sigmax_PAC = max(eigs_PAC)
        eigs_PAC[np.where(eigs_PAC < 1e-3)] = np.inf
        sigmin_PAC = min(eigs_PAC)

        gamma = (2*alpha*L)/(2*sigmax_PAC + sigmin_PAC)
        rho = 1/np.sqrt(gamma*sigmax_PAC)

        try:
            self._y0 = self._y0.tolist()
        except AttributeError as e:
            print("Please provide the initial conditions y0:", repr(e))

        # Dole out to the atoms or agents
        for j in range(1, self._N+1):
            i = j - 1
            curr_node_type: str = self.graph.node('bus_type')[j]
            n_neighbors = len(self.graph[j])
            y_num_elements: int = 0       # the number of vars/elements for the given node's y vector

            self.set_node_attribute(j, 'Aj', Aj_mats[i])
            self.set_node_attribute(j, 'bj', np.array(b_vecs[i][:, 0]).reshape((-1, 1)))
            self.set_node_attribute(j, 'Bj', Bj_mats[i])
            self.set_node_attribute(j, 'Gj', Gj_mats[i])
            self.set_node_attribute(j, 'Qmj', Qmj_mats)     # TODO: Make correct getter

            self.set_node_attribute(j, 'PL', (self.PLLow[i], self.PLUpp[i]))
            self.set_node_attribute(j, 'PG', (self.PGLow[i], self.PGUpp[i]))
            self.set_node_attribute(j, 'QL', (self.QLLow[i], self.QLUpp[i]))
            self.set_node_attribute(j, 'QG', (self.QGLow[i], self.QGUpp[i]))
            self.set_node_attribute(j, 'gamma', gamma)
            self.set_node_attribute(j, 'rho', rho)

            self.set_node_attribute(j, 'neighbors', dict(self.graph[j]))
            self.set_node_attribute(j, 'global_num_nodes', self._N)

            k = min(self.graph.node('neighbors')[j].keys())     # potential upstream node
            if not(i < j):
                # must be a feeder
                k = 'None'
            self.set_node_attribute(j, 'parent_node', k)

            # assuming only 1 feeder (root) in the whole network -- is given # 1
            # assign the correct length vector according to to the number of its neighbors
            if curr_node_type == 'feeder':      # feeder node
                # yj = [PLj, PGj, QLj, QGj, vj, {Pjk, Qjk}] --> use this
                y_num_elements: int = 5 + n_neighbors*2
            else:
                if n_neighbors == 1:    # end node
                    # yj = [Pij, Qij, lij, PLj, PGj, QLj, QGj, vj, {vi}] --> use this
                    y_num_elements: int = 9
                else:       # 'middle' node
                    # yj = [Pij, Qij, Lij, PLj, PGj, QLj, QGj, vj, {vi, Pjk, Qjk, . . .}] --> use this
                    y_num_elements: int = 9 + 2*(n_neighbors-1)

            y_vector = np.ones((y_num_elements, 1))     # set the initial conditions as ones
            # override the initial conditions assuming that y0 was provided in the mat file
            assert self._y0, "Please provide the initial conditions y0"
            y_vector = []
            for _ in range(y_num_elements):
                y_vector.append(self._y0.pop(0))
            y_vector = np.array(y_vector)

            self.set_node_attribute(j, 'y', y_vector)


class GridTopology3Node(GridTopologyBase):
    def __init__(self, riaps: bool = True, verbose: bool = False):
        self.xi: float = 1.0
        self.feeder_cost = 1.0
        if riaps:
            load_mat: str = f'{os.getcwd()}/libs/config/GridTopo3Node.mat'
        else:
            load_mat: str = 'config/GridTopo3Node.mat'

        super(GridTopology3Node, self).__init__(load_mat=load_mat, verbose=verbose)


class GridTopology10Node(GridTopologyBase):
    def __init__(self, riaps: bool = True, verbose: bool = False):
        self.xi: float = 1.0
        self.feeder_cost = 1.0
        if riaps:
            load_mat: str = f'{os.getcwd()}/libs/config/GridTopo10Node.mat'
        else:
            load_mat: str = 'config/GridTopo10Node.mat'

        super(GridTopology10Node, self).__init__(load_mat=load_mat, verbose=verbose)


class GridTopology26Node(GridTopologyBase):
    def __init__(self, riaps: bool = True, verbose: bool = False):
        self.xi: float = 1.0
        self.feeder_cost = 1.0
        if riaps:
            load_mat: str = f'{os.getcwd()}/libs/config/GridTopo26Node.mat'
        else:
            load_mat: str = 'config/GridTopo26Node.mat'

        super(GridTopology26Node, self).__init__(load_mat=load_mat, verbose=verbose)


if __name__ == '__main__':
    # grid = GridTopology3Node(riaps=False, verbose=True)
    grid = GridTopology10Node(riaps=False, verbose=True)
    # print(grid.graph.nodes(data=False))

