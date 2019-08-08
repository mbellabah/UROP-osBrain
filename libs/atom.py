from typing import Dict, Tuple
import numpy as np
from libs.solver import atomic_solve


class Atom(object):
    def __init__(self, atom_id: int, node_data_package: dict):
        self.atom_id = atom_id
        self.node_data_package = node_data_package
        self.first_time: bool = True
        self.previous_problem = None

        # set the attributes from the node data package
        for node_variable in self.node_data_package:
            setattr(self, '_'+node_variable, self.node_data_package[node_variable])

        self.mu = None
        self.mu_bar = None
        self.nu = None
        self.nu_bar = None

        self.adaptive_learning: bool = False
        self._gamma_mu = self._gamma
        self._gamma_nu = self._gamma
        self.round: int = 0         # round/iteration
        self.epsilon = 1e-6

        self.Gy_trajectory: list = []
        self.Ay_trajectory: list = []
        self._gamma_mu_trajectory = []

        # Later implement so don't have to broadcast to everyone
        self.global_atom_y: Dict[int, np.array] = {self.atom_id: self.get_y()}
        self.global_atom_nu_bar: Dict[int, np.array] = {}

        qmj_tuple = ()
        for m in range(self._global_num_nodes):
            qmj_tuple += (self._Qmj[m][0][self.atom_id - 1],)  # because indexing of atom starts at 1
        self._Qmj = np.vstack(qmj_tuple)

        # broadcast and receive to initialize things on the network - order
        '''
        (1) self.broadcast(msg_type='broadcast_y', msg=self._y)
        (2) self.receive(msg_type='receive_y')
        (3) self.init_dual_vars()
        (4) self.broadcast(msg_type='broadcast_nu_bar', msg=self.nu_bar)
        (5) self.receive(msg_type='receive_nu_bar')
        
        LOOP
        (6) Update y and mu, mu_bar 
        (7) broadcast y
        (8) receive y 
        (9) update nu, nu_bar
        (10) broadcast nu_bar
        '''

    # MARK: Getters
    def get_global_y(self):
        y_tuple: Tuple[np.array] = tuple([self.global_atom_y[key] for key in sorted(self.global_atom_y)])
        return np.vstack(y_tuple)

    def get_y(self) -> np.array:
        return self._y

    def get_nu_bar(self) -> np.array:
        return self.nu_bar

    def get_nu(self) -> np.array:
        return self.nu

    def get_global_nu_bar(self):
        nu_bar_tuple: Tuple[np.array] = tuple([self.global_atom_nu_bar[key] for key in sorted(self.global_atom_nu_bar)])
        return np.vstack(nu_bar_tuple)

    def init_dual_vars(self):
        self.mu: np.array = np.zeros_like(self._rho * self._gamma * self._Gj @ self.get_y())
        self.mu_bar: np.array = self.mu + self._rho * self._gamma * self._Gj @ self.get_y()
        global_y_mat: np.array = self._Aj @ self.get_global_y()

        self.nu: np.array = np.zeros_like(self._rho * self._gamma * global_y_mat)
        self.nu_bar: np.array = self.nu + self._rho * self._gamma * global_y_mat

        # set nu_bar
        self.global_atom_nu_bar[self.atom_id] = self.nu_bar

    # MARK: PAC
    def cost_function(self, var):
        xi = 1.0        # FIXME: Integrate with Network Grid topo

        gen_cost = 0.0
        load_util = 0.0
        loss = 0.0

        if self._bus_type == 'feeder':
            # yj = [PLj, PGj, QLj, QGj, vj, {Pjh, Qjh}]
            beta_pg = 1
            beta_qg = 1

            PG: float = var[1]
            QG: float = var[3]

            gen_cost: float = beta_pg*PG + beta_qg*QG

        else:
            # yj = [Pij, Qij, lij, PLj, PGj, QLj, QGj, vj, {vi}] (end node) OR
            # yj = [Pij, Qij, Lij, PLj, PGj, QLj, QGj, vj, {vi, Pjk, Qjk, . . .}] ('middle' node)

            PG: float = var[4]
            QG: float = var[6]
            PL: float = var[3]
            QL: float = var[5]
            Lij: float = var[2]       # the current flow on the upstream line

            gen_cost: float = self._beta_pg*(PG - self._PG[0])**2 + self._beta_qg*(QG - self._QG[0])**2
            load_util: float = self._beta_pl*(PL - self._PL[1])**2 + self._beta_ql*(QL - self._QL[1])**2

            parent_node: int = int(self._parent_node)
            upstream_line_resistance: float = self._neighbors[parent_node]['resistance']
            loss: float = xi*upstream_line_resistance*Lij

        return gen_cost + load_util + loss

    def solve_atomic_objective_function(self) -> np.array:
        parent_node: int = int(self._parent_node)
        upstream_line_thermal_limit: float = self._neighbors[parent_node]['thermal_limit']

        params = {'global_nu_bar': (self.get_global_nu_bar().shape, self.get_global_nu_bar()), 'mu_bar': (self.mu_bar.shape, self.mu_bar), 'prev_y': (self.get_y().shape, self.get_y())}
        if self.first_time:
            var, self.previous_problem = atomic_solve(self.cost_function, self._y.shape, Gj=self._Gj, rho=self._rho, Qmj=self._Qmj, Bj=self._Bj, bj=self._bj, bus_type=self._bus_type, thermal_limit=upstream_line_thermal_limit, prev_params=params)
        else:
            var, _ = atomic_solve(self.cost_function, self._y.shape, Gj=self._Gj, rho=self._rho, Qmj=self._Qmj, Bj=self._Bj, bj=self._bj, bus_type=self._bus_type, thermal_limit=upstream_line_thermal_limit, previous_problem=self.previous_problem, prev_params=params)

        return var

    def update_y_and_mu(self):
        try:
            self._y: np.array = self.solve_atomic_objective_function()
            self.first_time = False         # we've successfuly done our first time!
        except ValueError as e:
            print('Could not solve for y')
            raise e

        # update mu
        mat_product: np.array = self._Gj @ self.get_y()
        self.mu = self.mu + self._rho * self._gamma * mat_product

        PRODUCT = self._gamma * mat_product
        self.Gy_trajectory.append(mat_product)

        if self.adaptive_learning:
            H_round = sum([g@g.T for g in self.Gy_trajectory])
            n = H_round.shape[0]
            diagonalized_H_round = np.diag(np.diag(H_round))
            epsilon_identity = self.epsilon*np.identity(n)
            total = np.diag(1/np.sqrt(np.diag(epsilon_identity + diagonalized_H_round)))

            print(self.atom_id, "HEY", self._gamma * total)

            PRODUCT = self._gamma * total @ mat_product

        self.mu_bar = self.mu + self._rho * PRODUCT

        # update my y that exists in the global dict
        self.global_atom_y[self.atom_id] = self._y

    def update_nu(self):
        # update nu
        mat_product: np.array = self._Aj @ self.get_global_y()
        self.nu = self.nu + self._rho * self._gamma * mat_product

        PRODUCT = self._gamma * mat_product
        self.Ay_trajectory.append(mat_product)

        if self.adaptive_learning:
            H_round = sum([g@g.T for g in self.Ay_trajectory])
            n = H_round.shape[0]
            diagonalized_H_round = np.diag(np.diag(H_round))
            epsilon_identity = self.epsilon*np.identity(n)
            total = np.diag(1/np.sqrt(np.diag(epsilon_identity + diagonalized_H_round)))
            PRODUCT = (self._gamma*total) @ mat_product

        self.nu_bar = self.nu + self._rho * PRODUCT

        # update my belief of nu_bar
        self.global_atom_nu_bar[self.atom_id] = self.nu_bar

    def __str__(self):
        return f'I am atom-{self.atom_id}, with example: {self.get_y()}'
