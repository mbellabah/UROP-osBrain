from typing import Dict, Tuple
import numpy as np
import cvxpy as cp  # FIXME: Move this elsewhere!
from libs.solver import atomic_solve


class Atom(object):
    def __init__(self, atom_id: int, node_data_package: dict, rho: float = 0.17, gamma: float = 3.14):
        self.atom_id = atom_id

        self.rho = rho
        self.gamma = gamma
        self.node_data_package = node_data_package

        # set the attributes from the node data package
        for node_variable in self.node_data_package:
            setattr(self, '_'+node_variable, self.node_data_package[node_variable])

        self.mu = None
        self.mu_bar = None
        self.nu = None
        self.nu_bar = None

        # Later implement so don't have to broadcast to everyone
        self.global_atom_y: Dict[int, np.array] = {self.atom_id: self.get_y()}
        self.global_atom_nu_bar: Dict[int, np.array] = {}

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

    def get_global_nu_bar(self):
        nu_bar_tuple: Tuple[np.array] = tuple([self.global_atom_nu_bar[key] for key in sorted(self.global_atom_nu_bar)])
        return np.vstack(nu_bar_tuple)

    def init_dual_vars(self):
        self.mu: np.array = self.rho * self.gamma * self._Gj @ self._y
        self.mu_bar: np.array = np.ones_like(self.mu)

        global_y_mat: np.array = self._Aj @ self.get_global_y()
        self.nu: np.array = self.rho * self.gamma * global_y_mat
        self.nu_bar: np.array = self.nu + self.rho * self.gamma * global_y_mat

        # set nu_bar
        self.global_atom_nu_bar[self.atom_id] = self.nu_bar

    # MARK: PAC
    def cost_function(self, var):
        xi = 1.0        # FIXME: Integrate with Network Grid topo

        gen_cost = 0.0
        load_util = 0.0
        loss = 0.0

        if self._bus_type == 'feeder':
            # yj = [PGj, PLj, QGj, QLj, vj, {Pjh, Qjh}]
            beta_pg = 1
            beta_qg = 1
            gen_cost: float = beta_pg*var[0] + beta_qg*var[2]
        else:
            # yj = [PGj, PLj, QGj, QLj, vj, {vi, Pij, Qij, lij}] OR
            # yj = [PGj, PLj, QGj, QLj, vj, {vi, Pij, Qij, lij, Pjh, Qjh}]
            gen_cost: float = self._beta_pg*cp.square(var[0]) + self._beta_qg*cp.square(var[2])
            load_util: float = self._beta_pl*cp.square(var[1] - self._PL[0]) + self._beta_ql*cp.square(var[3] - self._QL[0])

        return gen_cost + load_util + loss      # FIXME: Implement atomic loss

    def atomic_objective_function(self, var):
        qmj_tuple = ()
        for m in range(self._global_num_nodes):
            qmj_tuple += (self._Qmj[m][0][self.atom_id-1],)     # because indexing of atom starts at 1
        Qmj = np.vstack(qmj_tuple)

        # print("atom: {}, globalnu: {}, qmj: {}, candidate_a: {}".format(self.atom_id, self.get_global_nu_bar().T.shape, Qmj.shape, var.shape))
        total = self.get_global_nu_bar().T@Qmj@var

        return self.cost_function(var) + self.mu_bar.T@self._Gj@var + total + (1/(2*self.rho)*cp.sum_squares(var-self.get_y()))

    def solve_atomic_objective_function(self) -> np.array:
        return atomic_solve(self.atomic_objective_function, self._y.shape, Bj=self._Bj, bj=self._bj)

    def update_y_and_mu(self):
        try:
            self._y: np.array = self.solve_atomic_objective_function()

            # update mu
            mat_product: np.array = self._Gj @ self._y
            self.mu += self.rho * self.gamma * mat_product
            self.mu_bar = self.mu + self.rho * self.gamma * mat_product

        except ValueError as e:
            print('Could not solve for y')
            raise e

    def update_nu(self):
        mat_product: np.array = self._Aj @ self.get_global_y()
        self.nu += self.rho * self.gamma * mat_product
        self.nu_bar = self.nu + self.rho * self.gamma * mat_product

    def __str__(self):
        return f'I am atom-{self.atom_id}, with example: {self.get_y()}'
