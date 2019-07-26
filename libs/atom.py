from typing import Dict, Tuple
import numpy as np
import cvxpy as cp  # FIXME: Move this elsewhere!
from libs.solver import atomic_solve


class Atom(object):
    def __init__(self, atom_id: int, node_data_package: dict):
        self.atom_id = atom_id
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
        self.mu: np.array = np.zeros_like(self._rho * self._gamma * self._Gj @ self._y)
        self.mu_bar: np.array = self.mu + self._rho * self._gamma * self._Gj @ self._y

        global_y_mat: np.array = self._Aj @ self.get_global_y()

        self.nu: np.array = np.zeros_like(self._rho * self._gamma * global_y_mat)
        self.nu_bar: np.array = self.nu + self._rho * self._gamma * global_y_mat

        # set nu_bar
        self.global_atom_nu_bar[self.atom_id] = self.nu_bar

    # MARK: PAC
    def cost_function(self, var):
        xi = 1.0        # FIXME: Integrate with Network Grid topo

        gen_cost: float = 0.0
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

            gen_cost: float = self._beta_pg*cp.square(PG - self._PG[0]) + self._beta_qg*cp.square(QG - self._QG[0])
            load_util: float = self._beta_pl*cp.square(PL - self._PL[1]) + self._beta_ql*cp.square(QL - self._QL[1])

            parent_node: int = int(self._parent_node)
            upstream_line_resistance: float = self._neighbors[parent_node]['resistance']
            loss: float = xi*upstream_line_resistance*Lij

        return gen_cost + load_util + loss

    def atomic_objective_function(self, var):
        qmj_tuple = ()
        for m in range(self._global_num_nodes):
            qmj_tuple += (self._Qmj[m][0][self.atom_id-1],)     # because indexing of atom starts at 1
        Qmj = np.vstack(qmj_tuple)

        total = self.get_global_nu_bar().T@Qmj@var
        return self.cost_function(var) + self.mu_bar.T@self._Gj@var + total + (1/(2*self._rho)*cp.norm(var-self.get_y())**2)

    # def _cost_function(self, var):
    #     xi = 1.0  # FIXME: Integrate with Network Grid topo
    #
    #     gen_cost: float = 0.0
    #     load_util = 0.0
    #     loss = 0.0
    #
    #     if self._bus_type == 'feeder':
    #         # yj = [PLj, PGj, QLj, QGj, vj, {Pjh, Qjh}]
    #         beta_pg = 1
    #         beta_qg = 1
    #
    #         PG: float = var[1]
    #         QG: float = var[3]
    #
    #         gen_cost: float = beta_pg * PG + beta_qg * QG
    #     else:
    #         # yj = [Pij, Qij, lij, PLj, PGj, QLj, QGj, vj, {vi}] (end node) OR
    #         # yj = [Pij, Qij, Lij, PLj, PGj, QLj, QGj, vj, {vi, Pjk, Qjk, . . .}] ('middle' node)
    #
    #         PG: float = var[4]
    #         QG: float = var[6]
    #         PL: float = var[3]
    #         QL: float = var[5]
    #         Lij: float = var[2]  # the current flow on the upstream line
    #
    #         gen_cost: float = self._beta_pg * np.square(PG - self._PG[0]) + self._beta_qg * np.square(QG - self._QG[0])
    #         load_util: float = self._beta_pl * np.square(PL - self._PL[1]) + self._beta_ql * np.square(QL - self._QL[1])
    #
    #         parent_node: int = int(self._parent_node)
    #         upstream_line_resistance: float = self._neighbors[parent_node]['resistance']
    #         loss: float = xi * upstream_line_resistance * Lij
    #
    #     return gen_cost + load_util + loss

    # def _atomic_objective_function(self, var):
    #     qmj_tuple = ()
    #     for m in range(self._global_num_nodes):
    #         qmj_tuple += (self._Qmj[m][0][self.atom_id-1],)     # because indexing of atom starts at 1
    #     Qmj = np.vstack(qmj_tuple)
    #
    #     total = self.get_global_nu_bar().T@Qmj@var
    #
    #     return self._cost_function(var) + self.mu_bar.T@self._Gj@var + total + (1/(2*self._rho)*np.linalg.norm(var-self.get_y())**2)

    def solve_atomic_objective_function(self) -> np.array:
        parent_node: int = int(self._parent_node)
        upstream_line_thermal_limit: float = self._neighbors[parent_node]['thermal_limit']

        # print(f"{self.atom_id}::: HEY HEY HEY::: {self._atomic_objective_function(var=np.ones_like(self.get_y()))}")

        return atomic_solve(self.atomic_objective_function, self._y.shape, Bj=self._Bj, bj=self._bj, bus_type=self._bus_type, thermal_limit=upstream_line_thermal_limit)

    def update_y_and_mu(self):
        try:
            self._y: np.array = self.solve_atomic_objective_function()
            # update mu
            mat_product: np.array = self._Gj @ self._y
            self.mu += self._rho * self._gamma * mat_product
            self.mu_bar = self.mu + self._rho * self._gamma * mat_product

            # update my y that exists in the global dict
            self.global_atom_y[self.atom_id] = self._y

        except ValueError as e:
            print('Could not solve for y')
            raise e

    def update_nu(self):
        mat_product: np.array = self._Aj @ self.get_global_y()
        # print("\n")
        self.nu += self._rho * self._gamma * mat_product
        self.nu_bar = self.nu + self._rho * self._gamma * mat_product

        # update my belief of nu_bar
        self.global_atom_nu_bar[self.atom_id] = self.nu_bar

    def __str__(self):
        return f'I am atom-{self.atom_id}, with example: {self.get_y()}'
