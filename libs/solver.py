import numpy as np
import cvxpy as cp

from libs.config.helper import timeit

SOLVER = cp.GUROBI


def atomic_solve(cost_function, a_shape: tuple, Gj: np.array, rho: float, Qmj: np.array, Bj: np.array, bj: np.array, bus_type: str, thermal_limit: float, previous_problem=None, prev_params=None) -> np.array:

    if not previous_problem:
        candidate_a = cp.Variable(a_shape)
        global_nu_bar = cp.Parameter(shape=prev_params['global_nu_bar'][0], name='global_nu_bar')
        mu_bar = cp.Parameter(shape=prev_params['mu_bar'][0], name='mu_bar')
        prev_y = cp.Parameter(shape=prev_params['prev_y'][0], name='prev_y')

        if prev_params:
            _, global_nu_bar.value = prev_params['global_nu_bar']
            _, mu_bar.value = prev_params['mu_bar']
            _, prev_y.value = prev_params['prev_y']

        quad_constraints = []
        if bus_type != 'feeder':
            """
            (eq. 1) Pij^2 + Qij^2 - Sij^2 <= 0 
            (eq. 2) Pij^2 + Qij^2 - vi*Lij <= 0  
            """
            n = a_shape[0]
            A = np.zeros(shape=(n, n))
            A[0, 0] = 1.0
            A[1, 1] = 1.0
            mat_product = A@candidate_a         # [Pij, Qij, 0, ...]

            quad_constraints = [
                cp.quad_over_lin(mat_product, candidate_a[8]) <= candidate_a[2],
                cp.sum_squares(mat_product) <= thermal_limit**2
            ]

        constraints = [Bj@candidate_a <= bj] + quad_constraints

        total = global_nu_bar.T @Qmj @ candidate_a
        objective_function = lambda var: cost_function(var) + mu_bar.T @ Gj @ var + total + (1 / (2 * rho)) * cp.sum_squares(var - prev_y)

        model_objective = cp.Minimize(objective_function(candidate_a))
        model_problem = cp.Problem(model_objective, constraints)
        model_problem.solve(solver=SOLVER, verbose=False)

        # print(f'status: {model_problem.status}\noptimal value: {model_problem.value}\noptimal var: {candidate_a.value}\n')
        return np.asarray(candidate_a.value), model_problem

    else:
        # Change the params
        for param in previous_problem.parameters():
            _, param.value = prev_params[param.name()]

        previous_problem.solve(solver=SOLVER, verbose=False)
        _candidate_a = previous_problem.variables()[0]

        return np.asarray(_candidate_a.value), None


def ropf(global_cost_func, num_edges, num_nodes, B, b, G, c):
    M = num_edges
    N = num_nodes

    candidate_x = cp.Variable((3*M + 5*N, 1))
    constraints = [
        G@candidate_x == c,
        B@candidate_x <= b
    ]

    model_objective = cp.Minimize(global_cost_func(candidate_x))
    model_problem = cp.Problem(model_objective, constraints)
    model_problem.solve(solver=SOLVER, verbose=True)

    return np.asarray(candidate_x.value), model_problem.value


if __name__ == '__main__':
    print(cp.installed_solvers())
