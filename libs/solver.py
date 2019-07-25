import numpy as np
import cvxpy as cp

SOLVER = cp.GUROBI


# MARK: Example Optimization Problems
# @timeit
# def pyomo_example_1():
#     # knapsack Example using pyomo
#
#     v = {'hammer': 8, 'wrench': 3, 'screwdriver': 6, 'towel': 11}
#     w = {'hammer': 8, 'wrench': 7, 'screwdriver': 4, 'towel': 3}
#
#     limit = 14
#     items = list(sorted(v.keys()))
#
#     m = ConcreteModel()
#
#     # Variables
#     m.x = Var(items, within=Binary)
#
#     # Objective
#     m.value = Objective(expr = sum(v[i]*m.x[i] for i in items), sense=maximize)
#     # Constraint
#     m.weight = Constraint(expr = sum(w[i]*m.x[i] for i in items) <= limit)
#     # Optimize
#     solver = SolverFactory('glpk')
#     status = solver.solve(m)
#
#     print("status = %s" % status.solver.termination_condition)
#
#     # Print the value of the variables at the optimum
#     for i in items:
#         print("%s = %f" % (m.x[i], value(m.x[i])))
#
#     # Print the value of the objective
#     print("Objective = %f" % value(m.value))
#
#
# @timeit
# def pyomo_example_2():
#     m = ConcreteModel()
#
#     m.x = Var(['x', 'y'], within=Binary)
#     m.Cost = Objective(expr=(m.x['x'] - m.x['y'])**2, sense=minimize)
#     m.constraints = ConstraintList()
#     m.constraints.add(expr=(m.x['x'] + m.x['y']) == 1)
#     m.constraints.add(expr=(m.x['x'] - m.x['y']) >= 1)
#
#     results = SolverFactory('cvxopt').solve(m)
#     results.write()
#
#
# @timeit
# def cvx_py_example_1():
#     # Create two scalar optimization variables.
#     x = cp.Variable()
#     y = cp.Variable()
#
#     # Create two constraints.
#     constraints = [x + y == 1,
#                    x - y >= 1]
#
#     # Form objective.
#     obj = cp.Minimize((x - y) ** 2)
#
#     # Form and solve problem.
#     prob = cp.Problem(obj, constraints)
#     prob.solve()  # Returns the optimal value.
#     print("status:", prob.status)
#     print("optimal value", prob.value)
#     print("optimal var", x.value, y.value)


# @timeit
def atomic_solve(optimize_equation_func, a_shape: tuple, Bj: np.array, bj: np.array, bus_type: str, thermal_limit: float) -> np.array:
    candidate_a = cp.Variable(a_shape)

    quad_constraints = []
    if bus_type != 'feeder':
        """
        (eq. 1) Pij^2 + Qij^2 - Sij <= 0 
        (eq. 2) Pij^2 + Qij^2 - vi*Lij <= 0 
        
        e.g. for candidate_a: (9x1) -- [x0, x1, x2 . . . x8] which maps to [Pij, Qij, Lij, . . .]
        We'd like standard form candidate_a.T @ A @ candidate_a = (eq.1, eq.2) 
        We've found that A = 
        1 . . .
        0 1 . . . . 
        .
        .
        .
        .
        .
        .
        . . -1 . . .
        Where the A: (9x9) will give x0^2 + x2^2 - x8*x2 
        
        note that cvxpy quad_form(x, P) is an alias for x.T @ P @ x 
        """
        n = a_shape[0]
        A = np.zeros(shape=(n, n))
        A[0, 0] = 1.0
        A[1, 1] = 1.0
        mat_product = A@candidate_a
        """
        quad_over_lin(x, y) = sum_i (x_i)^2 <= y 
        """
        quad_constraints = [
            cp.quad_over_lin(mat_product, candidate_a[8]) <= candidate_a[2],
            cp.quad_over_lin(mat_product, thermal_limit) <= thermal_limit
        ]

    constraints = [Bj@candidate_a <= bj] + quad_constraints
    model_objective = cp.Minimize(optimize_equation_func(candidate_a))
    model_problem = cp.Problem(model_objective, constraints)
    model_problem.solve(solver=SOLVER, verbose=False)

    # print(f'status: {model_problem.status}\noptimal value: {model_problem.value}\noptimal var: {candidate_a.value}\n')

    return np.asarray(candidate_a.value)


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
