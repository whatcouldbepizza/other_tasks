import numpy as np
from sympy import solve, symbols, Matrix, re, trace, lambdify
from math import fabs

from matplotlib import pyplot as plt

from copy import deepcopy


x, y, k1, k_1, k2, k3, k_3 = symbols("x y k1 k_1 k2 k3 k_3")


def one_parameter_analysis(f1, f2, param_values):
    """
    One-parameter analysis
    The analysis is performed by parameter k2
    """
    jac_A = Matrix([f1, f2]).jacobian(Matrix([x, y]))

    param_dict = deepcopy(param_values)
    del param_dict[k2]

    f1_subs = f1.subs([ (key, value) for key, value in param_dict.items() ])
    f2_subs = f2.subs([ (key, value) for key, value in param_dict.items() ])

    solution = solve([f1_subs, f2_subs], x, k2, dict=True)[0]

    y_list = np.linspace(0, 1, 1000)
    y_final = deepcopy(y_list)

    x_list = []
    k2_list = []
    j = 0

    for i, val in enumerate(y_list):
        curr_x = solution[x].subs({ y: val })
        curr_k2 = solution[k2].subs({ y: val })

        if not (0 <= curr_x <= 1) or not (0 <= val + curr_x <= 1) or fabs(curr_k2) > 2 * 50:
            y_final = np.delete(y_final, i - j)
            j += 1
            continue

        x_list.append(curr_x)
        k2_list.append(curr_k2)

    y_list = y_final

    jac_subs = jac_A.subs([ (key, value) for key, value in param_dict.items() ])
    jac_subs_listed = [ jac_subs.subs({ y: val, x: x_list[i], k2: k2_list[i] }) for i, val in enumerate(y_list) ]

    det_list = []
    trace_list = []

    for jac in jac_subs_listed:
        det_list.append(jac.det())
        trace_list.append(trace(jac))

    for i in range(2, len(det_list)):
        if det_list[i] * det_list[i - 1] <= 0:
            plt.plot(k2_list[i], x_list[i], 'r*', label='saddle-node')
            plt.plot(k2_list[i], y_list[i], 'r*', label='saddle-node')

        if trace_list[i] * trace_list[i - 1] <= 0:
            plt.plot(k2_list[i], x_list[i], 'go', label='hopf')
            plt.plot(k2_list[i], y_list[i], 'go', label='hopf')

    plt.plot(k2_list, x_list, linewidth=1.5, label='k2(x)')
    plt.plot(k2_list, y_list, linestyle='--', linewidth=1.5, label='k2(y)')

    plt.xlabel('k2')
    plt.ylabel('x, y')

    plt.show()


def two_parameter_analysis(f1, f2, param_dict):
    jac_A = Matrix([f1, f2]).jacobian(Matrix([x, y]))
    det_A = jac_A.det()
    trace_A = trace(jac_A)


    y_of_x = solve(f2, y)[0]
    k1_of_x = solve(f1, k1)[0]

    k1_trace = solve(trace_A, k1)[0]
    k_1_trace = solve(k1_trace - k1_of_x, k_1)[0]

    k1_det = solve(det_A, k1)[0]
    k_1_det = solve(k1_det - k1_of_x, k_1)[0]

    x_grid = np.linspace(0, 1, 1000)

    y_of_x_func = lambdify([x, k3, k_3], y_of_x, 'numpy')
    y_grid = y_of_x_func(x_grid, param_dict[k3], param_dict[k_3])

    new_x_grid = []
    new_y_grid = []

    for curr_x, curr_y in zip(x_grid, y_grid):
        if 0 <= curr_x + curr_y <= 1 and 0 <= curr_x <= 1:
            new_x_grid.append(curr_x)
            new_y_grid.append(curr_y)

    x_grid = np.array(new_x_grid)
    y_grid = np.array(new_y_grid)

    k_1_trace_subs = np.zeros(x_grid.shape)
    k1_trace_subs = np.zeros(x_grid.shape)
    k_1_det_subs = np.zeros(x_grid.shape)
    k1_det_subs = np.zeros(x_grid.shape)

    for i in range(x_grid.shape[0]):
        k_1_trace_subs[i] = k_1_trace.subs({
                                                  x: x_grid[i],
                                                  y: y_grid[i],
                                                  k2: param_dict[k2],
                                                  k3: param_dict[k3],
                                                  k_3: param_dict[k_3]
                                              })

        k1_trace_subs[i] = k1_of_x.subs({
                                               x: x_grid[i],
                                               y: y_grid[i],
                                               k_1: k_1_trace_subs[i],
                                               k2: param_dict[k2],
                                               k3: param_dict[k3],
                                               k_3: param_dict[k_3]
                                           })

        k_1_det_subs[i] = k_1_det.subs({
                                              x: x_grid[i],
                                              y: y_grid[i],
                                              k2: param_dict[k2],
                                              k3: param_dict[k3],
                                              k_3: param_dict[k_3]
                                          })

        k1_det_subs[i] = k1_of_x.subs({
                                             x: x_grid[i],
                                             y: y_grid[i],
                                             k_1: k_1_det_subs[i],
                                             k2: param_dict[k2],
                                             k3: param_dict[k3],
                                             k_3: param_dict[k_3]
                                         })

    e1, e2 = jac_A.eigenvals()

    for curr_x, curr_y, curr_k1, curr_k_1 in zip(x_grid, y_grid, k1_trace_subs, k_1_trace_subs):
        curr_e1 = e1.subs({
                              x: curr_x,
                              y: curr_y,
                              k1: curr_k1,
                              k_1: curr_k_1,
                              k2: param_dict[k2],
                              k3: param_dict[k3],
                              k_3: param_dict[k_3]
                          })

        curr_e2 = e2.subs({
                              x: curr_x,
                              y: curr_y,
                              k1: curr_k1,
                              k_1: curr_k_1,
                              k2: param_dict[k2],
                              k3: param_dict[k3],
                              k_3: param_dict[k_3]
                          })

        if re(curr_e1) < 50e-4 and re(curr_e2) < 50e-4:
            plt.plot(curr_k1, curr_k_1, 'X', color='g')

    k_1_det_diff = k_1_det.diff(x)
    k_1_det_diff_func = lambdify([x, y, k2, k3, k_3], k_1_det_diff, 'numpy')

    diff_arr = []

    for i in range(x_grid.shape[0]):
        if fabs(k_1_det_diff_func(x_grid[i], y_grid[i], param_dict[k2], param_dict[k3], param_dict[k_3])) < 10e-3:
            diff_arr.append(x_grid[i])

    c_x = sum(diff_arr) / len(diff_arr)
    c_y = y_of_x_func(c_x, param_dict[k3], param_dict[k_3])

    c_k_1 = k_1_det.subs({
                             x: c_x,
                             y: c_y,
                             k2: param_dict[k2],
                             k3: param_dict[k3],
                             k_3: param_dict[k_3]
                         })

    c_k1 = k1_of_x.subs({
                            x: c_x,
                            y: c_y,
                            k_1: c_k_1,
                            k2: param_dict[k2],
                            k3: param_dict[k3],
                            k_3: param_dict[k_3]
                        })

    plt.plot(k1_trace_subs, k_1_trace_subs, '--', label='neutrality')
    plt.plot(k1_det_subs, k_1_det_subs, label='multiplicity')

    plt.plot(c_k1, c_k_1, 'ro', color='r')

    plt.xlabel('k1')
    plt.ylabel('k_1')

    plt.show()


if __name__ == "__main__":
    param_values = { k1: 1, k_1: 0.01, k2: 2, k3: 0.0032, k_3: 0.002 }

    dxdt = k1 * (1 - x - y) - k_1 * x - k3 * x + k_3 * y - k2 * ((1 - x - y) ** 2) * x
    dydt = k3 * x - k_3 * y

    two_parameter_analysis(dxdt, dydt, param_values)
