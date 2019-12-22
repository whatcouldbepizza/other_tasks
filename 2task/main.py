import numpy as np
from sympy import solve, symbols, Matrix, re, trace, lambdify
from math import fabs
from scipy.integrate import odeint

from matplotlib import pyplot as plt

from copy import deepcopy


x, y, k1, k_1, k2, k3, k_3 = symbols("x y k1 k_1 k2 k3 k_3")


def one_parameter_analysis(f1, f2, param_values, k2_bot, k2_top, const1, const2):
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
        curr_x = re(solution[x].subs({ y: val }))
        curr_k2 = re(solution[k2].subs({ y: val }))

        if not (0 <= curr_x <= 1) or not (0 <= val + curr_x <= 1):
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
        if det_list[i] * det_list[i - 1] <= const1:
            plt.plot(k2_list[i], x_list[i], 'rx', label='saddle-node')
            plt.plot(k2_list[i], y_list[i], 'rx', label='saddle-node')

        if trace_list[i] * trace_list[i - 1] <= const2:
            plt.plot(k2_list[i], x_list[i], 'go', label='hopf')
            plt.plot(k2_list[i], y_list[i], 'go', label='hopf')

    plt.plot(k2_list, x_list, linewidth=1.5, label='k2(x)')
    plt.plot(k2_list, y_list, linestyle='--', linewidth=1.5, label='k2(y)')

    plt.xlabel('k2')
    plt.ylabel('x, y')

    plt.xlim(k2_bot, k2_top)
    plt.ylim(0.0, 1.0)

    plt.show()


def two_parameter_analysis(f1, f2, param_dict, cond_mult, cond_y, const1, const2):
    param1 = k_1
    param2 = k1

    jac_A = Matrix([f1, f2]).jacobian(Matrix([x, y]))
    det_A = jac_A.det()
    trace_A = trace(jac_A)

    y_of_x = solve(f2, y)[0]
    param2_of_x = solve(f1, param2)[0]

    param2_trace = solve(trace_A, param2)[0]
    param1_trace = solve(param2_trace - param2_of_x, param1)[0]

    param2_det = solve(det_A, param2)[0]
    param1_det = solve(param2_det - param2_of_x, param1)[0]

    x_grid = np.linspace(0, 1, 1000)

    y_of_x_func = lambdify([x, k3, k_3], y_of_x, 'numpy')
    y_grid = y_of_x_func(x_grid, param_dict[k3], param_dict[k_3])

    new_x_grid = []
    new_y_grid = []

    for curr_x, curr_y in zip(x_grid, y_grid):
        if 0 <= curr_x + cond_mult * curr_y <= 1 and 0 <= curr_y <= cond_y:
            new_x_grid.append(curr_x)
            new_y_grid.append(curr_y)

    x_grid = np.array(new_x_grid)
    y_grid = np.array(new_y_grid)

    param1_trace_subs = np.zeros(x_grid.shape)
    param2_trace_subs = np.zeros(x_grid.shape)
    param1_det_subs = np.zeros(x_grid.shape)
    param2_det_subs = np.zeros(x_grid.shape)

    for i in range(x_grid.shape[0]):
        param1_trace_subs[i] = param1_trace.subs({
                                                     x: x_grid[i],
                                                     y: y_grid[i],
                                                     k2: param_dict[k2],
                                                     k3: param_dict[k3],
                                                     k_3: param_dict[k_3]
                                                 })

        param2_trace_subs[i] = param2_of_x.subs({
                                                    x: x_grid[i],
                                                    y: y_grid[i],
                                                    k_1: param1_trace_subs[i],
                                                    k2: param_dict[k2],
                                                    k3: param_dict[k3],
                                                    k_3: param_dict[k_3]
                                                })

        param1_det_subs[i] = param1_det.subs({
                                                 x: x_grid[i],
                                                 y: y_grid[i],
                                                 k2: param_dict[k2],
                                                 k3: param_dict[k3],
                                                 k_3: param_dict[k_3]
                                             })

        param2_det_subs[i] = param2_of_x.subs({
                                                  x: x_grid[i],
                                                  y: y_grid[i],
                                                  k_1: param1_det_subs[i],
                                                  k2: param_dict[k2],
                                                  k3: param_dict[k3],
                                                  k_3: param_dict[k_3]
                                              })

    e1, e2 = jac_A.eigenvals()

    for curr_x, curr_y, curr_param2, curr_param1 in zip(x_grid, y_grid, param2_trace_subs, param1_trace_subs):
        curr_e1 = e1.subs({
                              x: curr_x,
                              y: curr_y,
                              k1: curr_param2,
                              k_1: curr_param1,
                              k2: param_dict[k2],
                              k3: param_dict[k3],
                              k_3: param_dict[k_3]
                          })

        curr_e2 = e2.subs({
                              x: curr_x,
                              y: curr_y,
                              k1: curr_param2,
                              k_1: curr_param1,
                              k2: param_dict[k2],
                              k3: param_dict[k3],
                              k_3: param_dict[k_3]
                          })

        if curr_e1.is_real and curr_e2.is_real and curr_e1 < const1 and curr_e2 < const1:
            plt.plot(curr_param2, curr_param1, 'X', color='g')

    param1_det_diff = param1_det.diff(x)
    param1_det_diff_func = lambdify([x, y, k2, k3, k_3], param1_det_diff, 'numpy')

    diff_arr = []

    for i in range(x_grid.shape[0]):
        if fabs(param1_det_diff_func(x_grid[i], y_grid[i], param_dict[k2], param_dict[k3], param_dict[k_3])) < const2:
            diff_arr.append(x_grid[i])

    c_x = sum(diff_arr) / len(diff_arr)
    c_y = y_of_x_func(c_x, param_dict[k3], param_dict[k_3])

    c_param1 = param1_det.subs({
                                   x: c_x,
                                   y: c_y,
                                   k2: param_dict[k2],
                                   #k1: param_dict[k1],
                                   k3: param_dict[k3],
                                   k_3: param_dict[k_3]
                               })

    c_param2 = param2_of_x.subs({
                                    x: c_x,
                                    y: c_y,
                                    k_1: c_param1,
                                    k2: param_dict[k2],
                                    #k1: param_dict[k1],
                                    k3: param_dict[k3],
                                    k_3: param_dict[k_3]
                                })

    plt.plot(param2_trace_subs, param1_trace_subs, '--', label='neutrality')
    plt.plot(param2_det_subs, param1_det_subs, label='multiplicity')

    plt.plot(c_param2, c_param1, 'ro', color='r')

    plt.xlabel('k1')
    plt.ylabel('k_1')

    plt.show()


def derivate(start, t, arg_k1, arg_k_1, arg_k2, arg_k3, arg_k_3, f1, f2):
    curr_f1 = f1.subs({
                          x: start[0],
                          y: start[1],
                          k1: arg_k1,
                          k_1: arg_k_1,
                          k2: arg_k2,
                          k3: arg_k3,
                          k_3: arg_k_3
                      })

    curr_f2 = f2.subs({
                          x: start[0],
                          y: start[1],
                          k1: arg_k1,
                          k_1: arg_k_1,
                          k2: arg_k2,
                          k3: arg_k3,
                          k_3: arg_k_3
                      })

    return [curr_f1, curr_f2]


def phase_portrait(f1, f2, param_dict, auto_k1, auto_k_1, cond_mult, cond_y, solution_a):
    x_grid = np.linspace(0, 1, 1000)

    y_of_x = solve(f2, y)[0]

    y_of_x_func = lambdify([x, k3, k_3], y_of_x, 'numpy')
    y_grid_f2 = y_of_x_func(x_grid, param_dict[k3], param_dict[k_3])

    new_x_grid = []
    new_y_grid = []

    for curr_x, curr_y in zip(x_grid, y_grid_f2):
        if 0 <= curr_x + cond_mult * curr_y <= 1 and 0 <= curr_y <= cond_y:
            new_x_grid.append(curr_x)
            new_y_grid.append(curr_y)

    x_grid = np.array(new_x_grid)
    y_grid_f2 = np.array(new_y_grid)

    ## f_2(x, y) = 0
    plt.plot(x_grid, y_grid_f2, 'g')

    y_from_f1 = solve(f1, y)
    y_from_f1_func = lambdify([x, k1, k_1, k2, k3, k_3], y_from_f1, 'numpy')
    y_grid_f1 = y_from_f1_func(x_grid, auto_k1, auto_k_1, param_dict[k2], param_dict[k3], param_dict[k_3])

    ## f_1(x, y) = 0
    plt.plot(x_grid, y_grid_f1[0], 'b')
    plt.plot(x_grid, y_grid_f1[1], 'b')

    t_grid = np.linspace(0, 10000, 1000)

    start_1 = [0.1, 0.1]
    solution_1 = odeint(
                            derivate,
                            start_1,
                            t_grid,
                            args=(auto_k1, auto_k_1, param_dict[k2], param_dict[k3], param_dict[k_3], f1, f2)
                        )

    start_2 = [0.55, 0.3]
    solution_2 = odeint(
                            derivate,
                            start_2,
                            t_grid,
                            args=(auto_k1, auto_k_1, param_dict[k2], param_dict[k3], param_dict[k_3], f1, f2)
                        )

    ## Trajectories
    plt.plot(solution_1[:, 0], solution_1[:, 1], '--', solution_2[:, 0], solution_2[:, 1], '--')

    solution_c = odeint(
                            derivate,
                            solution_a[-1],
                            t_grid,
                            args=(auto_k1, auto_k_1, param_dict[k2], param_dict[k3], param_dict[k_3], f1, f2)
                        )

    ## Cycle
    plt.quiver(
                  solution_c[:-1, 0],
                  solution_c[:-1, 1],
                  solution_c[1:, 0] - solution_c[:-1, 0],
                  solution_c[1:, 1] - solution_c[:-1, 1],
                  scale=2
              )

    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.show()


def auto_oscillation(f1, f2, param_dict, auto_k1, auto_k_1, cond_mult, cond_y):
    t_grid = np.linspace(0, 10000, 1000)
    start = [0.2, 0.2]

    solution = odeint(derivate, start, t_grid, args=(auto_k1, auto_k_1, param_dict[k2], param_dict[k3], param_dict[k_3], f1, f2))

    plt.plot(t_grid, solution[:, 0], 'b')

    plt.xlabel("t")
    plt.ylabel("x")
    plt.show()

    plt.plot(t_grid, solution[:, 1], 'r')

    plt.xlabel("t")
    plt.ylabel("y")
    plt.show()

    phase_portrait(f1, f2, param_dict, auto_k1, auto_k_1, cond_mult, cond_y, solution)


if __name__ == "__main__":
    ## Variant # 1
    param_values = { k1: 0.3, k_1: 0.003, k2: 2.0, k3: 0.0032, k_3: 0.003 }

    dxdt = k1 * (1 - x - y) - k_1 * x - k2 * ((1 - x - y) ** 2) * x
    dydt = k3 * (1 - x - y) - k_3 * y

    ## Variant #4
    #param_values = { k1: 1, k_1: 0.01, k2: 2, k3: 0.0032, k_3: 0.002 }

    #dxdt = k1 * (1 - x - y) - k_1 * x - k3 * x + k_3 * y - k2 * ((1 - x - y) ** 2) * x
    #dydt = k3 * x - k_3 * y

    ## Variant # 5
    #param_values = { k1: 0.12, k_1: 0.01, k2: 0.95, k3: 0.0032, k_3: 0.002 }

    #dxdt = k1 * (1 - x - 2 * y) - k_1 * x - k3 * x * (1 - x - 2 * y) + k_3 * y - k2 * ((1 - x - 2 * y) ** 2) * x
    #dydt = k3 * (1 - x - 2 * y) * x - k_3 * y

    #one_parameter_analysis(dxdt, dydt, param_values, -50, 50, 0, 0) ## Variant 1
    #one_parameter_analysis(dxdt, dydt, param_values, 0, 20, 9e-9, 6.5e-4) ## Variant 5

    #two_parameter_analysis(dxdt, dydt, param_values, 1, 1, 10e-4, 10e-1) ## Variant 1
    #two_parameter_analysis(dxdt, dydt, param_values, 2, 1, 10e-4, 100e-1)

    auto_oscillation(dxdt, dydt, param_values, 0.30066, 0.024783, 1, 1) ## Variant #1
    #auto_oscillation(dxdt, dydt, param_values, 0.121589, 0.011091, 2, 1) ## Variant #5
