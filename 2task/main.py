import numpy as np
from sympy import solve, symbols, Matrix, re, trace

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

    x_list = []
    k2_list = []

    for val in y_list:
        x_list.append(solution[x].subs({ y: val }))
        k2_list.append(solution[k2].subs({ y: val }))

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
            plt.plot(k2_list[i], y_list[i], 'r*', label='saddle-node')

    plt.plot(k2_list, x_list, linewidth=1.5, label='k2(x)')
    plt.plot(k2_list, y_list, linestyle='--', linewidth=1.5, label='k2(y)')

    plt.xlabel('k2')
    plt.ylabel('x, y')

    plt.show()


def two_parameter_analysis(f1, f2, param_values):
    """
    Two-parameter analysis
    The analysis is performed by parameters k_1 and k2
    """
    jac_A = Matrix([f1, f2]).jacobian(Matrix([x, y]))
    det_A = jac_A.det()
    trace_A = trace(jac_A)

    param_dict = deepcopy(param_values)
    del param_dict[k1]
    del param_dict[k_1]

    y_of_x = solve(f2, y)[0]
    f1_subs = f1.subs([ y, y_of_x ])
    k2_of_x = solve(f1, k2)[0]

    multi_line = det_A.subs({ y: y_of_x })
    k2_multi = solve(multi_line, k2)[0]

    neut_line = trace_A.subs({ y: y_of_x })
    k2_neut = solve(neut_line, k2)[0]

    k_1_multi = solve(k2_multi - k2_of_x, k1)[0]
    k_1_neut - solve(k2_neut - k2_of_x, k1)[0]

    k_1_multi_subs = k_1_multi.subs(param_dict.items())
    k_1_neut_subs = k_1_neut.subs(param_dict,items())



if __name__ == "__main__":
    param_values = { k1: 1, k_1: 0.01, k2: 2, k3: 0.0032, k_3: 0.002 }

    dxdt = k1 * (1 - x - y) - k_1 * x - k3 * x + k_3 * y - k2 * ((1 - x - y) ** 2) * x
    dydt = k3 * x - k_3 * y

    one_parameter_analysis(dxdt, dydt, param_values)
