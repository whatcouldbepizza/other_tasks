from sympy import solve, symbols, Matrix, re
import numpy as np

import matplotlib.pyplot as plt

from copy import deepcopy


x, y, k1, k_1, k3, k_3, k2 = symbols('x y k1 k_1 k3 k_3 k2')


def one_param_analisys(
    f1,
    f2,
    param_values
):
    jac_A = Matrix([f1, f2]).jacobian(Matrix([x, y]))

    param_dict = deepcopy(param_values)
    del param_dict[k2]

    f1_filled = f1.subs([(key, val) for key, val in param_dict.items()])
    f2_filled = f2.subs([(key, val) for key, val in param_dict.items()])

    sol = solve([f1_filled, f2_filled], x, k2, dict=True)[0]
    y_grid = np.linspace(0.001, 1, 1000, endpoint=True)

    k2_arr = np.array([ sol[k2].subs({ y: elem }) for elem in y_grid ])
    x_arr = np.array([ sol[x].subs({ y: elem }) for elem in y_grid ])

    jac_value = jac_A.subs([ (key, value) for key, value in param_dict.items() ])
    eigenvals = jac_value.eigenvals().keys()

    y_features = []

    for i, val in enumerate(eigenvals):
        eigenval = val.subs({ k2: sol[k2], x: sol[x] })
        y_arr = np.array([ re(eigenval.subs({ y: elem })) for elem in y_grid ])
        y_features.extend([ y_grid[j] for j in range(len(y_arr) - 1) if y_arr[i] * y_arr[i + 1] < 0 ])

    k1_features = [ sol[k2].subs({ y: elem }) for elem in y_features ]
    x_features = [ sol[x].subs({ y: elem }) for elem in y_features ]

    plt.plot(k2_arr, y_grid, linewidth=1.5, label='multi')
    plt.plot(k2_arr, x_arr, linestyle='--', linewidth=1.5, label='neutral')

    plt.plot(k1_features, y_features, 'ro')
    plt.plot(k1_features, x_features, 'X')

    plt.xlabel(r'$k2$')
    plt.ylabel(r'$x, y$')
    plt.xlim([0, 8])
    plt.ylim([0, 1])
    plt.show()


def main():
    param_values = { k1: 1, k_1: 0.01, k3: 0.0032, k_3: 0.0002, k2: 2 }

    dxdt = k1 * (1 - x - y) - k_1 * x - k3 * x + k_3 * y - k2 * ((1 - x - y) ** 2) * x
    dydt = k3 * x - k_3 * y

    one_param_analisys(dxdt, dydt, param_values)


if __name__ == "__main__":
    main()
