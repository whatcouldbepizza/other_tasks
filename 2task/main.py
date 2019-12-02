import numpy as np

from matplotlib import pyplot as plt


grid_power = 1000


def one_parameter_analisys():
    k1 = 1
    k_1 = 0.01
    k3 = 0.0032
    k_3 = 2

    x = np.linspace(0, 0.999, grid_power)
    y = np.zeros(grid_power)
    k2 = np.zeros(grid_power)

    s = np.zeros(grid_power)
    jac = np.zeros(grid_power)
    D = np.zeros(grid_power)

    y_hopf = []
    x_hopf = []
    k2_hopf = []

    y_saddle_node = []
    x_saddle_node = []
    k2_saddle_node = []

    y_node = []
    x_node = []
    k2_node = []

    for i in range(1, grid_power):
        y[i] = k3 * x[i] / k_3
        k2[i] = (k1 * (1 - x[i] - y[i]) - k_1 * x[i] - k3 * x[i] + k_3 * y[i]) / (x[i] * ((1 - x[i] - y[i]) ** 2))

        a11 = - k1 - k_1 - k3 - k2[i] * ((1 - x[i] - y[i]) ** 2) + 2 * x[i] * k2[i] * (1 - x[i] - y[i])
        a12 = - k1 + k_3 + 2 * x[i] * k2[i] * (1 - x[i] - y[i])
        a21 = k3
        a22 = - k_3

        s[i] = a11 + a22
        jac[i] = a11 * a22 - a12 * a21
        D[i] = s[i] ** 2 - 4 * jac[i]

        if i == 1:
            fig_1 = plt.figure()
            continue

        if s[i] * s[i - 1] <= 0:
            y_hopf.append(y[i])
            x_hopf.append(x[i])
            k2_hopf.append(k2[i])

            plt.plot(k2[i], x[i], 'go', label='hopf')
            plt.plot(k2[i], y[i], 'go', label='hopf')

        if jac[i] * jac[i - 1] <= 0:
            y_saddle_node.append(y[i])
            x_saddle_node.append(x[i])
            k2_saddle_node.append(k2[i])

            plt.plot(k2[i], x[i], 'r*', label='saddle-node')
            plt.plot(k2[i], y[i], 'r*', label='saddle-node')

        if D[i] * D[i - 1] <= 0:
            y_node.append(y[i])
            x_node.append(x[i])
            k2_node.append(k2[i])

            plt.plot(k2[i], x[i], 'b^', label='node')
            plt.plot(k2[i], y[i], 'b^', label='node')

    x_plot = fig_1.add_subplot(111)
    x_plot.plot(k2, x, label='k2(x)')

    y_plot = fig_1.add_subplot(111)
    y_plot.plot(k2, y, label='k2(y)')

    plt.title("One-parameter analisys")
    plt.show()


if __name__ == "__main__":
    one_parameter_analisys()
