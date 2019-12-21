import numpy as np

from fenics import *
from mshr import *

from matplotlib import tri
from matplotlib import pyplot as plt

from sympy import symbols, diff, ccode, sqrt

import imageio

x, y, t = symbols('x[0] x[1] t')


def laplassian(f):
    return diff(f, x, x) + diff(f, y, y)


def gradient(f):
    return [ diff(f, x), diff(f, y) ]


def boundary(x, on_boundary):
    if not on_boundary:
        return False

    if x[1] < 0:
        return True

    return False


def poisson_equation(u_e, alpha, name):
    """
    Solving Poisson equation
    """

    ## Generating finite elements
    domain = Circle(Point(0, 0), 1)
    mesh = generate_mesh(domain, 30)
    V = FunctionSpace(mesh, 'P', 2)

    ## Difference between trial and test functions is boundary conditions
    u = TrialFunction(V)
    v = TestFunction(V)

    ## Defining boundary conditions
    u_D = Expression(ccode(u_e), degree=2)
    bc = DirichletBC(V, u_D, boundary)

    ## Defining source term
    f = - laplassian(u_e) + alpha * u_e
    f = Expression(ccode(f), degree=2)

    grd = gradient(u_e)
    factor = sqrt(x ** 2 + y ** 2)

    g = (grd[0] * x + grd[1] * y) / factor
    g = Expression(ccode(g), degree=2)

    ## Defining variation problem
    a = dot(grad(u), grad(v)) * dx + alpha * u * v * dx
    L = f * v * dx + g * v * ds

    ## Forming and solving linear system
    u = Function(V)
    solve(a == L, u, bc)

    ## Computing the error
    l2_err = errornorm(u_D, u, 'L2')

    vert_vals_u_D = u_D.compute_vertex_values(mesh)
    vert_vals_u = u.compute_vertex_values(mesh)

    max_err = np.max(np.abs(vert_vals_u_D - vert_vals_u))

    print("Max error: {}, L2 error: {}".format(max_err, l2_err))

    n = mesh.num_vertices()
    dim = mesh.geometry().dim()
    mesh_coords = mesh.coordinates().reshape((n, dim))

    triangs = np.asarray([ cell.entities(0) for cell in cells(mesh) ])
    triangulation = tri.Triangulation(mesh_coords[:, 0], mesh_coords[:, 1], triangs)

    fig, (ax1, ax2) = plt.subplots(1, 2)

    zfaces1 = []
    zfaces2 = []

    for cell in cells(mesh):
        zfaces1.append(u(cell.midpoint()))
        zfaces2.append(u_D(cell.midpoint()))

    zfaces1 = np.asarray(zfaces1)
    zfaces2 = np.asarray(zfaces2)

    ax1_plot = ax1.tripcolor(triangulation, facecolors=zfaces1, edgecolors='k')
    ax2_plot = ax2.tripcolor(triangulation, facecolors=zfaces2, edgecolors='k')

    ax1.set_title('Approximate')
    ax2.set_title('Exact')

    fig.canvas.draw()

    fig_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(
        fig.canvas.get_width_height()[::-1] + (3,)
    )

    plt.close()

    imageio.imsave(f'poisson_{name}.png', fig_plot)


def heat_equation(u_e, alpha, step_count, name):
    """
    Solving heat equation
    """


if __name__ == "__main__":
    poisson_equation(1 - x ** 2 + y, 1, 'test1')

    # TODO: Need more examples
