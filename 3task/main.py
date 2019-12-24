import numpy as np

from fenics import *
from mshr import *

from matplotlib import tri
from matplotlib import pyplot as plt

from sympy import symbols, diff, ccode, sqrt, sin, cos

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


def make_animation(images, name, fps=10):
    with imageio.get_writer(f'{name}.avi', fps=fps) as wr:
        for img in images:
            wr.append_data(img)


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


def heat_equation(u_e, alpha, step_count, T, name):
    """
    Solving heat equation
    """
    global t

    domain = Circle(Point(0, 0), 1)
    mesh = generate_mesh(domain, 30)
    V = FunctionSpace(mesh, 'P', 2)

    u = TrialFunction(V)
    v = TestFunction(V)

    u_D = Expression(ccode(u_e), t=0, degree=2)
    bc = DirichletBC(V, u_D, boundary)

    u_I = interpolate(u_D, V)

    f = diff(u_e, t) - alpha * laplassian(u_e)
    f = Expression(ccode(f), t=0, degree=2)

    grd = gradient(u_e)
    factor = sqrt(x ** 2 + y ** 2)

    g = (grd[0] * x + grd[1] * y) / factor
    g = Expression(ccode(g), t=0, degree=2)

    t_step = T / step_count

    a = dot(grad(u), grad(v)) * t_step * dx + u * v * dx
    L = (u_I + t_step * f) * v * dx + t_step * g * v * ds

    u = Function(V)
    t = 0
    frames = []
    l2_errs = []
    max_errs = []

    for i in range(step_count):
        t += t_step
        u_D.t = t
        g.t = t
        f.t = t

        solve(a == L, u, bc)

        u_I.assign(u)
        u_e = interpolate(u_D, V)

        l2_e = errornorm(u_e, u, 'L2')
        max_e = np.abs(u_e.vector().get_local() - u.vector().get_local()).max()

        print('t: {}, L2 error: {}, max error: {}'.format(t, l2_e, max_e))

        ## -------------------------------------------- Geometry block --------------------------------------------
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
        ## ---------------------------------------------- End block -----------------------------------------------

        frames.append(fig_plot)
        l2_errs.append(l2_e)
        max_errs.append(max_e)

    make_animation(frames, f'heat_{name}')
    imageio.imsave(f'heat_{name}.png', frames[-1])

    t_grid = np.linspace(t_step, T, step_count)

    plt.plot(t_grid, l2_errs)
    plt.plot(t_grid, max_errs, '--')

    plt.savefig(f'error_{name}.png')
    plt.close()


if __name__ == "__main__":
    poisson_equation(-2 * x ** 2 + 4 * y + 1, 1, 'test1')
    poisson_equation(x ** 2 + y ** 2 + 1, 1, 'test2')
    poisson_equation(sin(x) + sin(y), 1, 'test3')

    heat_equation((x + y * t) * t, 1, 50, 5.0, 'test_h_1')
    heat_equation((x ** 2 * t + y ** 2) * t, 1, 50, 5.0, 'test_h_2')
    heat_equation(x * sin(3 * y * t) - 4 * cos(5 * x), 1, 50, 5.0, 'test_h_3')
