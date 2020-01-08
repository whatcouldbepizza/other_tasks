from calculations import particles_to_arrays

from copy import deepcopy
import numpy as np
cimport numpy as np


DTYPE = np.double
ctypedef np.double_t DTYPE_t


def get_acceleration(DTYPE_t[:, :] particles, DTYPE_t[:] p_masses, int index):
    cdef DTYPE_t[:] res = np.zeros((2,), dtype=DTYPE)
    cdef DTYPE_t[:] distance = np.zeros((2,), dtype=DTYPE)
    cdef double G = 6.6743015e-11
    cdef int i
    cdef double norm

    for i in range(len(particles)):
        if i == index:
            continue

        distance[0] = particles[i][0] - particles[index][0]
        distance[1] = particles[i][1] - particles[index][1]

        dist_sum = distance[0] ** 2 + distance[1] ** 2
        norm = dist_sum ** 0.5

        res[0] += G * p_masses[i] * distance[0] / (norm ** 3)
        res[1] += G * p_masses[i] * distance[1] / (norm ** 3)

    return res


def overall_verle_cython_inner(
    np.ndarray[DTYPE_t, ndim=3] particles_p,
    np.ndarray[DTYPE_t, ndim=1] p_masses_p,
    np.ndarray[DTYPE_t, ndim=1] t_grid_p
):
    cdef DTYPE_t[:, :, :] particles = deepcopy(particles_p)
    cdef DTYPE_t[:] p_masses = deepcopy(p_masses_p)
    cdef DTYPE_t[:] t_grid = deepcopy(t_grid_p)

    cdef double delta_t = t_grid[1] - t_grid[0]

    cdef DTYPE_t[:, :] acceleration_list = np.zeros((len(p_masses_p), 2), dtype=DTYPE)
    cdef DTYPE_t[:, :] new_acceleration_list = np.zeros((len(p_masses_p), 2), dtype=DTYPE)

    cdef DTYPE_t[:] curr_acceleration = np.zeros((2,), dtype=DTYPE)

    cdef int t_i, p_i, i

    for t_i in range(len(t_grid)):
        if t_i == 0:
            continue

        for p_i in range(len(particles[t_i])):
            particles[t_i][p_i][0] += particles[t_i][p_i][2] * delta_t + acceleration_list[p_i][0] / 2 * (delta_t ** 2)
            particles[t_i][p_i][1] += particles[t_i][p_i][3] * delta_t + acceleration_list[p_i][1] / 2 * (delta_t ** 2)

        for p_i in range(len(particles[t_i])):
            curr_acceleration = get_acceleration(particles[t_i], p_masses, p_i)

            new_acceleration_list[p_i][0] = curr_acceleration[0]
            new_acceleration_list[p_i][1] = curr_acceleration[1]

            particles[t_i][p_i][2] += (acceleration_list[p_i][0] + new_acceleration_list[p_i][0]) / 2 * delta_t
            particles[t_i][p_i][3] += (acceleration_list[p_i][1] + new_acceleration_list[p_i][1]) / 2 * delta_t

        for i in range(len(acceleration_list)):
            acceleration_list[i][0] = new_acceleration_list[i][0]
            acceleration_list[i][1] = new_acceleration_list[i][1]

        if t_i < len(t_grid) - 1:
            for p_i in range(len(particles[t_i])):
                particles[t_i + 1][p_i][0] = particles[t_i][p_i][0]
                particles[t_i + 1][p_i][1] = particles[t_i][p_i][1]
                particles[t_i + 1][p_i][2] = particles[t_i][p_i][2]
                particles[t_i + 1][p_i][3] = particles[t_i][p_i][3]

    return particles


def overall_verle_cython(particleList, tGrid):
    particles_p = np.zeros((len(tGrid), len(particleList), 4), dtype=DTYPE)
    p_masses_p = np.zeros((len(particleList),), dtype=DTYPE)
    t_grid_p = np.zeros((len(tGrid),), dtype=DTYPE)

    for p_i in range(len(particleList)):
        particles_p[0][p_i][0] = particleList[p_i].coordinates[0]
        particles_p[0][p_i][1] = particleList[p_i].coordinates[1]
        particles_p[0][p_i][2] = particleList[p_i].speed[0]
        particles_p[0][p_i][3] = particleList[p_i].speed[1]

        particles_p[1][p_i][0] = particleList[p_i].coordinates[0]
        particles_p[1][p_i][1] = particleList[p_i].coordinates[1]
        particles_p[1][p_i][2] = particleList[p_i].speed[0]
        particles_p[1][p_i][3] = particleList[p_i].speed[1]

        p_masses_p[p_i] = particleList[p_i].mass

    for t_i in range(len(tGrid)):
        t_grid_p[t_i] = tGrid[t_i]

    result = overall_verle_cython_inner(particles_p, p_masses_p, t_grid_p)

    return np.asarray(result[-1]), np.asarray(result)
