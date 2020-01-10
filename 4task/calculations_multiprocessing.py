from calculations import get_acceleration_verle

from multiprocessing import Array, Pool, Process
import ctypes
import numpy as np


PROCESS_COUNT = 4

particles = None
p_masses = None
acceleration_list = None
new_acceleration_list = None


def update_coords(
    t_i,
    delta_t,
    p_i_start,
    p_i_end
):
    """
    Here we update coordinates
    """
    global particles, acceleration_list

    for p_i in range(p_i_start, p_i_end):
        particles[t_i][p_i][0] += particles[t_i][p_i][2] * delta_t + acceleration_list[p_i][0] / 2 * (delta_t ** 2)
        particles[t_i][p_i][1] += particles[t_i][p_i][3] * delta_t + acceleration_list[p_i][1] / 2 * (delta_t ** 2)


def update_speed(
    t_i,
    delta_t,
    p_i_start,
    p_i_end
):
    """
    Updating speed
    """
    global particles, p_masses, acceleration_list, new_acceleration_list

    for p_i in range(p_i_start, p_i_end):
        new_acceleration_list[p_i] = get_acceleration_verle(particles[t_i], p_masses, p_i)

        particles[t_i][p_i][2] += (acceleration_list[p_i][0] + new_acceleration_list[p_i][0]) / 2 * delta_t
        particles[t_i][p_i][3] += (acceleration_list[p_i][1] + new_acceleration_list[p_i][1]) / 2 * delta_t


def one_time_layer_multiprocessing(
    t_i,
    delta_t,
    max_len
):
    global particles, p_masses, acceleration_list, new_acceleration_list

    block = len(particles[t_i]) // PROCESS_COUNT
    processes = []

    for i in range(PROCESS_COUNT):
        p_i_start = i * block
        p_i_end = (i + 1) * block if i < PROCESS_COUNT - 1 else len(particles[t_i])

        args = (t_i, delta_t, p_i_start, p_i_end)
        curr_process = Process(target=update_coords, args=args)

        processes.append(curr_process)
        curr_process.start()

    for proc in processes:
        proc.join()

    for i in range(len(new_acceleration_list)):
        new_acceleration_list[i][0] = 0
        new_acceleration_list[i][1] = 0

    processes = []

    for i in range(PROCESS_COUNT):
        p_i_start = i * block
        p_i_end = (i + 1) * block if i < PROCESS_COUNT - 1 else len(particles[t_i])

        args = (t_i, delta_t, p_i_start, p_i_end)

        curr_process = Process(target=update_speed, args=args)

        processes.append(curr_process)
        curr_process.start()

    for proc in processes:
        proc.join()

    for i in range(len(acceleration_list)):
        acceleration_list[i] = new_acceleration_list[i]

    if t_i < max_len:
        for p_i in range(len(particles[t_i])):
            particles[t_i + 1][p_i][0] = particles[t_i][p_i][0]
            particles[t_i + 1][p_i][1] = particles[t_i][p_i][1]
            particles[t_i + 1][p_i][2] = particles[t_i][p_i][2]
            particles[t_i + 1][p_i][3] = particles[t_i][p_i][3]


def overall_verle_multiprocessing(particleList, tGrid):
    """
    Verlet method with multiprocessing
    """
    global particles, p_masses, acceleration_list, new_acceleration_list

    t_rows = len(tGrid)
    p_rows = len(particleList)

    ## 4 stands for elems per particle: 2 coords + 2 speeds
    particles = np.frombuffer(Array(ctypes.c_double, t_rows * p_rows * 4).get_obj()).reshape(t_rows, p_rows, 4)
    p_masses = np.frombuffer(Array(ctypes.c_double, p_rows).get_obj())

    for p_i, p in enumerate(particleList):
        particles[0][p_i][0] = p.coordinates[0]
        particles[0][p_i][1] = p.coordinates[1]
        particles[0][p_i][2] = p.speed[0]
        particles[0][p_i][3] = p.speed[1]

        particles[1][p_i][0] = p.coordinates[0]
        particles[1][p_i][1] = p.coordinates[1]
        particles[1][p_i][2] = p.speed[0]
        particles[1][p_i][3] = p.speed[1]

        p_masses[p_i] = p.mass

    ## 2 stands for 2 acceleration components
    acceleration_list = np.frombuffer(Array(ctypes.c_double, p_rows * 2).get_obj()).reshape(p_rows, 2)
    new_acceleration_list = np.frombuffer(Array(ctypes.c_double, p_rows * 2).get_obj()).reshape(p_rows, 2)

    delta_t = tGrid[1] - tGrid[0]
    max_len = len(tGrid) - 1

    for t_i, _ in enumerate(tGrid):
        if t_i == 0:
            continue

        one_time_layer_multiprocessing(
            t_i,
            delta_t,
            max_len
        )

    return particles[-1], particles
