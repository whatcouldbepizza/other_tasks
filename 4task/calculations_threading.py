from calculations import particles_to_arrays, get_acceleration_verle

import threading


THREAD_COUNT = 3


def update_coords(
    particles,
    acceleration_list,
    t_i,
    delta_t,
    p_i_start,
    p_i_end
):
    """
    Here we update coordinates
    """
    for p_i in range(p_i_start, p_i_end):
        particles[t_i][p_i][0] += particles[t_i][p_i][2] * delta_t + acceleration_list[p_i][0] / 2 * (delta_t ** 2)
        particles[t_i][p_i][1] += particles[t_i][p_i][3] * delta_t + acceleration_list[p_i][1] / 2 * (delta_t ** 2)


def update_speed(
    particles,
    p_masses,
    acceleration_list,
    new_acceleration_list,
    t_i,
    delta_t,
    p_i_start,
    p_i_end
):
    """
    Updating speed
    """
    for p_i in range(p_i_start, p_i_end):
        new_acceleration_list[p_i] = get_acceleration_verle(particles[t_i], p_masses, p_i)

        particles[t_i][p_i][2] += (acceleration_list[p_i][0] + new_acceleration_list[p_i][0]) / 2 * delta_t
        particles[t_i][p_i][3] += (acceleration_list[p_i][1] + new_acceleration_list[p_i][1]) / 2 * delta_t


def copy_to_next_layer(particles, t_i, p_i):
    """
    Copying coordinates and speed values to next layer
    """
    particles[t_i + 1][p_i][0] = particles[t_i][p_i][0]
    particles[t_i + 1][p_i][1] = particles[t_i][p_i][1]
    particles[t_i + 1][p_i][2] = particles[t_i][p_i][2]
    particles[t_i + 1][p_i][3] = particles[t_i][p_i][3]


def one_time_layer(
    particles,
    p_masses,
    acceleration_list,
    t_i,
    delta_t,
    max_len
):
    """
    Here we perform computations for one time layer
    """
    block = len(particles[t_i]) // THREAD_COUNT
    threads = []

    ## Updating coordinates
    for i in range(THREAD_COUNT):
        p_i_start = i * block
        p_i_end = (i + 1) * block if i < THREAD_COUNT - 1 else len(particles[t_i])

        args = (particles, acceleration_list, t_i, delta_t, p_i_start, p_i_end)
        curr_thread = threading.Thread(target=update_coords, args=args)

        threads.append(curr_thread)
        curr_thread.start()

    for thread in threads:
        thread.join()
    ## Done updating coordinates

    new_acceleration_list = [[0, 0] for _ in acceleration_list]
    threads = []

    ## Updating speed
    for i in range(THREAD_COUNT):
        p_i_start = i * block
        p_i_end = (i + 1) * block if i < THREAD_COUNT - 1 else len(particles[t_i])

        args = (particles, p_masses, acceleration_list, new_acceleration_list, t_i, delta_t, p_i_start, p_i_end)
        curr_thread = threading.Thread(target=update_speed, args=args)

        threads.append(curr_thread)
        curr_thread.start()

    for thread in threads:
        thread.join()
    ## Done updating speed

    for i in range(len(acceleration_list)):
        acceleration_list[i] = new_acceleration_list[i]

    if t_i < max_len:
        for p_i in range(len(particles[t_i])):
            copy_to_next_layer(particles, t_i, p_i)


def overall_verle_threading(particleList, tGrid):
    """
    Verlet method realized with threading parallelizing
    """
    particles, p_masses = particles_to_arrays(particleList, tGrid)
    acceleration_list = [[0, 0] for _ in range(len(particles[0]))]
    delta_t = tGrid[1] - tGrid[0]
    max_len = len(tGrid) - 1

    for t_i, _ in enumerate(tGrid):
        if t_i == 0:
            continue

        one_time_layer(particles, p_masses, acceleration_list, t_i, delta_t, max_len)

    #print(particles)

    return particles[-1], particles
