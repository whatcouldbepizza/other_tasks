from scipy.integrate import odeint
import numpy as np
import math
from classes import Particle
import copy


G = 6.6743015 * (10 ** -11)


def to_particle_list(data, old_list):
    result = []

    for i in range(len(old_list)):
        result.append(
            Particle(
                        coordinates=[data[i][0], data[i][1]],
                        speed=[data[i][2], data[i][3]],
                        mass=old_list[i].mass,
                        color=old_list[i].color,
                        living_time=old_list[i].living_time
                    )
            )

    return result


def supercopy(lst):
    result = []

    for elem in lst:
        newParticle = Particle(
                               [
                                   elem.coordinates[0],
                                   elem.coordinates[1]
                               ],
                               [
                                   elem.speed[0],
                                   elem.speed[1]
                               ],
                               elem.mass,
                               elem.color,
                               elem.living_time
                               )
        result.append(newParticle)

    return result


def get_acceleration(particleList, index):
    """
    Getting acceleration for one point dictated by other points
    """
    res = np.array([0.0, 0.0])

    for i in range(len(particleList)):
        if i == index:
            continue

        distance = np.array([particleList[i].coordinates[0] - particleList[index].coordinates[0],
                             particleList[i].coordinates[1] - particleList[index].coordinates[1]])

        if np.linalg.norm(distance) <= particleList[index].mass * 5 + particleList[i].mass * 5:
            continue

        res += G * particleList[i].mass * distance / (np.linalg.norm(distance) ** 3)

    return res


def overall_pend(prev, t, p_masses):
    acceleration_list = []

    for i in range(len(prev) // 4):
        partial_acceleration = np.array([0.0, 0.0])

        for j in range(len(prev) // 4):
            if i == j:
                continue

            distance = np.array([
                                    prev[j * 4] - prev[i * 4],
                                    prev[j * 4 + 1] - prev[i * 4 + 1]
                                ])

            norm = math.pow(math.pow(distance[0], 2) + math.pow(distance[1], 2), 3/2)

            partial_acceleration += G * p_masses[j] * distance / norm

        acceleration_list.append([partial_acceleration[0], partial_acceleration[1]])

    result = []

    for i in range(len(prev) // 4):
        result.extend([
                          prev[i * 4 + 2],
                          prev[i * 4 + 3],
                          acceleration_list[i][0],
                          acceleration_list[i][1]
                      ])

    return result


def overall_odeint(particleList, tGrid):
    particles = []

    y0 = []

    for _, p in enumerate(particleList):
        y0.extend([
                      p.coordinates[0],
                      p.coordinates[1],
                      p.speed[0],
                      p.speed[1]
                  ])

    result = odeint(func=overall_pend, y0=y0, t=tGrid, args=([p.mass for _, p in enumerate(particleList)],))

    for i in range(len(result[-1]) // 4):
        particles.append([
                             result[-1][i * 4],
                             result[-1][i * 4 + 1],
                             result[-1][i * 4 + 2],
                             result[-1][i * 4 + 3],
                         ])

    res = []

    for i in range(len(result)):
        tmp = []

        for j in range(len(result[i]) // 4):
            tmp2 = []
            tmp2.append(result[i][4 * j])
            tmp2.append(result[i][4 * j + 1])
            tmp2.append(result[i][4 * j + 2])
            tmp2.append(result[i][4 * j + 3])
            tmp.append(tmp2)

        res.append(tmp)

    return particles, res


def get_acceleration_verle(particleList, p_masses, index):
    """
    Getting acceleration for one point dictated by other points
    """
    res = np.array([0.0, 0.0])

    for i in range(len(particleList)):
        if i == index:
            continue

        distance = np.array([particleList[i][0] - particleList[index][0],
                             particleList[i][1] - particleList[index][1]])

        norm = math.pow(math.pow(distance[0], 2) + math.pow(distance[1], 2), 3/2)

        res += G * p_masses[i] * distance / norm

    return res


def particles_to_arrays(particleList, tGrid):
    particles = [[[0, 0, 0, 0] for _ in particleList] for _ in tGrid]
    p_masses = [p.mass for p in particleList]

    for p_i, p in enumerate(particleList):
        particles[0][p_i][0] = p.coordinates[0]
        particles[0][p_i][1] = p.coordinates[1]
        particles[0][p_i][2] = p.speed[0]
        particles[0][p_i][3] = p.speed[1]

        particles[1][p_i][0] = p.coordinates[0]
        particles[1][p_i][1] = p.coordinates[1]
        particles[1][p_i][2] = p.speed[0]
        particles[1][p_i][3] = p.speed[1]

    return particles, p_masses


def overall_verle(particleList, tGrid):
    """
    Consequent Verlet method
    """
    particles, p_masses = particles_to_arrays(particleList, tGrid)
    acceleration_list = [[0, 0] for _ in range(len(particles[0]))]
    delta_t = tGrid[1] - tGrid[0]

    for t_i, _ in enumerate(tGrid):
        if t_i == 0:
            continue

        #print(particles)

        for p_i in range(len(particleList)):
            old_acceleration = acceleration_list[p_i]

            particles[t_i][p_i][0] += particles[t_i][p_i][2] * delta_t + old_acceleration[0] / 2 * (delta_t ** 2)
            particles[t_i][p_i][1] += particles[t_i][p_i][3] * delta_t + old_acceleration[1] / 2 * (delta_t ** 2)

        new_acceleration_list = []

        for p_i in range(len(particles[t_i])):
            new_acceleration_list.append(get_acceleration_verle(particles[t_i], p_masses, p_i))

            particles[t_i][p_i][2] += (acceleration_list[p_i][0] + new_acceleration_list[p_i][0]) / 2 * delta_t
            particles[t_i][p_i][3] += (acceleration_list[p_i][1] + new_acceleration_list[p_i][1]) / 2 * delta_t

        acceleration_list = copy.deepcopy(new_acceleration_list)

        if t_i < len(tGrid) - 1:
            for p_i in range(len(particles[t_i])):
                particles[t_i + 1][p_i][0] = particles[t_i][p_i][0]
                particles[t_i + 1][p_i][1] = particles[t_i][p_i][1]
                particles[t_i + 1][p_i][2] = particles[t_i][p_i][2]
                particles[t_i + 1][p_i][3] = particles[t_i][p_i][3]

    return particles[-1], particles
