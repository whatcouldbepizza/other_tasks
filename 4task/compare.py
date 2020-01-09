import sys
import json

import numpy as np
import datetime
import time
import matplotlib.pyplot as plt

from numpy.random import uniform

from calculations import supercopy, to_particle_list, overall_odeint, overall_verle
from calculations_threading import overall_verle_threading
from calculations_multiprocessing import overall_verle_multiprocessing
from calculations_cython import overall_verle_cython
from calculations_opencl import overall_verle_opencl

from classes import Particle


def print_particle_list(lst=None):
    for elem in lst:
        print(elem.coordinates)

    print("-----")


def compare(particleList):
    """
    Compare odeint and Verle methods
    """
    T = 1
    delta_t = 0.5
    tGrid = np.linspace(0, T, T / delta_t + 1)

    #result_list = []

    #if len(particleList) == 1:
    #    first_result = \
    #    [
    #        particleList[0].coordinates[0],
    #        particleList[0].coordinates[1],
    #        particleList[0].speed[0],
    #        particleList[0].speed[1]
    #    ]
    #    result_list.append(first_result)

    #    for i, _ in enumerate(tGrid):
    #        partial_result = \
    #        [
    #            result_list[-1][0] + result_list[-1][2],
    #            result_list[-1][1] + result_list[-1][3],
    #            result_list[-1][2],
    #            result_list[-1][3]
    #        ]
    #        result_list.append(partial_result)
    #        # TODO: do something

    odeint_list = supercopy(particleList)
    verle_list = supercopy(particleList)

    start_time = datetime.datetime.now()
    odeint_result, all_odeint = overall_odeint(odeint_list, tGrid)
    print("Odeint time: " + str(datetime.datetime.now() - start_time))

    start_time = datetime.datetime.now()
    verle_result, all_verle = overall_verle(verle_list, tGrid)
    print("Verle time: " + str(datetime.datetime.now() - start_time))

    #print("odeint: " + str(odeint_result))
    #print("\n\n\nverle: " + str(verle_result))


def initialize_solar_system(data_file="solar_system.json"):
    """
    Function that prepares solar system example
    """
    particleList = []

    with open(data_file, "r") as descr:
        text_content = descr.read()

    json_content = json.loads(text_content)

    for val in json_content["particles"].values():

        particle = Particle(coordinates=[val["x"], val["y"]],
                            speed=[val["u"], val["v"]],
                            mass=val["m"],
                            color=val["color"],
                            living_time=val["lifetime"])

        particleList.append(particle)

    solar_mode = True

    return particleList


def generate_random_particles(count):
    result = []

    for _ in range(count):
        p = Particle(
            coordinates=[uniform(-100, 100), uniform(-100, 100)],
            speed=[uniform(-100, 100), uniform(-100, 100)],
            mass=uniform(10 ** 3, 10 ** 5)
        )

        result.append(p)

    return result


def compare_accuracy():
    solar_system = initialize_solar_system()

    odeint_list_0 = supercopy(solar_system)
    verle_list_0 = supercopy(solar_system)
    verle_list_t_0 = supercopy(solar_system)
    verle_list_m_0 = supercopy(solar_system)
    verle_list_c_0 = supercopy(solar_system)
    verle_list_o_0 = supercopy(solar_system)

    tGrid = np.linspace(0, 5, 500)

    odeint_list = overall_odeint(odeint_list_0, tGrid)[1]
    verle_list = overall_verle(verle_list_0, tGrid)[1]
    verle_list_t = overall_verle_threading(verle_list_t_0, tGrid)[1]
    verle_list_m = overall_verle_multiprocessing(verle_list_m_0, tGrid)[1]
    verle_list_c = overall_verle_cython(verle_list_c_0, tGrid)[1]
    verle_list_o = overall_verle_opencl(verle_list_o_0, tGrid)[1]

    inacc_v = []
    inacc_vt = []
    inacc_vm = []
    inacc_vc = []
    inacc_vo = []

    for t in range(len(tGrid)):
        metric_v = .0
        metric_vt = .0
        metric_vm = .0
        metric_vc = .0
        metric_vo = .0

        for p_o, p_v, p_vt, p_vm, p_vc, p_vo in zip(odeint_list[t], verle_list[t], verle_list_t[t],
                                                    verle_list_m[t], verle_list_c[t], verle_list_o[t]):
            dist_v = (np.array(p_o[:2]) - np.array(p_v[:2]))
            dist_vt = (np.array(p_o[:2]) - np.array(p_vt[:2]))
            dist_vm = (np.array(p_o[:2]) - np.array(p_vm[:2]))
            dist_vc = (np.array(p_o[:2]) - np.array(p_vc[:2]))
            dist_vo = (np.array(p_o[:2]) - np.array(p_vo[:2]))

            metric_v += np.linalg.norm(dist_v)
            metric_vt += np.linalg.norm(dist_vt)
            metric_vm += np.linalg.norm(dist_vm)
            metric_vc += np.linalg.norm(dist_vc)
            metric_vo += np.linalg.norm(dist_vo)

        inacc_v.append(metric_v)
        inacc_vt.append(metric_vt)
        inacc_vm.append(metric_vm)
        inacc_vc.append(metric_vc)
        inacc_vo.append(metric_vo)

    plt.plot(tGrid, inacc_v, label='Verle')
    plt.plot(tGrid, inacc_vt, label='Threading')
    plt.plot(tGrid, inacc_vm, label='Multiprocessing')
    plt.plot(tGrid, inacc_vc, label='Cython')
    plt.plot(tGrid, inacc_vo, label='OpenCL')

    plt.legend()
    plt.show()


def compare_performance():
    counts = [10, 20]
    labels = ['Verle', 'Threading', 'Multiprocessing', 'Cython', 'OpenCL']

    m_lists = {
        overall_verle: [],
        overall_verle_threading: [],
        overall_verle_multiprocessing: [],
        overall_verle_cython: [],
        overall_verle_opencl: []
    }

    tGrid = np.linspace(0, 5, 500)

    for count in counts:
        particleList = generate_random_particles(count)

        for i, method in enumerate(m_lists.keys()):
            start_time = time.time()
            particle_list = method(supercopy(particleList), tGrid)[1]
            total = time.time() - start_time

            m_lists[method].append(total)

            print(labels[i] + " done computing " + str(count) + ": " + str(total))

    for i, lst in zip(range(len(labels)), m_lists.values()):
        plt.plot(counts, lst, label=labels[i])

    plt.legend()
    plt.show()


if __name__ == "__main__":
    if sys.argv[1] == "acc":
        compare_accuracy()
    elif sys.argv[1] == "perf":
        compare_performance()
