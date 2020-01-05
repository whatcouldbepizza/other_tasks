import numpy as np
import datetime
import matplotlib.pyplot as plt

from calculations import supercopy, overall_odeint, overall_verle


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
