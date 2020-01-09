from calculations import particles_to_arrays

import pyopencl as cl
import numpy as np

from pyopencl import cltypes


ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

mf = cl.mem_flags

prg = cl.Program(
    ctx,
    """
    #ifdef cl_khr_fp64
        #pragma OPENCL EXTENSION cl_khr_fp64 : enable
    #elif defined(cl_amd_fp64)
        #pragma OPENCL EXTENSION cl_amd_fp64 : enable
    #else
        #error "Double precision floating point not supported by OpenCL implementation."
    #endif

    void get_acceleration(
        __global double *particles,
        __global double *p_masses,
        int t_i,
        int index,
        int Pi,
        __global double *acc_tmp
    )
    {
        double G = 6.6743015 * 0.0000000001;
        double norm;
        double distance[2];

        acc_tmp[0] = 0.0;
        acc_tmp[1] = 0.0;

        for (int p_i = 0; p_i < Pi; ++p_i)
        {
            if (p_i != index)
            {
                distance[0] = particles[(Pi * t_i + p_i) * 4 + 0] - particles[(Pi * t_i + index) * 4 + 0];
                distance[1] = particles[(Pi * t_i + p_i) * 4 + 1] - particles[(Pi * t_i + index) * 4 + 1];

                norm = sqrt(distance[0] * distance[0] + distance[1] * distance[1]);

                acc_tmp[0] += G * p_masses[p_i] * distance[0] / norm;
                acc_tmp[1] += G * p_masses[p_i] * distance[1] / norm;
            }
        }
    }

    __kernel void verle_opencl(
        __global double *particles,
        __global double *p_masses,
        __global double *accelerations,
        __global double *acc_tmp,
        __global int *Ti_buf,
        __global int *Pi_buf,
        __global double *dt_buf,
        __global double *result
    )
    {
        int Ti = *Ti_buf;
        int Pi = *Pi_buf;
        double dt = *dt_buf;

        for (int t_i = 1; t_i < Ti; ++t_i)
        {
            for (int p_i = 0; p_i < Pi; ++p_i)
            {
                particles[(Pi * t_i + p_i) * 4 + 0] += particles[(Pi * t_i + p_i) * 4 + 2] * dt
                                                       + accelerations[p_i * 2 + 0] / 2 * dt * dt;

                particles[(Pi * t_i + p_i) * 4 + 1] += particles[(Pi * t_i + p_i) * 4 + 3] * dt
                                                       + accelerations[p_i * 2 + 2] / 2 * dt * dt;
            }

            for (int p_i = 0; p_i < Pi; ++p_i)
            {
                get_acceleration(particles, p_masses, t_i, p_i, Pi, acc_tmp);

                particles[(Pi * t_i + p_i) * 4 + 2] += (accelerations[p_i * 2 + 0] + acc_tmp[0]) / 2 * dt;
                particles[(Pi * t_i + p_i) * 4 + 3] += (accelerations[p_i * 2 + 1] + acc_tmp[1]) / 2 * dt;

                accelerations[p_i * 2 + 0] = acc_tmp[0];
                accelerations[p_i * 2 + 1] = acc_tmp[1];
            }

            if (t_i < Ti - 1)
            {
                for (int p_i = 0; p_i < Pi; ++p_i)
                {
                    particles[(Pi * (t_i + 1) + p_i) * 4 + 0] = particles[(Pi * t_i + p_i) * 4 + 0];
                    particles[(Pi * (t_i + 1) + p_i) * 4 + 1] = particles[(Pi * t_i + p_i) * 4 + 1];
                    particles[(Pi * (t_i + 1) + p_i) * 4 + 2] = particles[(Pi * t_i + p_i) * 4 + 2];
                    particles[(Pi * (t_i + 1) + p_i) * 4 + 3] = particles[(Pi * t_i + p_i) * 4 + 3];
                }
            }
        }

        for (int t_i = 0; t_i < Ti; ++t_i)
            for (int p_i = 0; p_i < Pi; ++p_i)
            {
                result[(Pi * t_i + p_i) * 4 + 0] = particles[(Pi * t_i + p_i) * 4 + 0];
                result[(Pi * t_i + p_i) * 4 + 1] = particles[(Pi * t_i + p_i) * 4 + 1];
                result[(Pi * t_i + p_i) * 4 + 2] = particles[(Pi * t_i + p_i) * 4 + 2];
                result[(Pi * t_i + p_i) * 4 + 3] = particles[(Pi * t_i + p_i) * 4 + 3];
            }
    }
    """
)


def overall_verle_opencl(particleList, tGrid):
    particles, p_masses = particles_to_arrays(particleList, tGrid)

    Ti = np.array(len(tGrid))
    Pi = np.array(len(particleList))
    dt = np.array(tGrid[1] - tGrid[0])

    particles = np.array(particles, dtype=cltypes.double)
    p_masses = np.array(p_masses, dtype=cltypes.double)

    acceleration_list = np.zeros((Ti, Pi, 2), dtype=cltypes.double)
    acc_tmp = np.zeros((2,), dtype=cltypes.double)

    result = np.zeros((Ti, Pi, 4), dtype=cltypes.double)

    Ti_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=Ti)
    Pi_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=Pi)
    dt_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=dt)

    particles_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=particles)
    p_masses_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=p_masses)

    acceleration_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=acceleration_list)
    acc_tmp_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=acc_tmp)

    result_out_buf = cl.Buffer(ctx, mf.WRITE_ONLY, result.nbytes)

    prg.build()

    prg.verle_opencl(
        queue,
        (1,),
        None,
        particles_buf,
        p_masses_buf,
        acceleration_buf,
        acc_tmp_buf,
        Ti_buf,
        Pi_buf,
        dt_buf,
        result_out_buf
    )

    cl.enqueue_copy(queue, result, result_out_buf)

    return result[-1], result
