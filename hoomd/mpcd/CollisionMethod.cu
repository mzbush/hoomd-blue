// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/CollisionMethod.cu
 * \brief Defines GPU functions and kernels used by mpcd::CollisionMethod
 */

#include "CollisionMethod.cuh"

namespace hoomd
    {
namespace mpcd
    {
namespace gpu
    {
namespace kernel
    {

__global__ void store_initial_embedded_group_velocities(Scalar4* d_initial_vel,
                                                        const Scalar4* d_vel_embed,
                                                        const unsigned int* d_embed_group,
                                                        const unsigned int num_group)
    {
    // one thread per particle
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_group)
        return;
    }

__global__ void accumulate_rigid_body_momenta(Scalar3* d_linmom_accum,
                                              Scalar3* d_angmom_accum,
                                              const Scalar4* d_initial_vel,
                                              const unsigned int* d_embed_group,
                                              const Scalar4* d_postype,
                                              const Scalar4* d_velocity,
                                              const int3* d_image,
                                              const unsigned int* d_body,
                                              const unsigned int* d_rtag,
                                              const BoxDim global_box,
                                              const unsigned int num_group)
    {
    // one thread per particle
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_group)
        return;
    }

__global__ void transfer_rigid_body_momenta(Scalar3* d_linmom_accum,
                                            Scalar3* d_angmom_accum,
                                            Scalar4* d_velocity,
                                            const Scalar4* d_orientation,
                                            Scalar4* d_angmom,
                                            const Scalar3* d_inertia,
                                            const unsigned int* d_body,
                                            const unsigned int* d_rtag,
                                            const unsigned int num_total)
    {
    // one thread per particle
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_total)
        return;
    }

    } // end namespace kernel

cudaError_t store_initial_embedded_group_velocities(Scalar4* d_initial_vel,
                                                    const Scalar4* d_vel_embed,
                                                    const unsigned int* d_embed_group,
                                                    const unsigned int num_group,
                                                    const unsigned int block_size)
    {
    unsigned int max_block_size;
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr,
                          (const void*)mpcd::gpu::kernel::store_initial_embedded_group_velocities);
    max_block_size = attr.maxThreadsPerBlock;

    unsigned int run_block_size = min(block_size, max_block_size);

    dim3 grid(num_group / run_block_size + 1);
    mpcd::gpu::kernel::store_initial_embedded_group_velocities<<<grid, run_block_size>>>(
        d_initial_vel,
        d_vel_embed,
        d_embed_group,
        num_group);

    return cudaSuccess;
    }

cudaError_t accumulate_rigid_body_momenta(Scalar3* d_linmom_accum,
                                          Scalar3* d_angmom_accum,
                                          const Scalar4* d_initial_vel,
                                          const unsigned int* d_embed_group,
                                          const Scalar4* d_postype,
                                          const Scalar4* d_velocity,
                                          const int3* d_image,
                                          const unsigned int* d_body,
                                          const unsigned int* d_rtag,
                                          const BoxDim global_box,
                                          const unsigned int num_group,
                                          const unsigned int block_size)
    {
    unsigned int max_block_size;
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, (const void*)mpcd::gpu::kernel::accumulate_rigid_body_momenta);
    max_block_size = attr.maxThreadsPerBlock;

    unsigned int run_block_size = min(block_size, max_block_size);

    dim3 grid(num_group / run_block_size + 1);
    mpcd::gpu::kernel::accumulate_rigid_body_momenta<<<grid, run_block_size>>>(d_linmom_accum,
                                                                               d_angmom_accum,
                                                                               d_initial_vel,
                                                                               d_embed_group,
                                                                               d_postype,
                                                                               d_velocity,
                                                                               d_image,
                                                                               d_body,
                                                                               d_rtag,
                                                                               global_box,
                                                                               num_group);

    return cudaSuccess;
    }

cudaError_t transfer_rigid_body_momenta(Scalar3* d_linmom_accum,
                                        Scalar3* d_angmom_accum,
                                        Scalar4* d_velocity,
                                        const Scalar4* d_orientation,
                                        Scalar4* d_angmom,
                                        const Scalar3* d_inertia,
                                        const unsigned int* d_body,
                                        const unsigned int* d_rtag,
                                        const unsigned int num_total,
                                        const unsigned int block_size)
    {
    unsigned int max_block_size;
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, (const void*)mpcd::gpu::kernel::transfer_rigid_body_momenta);
    max_block_size = attr.maxThreadsPerBlock;

    unsigned int run_block_size = min(block_size, max_block_size);

    dim3 grid(num_total / run_block_size + 1);
    mpcd::gpu::kernel::transfer_rigid_body_momenta<<<grid, run_block_size>>>(d_linmom_accum,
                                                                             d_angmom_accum,
                                                                             d_velocity,
                                                                             d_orientation,
                                                                             d_angmom,
                                                                             d_inertia,
                                                                             d_body,
                                                                             d_rtag,
                                                                             num_total);

    return cudaSuccess;
    }
    } // end namespace gpu
    } // end namespace mpcd
    } // end namespace hoomd
