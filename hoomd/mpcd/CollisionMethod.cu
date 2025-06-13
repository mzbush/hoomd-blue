// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/CollisionMethod.cu
 * \brief Defines GPU functions and kernels used by mpcd::CollisionMethod
 */

#include "CollisionMethod.cuh"
#include "hoomd/ParticleData.cuh"

namespace hoomd
    {
namespace mpcd
    {
namespace gpu
    {
namespace kernel
    {

__global__ void store_initial_embedded_group_velocities(Scalar4* d_initial_vel,
                                                        const Scalar4* d_velocity,
                                                        const unsigned int* d_embed_group,
                                                        const unsigned int num_group)
    {
    // one thread per particle
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_group)
        return;

    // store the initial velocity
    const unsigned int particle_idx = d_embed_group[idx];
    d_initial_vel[idx] = d_velocity[particle_idx];
    }

__global__ void draw_velocities_constituent_particles(Scalar3* d_linmom_accum,
                                                      Scalar3* d_angmom_accum,
                                                      Scalar4* d_alt_vel,
                                                      const Scalar4* d_postype,
                                                      const Scalar4* d_velocity,
                                                      const int3* d_image,
                                                      const unsigned int* d_body,
                                                      const unsigned int* d_tag,
                                                      const unsigned int* d_rtag,
                                                      const BoxDim& global_box,
                                                      const uint64_t timestep,
                                                      const uint16_t seed,
                                                      const Scalar T,
                                                      const unsigned int num_total)
    {
    // one thread per particle
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_total)
        return;

    // check if in a rigid body
    const unsigned int central_tag = d_body[idx];
    if (central_tag >= MIN_FLOPPY)
        {
        return;
        }
    const unsigned int central_idx = d_rtag[central_tag];
    // if the central particle is not local, cannot read or write to it.
    assert(central_idx != NOT_LOCAL);

    // do not thermalize central particle
    if (idx == central_idx)
        {
        return;
        }
    }

__global__ void get_net_velocity_rigid_body(Scalar3* d_linmom_accum,
                                            Scalar3* d_angmom_accum,
                                            Scalar4* d_alt_vel,
                                            const Scalar4* d_velocity,
                                            const Scalar4* d_orientation,
                                            const Scalar3* d_inertia,
                                            const unsigned int* d_body,
                                            const unsigned int* d_rtag,
                                            const unsigned int num_total)
    {
    // one thread per particle
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_total)
        return;

    // check if in a rigid body
    const unsigned int central_tag = d_body[idx];
    if (central_tag >= MIN_FLOPPY)
        {
        return;
        }
    const unsigned int central_idx = d_rtag[central_tag];
    // if the central particle is not local, cannot read or write to it.
    assert(central_idx != NOT_LOCAL);

    // must be central particle
    if (idx != central_idx)
        {
        return;
        }
    }

__global__ void apply_thermalized_velocity_vectors(Scalar3* d_linmom_accum,
                                                   Scalar3* d_angmom_accum,
                                                   Scalar4* d_alt_vel,
                                                   const Scalar4* d_postype,
                                                   const Scalar4* d_velocity,
                                                   const int3* d_image,
                                                   const unsigned int* d_body,
                                                   const unsigned int* d_rtag,
                                                   const BoxDim& global_box,
                                                   const unsigned int num_total)
    {
    // one thread per particle
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_total)
        return;

    // check if in a rigid body
    const unsigned int central_tag = d_body[idx];
    if (central_tag >= MIN_FLOPPY)
        {
        return;
        }
    const unsigned int central_idx = d_rtag[central_tag];
    // if the central particle is not local, cannot read or write to it.
    assert(central_idx != NOT_LOCAL);

    // do not thermalize central particle
    if (idx == central_idx)
        {
        return;
        }
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
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_group)
        return;

    // get the index from the embedded group and check if in a rigid body
    unsigned int particle_idx = d_embed_group[idx];
    const unsigned int central_tag = d_body[particle_idx];
    if (central_tag >= MIN_FLOPPY)
        {
        return;
        }
    const unsigned int central_idx = d_rtag[central_tag];
    // if the central particle is not local, cannot read or write to it.
    assert(central_idx != NOT_LOCAL);

    // collision on central particle itself already taken care of by collision rule
    if (particle_idx == central_idx)
        {
        return;
        }
    // get velocities and masses
    const Scalar4 vel_mass_const = d_velocity[particle_idx];
    const vec3<Scalar> vel_const(vel_mass_const);
    const Scalar mass_const = vel_mass_const.w;

    // get displacement
    const Scalar4 postype_const = d_postype[particle_idx];
    const vec3<Scalar> pos_const(postype_const);
    const int3 img_const = d_image[particle_idx];

    const Scalar4 postype_central = d_postype[central_idx];
    const vec3<Scalar> pos_central(postype_central);
    const int3 img_central = d_image[central_idx];

    vec3<Scalar> displacement = pos_const - pos_central;
    const int3 displacement_img = make_int3(img_const.x - img_central.x,
                                            img_const.y - img_central.y,
                                            img_const.z - img_central.z);
    displacement = global_box.shift(displacement, displacement_img);

    // change in linear and angular momentum
    const vec3<Scalar> initial_vel_const(d_initial_vel[idx]);
    const vec3<Scalar> linmom_change = mass_const * (vel_const - initial_vel_const);
    const vec3<Scalar> angmom_change = cross(displacement, linmom_change);

    // accumulate onto central particle
    atomicAdd(&d_linmom_accum[central_idx].x, linmom_change.x);
    atomicAdd(&d_linmom_accum[central_idx].y, linmom_change.y);
    atomicAdd(&d_linmom_accum[central_idx].z, linmom_change.z);
    atomicAdd(&d_angmom_accum[central_idx].x, angmom_change.x);
    atomicAdd(&d_angmom_accum[central_idx].y, angmom_change.y);
    atomicAdd(&d_angmom_accum[central_idx].z, angmom_change.z);
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
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_total)
        {
        return;
        }

    // check that the particle is in a rigid body and a central particle
    const unsigned int central_tag = d_body[idx];
    if (central_tag >= MIN_FLOPPY)
        {
        return;
        }
    const unsigned int central_idx = d_rtag[central_tag];
    if (central_idx != idx)
        {
        return;
        }

    // get accumulated momentum for particle
    const Scalar3 linmom_accum(d_linmom_accum[idx]);
    const vec3<Scalar> angmom_accum(d_angmom_accum[idx]);

    // compute and store new velocity
    Scalar4 vel_mass = d_velocity[idx];
    const Scalar mass = vel_mass.w;
    if (mass > 0)
        {
        vel_mass.x += linmom_accum.x / mass;
        vel_mass.y += linmom_accum.y / mass;
        vel_mass.z += linmom_accum.z / mass;
        d_velocity[idx] = vel_mass;
        }

    // compute new angular momentum
    quat<Scalar> angmom(d_angmom[idx]);
    const quat<Scalar> orientation(d_orientation[idx]);

    // convert angular momentum to quaternion and update
    const vec3<Scalar> inertia(d_inertia[central_idx]);
    vec3<Scalar> angmom_change_body = rotate(conj(orientation), angmom_accum);
    if (inertia.x == Scalar(0))
        {
        angmom_change_body.x = Scalar(0);
        }

    if (inertia.y == Scalar(0))
        {
        angmom_change_body.y = Scalar(0);
        }

    if (inertia.z == Scalar(0))
        {
        angmom_change_body.z = Scalar(0);
        }
    angmom += Scalar(2.0) * orientation * quat(0.0, angmom_change_body);

    // save update
    d_angmom[idx] = quat_to_scalar4(angmom);
    }
    } // end namespace kernel

cudaError_t store_initial_embedded_group_velocities(Scalar4* d_initial_vel,
                                                    const Scalar4* d_velocity,
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
        d_velocity,
        d_embed_group,
        num_group);

    return cudaSuccess;
    }

cudaError_t draw_velocities_constituent_particles(Scalar3* d_linmom_accum,
                                                  Scalar3* d_angmom_accum,
                                                  Scalar4* d_alt_vel,
                                                  const Scalar4* d_postype,
                                                  const Scalar4* d_velocity,
                                                  const int3* d_image,
                                                  const unsigned int* d_body,
                                                  const unsigned int* d_tag,
                                                  const unsigned int* d_rtag,
                                                  const BoxDim& global_box,
                                                  const uint64_t timestep,
                                                  const uint16_t seed,
                                                  const Scalar T,
                                                  const unsigned int num_total,
                                                  const unsigned int block_size)
    {
    unsigned int max_block_size;
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr,
                          (const void*)mpcd::gpu::kernel::draw_velocities_constituent_particles);
    max_block_size = attr.maxThreadsPerBlock;

    unsigned int run_block_size = min(block_size, max_block_size);

    dim3 grid(num_total / run_block_size + 1);
    mpcd::gpu::kernel::draw_velocities_constituent_particles<<<grid, run_block_size>>>(
        d_linmom_accum,
        d_angmom_accum,
        d_alt_vel,
        d_postype,
        d_velocity,
        d_image,
        d_body,
        d_tag,
        d_rtag,
        global_box,
        timestep,
        seed,
        T,
        num_total);
    return cudaSuccess;
    }

cudaError_t get_net_velocity_rigid_body(Scalar3* d_linmom_accum,
                                        Scalar3* d_angmom_accum,
                                        Scalar4* d_alt_vel,
                                        const Scalar4* d_velocity,
                                        const Scalar4* d_orientation,
                                        const Scalar3* d_inertia,
                                        const unsigned int* d_body,
                                        const unsigned int* d_rtag,
                                        const unsigned int num_total,
                                        const unsigned int block_size)
    {
    unsigned int max_block_size;
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, (const void*)mpcd::gpu::kernel::get_net_velocity_rigid_body);
    max_block_size = attr.maxThreadsPerBlock;

    unsigned int run_block_size = min(block_size, max_block_size);

    dim3 grid(num_total / run_block_size + 1);
    mpcd::gpu::kernel::get_net_velocity_rigid_body<<<grid, run_block_size>>>(d_linmom_accum,
                                                                             d_angmom_accum,
                                                                             d_alt_vel,
                                                                             d_velocity,
                                                                             d_orientation,
                                                                             d_inertia,
                                                                             d_body,
                                                                             d_rtag,
                                                                             num_total);
    return cudaSuccess;
    }

cudaError_t apply_thermalized_velocity_vectors(Scalar3* d_linmom_accum,
                                               Scalar3* d_angmom_accum,
                                               Scalar4* d_alt_vel,
                                               const Scalar4* d_postype,
                                               const Scalar4* d_velocity,
                                               const int3* d_image,
                                               const unsigned int* d_body,
                                               const unsigned int* d_rtag,
                                               const BoxDim& global_box,
                                               const unsigned int num_total,
                                               const unsigned int block_size)
    {
    unsigned int max_block_size;
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr,
                          (const void*)mpcd::gpu::kernel::apply_thermalized_velocity_vectors);
    max_block_size = attr.maxThreadsPerBlock;

    unsigned int run_block_size = min(block_size, max_block_size);

    dim3 grid(num_total / run_block_size + 1);
    mpcd::gpu::kernel::apply_thermalized_velocity_vectors<<<grid, run_block_size>>>(d_linmom_accum,
                                                                                    d_angmom_accum,
                                                                                    d_alt_vel,
                                                                                    d_postype,
                                                                                    d_velocity,
                                                                                    d_image,
                                                                                    d_body,
                                                                                    d_rtag,
                                                                                    global_box,
                                                                                    num_total);
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
                                          const BoxDim& global_box,
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
