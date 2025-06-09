// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/CollisionMethod.cu
 * \brief Defines GPU functions and kernels used by mpcd::CollisionMethod
 */

#include "CollisionMethod.cuh"
#include "hoomd/ParticleData.cuh"
#include "hoomd/RNGIdentifiers.h"
#include "hoomd/RandomNumbers.h"

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

__global__ void accumulate_rigid_body_momenta(Scalar3* d_linmom_accum,
                                              Scalar3* d_angmom_accum,
                                              Scalar* d_ke_accum,
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

    // change in kinetic energy
    const Scalar ke_change
        = Scalar(0.5) * mass_const
          * (dot(vel_const, vel_const) - dot(initial_vel_const, initial_vel_const));

    // accumulate onto central particle
    atomicAdd(&d_linmom_accum[central_idx].x, linmom_change.x);
    atomicAdd(&d_linmom_accum[central_idx].y, linmom_change.y);
    atomicAdd(&d_linmom_accum[central_idx].z, linmom_change.z);
    atomicAdd(&d_angmom_accum[central_idx].x, angmom_change.x);
    atomicAdd(&d_angmom_accum[central_idx].y, angmom_change.y);
    atomicAdd(&d_angmom_accum[central_idx].z, angmom_change.z);
    atomicAdd(&d_ke_accum[central_idx], ke_change);
    }

__global__ void transfer_rigid_body_momenta(Scalar3* d_linmom_accum,
                                            Scalar3* d_angmom_accum,
                                            Scalar* d_ke_accum,
                                            Scalar4* d_velocity,
                                            const Scalar4* d_orientation,
                                            Scalar4* d_angmom,
                                            const Scalar3* d_inertia,
                                            const unsigned int* d_body,
                                            const unsigned int* d_rtag,
                                            const unsigned int num_total,
                                            const uint16_t seed,
                                            const uint64_t timestep,
                                            const Scalar T_set,
                                            const unsigned int n_dimensions,
                                            uint3* d_errors)
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

    // check for zero moment of inertia
    const vec3<Scalar> I(d_inertia[central_idx]);
    bool x_zero, y_zero, z_zero;
    x_zero = (I.x == 0);
    y_zero = (I.y == 0);
    z_zero = (I.z == 0);

    // get accumulated momentum for particle
    const Scalar3 linmom_accum(d_linmom_accum[idx]);
    const vec3<Scalar> angmom_accum(d_angmom_accum[idx]);
    const Scalar ke_accum(d_ke_accum[idx]);

    // get central velocity
    Scalar4 vel_mass = d_velocity[idx];
    const Scalar mass = vel_mass.w;

    // compute initial kinetic energy
    Scalar ke_tra_change
        = -Scalar(0.5) * mass
          * (vel_mass.x * vel_mass.x + vel_mass.y * vel_mass.y + vel_mass.z * vel_mass.z);

    // compute and store change in velocity
    if (mass > 0)
        {
        vel_mass.x += linmom_accum.x / mass;
        vel_mass.y += linmom_accum.y / mass;
        vel_mass.z += linmom_accum.z / mass;
        d_velocity[idx] = vel_mass;
        }

    ke_tra_change
        += Scalar(0.5) * mass
           * (vel_mass.x * vel_mass.x + vel_mass.y * vel_mass.y + vel_mass.z * vel_mass.z);

    // compute new angular momentum
    quat<Scalar> angmom(d_angmom[idx]);
    const quat<Scalar> orientation(d_orientation[idx]);

    // convert angular momentum to quaternion and update
    const vec3<Scalar> inertia(d_inertia[central_idx]);
    vec3<Scalar> angmom_change_body = rotate(conj(orientation), angmom_accum);
    if (x_zero)
        {
        angmom_change_body.x = Scalar(0);
        }

    if (y_zero)
        {
        angmom_change_body.y = Scalar(0);
        }

    if (z_zero)
        {
        angmom_change_body.z = Scalar(0);
        }

    // calculate scaling factor
    Scalar a = 0;
    Scalar b = 0;
    Scalar c = 0;
    Scalar s = 0;

    const quat<Scalar> initial_angmom_body = Scalar(0.5) * conj(orientation) * angmom;
    if (!x_zero)
        {
        a += angmom_change_body.x * angmom_change_body.x / I.x;
        b += initial_angmom_body.v.x * angmom_change_body.x / I.x;
        }

    if (!y_zero)
        {
        a += angmom_change_body.y * angmom_change_body.y / I.y;
        b += initial_angmom_body.v.y * angmom_change_body.y / I.y;
        }

    if (!z_zero)
        {
        a += angmom_change_body.z * angmom_change_body.z / I.z;
        b += initial_angmom_body.v.z * angmom_change_body.z / I.z;
        }
    a *= Scalar(0.5);
    c = ke_tra_change - ke_accum;

    // check if there are imaginary roots
    Scalar d = b * b - 4 * a * c;
    if (d < 0.0)
        {
        (*d_errors).x = 1;
        return;
        }

    // choose the root for the scaling factor
    Scalar root1 = (-b - sqrt(d)) / (2 * a);
    Scalar root2 = (-b + sqrt(d)) / (2 * a);

    if (root1 <= 0.0 && root2 <= 0.0)
        {
        (*d_errors).y = 1;
        return;
        }
    else if (root1 > 0 && root2 > 0)
        {
        s = (fabs(root1 - 1) > fabs(root2 - 1)) ? root2 : root1;
        }
    else
        {
        s = (root1 <= 0) ? root2 : root1;
        }

    // scale change in angular momentum
    angmom_change_body.x *= s;
    angmom_change_body.y *= s;
    angmom_change_body.z *= s;

    // add scale factor to final angular momentum
    vec3<Scalar> new_angmom = initial_angmom_body.v;
    new_angmom.x += angmom_change_body.x;
    new_angmom.y += angmom_change_body.y;
    new_angmom.z += angmom_change_body.z;

    Scalar new_rot_ke = 0;
    if (!x_zero)
        {
        new_rot_ke += Scalar(0.5) * new_angmom.x * new_angmom.x / I.x;
        }
    if (!y_zero)
        {
        new_rot_ke += Scalar(0.5) * new_angmom.y * new_angmom.y / I.y;
        }
    if (!z_zero)
        {
        new_rot_ke += Scalar(0.5) * new_angmom.z * new_angmom.z / I.z;
        }

    hoomd::RandomGenerator rng(hoomd::Seed(hoomd::RNGIdentifier::CollisionMethod, timestep, seed),
                               hoomd::Counter(idx));
    const double alpha = n_dimensions / (double)2.0;

    hoomd::GammaDistribution<double> gamma_gen(alpha, T_set);
    const Scalar random_ke = Scalar(gamma_gen(rng));
    const Scalar scale = sqrt(random_ke / new_rot_ke);

    new_angmom.x *= scale;
    new_angmom.y *= scale;
    new_angmom.z *= scale;
    // rotate to space frame
    quat<Scalar> new_angmom_space = Scalar(2.0) * orientation * quat(0.0, new_angmom);

    // save update
    d_angmom[idx] = quat_to_scalar4(new_angmom_space);
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

cudaError_t accumulate_rigid_body_momenta(Scalar3* d_linmom_accum,
                                          Scalar3* d_angmom_accum,
                                          Scalar* d_ke_accum,
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
                                                                               d_ke_accum,
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
                                        Scalar* d_ke_accum,
                                        Scalar4* d_velocity,
                                        const Scalar4* d_orientation,
                                        Scalar4* d_angmom,
                                        const Scalar3* d_inertia,
                                        const unsigned int* d_body,
                                        const unsigned int* d_rtag,
                                        const unsigned int num_total,
                                        const uint16_t seed,
                                        const uint64_t timestep,
                                        const Scalar T_set,
                                        const unsigned int n_dimensions,
                                        uint3* d_errors,
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
                                                                             d_ke_accum,
                                                                             d_velocity,
                                                                             d_orientation,
                                                                             d_angmom,
                                                                             d_inertia,
                                                                             d_body,
                                                                             d_rtag,
                                                                             num_total,
                                                                             seed,
                                                                             timestep,
                                                                             T_set,
                                                                             n_dimensions,
                                                                             d_errors);

    return cudaSuccess;
    }
    } // end namespace gpu
    } // end namespace mpcd
    } // end namespace hoomd
