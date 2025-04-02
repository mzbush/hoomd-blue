// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/CollisionMethod.cuh
 * \brief Declaration of CUDA kernels for mpcd::CollisionMethod
 */

#include "hoomd/BoxDim.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/Index1D.h"

namespace hoomd
    {
namespace mpcd
    {
namespace kernel
    {

void gpu_check_collision_warnings(Scalar4 d_initial_velo,
                                  const Scalar4* d_vel_embed,
                                  const unsigned int* d_embed_group,
                                  const unsigned int* d_body,
                                  const unsigned int* d_rtag,
                                  const uint64_t timestep);

void gpu_store_initial_embedded_group_velocities(Scalar4 d_initial_velo,
                                                 const Scalar4* d_vel_embed,
                                                 const unsigned int* d_embed_group,
                                                 const uint64_t timestep);

void gpu_accumulate_rigid_body_momenta(Scalar3 d_linmom_accum,
                                       Scalar3 d_angmom_accum,
                                       const Scalar4 d_initial_velo,
                                       const unsigned int* d_embed_group,
                                       const Scalar4* d_postype,
                                       const Scalar4* d_velocity,
                                       const int3* d_image,
                                       const unsigned int* d_body,
                                       const unsigned int* d_rtag,
                                       const BoxDim global_box,
                                       const uint64_t timestep);

void gpu_transfer_rigid_body_momenta(Scalar3 d_linmom_accum,
                                     Scalar3 d_angmom_accum,
                                     Scalar4* d_velocity,
                                     const Scalar4* d_orientation,
                                     Scalar4* d_angmom,
                                     const Scalar3* d_inertia,
                                     const unsigned int* d_body,
                                     const unsigned int* d_rtag,
                                     const uint64_t timestep);

    } // end namespace kernel
    } // end namespace mpcd
    } // end namespace hoomd
