// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/CellListGPU.cu
 * \brief Defines GPU functions and kernels used by mpcd::CellListGPU
 */
#include <hipcub/hipcub.hpp>

#include "CellListGPU.cuh"

#include <thrust/sort.h>

namespace hoomd
    {
namespace mpcd
    {
namespace gpu
    {
namespace kernel
    {
//! Kernel to compute the MPCD cell list on the GPU
/*!
 * \param d_cell_np Array of number of particles per cell
 * \param d_cell_vel Array of center of mass velocity + cell mass per cell
 * \param d_cell_energy Array of per cell kinetic energy
 * \param d_conditions Conditions flags for error reporting
 * \param d_vel MPCD particle velocities
 * \param mpcd_mass MPCD particle mass
 * \param d_embed_cell_ids Cell indexes of embedded particles
 * \param d_pos MPCD particle positions
 * \param d_pos_embed Particle positions
 * \param d_vel_embed Particle velocities
 * \param d_embed_member_idx Indexes of embedded particles in \a d_pos_embed
 * \param periodic Flags if local simulation is periodic
 * \param origin_idx Global origin index for the local box
 * \param grid_shift Random grid shift vector
 * \param global_box Global simulation box
 * \param global_cell_dim Global cell dimensions, no padding
 * \param cell_indexer 3D indexer for cell id
 * \param cell_indexer 3D indexer for global cell id
 * \param d_mpcd_comm_key directions to send MPCD particles as ghosts
 * \param rank_size the size of the local rank
 * \param is_decomposition whether there is domain decomposition
 * \param N_mpcd Number of MPCD particles
 * \param N_tot Total number of particle (MPCD + embedded)
 * \param need_energy True if computing the energy
 *
 * \b Implementation
 * One thread is launched per particle. The particle is floored into a bin subject to a random grid
 * shift. The number of particles in that bin is atomically incremented and the contribution of the
 * particle's properties to the cell's properties is added to a running sum. If the addition of the
 * particle will not overflow the allocated memory, the particle is written into that bin.
 * Otherwise, a flag is set to resize the cell list and recompute. The MPCD particle's cell id is
 * stashed into the velocity array.
 */
__global__ void compute_cell_list(unsigned int* d_cell_np,
                                  double4* d_cell_vel,
                                  double* d_cell_energy,
                                  uint3* d_conditions,
                                  Scalar4* d_vel,
                                  double mpcd_mass,
                                  unsigned int* d_embed_cell_ids,
                                  const Scalar4* d_pos,
                                  const Scalar4* d_pos_embed,
                                  const Scalar4* d_vel_embed,
                                  const unsigned int* d_embed_member_idx,
                                  const uchar3 periodic,
                                  const uint3 origin_idx,
                                  const Scalar3 grid_shift,
                                  const BoxDim global_box,
                                  const uint3 global_cell_dim,
                                  const Index3D cell_indexer,
                                  const Index3D global_cell_indexer,
                                  uint2* d_mpcd_comm_key,
                                  const uint3 rank_size,
                                  const bool is_decomposition,
                                  const unsigned int N_mpcd,
                                  const unsigned int N_tot,
                                  const bool need_energy)
    {
    // one thread per particle
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_tot)
        return;

    Scalar4 postype_i;
    Scalar4 vel_mass_i;
    double mass_i;
    if (idx < N_mpcd)
        {
        postype_i = d_pos[idx];
        vel_mass_i = d_vel[idx];
        mass_i = mpcd_mass;
        }
    else
        {
        postype_i = d_pos_embed[d_embed_member_idx[idx - N_mpcd]];
        vel_mass_i = d_vel_embed[d_embed_member_idx[idx - N_mpcd]];
        mass_i = vel_mass_i.w;
        }
    const Scalar3 pos_i = make_scalar3(postype_i.x, postype_i.y, postype_i.z);
    const double3 vel_i = make_double3(vel_mass_i.x, vel_mass_i.y, vel_mass_i.z);

    if (isnan(pos_i.x) || isnan(pos_i.y) || isnan(pos_i.z))
        {
        (*d_conditions).y = idx + 1;
        return;
        }

    // bin particle with grid shift
    const Scalar3 fractional_pos_i = global_box.makeFraction(pos_i) - grid_shift;
    int3 global_bin = make_int3((int)std::floor(fractional_pos_i.x * global_cell_dim.x),
                                (int)std::floor(fractional_pos_i.y * global_cell_dim.y),
                                (int)std::floor(fractional_pos_i.z * global_cell_dim.z));

    // wrap cell back through the boundaries (grid shifting may send +/- 1 outside of range)
    // this is done using periodic from the global box
    if (periodic.x)
        {
        if (global_bin.x >= (int)global_cell_dim.x)
            global_bin.x -= global_cell_dim.x;
        else if (global_bin.x < 0)
            global_bin.x += global_cell_dim.x;
        }
    if (periodic.y)
        {
        if (global_bin.y >= (int)global_cell_dim.y)
            global_bin.y -= global_cell_dim.y;
        else if (global_bin.y < 0)
            global_bin.y += global_cell_dim.y;
        }
    if (periodic.z)
        {
        if (global_bin.z >= (int)global_cell_dim.z)
            global_bin.z -= global_cell_dim.z;
        else if (global_bin.z < 0)
            global_bin.z += global_cell_dim.z;
        }

    // validate and make sure no particles blew out of the global box
    if ((global_bin.x < 0 || global_bin.x >= (int)global_cell_dim.x)
        || (global_bin.y < 0 || global_bin.y >= (int)global_cell_dim.y)
        || (global_bin.z < 0 || global_bin.z >= (int)global_cell_dim.z))
        {
        (*d_conditions).z = idx + 1;
        return;
        }

    // compute the local cell
    int3 bin = make_int3(global_bin.x - origin_idx.x,
                         global_bin.y - origin_idx.y,
                         global_bin.z - origin_idx.z);

    // check if the particle is still in the local box
    const bool is_local = ((bin.x >= 0 && bin.x < (int)cell_indexer.getW())
                           && (bin.y >= 0 && bin.y < (int)cell_indexer.getH())
                           && (bin.z >= 0 && bin.z < (int)cell_indexer.getD()));

    // check if particles outside the local box is allowed
    if (!is_local && !is_decomposition)
        {
        (*d_conditions).z = idx + 1;
        return;
        }

    unsigned int bin_idx;
    if (is_local)
        {
        bin_idx = cell_indexer(bin.x, bin.y, bin.z);
        // set the MPI communication flag
#ifdef ENABLE_MPI
        if (is_decomposition && idx < N_mpcd)
            {
            d_mpcd_comm_key[idx] = make_uint2(0xffffffff, idx);
            }
#endif // ENABLE_MPI
        }
#ifdef ENABLE_MPI
    else
        {
        if (is_decomposition && idx < N_mpcd)
            {
            // determine from the bin which rank the particle's cell belongs to
            int ix = 0;
            if (bin.x >= (int)cell_indexer.getW())
                ix = (rank_size.x > 2) ? 1 : -1;
            else if (bin.x < 0)
                ix = -1;

            int iy = 0;
            if (bin.y >= (int)cell_indexer.getH())
                iy = (rank_size.y > 2) ? 1 : -1;
            else if (bin.y < 0)
                iy = -1;

            int iz = 0;
            if (bin.z >= (int)cell_indexer.getD())
                iz = (rank_size.z > 2) ? 1 : -1;
            else if (bin.z < 0)
                iz = -1;

            // get shifted direction index
            int dir = ((iz + 1) * 3 + (iy + 1)) * 3 + (ix + 1);
            dir = dir + ((ix == 1) ? -2 : 1) + ((iy == 1) ? -6 : 3) + ((iz == 1) ? -12 : 9);
            // mark particle to be sent to neighboring rank
            d_mpcd_comm_key[idx] = make_uint2(dir, idx);

            // set the bin idx to be the global index with highest bit set to 1
            bin_idx = global_cell_indexer(global_bin.x, global_bin.y, global_bin.z);
            bin_idx |= 1 << 31;
            }
        }
#endif // ENABLE_MPI

    // stash the current particle bin into the velocity array
    if (idx < N_mpcd)
        {
        d_vel[idx].w = __int_as_scalar(bin_idx);
        }
    else
        {
        d_embed_cell_ids[idx - N_mpcd] = bin_idx;
        }

    if (!is_local)
        {
        return;
        }

    // compute the contribution of the particle to cell properties
    const unsigned int offset = atomicInc(&d_cell_np[bin_idx], 0xffffffff);
    double4& cell_vel = d_cell_vel[bin_idx];
    atomicAdd(&cell_vel.x, vel_i.x * mass_i);
    atomicAdd(&cell_vel.y, vel_i.y * mass_i);
    atomicAdd(&cell_vel.z, vel_i.z * mass_i);
    atomicAdd(&cell_vel.w, mass_i);

    if (need_energy)
        {
        double ke = 0.5 * mass_i * (vel_i.x * vel_i.x + vel_i.y * vel_i.y + vel_i.z * vel_i.z);
        atomicAdd(&d_cell_energy[bin_idx], ke);
        }
    }

/*!
 * \param d_cell_np Array of number of particles per cell
 * \param d_cell_vel Cell velocity to finish computing
 * \param d_cell_energy Cell kinetic energy
 * \param d_cell_temp Cell temperature to finish computing
 * \param N_cells Number of cells
 * \param N_dim Number of dimensions
 * \param need_energy If true, compute the cell-level energy properties.
 *
 * \b Implementation details:
 * Using one thread per cell, the cell properties are normalized from the
 * additive contributions of the particles to their final values, e.g. the cell
 * velocities are computed from the net momentum of the cell divided by the final
 * mass. The temperature of each cell is calculated.
 */
__global__ void finish_cell_properties(const unsigned int* d_cell_np,
                                       double4* d_cell_vel,
                                       const double* d_cell_energy,
                                       double* d_cell_temp,
                                       const unsigned int N_cells,
                                       const unsigned int N_dim,
                                       const bool need_energy)
    {
    // one thread per cell
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_cells)
        return;

    const double4 vel_cell = d_cell_vel[idx];
    double3 vel_cm = make_double3(vel_cell.x, vel_cell.y, vel_cell.z);
    const double mass = vel_cell.w;

    // get center of mass velocity from momentum
    if (mass > 0.)
        {
        // average velocity is only defined when there is some mass in the cell
        vel_cm.x /= mass;
        vel_cm.y /= mass;
        vel_cm.z /= mass;
        }
    d_cell_vel[idx] = make_double4(vel_cm.x, vel_cm.y, vel_cm.z, mass);

    if (need_energy)
        {
        const double cell_energy = d_cell_energy[idx];
        double temp(0.0);
        const unsigned int np = d_cell_np[idx];

        if (np > 1)
            {
            const double ke_cm
                = 0.5 * mass * (vel_cm.x * vel_cm.x + vel_cm.y * vel_cm.y + vel_cm.z * vel_cm.z);
            temp = 2. * (cell_energy - ke_cm) / (N_dim * (np - 1));
            }
        d_cell_temp[idx] = temp;
        }
    }

/*!
 * \param d_tmp_thermo Temporary cell packed thermo element
 * \param d_cell_np Number of particles per cell
 * \param d_cell_vel Cell velocity
 * \param d_cell_energy Cell kinetic energy
 * \param d_cell_temperature Cell temperature
 * \param N_cells Number of cells
 * \tparam need_energy If true, compute the cell-level energy properties.
 *
 * \b Implementation details:
 * Using one thread per \a temporary cell, the cell properties are normalized
 * in a way suitable for reduction of net properties, e.g. the cell velocities
 * are converted to momentum. The temperature is set to the cell energy, and a
 * flag is set to 1 or 0 to indicate whether this cell has an energy that should
 * be used in averaging the total temperature.
 */
template<bool need_energy>
__global__ void stage_net_cell_thermo(mpcd::detail::cell_thermo_element* d_tmp_thermo,
                                      const unsigned int* d_cell_np,
                                      const double4* d_cell_vel,
                                      const double* d_cell_energy,
                                      const double* d_cell_temp,
                                      const unsigned int N_cells)
    {
    // one thread per cell
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_cells)
        return;

    const double4 vel_cell = d_cell_vel[idx];
    const double mass = vel_cell.w;

    // add accumulated momentum to net momentum
    mpcd::detail::cell_thermo_element thermo;
    thermo.momentum = make_double3(vel_cell.x * mass, vel_cell.y * mass, vel_cell.z * mass);

    if (need_energy)
        {
        if (d_cell_np[idx] > 1)
            {
            thermo.temperature = d_cell_temp[idx];
            thermo.flag = 1;
            }
        else
            {
            thermo.temperature = 0.;
            thermo.flag = 0;
            }
        thermo.energy = d_cell_energy[idx];
        }
    else
        {
        thermo.energy = 0.;
        thermo.temperature = 0.;
        thermo.flag = 0;
        }

    d_tmp_thermo[idx] = thermo;
    }

/*!
 * \param d_migrate_flag Flag signaling migration is required (output)
 * \param d_pos Embedded particle positions
 * \param d_group Indexes into \a d_pos for particles in embedded group
 * \param box Box covered by this domain
 * \param num_dim Dimensionality of system
 * \param N Number of particles in group
 *
 * \b Implementation
 * Using one thread per particle, each particle position is compared to the
 * bounds of the simulation box. If a particle lies outside the box, \a d_migrate_flag
 * has its bits set using an atomicMax transaction. The caller should then trigger
 * a communication step to migrate particles to their appropriate ranks.
 */
__global__ void cell_check_migrate_embed(unsigned int* d_migrate_flag,
                                         const Scalar4* d_pos,
                                         const unsigned int* d_group,
                                         const BoxDim box,
                                         const unsigned int num_dim,
                                         const unsigned int N)
    {
    // one thread per particle in group
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N)
        return;

    const unsigned int idx = d_group[tid];
    const Scalar4 postype = d_pos[idx];
    const Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);

    const uchar3 periodic = box.getPeriodic();
    const Scalar3 fractional_pos = box.makeFraction(pos);
    if ((!periodic.x && (fractional_pos.x >= Scalar(1.0) || fractional_pos.x < Scalar(0.0)))
        || (!periodic.y && (fractional_pos.y >= Scalar(1.0) || fractional_pos.y < Scalar(0.0)))
        || (!periodic.z && num_dim == 3
            && (fractional_pos.z >= Scalar(1.0) || fractional_pos.z < Scalar(0.0))))
        {
        atomicMax(d_migrate_flag, 1);
        }
    }

/*!
 * \param d_mpcd_comm_key directions to send MPCD particles as ghosts
 * \param d_mpcd_send_offsets starting index of points sent to each neighbor
 * \param num_mpcd_ghosts_send the total number of MPCD particles being sent
 * \param N Number of particles in group
 *
 * \b Implementation
 * Determines the starting index of the list of ghost particles in mpcd_comm_key
 * and the total number of ghost particles to be sent. This is done by checking
 * the position 1 to the left in the array and seeing if it is a different
 * direction. If so, then the current position is the start indexing of the
 * current direction.

 */
__global__ void find_num_ghost_send(uint2* d_mpcd_comm_key,
                                    unsigned int* d_mpcd_send_offsets,
                                    unsigned int& num_mpcd_ghosts_send,
                                    const unsigned int N)
    {
    // one thread per particle in group
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;

    unsigned int dir = d_mpcd_comm_key[idx].x;
    if (idx == 0)
        {
        if (dir < 27)
            {
            d_mpcd_send_offsets[dir] = 0;
            }
        return;
        }

    unsigned int left_dir = d_mpcd_comm_key[idx - 1].x;

    // exit if not at the start of a new index
    if (dir == left_dir)
        {
        return;
        }

    if (dir < 27)
        {
        d_mpcd_send_offsets[dir] = idx;
        }
    else
        {
        num_mpcd_ghosts_send = idx;
        }
    }

/*!
 * \param d_mpcd_comm_key directions to send MPCD particles as ghosts
 * \param d_vel MPCD particle velocities
 * \param d_mpcd_vel_sendbuf buffer for MPCD ghost velocities to be sent
 * \param num_mpcd_ghosts_send the total number of MPCD particles being sent
 *
 * \b Implementation
 * Fills the velocity buffer with ghost particles to send

 */
__global__ void fill_buffer(uint2* d_mpcd_comm_key,
                            Scalar4* d_vel,
                            Scalar4* d_mpcd_vel_sendbuf,
                            const unsigned int num_mpcd_ghosts_send)
    {
    // one thread per particle in group
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_mpcd_ghosts_send)
        return;

    const unsigned int particle_index = d_mpcd_comm_key[idx].y;

    // add particle data to send buffers
    const Scalar4 vel_mass = d_vel[particle_index];
    d_mpcd_vel_sendbuf[idx] = vel_mass;
    }
    } // end namespace kernel
    } // end namespace gpu
    } // end namespace mpcd

/*!
 * \param d_cell_np Array of number of particles per cell
 * \param d_cell_vel Array of center of mass velocity + cell mass per cell
 * \param d_cell_energy Array of per cell kinetic energy, temperature, and dof
 * \param d_conditions Conditions flags for error reporting
 * \param d_vel MPCD particle velocities
 * \param mpcd_mass MPCD particle mass
 * \param d_embed_cell_ids Cell indexes of embedded particles
 * \param d_pos MPCD particle positions
 * \param d_pos_embed Particle positions
 * \param d_vel_embed Particle velocities
 * \param d_embed_member_idx Indexes of embedded particles in \a d_pos_embed
 * \param periodic Flags if local simulation is periodic
 * \param origin_idx Global origin index for the local box
 * \param grid_shift Random grid shift vector
 * \param global_box Global simulation box
 * \param global_cell_dim Global cell dimensions, no padding
 * \param cell_indexer 3D indexer for cell id
 * \param N_mpcd Number of MPCD particles
 * \param N_tot Total number of particle (MPCD + embedded)
 * \param need_energy True if computing the energy
 * \param block_size Number of threads per block
 *
 * \returns cudaSuccess on completion, or an error on failure
 */
cudaError_t mpcd::gpu::compute_cell_list(unsigned int* d_cell_np,
                                         double4* d_cell_vel,
                                         double* d_cell_energy,
                                         uint3* d_conditions,
                                         Scalar4* d_vel,
                                         double mpcd_mass,
                                         unsigned int* d_embed_cell_ids,
                                         const Scalar4* d_pos,
                                         const Scalar4* d_pos_embed,
                                         const Scalar4* d_vel_embed,
                                         const unsigned int* d_embed_member_idx,
                                         const uchar3& periodic,
                                         const uint3& origin_idx,
                                         const Scalar3& grid_shift,
                                         const BoxDim& global_box,
                                         const uint3& global_cell_dim,
                                         const Index3D& cell_indexer,
                                         const Index3D& global_cell_indexer,
                                         uint2* d_mpcd_comm_key,
                                         const uint3& rank_size,
                                         const bool is_decomposition,
                                         const unsigned int N_mpcd,
                                         const unsigned int N_tot,
                                         const bool need_energy,
                                         const unsigned int block_size)
    {
    // set the number of particles in each cell to zero
    cudaError_t error
        = cudaMemset(d_cell_np, 0, sizeof(unsigned int) * cell_indexer.getNumElements());
    if (error != cudaSuccess)
        return error;
    cudaError_t error_vel
        = cudaMemset(d_cell_vel, 0, sizeof(double4) * cell_indexer.getNumElements());
    if (error_vel != cudaSuccess)
        return error_vel;
    if (need_energy)
        {
        cudaError_t error_energy
            = cudaMemset(d_cell_energy, 0, sizeof(double) * cell_indexer.getNumElements());
        if (error_energy != cudaSuccess)
            return error_energy;
        }

    unsigned int max_block_size;
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, (const void*)mpcd::gpu::kernel::compute_cell_list);
    max_block_size = attr.maxThreadsPerBlock;

    unsigned int run_block_size = min(block_size, max_block_size);
    dim3 grid(N_tot / run_block_size + 1);
    mpcd::gpu::kernel::compute_cell_list<<<grid, run_block_size>>>(d_cell_np,
                                                                   d_cell_vel,
                                                                   d_cell_energy,
                                                                   d_conditions,
                                                                   d_vel,
                                                                   mpcd_mass,
                                                                   d_embed_cell_ids,
                                                                   d_pos,
                                                                   d_pos_embed,
                                                                   d_vel_embed,
                                                                   d_embed_member_idx,
                                                                   periodic,
                                                                   origin_idx,
                                                                   grid_shift,
                                                                   global_box,
                                                                   global_cell_dim,
                                                                   cell_indexer,
                                                                   global_cell_indexer,
                                                                   d_mpcd_comm_key,
                                                                   rank_size,
                                                                   is_decomposition,
                                                                   N_mpcd,
                                                                   N_tot,
                                                                   need_energy);

    return cudaSuccess;
    }

/*!
 * \param d_cell_np Number of particles in cells
 * \param d_cell_vel Cell velocity to reduce
 * \param d_cell_energy Cell kinetic energy
 * \param d_cell_temp Cell temperature to finish computing
 * \param N_cells Number of total cells
 * \param N_dim Number of dimensions
 * \param need_energy If true, compute the cell-level energy properties
 * \param block_size Number of threads per block
 *
 * \returns cudaSuccess on completion
 *
 * \sa mpcd::gpu::kernel::finish_cell_properties
 */
cudaError_t mpcd::gpu::finish_cell_properties(const unsigned int* d_cell_np,
                                              double4* d_cell_vel,
                                              const double* d_cell_energy,
                                              double* d_cell_temp,
                                              const unsigned int N_cells,
                                              const unsigned int N_dim,
                                              const bool need_energy,
                                              const unsigned int block_size)
    {
    // set all temperatures to zero
    if (need_energy)
        {
        cudaError_t error_temp = cudaMemset(d_cell_temp, 0, sizeof(double) * N_cells);
        if (error_temp != cudaSuccess)
            return error_temp;
        }

    unsigned int max_block_size;
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, (const void*)mpcd::gpu::kernel::finish_cell_properties);
    max_block_size = attr.maxThreadsPerBlock;

    unsigned int run_block_size = min(block_size, max_block_size);
    dim3 grid(N_cells / run_block_size + 1);
    mpcd::gpu::kernel::finish_cell_properties<<<grid, run_block_size>>>(d_cell_np,
                                                                        d_cell_vel,
                                                                        d_cell_energy,
                                                                        d_cell_temp,
                                                                        N_cells,
                                                                        N_dim,
                                                                        need_energy);

    return cudaSuccess;
    }

/*!
 * \param d_tmp_thermo Temporary cell packed thermo element
 * \param d_cell_np Number of particles per cell
 * \param d_cell_vel Cell velocity
 * \param d_cell_energy Cell kinetic energy
 * \param d_cell_temp Cell temperature
 * \param N_cells Number of total cells
 * \param need_energy If true, compute the cell-level energy properties
 * \param block_size Number of threads per block
 *
 * \returns cudaSuccess on completion
 *
 * \sa mpcd::gpu::kernel::stage_net_cell_thermo
 */
cudaError_t mpcd::gpu::stage_net_cell_thermo(mpcd::detail::cell_thermo_element* d_tmp_thermo,
                                             const unsigned int* d_cell_np,
                                             const double4* d_cell_vel,
                                             const double* d_cell_energy,
                                             const double* d_cell_temp,
                                             const unsigned int N_cells,
                                             const bool need_energy,
                                             const unsigned int block_size)
    {
    if (need_energy)
        {
        unsigned int max_block_size_energy;
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, (const void*)mpcd::gpu::kernel::stage_net_cell_thermo<true>);
        max_block_size_energy = attr.maxThreadsPerBlock;

        unsigned int run_block_size = min(block_size, max_block_size_energy);
        dim3 grid(N_cells / run_block_size + 1);
        mpcd::gpu::kernel::stage_net_cell_thermo<true><<<grid, run_block_size>>>(d_tmp_thermo,
                                                                                 d_cell_np,
                                                                                 d_cell_vel,
                                                                                 d_cell_energy,
                                                                                 d_cell_temp,
                                                                                 N_cells);
        }
    else
        {
        unsigned int max_block_size_noenergy;
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, (const void*)mpcd::gpu::kernel::stage_net_cell_thermo<false>);
        max_block_size_noenergy = attr.maxThreadsPerBlock;

        unsigned int run_block_size = min(block_size, max_block_size_noenergy);
        dim3 grid(N_cells / run_block_size + 1);
        mpcd::gpu::kernel::stage_net_cell_thermo<false><<<grid, run_block_size>>>(d_tmp_thermo,
                                                                                  d_cell_np,
                                                                                  d_cell_vel,
                                                                                  d_cell_energy,
                                                                                  d_cell_temp,
                                                                                  N_cells);
        }
    return cudaSuccess;
    }

/*!
 * \param d_reduced Cell thermo properties reduced across all cells (output on second call)
 * \param d_tmp Temporary storage for reduction (output on first call)
 * \param tmp_bytes Number of bytes allocated for temporary storage (output on first call)
 * \param d_tmp_thermo Cell thermo properties to reduce
 * \param N_cells The number of cells to reduce across
 *
 * \returns cudaSuccess on completion
 *
 * \b Implementation details:
 * CUB DeviceReduce is used to perform the reduction. Hence, this function requires
 * two calls to perform the reduction. The first call sizes the temporary storage,
 * which is returned in \a d_tmp and \a tmp_bytes. The caller must then allocate
 * the required bytes, and call the function a second time. This performs the
 * reduction and returns the result in \a d_reduced.
 */
cudaError_t mpcd::gpu::reduce_net_cell_thermo(mpcd::detail::cell_thermo_element* d_reduced,
                                              void* d_tmp,
                                              size_t& tmp_bytes,
                                              const mpcd::detail::cell_thermo_element* d_tmp_thermo,
                                              const size_t N_cells)
    {
    cub::DeviceReduce::Sum(d_tmp, tmp_bytes, d_tmp_thermo, d_reduced, (unsigned int)N_cells);
    return cudaSuccess;
    }

/*!
 * \param d_migrate_flag Flag signaling migration is required (output)
 * \param d_pos Embedded particle positions
 * \param d_group Indexes into \a d_pos for particles in embedded group
 * \param box Box covered by this domain
 * \param N Number of particles in group
 * \param block_size Number of threads per block
 *
 * \sa mpcd::gpu::kernel::cell_check_migrate_embed
 */
cudaError_t mpcd::gpu::cell_check_migrate_embed(unsigned int* d_migrate_flag,
                                                const Scalar4* d_pos,
                                                const unsigned int* d_group,
                                                const BoxDim& box,
                                                const unsigned int num_dim,
                                                const unsigned int N,
                                                const unsigned int block_size)
    {
    // ensure that the flag is always zeroed even if the caller forgets
    cudaMemset(d_migrate_flag, 0, sizeof(unsigned int));

    unsigned int max_block_size;
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, (const void*)mpcd::gpu::kernel::cell_check_migrate_embed);
    max_block_size = attr.maxThreadsPerBlock;

    unsigned int run_block_size = min(block_size, max_block_size);
    dim3 grid(N / run_block_size + 1);
    mpcd::gpu::kernel::cell_check_migrate_embed<<<grid, run_block_size>>>(d_migrate_flag,
                                                                          d_pos,
                                                                          d_group,
                                                                          box,
                                                                          num_dim,
                                                                          N);

    return cudaSuccess;
    }

/*!
 * \param d_mpcd_comm_key directions to send MPCD particles as ghosts
 * \param d_mpcd_send_offsets starting index of points sent to each neighbor
 * \param num_mpcd_ghosts_send the total number of MPCD particles being sent
 * \param N Number of particles in group
 * \param block_size Number of threads per block
 *
 * \sa mpcd::gpu::kernel::find_num_ghost_send
 */
cudaError_t mpcd::gpu::find_num_ghost_send(uint2* d_mpcd_comm_key,
                                           unsigned int* d_mpcd_send_offsets,
                                           unsigned int& num_mpcd_ghosts_send,
                                           const unsigned int N,
                                           const unsigned int block_size)
    {
    // // sort communication keys
    // thrust::sort(thrust::device, d_mpcd_comm_key, d_mpcd_comm_key + N);

    // fill the starting indices with invalid values
    cudaError_t error = cudaMemset(d_mpcd_send_offsets, 0xffffffff, sizeof(unsigned int) * 27);
    if (error != cudaSuccess)
        return error;
    // prepare kernel
    unsigned int max_block_size;
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, (const void*)mpcd::gpu::kernel::find_num_ghost_send);
    max_block_size = attr.maxThreadsPerBlock;

    unsigned int run_block_size = min(block_size, max_block_size);
    dim3 grid(N / run_block_size + 1);
    mpcd::gpu::kernel::find_num_ghost_send<<<grid, run_block_size>>>(d_mpcd_comm_key,
                                                                     d_mpcd_send_offsets,
                                                                     num_mpcd_ghosts_send,
                                                                     N);

    return cudaSuccess;
    }

/*!
 * \param d_mpcd_comm_key directions to send MPCD particles as ghosts
 * \param d_vel MPCD particle velocities
 * \param d_mpcd_vel_sendbuf buffer for MPCD ghost velocities to be sent
 * \param num_mpcd_ghosts_send the total number of MPCD particles being sent
 * \param block_size Number of threads per block
 *
 * \sa mpcd::gpu::kernel::fill_buffer
 */
cudaError_t mpcd::gpu::fill_buffer(uint2* d_mpcd_comm_key,
                                   Scalar4* d_vel,
                                   Scalar4* d_mpcd_vel_sendbuf,
                                   unsigned int num_mpcd_ghosts_send,
                                   const unsigned int block_size)
    {
    unsigned int max_block_size;
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, (const void*)mpcd::gpu::kernel::fill_buffer);
    max_block_size = attr.maxThreadsPerBlock;

    unsigned int run_block_size = min(block_size, max_block_size);
    dim3 grid(num_mpcd_ghosts_send / run_block_size + 1);
    mpcd::gpu::kernel::fill_buffer<<<grid, run_block_size>>>(d_mpcd_comm_key,
                                                             d_vel,
                                                             d_mpcd_vel_sendbuf,
                                                             num_mpcd_ghosts_send);

    return cudaSuccess;
    }

    } // end namespace hoomd
