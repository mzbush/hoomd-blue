// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/CollisionMethod.h
 * \brief Definition of mpcd::CollisionMethod
 */

#include "CollisionMethod.h"
#include "hoomd/RNGIdentifiers.h"
#include "hoomd/RandomNumbers.h"
#ifdef ENABLE_MPI
#include "hoomd/Communicator.h"
#endif // ENABLE_MPI
namespace hoomd
    {
/*!
 * \param sysdef System definition
 * \param cur_timestep Current system timestep
 * \param period Number of timesteps between collisions
 * \param phase Phase shift for periodic updates
 * \param seed Seed to pseudo-random number generator
 */
mpcd::CollisionMethod::CollisionMethod(std::shared_ptr<SystemDefinition> sysdef,
                                       uint64_t cur_timestep,
                                       uint64_t period,
                                       int phase)
    : m_sysdef(sysdef), m_pdata(m_sysdef->getParticleData()),
      m_mpcd_pdata(sysdef->getMPCDParticleData()), m_exec_conf(m_pdata->getExecConf()),
      m_period(period), m_checked_collision_warnings(false), m_initial_velocity(m_exec_conf),
      m_linmom_accum(m_exec_conf), m_angmom_accum(m_exec_conf), m_linmom_accum_copybuf(m_exec_conf),
      m_angmom_accum_copybuf(m_exec_conf)
    {
    // setup next timestep for collision
    m_next_timestep = cur_timestep;
    if (phase >= 0)
        {
        // determine next step that is in line with period + phase
        uint64_t multiple = cur_timestep / m_period + (cur_timestep % m_period != 0);
        m_next_timestep = multiple * m_period + phase;
        }

#ifdef ENABLE_HIP
    if (m_exec_conf->isCUDAEnabled())
        {
        m_store_tuner.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)},
                                             m_exec_conf,
                                             "mpcd_rigid_store"));
        m_accumulate_tuner.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)},
                                                  m_exec_conf,
                                                  "mpcd_rigid_accum"));
        m_transfer_tuner.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)},
                                                m_exec_conf,
                                                "mpcd_rigid_transfer"));
        m_autotuners.insert(m_autotuners.end(),
                            {m_store_tuner, m_accumulate_tuner, m_transfer_tuner});
        }
#endif // ENABLE_HIP
    }

void mpcd::CollisionMethod::collide(uint64_t timestep)
    {
    if (!shouldCollide(timestep))
        return;

    if (!m_cl)
        {
        throw std::runtime_error("Cell list has not been set");
        }

    // sync the embedded group
    m_cl->setEmbeddedGroup(m_embed_group);

    // check for collision warnings
    checkCollisionWarnings(timestep);

    // set random grid shift
    m_cl->drawGridShift(timestep);

    // update cell list
    m_cl->compute(timestep);

    // create auxillary array for rigid bodies
    const bool rigid_body_collision = m_embed_group && m_rigid_bodies;
    if (rigid_body_collision)
        {
        // resize initial velocity array
        const unsigned int num_group = m_embed_group->getNumMembers();
        if (num_group > m_initial_velocity.getNumElements())
            {
            GPUArray<Scalar4> initial_velocity(num_group, m_exec_conf);
            m_initial_velocity.swap(initial_velocity);
            }

        const unsigned int num_total = m_pdata->getN() + m_pdata->getNGhosts();
        if (num_total > m_linmom_accum.getNumElements())
            {
            GPUArray<Scalar3> linmom_accum(num_total, m_exec_conf);
            m_linmom_accum.swap(linmom_accum);
            GPUArray<Scalar3> angmom_accum(num_total, m_exec_conf);
            m_angmom_accum.swap(angmom_accum);
#ifdef ENABLE_MPI
            GPUArray<Scalar3> linmom_accum_copybuf(num_total, m_exec_conf);
            m_linmom_accum_copybuf.swap(linmom_accum);
            GPUArray<Scalar3> angmom_accum_copybuf(num_total, m_exec_conf);
            m_angmom_accum_copybuf.swap(angmom_accum);
#endif
            }

#ifdef ENABLE_HIP
        if (m_exec_conf->isCUDAEnabled())
            {
            storeInitialEmbeddedGroupVelocitiesGPU(timestep);
            }
        else
#endif
            {
            storeInitialEmbeddedGroupVelocities(timestep);
            }
        }

    // apply collisions
    rule(timestep);

    // apply collisions to rigid bodies
    if (rigid_body_collision)
        {
#ifdef ENABLE_HIP
        if (m_exec_conf->isCUDAEnabled())
            {
            accumulateRigidBodyMomentaGPU(timestep);
            transferRigidBodyMomentaGPU(timestep);
            }
        else
#endif
            {
            accumulateRigidBodyMomenta(timestep);
            transferRigidBodyMomenta(timestep);
            }
        m_rigid_bodies->updateCompositeParticles(timestep);
        }
    }

void mpcd::CollisionMethod::checkCollisionWarnings(uint64_t timestep)
    {
    if (m_checked_collision_warnings)
        {
        return;
        }

    if (m_embed_group)
        {
        const unsigned int num_group = m_embed_group->getNumMembers();
        ArrayHandle<unsigned int> h_embed_group(m_embed_group->getIndexArray(),
                                                access_location::host,
                                                access_mode::read);
        ArrayHandle<Scalar4> h_velocity(m_pdata->getVelocities(),
                                        access_location::host,
                                        access_mode::read);
        ArrayHandle<unsigned int> h_body_embed(m_pdata->getBodies(),
                                               access_location::host,
                                               access_mode::read);
        ArrayHandle<unsigned int> h_rtag_embed(m_pdata->getRTags(),
                                               access_location::host,
                                               access_mode::read);

        // check if some of the masses are less or equal to 0
        bool invalid_mass = false;
        bool central_interacting = false;
        for (unsigned int idx = 0; idx < num_group && !(invalid_mass && central_interacting); ++idx)
            {
            // get the index from the embedded group
            const unsigned int particle_index = h_embed_group.data[idx];

            // check mass
            const Scalar4 vel_mass = h_velocity.data[particle_index];
            const Scalar mass = vel_mass.w;
            if (mass <= Scalar(0))
                {
                invalid_mass = true;
                }

            // check if particle is central particle in rigid body
            const unsigned int central_tag = h_body_embed.data[particle_index];
            if (central_tag < MIN_FLOPPY)
                {
                const unsigned int central_idx = h_rtag_embed.data[central_tag];
                if (particle_index == central_idx)
                    {
                    central_interacting = true;
                    }
                }
            }

#ifdef ENABLE_MPI
        if (m_exec_conf->getNRanks() > 1)
            {
            MPI_Allreduce(MPI_IN_PLACE,
                          &invalid_mass,
                          1,
                          MPI_CXX_BOOL,
                          MPI_LOR,
                          m_exec_conf->getMPICommunicator());
            MPI_Allreduce(MPI_IN_PLACE,
                          &central_interacting,
                          1,
                          MPI_CXX_BOOL,
                          MPI_LOR,
                          m_exec_conf->getMPICommunicator());
            }
#endif // ENABLE_MPI

        if (invalid_mass)
            {
            m_exec_conf->msg->warning() << "Some particles have a mass <= 0, may lead to "
                                           "invalid results during MPCD collision."
                                        << std::endl;
            }

        if (central_interacting)
            {
            m_exec_conf->msg->warning() << "Central particle of rigid body included in "
                                           "MPCD collision. Check if this is intentional."
                                        << std::endl;
            }
        }
    m_checked_collision_warnings = true;
    }

void mpcd::CollisionMethod::storeInitialEmbeddedGroupVelocities(uint64_t timestep)
    {
    const unsigned int num_group = m_embed_group->getNumMembers();
    ArrayHandle<Scalar4> h_initial_vel(m_initial_velocity,
                                       access_location::host,
                                       access_mode::overwrite);
    ArrayHandle<unsigned int> h_embed_group(m_embed_group->getIndexArray(),
                                            access_location::host,
                                            access_mode::read);
    ArrayHandle<Scalar4> h_velocity(m_pdata->getVelocities(),
                                    access_location::host,
                                    access_mode::read);
    for (unsigned int idx = 0; idx < num_group; ++idx)
        {
        // collect the initial velocities of the embedded particles
        const unsigned int particle_index = h_embed_group.data[idx];
        h_initial_vel.data[idx] = h_velocity.data[particle_index];
        }
    }

void mpcd::CollisionMethod::accumulateRigidBodyMomenta(uint64_t timestep)
    {
    // zero accumulators
    m_linmom_accum.zeroFill();
    m_angmom_accum.zeroFill();

    const unsigned int num_group = m_embed_group->getNumMembers();
    ArrayHandle<unsigned int> h_embed_group(m_embed_group->getIndexArray(),
                                            access_location::host,
                                            access_mode::read);
    ArrayHandle<Scalar4> h_initial_vel(m_initial_velocity,
                                       access_location::host,
                                       access_mode::read);

    ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(),
                                   access_location::host,
                                   access_mode::read);
    ArrayHandle<Scalar4> h_velocity(m_pdata->getVelocities(),
                                    access_location::host,
                                    access_mode::read);
    ArrayHandle<unsigned int> h_body(m_pdata->getBodies(),
                                     access_location::host,
                                     access_mode::read);
    ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);
    ArrayHandle<int3> h_image(m_pdata->getImages(), access_location::host, access_mode::read);
    const BoxDim& global_box = m_pdata->getGlobalBox();

    ArrayHandle<Scalar3> h_linmom_accum(m_linmom_accum,
                                        access_location::host,
                                        access_mode::readwrite);
    ArrayHandle<Scalar3> h_angmom_accum(m_angmom_accum,
                                        access_location::host,
                                        access_mode::readwrite);

    for (unsigned int idx = 0; idx < num_group; ++idx)
        {
        // get the index from the embedded group and check if in a rigid body
        const unsigned int particle_index = h_embed_group.data[idx];
        const unsigned int central_tag = h_body.data[particle_index];
        if (central_tag >= MIN_FLOPPY)
            {
            continue;
            }
        const unsigned int central_idx = h_rtag.data[central_tag];
        // if the central particle is not local, cannot read or write to it.
        assert(central_idx != NOT_LOCAL);

        // collision on central particle itself already taken care of by collision rule
        if (particle_index == central_idx)
            {
            continue;
            }

        // get velocities and masses
        const Scalar4 vel_mass_const = h_velocity.data[particle_index];
        const vec3<Scalar> vel_const(vel_mass_const);
        const Scalar mass_const = vel_mass_const.w;

        // get displacement
        const Scalar4 postype_const = h_postype.data[particle_index];
        const vec3<Scalar> pos_const(postype_const);
        const int3 img_const = h_image.data[particle_index];

        const Scalar4 postype_central = h_postype.data[central_idx];
        const vec3<Scalar> pos_central(postype_central);
        const int3 img_central = h_image.data[central_idx];

        vec3<Scalar> displacement = pos_const - pos_central;
        const int3 displacement_img = make_int3(img_const.x - img_central.x,
                                                img_const.y - img_central.y,
                                                img_const.z - img_central.z);
        displacement = global_box.shift(displacement, displacement_img);

        // change in linear and angular momentum
        const vec3<Scalar> initial_vel_const(h_initial_vel.data[idx]);
        const vec3<Scalar> linmom_change = mass_const * (vel_const - initial_vel_const);
        const vec3<Scalar> angmom_change = cross(displacement, linmom_change);

        // accumulate onto central particle
        h_linmom_accum.data[central_idx] += vec_to_scalar3(linmom_change);
        h_angmom_accum.data[central_idx] += vec_to_scalar3(angmom_change);
        }
    }

void mpcd::CollisionMethod::transferRigidBodyMomenta(uint64_t timestep)
    {
    m_thermo_rigid_thermostat->compute(timestep);
    ArrayHandle<Scalar3> h_linmom_accum(m_linmom_accum, access_location::host, access_mode::read);
    ArrayHandle<Scalar3> h_angmom_accum(m_angmom_accum, access_location::host, access_mode::read);

    ArrayHandle<Scalar4> h_velocity(m_pdata->getVelocities(),
                                    access_location::host,
                                    access_mode::readwrite);
    ArrayHandle<Scalar4> h_angmom(m_pdata->getAngularMomentumArray(),
                                  access_location::host,
                                  access_mode::readwrite);
    ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(),
                                       access_location::host,
                                       access_mode::read);
    ArrayHandle<Scalar3> h_inertia(m_pdata->getMomentsOfInertiaArray(),
                                   access_location::host,
                                   access_mode::read);
    ArrayHandle<unsigned int> h_body(m_pdata->getBodies(),
                                     access_location::host,
                                     access_mode::read);
    ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);

    // for rescaling the velocity to match the thermostat
    ArrayHandle<double4> h_cell_vel(m_thermo_rigid_thermostat->getCellVelocities(),
                                    access_location::host,
                                    access_mode::read);
    ArrayHandle<double3> h_cell_energy(m_thermo_rigid_thermostat->getCellEnergies(),
                                       access_location::host,
                                       access_mode::read);
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_cell_size(m_cl->getCellSizeArray(),
                                          access_location::host,
                                          access_mode::read);
    ArrayHandle<unsigned int> h_cell_list(m_cl->getCellList(),
                                          access_location::host,
                                          access_mode::read);
    ArrayHandle<Scalar4> h_vel_mpcd(m_mpcd_pdata->getVelocities(),
                                    access_location::host,
                                    access_mode::readwrite);
    const Index2D& cli = m_cl->getCellListIndexer();
    const BoxDim& box = m_pdata->getBox();
    uchar3 periodic = box.getPeriodic();
    uint3 cell_dim = m_cl->getDim();
    const BoxDim& global_box = m_pdata->getGlobalBox();
    const Scalar3 grid_shift = m_cl->getGridShift();
    uint3 n_global_cells = m_cl->getGlobalDim();
    uint3 h_global_cell_dim = m_cl->getGlobalDim();
    const Index3D& ci = m_cl->getCellIndexer();
    const Index3D& global_ci = m_cl->getGlobalCellIndexer();
    const int3& h_origin_idx = m_cl->getOriginIndex();

    // add accumulated momentum to the central particle
    unsigned int num_total = m_pdata->getN();
    for (unsigned int idx = 0; idx < num_total; ++idx)
        {
        // check that the particle is in a rigid body and a central particle
        const unsigned int central_tag = h_body.data[idx];
        if (central_tag >= MIN_FLOPPY)
            {
            continue;
            }
        const unsigned int central_idx = h_rtag.data[central_tag];
        if (central_idx != idx)
            {
            continue;
            }

        // get accumulated momentum for particle
        const Scalar3 linmom_accum(h_linmom_accum.data[idx]);
        const vec3<Scalar> angmom_accum(h_angmom_accum.data[idx]);

        // compute and store new velocity
        Scalar4 vel_mass = h_velocity.data[idx];
        const Scalar mass = vel_mass.w;
        if (mass > 0)
            {
            vel_mass.x += linmom_accum.x / mass;
            vel_mass.y += linmom_accum.y / mass;
            vel_mass.z += linmom_accum.z / mass;
            h_velocity.data[idx] = vel_mass;
            }

        // compute new angular momentum
        quat<Scalar> angmom(h_angmom.data[idx]);
        const quat<Scalar> orientation(h_orientation.data[idx]);

        // convert angular momentum to quaternion and update
        const vec3<Scalar> inertia(h_inertia.data[central_idx]);
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
        h_angmom.data[idx] = quat_to_scalar4(angmom);

        // find the cell this particle would belong to
        Scalar4 postype_i = h_pos.data[idx];
        Scalar3 pos_i = make_scalar3(postype_i.x, postype_i.y, postype_i.z);
        const Scalar3 fractional_pos_i = global_box.makeFraction(pos_i) - grid_shift;
        int3 global_bin = make_int3((int)std::floor(fractional_pos_i.x * h_global_cell_dim.x),
                                    (int)std::floor(fractional_pos_i.y * h_global_cell_dim.y),
                                    (int)std::floor(fractional_pos_i.z * h_global_cell_dim.z));

        // wrap cell back through the boundaries (grid shifting may send +/- 1 outside of range)
        // this is done using periodic from the "local" box, since this will be periodic
        // only when there is one rank along the dimension
        if (periodic.x)
            {
            if (global_bin.x == (int)n_global_cells.x)
                global_bin.x = 0;
            else if (global_bin.x == -1)
                global_bin.x = n_global_cells.x - 1;
            }
        if (periodic.y)
            {
            if (global_bin.y == (int)n_global_cells.y)
                global_bin.y = 0;
            else if (global_bin.y == -1)
                global_bin.y = n_global_cells.y - 1;
            }
        if (periodic.z)
            {
            if (global_bin.z == (int)n_global_cells.z)
                global_bin.z = 0;
            else if (global_bin.z == -1)
                global_bin.z = n_global_cells.z - 1;
            }

        // compute the local cell
        int3 bin = make_int3(global_bin.x - h_origin_idx.x,
                             global_bin.y - h_origin_idx.y,
                             global_bin.z - h_origin_idx.z);
        // these checks guard against round-off errors with domain decomposition
        if (!periodic.x)
            {
            if (bin.x == -1)
                bin.x = 0;
            else if (bin.x == (int)cell_dim.x)
                bin.x = cell_dim.x - 1;
            }
        if (!periodic.y)
            {
            if (bin.y == -1)
                bin.y = 0;
            else if (bin.y == (int)cell_dim.y)
                bin.y = cell_dim.y - 1;
            }
        if (!periodic.z)
            {
            if (bin.z == -1)
                bin.z = 0;
            else if (bin.z == (int)cell_dim.z)
                bin.z = cell_dim.z - 1;
            }

        // validate and make sure no particles blew out of the box
        if ((bin.x < 0 || bin.x >= (int)cell_dim.x) || (bin.y < 0 || bin.y >= (int)cell_dim.y)
            || (bin.z < 0 || bin.z >= (int)cell_dim.z))
            {
            continue;
            }

        unsigned int bin_idx = ci(bin.x, bin.y, bin.z);

        // get the cell velocity, kinetic energy, and new temperature
        const Scalar4 vel_mass_cell = h_cell_vel.data[bin_idx];
        const vec3<Scalar> vel_cell(vel_mass_cell);
        const Scalar mass_cell = vel_mass_cell.w;
        const double3 cell_energy = h_cell_energy.data[bin_idx];
        const unsigned int np = __double_as_int(cell_energy.z);

        Scalar kinetic_energy
            = cell_energy.x
              + 0.5 * mass
                    * (vel_mass.x * vel_mass.x + vel_mass.y * vel_mass.y + vel_mass.z * vel_mass.z);
        Scalar temperature = 2 * kinetic_energy / (3 * (np - 1 + 1));

        // thermostat relative velocities
        // get the global index from the bin numbers found
        const int3 global_cell = m_cl->getGlobalCell(bin);
        const unsigned int global_idx = global_ci(global_cell.x, global_cell.y, global_cell.z);
        // get the rng
        uint16_t seed = m_sysdef->getSeed();
        hoomd::RandomGenerator rng(
            hoomd::Seed(hoomd::RNGIdentifier::CollisionMethod, timestep, seed),
            hoomd::Counter(global_idx));

        // the total number of degrees of freedom in the cell divided by 2
        const double alpha = m_sysdef->getNDimensions() * (np - 1 + 1) / (double)2.;

        // rescale velocity if there are other particles in the cell
        if (alpha > 0)
            {
            // draw a random kinetic energy for the cell at the set temperature
            hoomd::GammaDistribution<double> gamma_gen(alpha, 1.0);
            const double rand_ke = gamma_gen(rng);

            // generate the scale factor from the current temperature
            // (don't use the kinetic energy of this cell, since this
            // is total not relative to COM)
            const double cur_ke = alpha * temperature;
            double factor = (cur_ke > 0.) ? fast::sqrt(rand_ke / cur_ke) : 1.;

            // rescale with the temperature of the cell
            Scalar4 vel_com;
            if (mass + mass_cell > 0)
                {
                vel_com.x = (vel_mass.x * mass + vel_mass_cell.x * mass_cell) / (mass + mass_cell);
                vel_com.y = (vel_mass.y * mass + vel_mass_cell.y * mass_cell) / (mass + mass_cell);
                vel_com.z = (vel_mass.z * mass + vel_mass_cell.z * mass_cell) / (mass + mass_cell);
                vel_com.w = mass + mass_cell;

                vel_mass.x = vel_com.x + (vel_mass.x - vel_com.x) * factor;
                vel_mass.y = vel_com.y + (vel_mass.y - vel_com.y) * factor;
                vel_mass.z = vel_com.z + (vel_mass.z - vel_com.z) * factor;
                h_velocity.data[idx] = vel_mass;

                // rescale the other particles in the cell
                const unsigned int np = h_cell_size.data[bin_idx];
                const unsigned int N_mpcd = m_mpcd_pdata->getN() + m_mpcd_pdata->getNVirtual();
                for (unsigned int offset = 0; offset < np; ++offset)
                    {
                    // Load particle data
                    // NOTE: unsure if bin_idx is correct here
                    const unsigned int cur_p = h_cell_list.data[cli(offset, bin_idx)];
                    if (cur_p < N_mpcd)
                        {
                        Scalar4 vel_solv = h_vel_mpcd.data[cur_p];
                        vel_solv.x = vel_com.x + (vel_solv.x - vel_com.x) * factor;
                        vel_solv.y = vel_com.y + (vel_solv.y - vel_com.y) * factor;
                        vel_solv.z = vel_com.z + (vel_solv.z - vel_com.z) * factor;
                        h_vel_mpcd.data[idx] = vel_solv;
                        }
                    else
                        {
                        }
                    }
                }
            }
        }
    }

#ifdef ENABLE_HIP
//! Begin process of applying collisions to rigid bodies (GPU version)
void mpcd::CollisionMethod::storeInitialEmbeddedGroupVelocitiesGPU(uint64_t timestep)
    {
    ArrayHandle<Scalar4> d_initial_vel(m_initial_velocity,
                                       access_location::device,
                                       access_mode::overwrite);
    ArrayHandle<unsigned int> d_embed_group(m_embed_group->getIndexArray(),
                                            access_location::device,
                                            access_mode::read);
    ArrayHandle<Scalar4> d_velocity(m_pdata->getVelocities(),
                                    access_location::device,
                                    access_mode::read);
    m_store_tuner->begin();
    mpcd::gpu::store_initial_embedded_group_velocities(d_initial_vel.data,
                                                       d_velocity.data,
                                                       d_embed_group.data,
                                                       m_embed_group->getNumMembers(),
                                                       m_store_tuner->getParam()[0]);
    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    m_store_tuner->end();
    }

//! Accumulate momenta changes of constituent particles of rigid bodies (GPU version)
void mpcd::CollisionMethod::accumulateRigidBodyMomentaGPU(uint64_t timestep)
    {
    // zero accumulators
    m_linmom_accum.zeroFill();
    m_angmom_accum.zeroFill();

    ArrayHandle<unsigned int> d_embed_group(m_embed_group->getIndexArray(),
                                            access_location::device,
                                            access_mode::read);
    ArrayHandle<Scalar4> d_initial_vel(m_initial_velocity,
                                       access_location::device,
                                       access_mode::read);

    ArrayHandle<Scalar4> d_postype(m_pdata->getPositions(),
                                   access_location::device,
                                   access_mode::read);
    ArrayHandle<Scalar4> d_velocity(m_pdata->getVelocities(),
                                    access_location::device,
                                    access_mode::read);
    ArrayHandle<unsigned int> d_body(m_pdata->getBodies(),
                                     access_location::device,
                                     access_mode::read);
    ArrayHandle<unsigned int> d_rtag(m_pdata->getRTags(),
                                     access_location::device,
                                     access_mode::read);
    ArrayHandle<int3> d_image(m_pdata->getImages(), access_location::device, access_mode::read);
    const BoxDim& global_box = m_pdata->getGlobalBox();

    ArrayHandle<Scalar3> d_linmom_accum(m_linmom_accum,
                                        access_location::device,
                                        access_mode::readwrite);
    ArrayHandle<Scalar3> d_angmom_accum(m_angmom_accum,
                                        access_location::device,
                                        access_mode::readwrite);
    m_accumulate_tuner->begin();
    mpcd::gpu::accumulate_rigid_body_momenta(d_linmom_accum.data,
                                             d_angmom_accum.data,
                                             d_initial_vel.data,
                                             d_embed_group.data,
                                             d_postype.data,
                                             d_velocity.data,
                                             d_image.data,
                                             d_body.data,
                                             d_rtag.data,
                                             global_box,
                                             m_embed_group->getNumMembers(),
                                             m_accumulate_tuner->getParam()[0]);
    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    m_accumulate_tuner->end();
    }

//! Finish process of applying collisions to rigid bodies (GPU version)
void mpcd::CollisionMethod::transferRigidBodyMomentaGPU(uint64_t timestep)
    {
    ArrayHandle<Scalar3> d_linmom_accum(m_linmom_accum, access_location::device, access_mode::read);
    ArrayHandle<Scalar3> d_angmom_accum(m_angmom_accum, access_location::device, access_mode::read);

    ArrayHandle<Scalar4> d_velocity(m_pdata->getVelocities(),
                                    access_location::device,
                                    access_mode::readwrite);
    ArrayHandle<Scalar4> d_angmom(m_pdata->getAngularMomentumArray(),
                                  access_location::device,
                                  access_mode::readwrite);
    ArrayHandle<Scalar4> d_orientation(m_pdata->getOrientationArray(),
                                       access_location::device,
                                       access_mode::read);
    ArrayHandle<Scalar3> d_inertia(m_pdata->getMomentsOfInertiaArray(),
                                   access_location::device,
                                   access_mode::read);
    ArrayHandle<unsigned int> d_body(m_pdata->getBodies(),
                                     access_location::device,
                                     access_mode::read);
    ArrayHandle<unsigned int> d_rtag(m_pdata->getRTags(),
                                     access_location::device,
                                     access_mode::read);

    m_transfer_tuner->begin();
    mpcd::gpu::transfer_rigid_body_momenta(d_linmom_accum.data,
                                           d_angmom_accum.data,
                                           d_velocity.data,
                                           d_orientation.data,
                                           d_angmom.data,
                                           d_inertia.data,
                                           d_body.data,
                                           d_rtag.data,
                                           m_pdata->getN(),
                                           m_transfer_tuner->getParam()[0]);
    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    m_transfer_tuner->end();
    }

#endif // ENABLE_HIP

/*!
 * \param timestep Current timestep
 * \returns True when \a timestep is a \a m_period multiple of the the next timestep the collision
 * should occur
 *
 * Using a multiple allows the collision method to be disabled and then reenabled later if the \a
 * timestep has already exceeded the \a m_next_timestep.
 */
bool mpcd::CollisionMethod::peekCollide(uint64_t timestep) const
    {
    if (timestep < m_next_timestep)
        return false;
    else
        return ((timestep - m_next_timestep) % m_period == 0);
    }

/*!
 * \param cur_timestep Current simulation timestep
 * \param period New period
 *
 * The collision method period is updated to \a period only if collision would occur at \a
 * cur_timestep. It is the caller's responsibility to ensure this condition is valid.
 */
void mpcd::CollisionMethod::setPeriod(uint64_t cur_timestep, uint64_t period)
    {
    if (!peekCollide(cur_timestep))
        {
        m_exec_conf->msg->error()
            << "MPCD CollisionMethod period can only be changed on multiple of original period"
            << std::endl;
        throw std::runtime_error(
            "Collision period can only be changed on multiple of original period");
        }

    // try to update the period
    const uint64_t old_period = m_period;
    m_period = period;

    // validate the new period, resetting to the old one before erroring out if it doesn't match
    if (!peekCollide(cur_timestep))
        {
        m_period = old_period;
        m_exec_conf->msg->error()
            << "MPCD CollisionMethod period can only be changed on multiple of new period"
            << std::endl;
        throw std::runtime_error("Collision period can only be changed on multiple of new period");
        }
    }

/*!
 * \param timestep Current timestep
 * \returns True when \a timestep is a \a m_period multiple of the the next timestep the collision
 * should occur
 *
 * \post The next timestep is also advanced to the next timestep the collision should occur after \a
 * timestep. If this behavior is not desired, then use peekCollide() instead.
 */
bool mpcd::CollisionMethod::shouldCollide(uint64_t timestep)
    {
    if (peekCollide(timestep))
        {
        m_next_timestep = timestep + m_period;
        return true;
        }
    else
        {
        return false;
        }
    }

namespace mpcd
    {
namespace detail
    {
/*!
 * \param m Python module to export to
 */
void export_CollisionMethod(pybind11::module& m)
    {
    pybind11::class_<mpcd::CollisionMethod, std::shared_ptr<mpcd::CollisionMethod>>(
        m,
        "CollisionMethod")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, uint64_t, uint64_t, int>())
        .def_property_readonly("embedded_particles",
                               [](const std::shared_ptr<mpcd::CollisionMethod> method)
                               {
                                   auto group = method->getEmbeddedGroup();
                                   return (group) ? group->getFilter()
                                                  : std::shared_ptr<hoomd::ParticleFilter>();
                               })
        .def("setEmbeddedGroup", &mpcd::CollisionMethod::setEmbeddedGroup)
        .def_property_readonly("period", &mpcd::CollisionMethod::getPeriod);
    }
    } // namespace detail
    } // namespace mpcd
    } // end namespace hoomd
