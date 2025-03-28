// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/CollisionMethod.h
 * \brief Definition of mpcd::CollisionMethod
 */

#include "CollisionMethod.h"

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
      m_period(period), m_checked_collision_warnings(false), m_initial_velocity(m_exec_conf)
    {
    // setup next timestep for collision
    m_next_timestep = cur_timestep;
    if (phase >= 0)
        {
        // determine next step that is in line with period + phase
        uint64_t multiple = cur_timestep / m_period + (cur_timestep % m_period != 0);
        m_next_timestep = multiple * m_period + phase;
        }
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
    if (m_embed_group)
        {
        // resize initial velocity array
        const unsigned int num_group = m_embed_group->getNumMembers();
        if (num_group > m_initial_velocity.getNumElements())
            {
            GPUArray<Scalar4> initial_velocity(num_group, m_exec_conf);
            m_initial_velocity.swap(initial_velocity);
            }

        const unsigned int num_total = m_pdata->getN();
        if (num_total > m_linmom_accum.getNumElements())
            {
            GPUArray<Scalar3> linmom_accum(num_total, m_exec_conf);
            m_linmom_accum.swap(linmom_accum);
            GPUArray<Scalar4> angmom_accum(num_total, m_exec_conf);
            m_angmom_accum.swap(angmom_accum);
            }
        beginRigidBodyCollision(timestep);
        }

    // apply collisions
    rule(timestep);

    // apply collisions to rigid bodies
    if (m_embed_group)
        {
        accumulateRigidBodyMomenta(timestep);
        finishRigidBodyCollision(timestep);
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
        ArrayHandle<Scalar4> h_vel_embed(m_pdata->getVelocities(),
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
            const Scalar4 vel_mass = h_vel_embed.data[particle_index];
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

void mpcd::CollisionMethod::beginRigidBodyCollision(uint64_t timestep)
    {
    unsigned int num_group = m_embed_group->getNumMembers();
    ArrayHandle<Scalar4> h_initial_vel(m_initial_velocity,
                                       access_location::host,
                                       access_mode::overwrite);
    ArrayHandle<unsigned int> h_embed_group(m_embed_group->getIndexArray(),
                                            access_location::host,
                                            access_mode::read);
    ArrayHandle<Scalar4> h_vel_embed(m_pdata->getVelocities(),
                                     access_location::host,
                                     access_mode::read);
    for (unsigned int idx = 0; idx < num_group; ++idx)
        {
        // collect the initial velocities of the embedded particles
        const unsigned int particle_index = h_embed_group.data[idx];
        h_initial_vel.data[idx] = h_vel_embed.data[particle_index];
        }
    }

void mpcd::CollisionMethod::accumulateRigidBodyMomenta(uint64_t timestep)
    {
    unsigned int num_group = m_embed_group->getNumMembers();
    ArrayHandle<Scalar4> h_initial_vel(m_initial_velocity,
                                       access_location::host,
                                       access_mode::read);
    ArrayHandle<Scalar3> h_linmom_accum(m_linmom_accum,
                                        access_location::host,
                                        access_mode::readwrite);
    ArrayHandle<Scalar4> h_angmom_accum(m_angmom_accum,
                                        access_location::host,
                                        access_mode::readwrite);
    ArrayHandle<unsigned int> h_embed_group(m_embed_group->getIndexArray(),
                                            access_location::host,
                                            access_mode::read);
    ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(),
                                   access_location::host,
                                   access_mode::read);
    ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(),
                                       access_location::host,
                                       access_mode::read);
    ArrayHandle<Scalar4> h_velocity(m_pdata->getVelocities(),
                                    access_location::host,
                                    access_mode::read);
    ArrayHandle<unsigned int> h_body(m_pdata->getBodies(),
                                     access_location::host,
                                     access_mode::read);
    ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);
    for (unsigned int idx = 0; idx < num_group; ++idx)
        {
        // get the index from the embedded group
        const unsigned int particle_index = h_embed_group.data[idx];
        // check if particle is in a rigid body
        unsigned int central_tag = h_body.data[particle_index];
        if (central_tag >= MIN_FLOPPY)
            {
            continue;
            }
        // if the central particle is not local, cannot read or write to it.
        unsigned int central_idx = h_rtag.data[central_tag];
        if (central_idx == NOT_LOCAL)
            {
            continue;
            }
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
        const Scalar4 postype_central = h_postype.data[central_idx];
        const vec3<Scalar> pos_central(postype_central);
        vec3<Scalar> displacement = pos_const - pos_central;

        // change in linear and angular momentum
        const vec3<Scalar> initial_vel_const(h_initial_vel.data[idx]);
        const vec3<Scalar> linmom_change = mass_const * (vel_const - initial_vel_const);
        const vec3<Scalar> angmom_change = cross(displacement, linmom_change);

        // accumulate onto central particle
        h_linmom_accum.data[central_idx] += linmom_change;
        h_angmom_accum.data[central_idx] += angmom_change;
        }
    }

void mpcd::CollisionMethod::finishRigidBodyCollision(uint64_t timestep)
    {
    ArrayHandle<Scalar3> h_linmom_accum(m_linmom_accum, access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_angmom_accum(m_angmom_accum, access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_velocity(m_pdata->getVelocities(),
                                    access_location::host,
                                    access_mode::readwrite);
    ArrayHandle<Scalar4> h_angmom(m_pdata->getAngularMomentumArray(),
                                  access_location::host,
                                  access_mode::readwrite);

    // add accumulated momentum to the central particle
    unsigned int num_total = m_pdata->getN();
    for (unsigned int idx = 0; idx < num_total; ++idx)
        {
        // get accumulated momentum for particle
        vec3<Scalar> linmom_accum(h_linmom_accum.data[idx]);
        quat<Scalar> angmom_accum(h_angmom_accum.data[idx]);

        // get velocity, mass, and angular momentum
        Scalar4 vel_mass = h_velocity.data[idx];
        quat<Scalar> angmom(h_angmom.data[idx]);
        vec3<Scalar> vel(vel_mass);
        const Scalar mass = vel_mass.w;

        // update velocity and angular momentum
        vec3<Scalar> updated_vel = vel;
        if (mass > 0)
            {
            updated_vel += linmom_accum / mass;
            }
        quat<Scalar> updated_angmom = angmom + angmom_accum;

        // save update
        h_angmom.data[idx] = quat_to_scalar4(updated_angmom);
        h_velocity.data[idx] = make_scalar4(updated_vel.x, updated_vel.y, updated_vel.z, mass);
        }
    }
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
