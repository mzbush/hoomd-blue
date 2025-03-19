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
      m_period(period)
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

    // set random grid shift
    m_cl->drawGridShift(timestep);

    // update cell list
    m_cl->compute(timestep);

    rule(timestep);
    }

void mpcd::CollisionMethod::checkCollisionWarnings(uint64_t timestep)
    {
    if (m_checked_collision_warnings)
        {
        return;
        }
    std::unique_ptr<ArrayHandle<unsigned int>> h_embed_group;
    std::unique_ptr<ArrayHandle<Scalar4>> h_vel_embed;
    std::unique_ptr<ArrayHandle<unsigned int>> h_bodies_embed;
    std::unique_ptr<ArrayHandle<unsigned int>> h_rtag_embed;
    if (m_embed_group)
        {
        unsigned int N_tot = m_embed_group->getNumMembers();
        h_embed_group.reset(new ArrayHandle<unsigned int>(m_embed_group->getIndexArray(),
                                                          access_location::host,
                                                          access_mode::read));
        h_vel_embed.reset(new ArrayHandle<Scalar4>(m_pdata->getVelocities(),
                                                   access_location::host,
                                                   access_mode::read));
        h_bodies_embed.reset(new ArrayHandle<unsigned int>(m_pdata->getBodies(),
                                                           access_location::host,
                                                           access_mode::read));
        h_embed_group.reset(new ArrayHandle<unsigned int>(m_pdata->getRTags(),
                                                          access_location::host,
                                                          access_mode::read));

        // check if some of the masses are less or equal to 0
        for (unsigned int idx = 0; idx < N_tot; ++idx)
            {
            unsigned int particle_index = h_embed_group->data[idx];
            const Scalar4 vel_mass = h_vel_embed->data[particle_index];
            const Scalar mass = vel_mass.w;
            if (mass <= 0)
                {
                m_exec_conf->msg->warning() << "Some particles have a mass <= 0, may lead to "
                                               "invalid results during momentum transfer."
                                            << std::endl;
                break;
                }
            }
        }
    m_checked_collision_warnings = true;
    }
void mpcd::CollisionMethod::beginRigidBodyCollision(uint64_t timestep) { }
void mpcd::CollisionMethod::finishRigidBodyCollision(uint64_t timestep) { }
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
