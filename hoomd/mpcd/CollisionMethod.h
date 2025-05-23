// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/CollisionMethod.h
 * \brief Declaration of mpcd::CollisionMethod
 */

#ifndef MPCD_COLLISION_METHOD_H_
#define MPCD_COLLISION_METHOD_H_

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#ifdef ENABLE_HIP
#include "CollisionMethod.cuh"
#endif

#include "CellList.h"
#include "hoomd/md/ForceComposite.h"

#include "hoomd/Autotuned.h"
#include "hoomd/ParticleGroup.h"
#include "hoomd/SystemDefinition.h"
#include <pybind11/pybind11.h>

namespace hoomd
    {
namespace mpcd
    {
//! MPCD collision method
/*!
 * This class forms the generic base for an MPCD collision method. It handles the boiler plate of
 * setting up the method and implementing the collision. Each deriving class should implement a
 * rule() that gives the physics of the collision.
 */
class PYBIND11_EXPORT CollisionMethod : public Autotuned
    {
    public:
    //! Constructor
    CollisionMethod(std::shared_ptr<SystemDefinition> sysdef,
                    uint64_t cur_timestep,
                    uint64_t period,
                    int phase);
    //! Destructor
    virtual ~CollisionMethod() { }

    //! Implementation of the collision rule
    void collide(uint64_t timestep);

    //! Peek if a collision will occur on this timestep
    virtual bool peekCollide(uint64_t timestep) const;

    //! Get the particle group that is coupled to the MPCD solvent through the collision step.
    std::shared_ptr<ParticleGroup> getEmbeddedGroup()
        {
        return m_embed_group;
        }

    //! Sets a group of particles that is coupled to the MPCD solvent through the collision step
    /*!
     * \param embed_group Group to embed
     */
    void setEmbeddedGroup(std::shared_ptr<ParticleGroup> embed_group)
        {
        if (embed_group != m_embed_group)
            {
            m_checked_collision_warnings = false;
            }
        m_embed_group = embed_group;
        if (m_cl)
            {
            m_cl->setEmbeddedGroup(embed_group);
            }
        }

    //! Get the period of the collision method
    uint64_t getPeriod() const
        {
        return m_period;
        }

    //! Set the period of the collision method
    void setPeriod(uint64_t cur_timestep, uint64_t period);

    //! Get the cell list used for collisions
    std::shared_ptr<mpcd::CellList> getCellList() const
        {
        return m_cl;
        }

    //! Set the cell list used for collisions
    virtual void setCellList(std::shared_ptr<mpcd::CellList> cl)
        {
        m_cl = cl;
        if (m_cl)
            {
            m_cl->setEmbeddedGroup(m_embed_group);
            }
        }

    //! Get the rigid body definitions
    std::shared_ptr<hoomd::md::ForceComposite> getRigid()
        {
        return m_rigid_bodies;
        }

    //! Set the rigid body definitions
    void setRigid(std::shared_ptr<hoomd::md::ForceComposite> new_rigid)
        {
        m_rigid_bodies = new_rigid;
        }

    protected:
    std::shared_ptr<SystemDefinition> m_sysdef;                //!< HOOMD system definition
    std::shared_ptr<hoomd::ParticleData> m_pdata;              //!< HOOMD particle data
    std::shared_ptr<mpcd::ParticleData> m_mpcd_pdata;          //!< MPCD particle data
    std::shared_ptr<const ExecutionConfiguration> m_exec_conf; //!< Execution configuration

    std::shared_ptr<mpcd::CellList> m_cl;                      //!< MPCD cell list
    std::shared_ptr<ParticleGroup> m_embed_group;              //!< Embedded particles
    std::shared_ptr<hoomd::md::ForceComposite> m_rigid_bodies; //!< definition for rigid bodies

    uint64_t m_period;        //!< Number of timesteps between collisions
    uint64_t m_next_timestep; //!< Timestep next collision should be performed

    bool m_checked_collision_warnings; //!< True if collision related warnings have been checked

    GPUArray<Scalar4> m_initial_velocity; //!< Initial velocities of the embedded particles
    GPUArray<Scalar3> m_linmom_accum;     //!< Accumulated change in linear momentum of rigid bodies
    GPUArray<Scalar3> m_angmom_accum; //!< Accumulated change in angular momentum of rigid bodies

    //! Check if a collision should occur and advance the timestep counter
    virtual bool shouldCollide(uint64_t timestep);

    //! Check for issues related to applying collision to rigid bodies
    void checkCollisionWarnings(uint64_t timestep);

    //! Begin process of applying collisions to rigid bodies
    void storeInitialEmbeddedGroupVelocities(uint64_t timestep);

    //! Accumulate momenta changes of constituent particles of rigid bodies
    void accumulateRigidBodyMomenta(uint64_t timestep);

    //! Finish process of applying collisions to rigid bodies
    void transferRigidBodyMomenta(uint64_t timestep);

#ifdef ENABLE_HIP
    //! Begin process of applying collisions to rigid bodies (GPU version)
    void storeInitialEmbeddedGroupVelocitiesGPU(uint64_t timestep);

    //! Accumulate momenta changes of constituent particles of rigid bodies (GPU version)
    void accumulateRigidBodyMomentaGPU(uint64_t timestep);

    //! Finish process of applying collisions to rigid bodies (GPU version)
    void transferRigidBodyMomentaGPU(uint64_t timestep);

    std::shared_ptr<Autotuner<1>> m_store_tuner;      //!< Tuner for storing velocities
    std::shared_ptr<Autotuner<1>> m_accumulate_tuner; //!< Tuner for accumulating momenta
    std::shared_ptr<Autotuner<1>> m_transfer_tuner;   //!< Tuner for transfering momenta
#endif                                                // ENABLE_HIP

    //! Call the collision rule
    virtual void rule(uint64_t timestep) { }
    };
    } // end namespace mpcd
    } // end namespace hoomd
#endif // MPCD_COLLISION_METHOD_H_
