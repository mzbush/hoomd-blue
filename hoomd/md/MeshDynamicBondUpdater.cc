// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file MeshDynamicBondUpdater.cc
    \brief Defines the MeshDynamicBondUpdater class
*/

#include "MeshDynamicBondUpdater.h"
#include "hoomd/RNGIdentifiers.h"
#include "hoomd/RandomNumbers.h"

#include <iostream>

using namespace std;

namespace hoomd
    {
namespace md
    {
/*! \param sysdef System definition
 *  \param rotational_diffusion The diffusion across time
 *  \param group the particles to diffusion rotation on
 */

MeshDynamicBondUpdater::MeshDynamicBondUpdater(std::shared_ptr<SystemDefinition> sysdef,
                                               std::shared_ptr<Trigger> trigger,
                                               std::shared_ptr<Integrator> integrator,
                                               std::shared_ptr<MeshDefinition> mesh,
                                               Scalar T)
    : Updater(sysdef, trigger), m_integrator(integrator), m_mesh(mesh), m_inv_T(1.0 / T)
    {
    assert(m_pdata);
    assert(m_integrator);
    assert(m_mesh);
    m_exec_conf->msg->notice(5) << "Constructing MeshDynamicBondUpdater" << endl;
    }

MeshDynamicBondUpdater::~MeshDynamicBondUpdater()
    {
    m_exec_conf->msg->notice(5) << "Destroying MeshDynamicBondUpdater" << endl;
    }


/** Perform the needed calculations to update particle orientations
    \param timestep Current time step of the simulation
*/
void MeshDynamicBondUpdater::update(uint64_t timestep)
    {
    std::vector<std::shared_ptr<ForceCompute>> forces = m_integrator->getForces();

    uint16_t seed = m_sysdef->getSeed();

    for (auto& force : forces)
        {
        force->precomputeParameter();
        }

    ArrayHandle<typename MeshBond::members_t> h_bonds(m_mesh->getMeshBondData()->getMembersArray(),
                                                      access_location::host,
                                                      access_mode::readwrite);

    ArrayHandle<unit2> h_neigh_bonds(m_mesh->getNeighToBond()->getMembersArray(),
                                                      access_location::host,
                                                      access_mode::readwrite);

    ArrayHandle<typename MeshTriangle::members_t> h_triangles(
        m_mesh->getMeshTriangleData()->getMembersArray(),
        access_location::host,
        access_mode::readwrite);

    ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);

    ArrayHandle<typeval_t> h_typeval(m_mesh->getMeshBondData()->getTypeValArray(),
                                     access_location::host,
                                     access_mode::read);

    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);

    // for each of the angles
    const unsigned int size = (unsigned int)m_mesh->getMeshBondData()->getN();

    bool changeDetected = false;

    std::vector<unsigned int> changed;

    for (unsigned int i = 0; i < size; i++)
        {
        const typename MeshBond::members_t& bond = h_bonds.data[i];
        assert(bond.tag[0] < m_pdata->getMaximumTag() + 1);
        assert(bond.tag[1] < m_pdata->getMaximumTag() + 1);

        // transform a and b into indices into the particle data arrays
        // (MEM TRANSFER: 4 integers)
        unsigned int tag_a = bond.tag[0];
        unsigned int tag_b = bond.tag[1];

        unsigned int idx_a = h_rtag.data[tag_a];
        unsigned int idx_b = h_rtag.data[tag_b];

        unsigned int type_a = __scalar_as_int(h_pos.data[idx_a].w);
        unsigned int type_b = __scalar_as_int(h_pos.data[idx_b].w);

        //bool already_used = false;
        //for (unsigned int j = 0; j < changed.size(); j++)
        //    {
        //    if (tag_a == changed[j] || tag_b == changed[j])
        //        {
        //        already_used = true;
        //        break;
        //        }
        //    }

        //if (already_used)
        //    continue;

        unsigned int tag_c = bond.tag[2];
        unsigned int tag_d = bond.tag[3];

        if (tag_c == tag_d)
            continue;

        unsigned int idx_c = h_rtag.data[tag_c];
        unsigned int idx_d = h_rtag.data[tag_d];

	unsigned int tr_idx1 = h_neigh_bonds.data[i].tag[0];
	unsigned int tr_idx2 = h_neigh_bonds.data[i].tag[1];

        const typename MeshTriangle::members_t& triangle1 = h_triangles.data[tr_idx1];

        unsigned int iterator = 0;

        bool a_before_b = false;
        bool tr1_with_c = false;

        while (tag_a != triangle1.tag[iterator])
            iterator++;

        iterator = (iterator + 1) % 3;

        if (tag_b == triangle1.tag[iterator])
	    {
            a_before_b = true;
            iterator = (iterator + 1) % 3;
	    }
	
	if(tag_c == triangle1.tag[iterator])
	    tr1_with_c = true;

        unsigned int type_id = h_typeval.data[i].type;

        Scalar energyDifference = 0;

        unsigned int idx_cc = idx_d;
        unsigned int idx_dd = idx_c;

        if (a_before_b)
            {
	    idx_cc = idx_c;
	    idx_dd = idx_d;
	    }

	bool have_to_check_surrounding = false;
        for (auto& force : forces)
	    {
            energyDifference += force->energyDiff(idx_a, idx_b, idx_cc, idx_dd, type_id);

	    if(force->checkSurrounding())
		    have_to_check_surrounding = true;
	    }

	if(have_to_check_surrounding)
           {
	   std::vector<unsigned int> tr_idx(6);
	   counter = 2;
	   tr_idx[0] = tr_idx1;
	   tr_idx[1] = tr_idx2;

	   for(unsigned int j = 0; j < 2; ++j)
	   	{
	   	for(unsigned int k = 0; k < 3; ++k)
	   		{
		   	unsigned int tr_idx_candidate = h_neigh_triags.data[tr_idx[j]].tag[k];
			if(tr_idx_candidate != i)
				{
				tr_idx[counter] = tr_idx_candidate;
				counter++;
	   			}
	   		}
		}

           for (auto& force : forces)
	      	energyDifference += force->energyDiffSurrounding(tr_idx, type_id);

	   }

        // Initialize the RNG
        RandomGenerator rng(hoomd::Seed(RNGIdentifier::MeshDynamicBondUpdater, timestep, seed),
                            hoomd::Counter(i));

        // compute the random force
        UniformDistribution<Scalar> uniform(0, Scalar(1));

        if (exp(-m_inv_T * energyDifference) > uniform(rng))
            {
            changeDetected = true;

	    h_bonds.data[i]; // bond a,b to c,d

	    h_triangles[tr_idx1]; // triangle a,b,c/d to a,c,d 
	    h_triangles[tr_id2]; // triangle a,b,c/d to b,c,d 
	
	    h_bonds.data[b_idx1]; // bond a,c,b,x to a,c,d,x
	    h_bonds.data[b_idx2]; // bond a,d,b,x to a,d,c,x

	    h_bonds.data[b_idx3]; // bond b,c,b,x to b,c,d,x
	    h_bonds.data[b_idx4]; // bond b,d,b,x to b,d,c,x
				  //
	    h_neight_bonds.data[b_idx1]; // nbond tr1/2,trx,try to tr1,trx,try
	    h_neight_bonds.data[b_idx2]; // nbond tr1/2,trx,try to tr1,trx,try 

	    h_neight_bonds.data[b_idx3]; // nbond tr1/2,trx,try to tr2,trx,try
	    h_neight_bonds.data[b_idx4]; // nbond tr1/2,trx,try to tr2,trx,try 
					 //
	    h_neight_triags.data[tr_idx1]; // nbond i,b_idx1/2,b_idx3/4 to i,b_idx1,b_idx2 
	    h_neight_triags.data[tr_idx2]; // nbond i,b_idx2/1,b_idx4/3 to i,b_idx3,b_idx4
	
			



            changed.push_back(tag_a);
            changed.push_back(tag_b);
            changed.push_back(tag_c);
            changed.push_back(tag_d);

            typename MeshBond::members_t bond_n;
            typename MeshTriangle::members_t triangle1_n;
            typename MeshTriangle::members_t triangle2_n;

            bond_n.tag[0] = tag_c;
            bond_n.tag[1] = tag_d;
            bond_n.tag[2] = tr_idx1;
            bond_n.tag[3] = tr_idx2;

            h_bonds.data[i] = bond_n;

            bool needs_flipping = true;

            if (iterator < 2)
                {
                if (triangle2.tag[iterator + 1] == tag_a)
                    needs_flipping = false;
                }
            else
                {
                if (triangle2.tag[0] == tag_a)
                    needs_flipping = false;
                }

            triangle1_n.tag[0] = tag_a;
            triangle2_n.tag[0] = tag_b;

            if (needs_flipping)
                {
                triangle1_n.tag[2] = tag_c;
                triangle1_n.tag[1] = tag_d;
                triangle2_n.tag[2] = tag_d;
                triangle2_n.tag[1] = tag_c;
                }
            else
                {
                triangle1_n.tag[1] = tag_c;
                triangle1_n.tag[2] = tag_d;
                triangle2_n.tag[1] = tag_d;
                triangle2_n.tag[2] = tag_c;
                }

            for (int j = 3; j < 6; j++)
                {
                int k = triangle1.tag[j];
                if (k != i)
                    {
                    typename MeshBond::members_t& bond_s = h_bonds.data[k];

                    unsigned int tr_idx;
                    if (bond_s.tag[0] == tag_a || bond_s.tag[1] == tag_a)
                        {
                        tr_idx = tr_idx1;
                        triangle1_n.tag[3] = k;
                        }
                    else
                        {
                        tr_idx = tr_idx2;
                        triangle2_n.tag[3] = k;
                        }

                    if (bond_s.tag[2] == tr_idx1 || bond_s.tag[2] == tr_idx2)
                        bond_s.tag[2] = tr_idx;
                    else
                        bond_s.tag[3] = tr_idx;
                    h_bonds.data[k] = bond_s;
                    }
                k = triangle2.tag[j];
                if (k != i)
                    {
                    typename MeshBond::members_t& bond_s = h_bonds.data[k];

                    unsigned int tr_idx;
                    if (bond_s.tag[0] == tag_a || bond_s.tag[1] == tag_a)
                        {
                        tr_idx = tr_idx1;
                        triangle1_n.tag[4] = k;
                        }
                    else
                        {
                        tr_idx = tr_idx2;
                        triangle2_n.tag[4] = k;
                        }

                    if (bond_s.tag[2] == tr_idx1 || bond_s.tag[2] == tr_idx2)
                        bond_s.tag[2] = tr_idx;
                    else
                        bond_s.tag[3] = tr_idx;
                    h_bonds.data[k] = bond_s;
                    }
                }

            triangle1_n.tag[5] = i;
            triangle2_n.tag[5] = i;

            h_triangles.data[tr_idx1] = triangle1_n;
            h_triangles.data[tr_idx2] = triangle2_n;

            if (a_before_b)
                {
                for (auto& force : forces)
                    {
                    force->postcompute(idx_a, idx_b, idx_c, idx_d);
                    }
                }
            else
                {
                for (auto& force : forces)
                    {
                    force->postcompute(idx_a, idx_b, idx_d, idx_c);
                    }
                }
            }
        }

    if (changeDetected)
        {
        m_mesh->getMeshBondData()->meshChanged();
        m_mesh->getMeshTriangleData()->meshChanged();
        }
    }

namespace detail
    {
void export_MeshDynamicBondUpdater(pybind11::module& m)
    {
    pybind11::class_<MeshDynamicBondUpdater, Updater, std::shared_ptr<MeshDynamicBondUpdater>>(
        m,
        "MeshDynamicBondUpdater")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<Trigger>,
                            std::shared_ptr<Integrator>,
                            std::shared_ptr<MeshDefinition>,
                            Scalar>())
        .def_property("kT", &MeshDynamicBondUpdater::getT, &MeshDynamicBondUpdater::setT);
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
