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
    assert(m_mesh);

    m_mesh->createMeshNeighborhood();

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
    uint16_t seed = m_sysdef->getSeed();

    std::vector<std::shared_ptr<ForceCompute>> forces = m_integrator->getForces();

    //for (auto& force : m_forces)
    for (auto& force : forces)
        {
        force->precomputeParameter();
        }

    ArrayHandle<typename MeshBond::members_t> h_bonds(m_mesh->getMeshBondData()->getMembersArray(),
                                                      access_location::host,
                                                      access_mode::readwrite);

    ArrayHandle<uint2> h_neigh_bonds(m_mesh->getNeighToBond(),
                                                      access_location::host,
                                                      access_mode::readwrite);

    ArrayHandle<typename Angle::members_t> h_triangles(
        m_mesh->getMeshTriangleData()->getMembersArray(),
        access_location::host,
        access_mode::readwrite);

    ArrayHandle<uint3> h_neigh_triags(m_mesh->getNeighToTriag(),
                                                      access_location::host,
                                                      access_mode::readwrite);

    ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);

    ArrayHandle<typeval_t> h_typeval(m_mesh->getMeshBondData()->getTypeValArray(),
                                     access_location::host,
                                     access_mode::read);

    // for each of the angles
    const unsigned int size = (unsigned int)m_mesh->getMeshBondData()->getN();

    std::vector<unsigned int> changed;

    for (unsigned int i = 0; i < size; i++)
        {
        typename MeshBond::members_t& bond = h_bonds.data[i];
        assert(bond.tag[0] < m_pdata->getMaximumTag() + 1);
        assert(bond.tag[1] < m_pdata->getMaximumTag() + 1);

        // transform a and b into indices into the particle data arrays
        // (MEM TRANSFER: 4 integers)
        unsigned int tag_a = bond.tag[0];
        unsigned int tag_b = bond.tag[1];

        unsigned int idx_a = h_rtag.data[tag_a];
        unsigned int idx_b = h_rtag.data[tag_b];

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

        unsigned int idx_c = h_rtag.data[tag_c];
        unsigned int idx_d = h_rtag.data[tag_d];

        if (tag_c == tag_d)
            continue;

	unsigned int tr_idx1 = h_neigh_bonds.data[i].x;
	unsigned int tr_idx2 = h_neigh_bonds.data[i].y;

        typename Angle::members_t& triangle1 = h_triangles.data[tr_idx1];

        unsigned int iterator = 0;

        bool a_before_b = false;

        while (tag_a != triangle1.tag[iterator])
            iterator++;

        iterator = (iterator + 1) % 3;

        if (tag_b == triangle1.tag[iterator])
	    {
            a_before_b = true;
            iterator = (iterator + 1) % 3;
	    }
	
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
        //for (auto& force : m_forces)
        for (auto& force : forces)
	    {
            energyDifference += force->energyDiff(idx_a, idx_b, idx_cc, idx_dd, type_id);
	    if(force->checkSurrounding())
		    have_to_check_surrounding = true;
	    }



        // Initialize the RNG
        RandomGenerator rng(hoomd::Seed(RNGIdentifier::MeshDynamicBondUpdater, timestep, seed),
                            hoomd::Counter(i));

        // compute the random force
        UniformDistribution<Scalar> uniform(0, Scalar(1));

	Scalar rand_number = uniform(rng);
	Scalar part_func = exp(-m_inv_T * energyDifference);

	std::vector<unsigned int> tr_idx(6);
	std::vector<unsigned int> b_idx(4);
	std::vector<unsigned int> v_idx(8);

        if (have_to_check_surrounding || part_func > rand_number )
           {
	   tr_idx[1] = tr_idx1;
	   tr_idx[0] = tr_idx2;

	   v_idx[0]=tag_a;
	   v_idx[1]=tag_b;
	   v_idx[2]=tag_d;
	   v_idx[3]=tag_c;

	   if(a_before_b)
	   	{
	   	tr_idx[0] = tr_idx1;
	   	tr_idx[1] = tr_idx2;
	   	v_idx[2]=tag_c;
	   	v_idx[3]=tag_d;
		}
	   unsigned int counter = 4;

	   for(unsigned int j = 0; j < 2; ++j)
	   	{
		unsigned int bic = h_neigh_triags.data[tr_idx[j]].x;
		unsigned int bic2 = h_neigh_triags.data[tr_idx[j]].y;
		if(bic == i)
			{
			bic = bic2;
			bic2 = h_neigh_triags.data[tr_idx[j]].z;
			}
		else
			{
			if(bic2 == i)
				bic2 =  h_neigh_triags.data[tr_idx[j]].z;
			}
		if( h_bonds.data[bic].tag[0] == tag_a || h_bonds.data[bic].tag[1] == tag_a)
			{
				b_idx[2*j] = bic;
				b_idx[2*j+1] = bic2;
			}
		else
			{
				b_idx[2*j+1] = bic;
				b_idx[2*j] = bic2;
			}
	   	for(unsigned int k = 0; k < 2; ++k)
			{
			unsigned int tic =  h_neigh_bonds.data[b_idx[2*j+k]].x;
			if(tic == tr_idx[j])
				tic =  h_neigh_bonds.data[b_idx[2*j+k]].y;
			unsigned int zaehler = 1;
			unsigned int nv_idx =  h_triangles.data[tic].tag[0];
			while(nv_idx == v_idx[k] || nv_idx == v_idx[2+j])
				{
				nv_idx =  h_triangles.data[tic].tag[zaehler];
				zaehler++;
				}
			v_idx[counter] = nv_idx;
			tr_idx[counter-2] = tic;
			counter++;
			}
		}
	   if (have_to_check_surrounding)
		   {
		   //for (auto& force : m_forces)
		   for (auto& force : forces)
			energyDifference += force->energyDiffSurrounding(h_rtag.data[v_idx[0]],h_rtag.data[v_idx[1]],h_rtag.data[v_idx[2]],h_rtag.data[v_idx[3]],h_rtag.data[v_idx[4]],h_rtag.data[v_idx[5]],h_rtag.data[v_idx[6]],h_rtag.data[v_idx[7]], type_id);
		   part_func = exp(-m_inv_T * energyDifference);
		   }
	   }

        if (part_func > rand_number)
            {
	    h_bonds.data[i].tag[0] = v_idx[2];
	    h_bonds.data[i].tag[1] = v_idx[3];
	    if(v_idx[2] > v_idx[3])
		{
	    	h_bonds.data[i].tag[1] = v_idx[2];
	    	h_bonds.data[i].tag[0] = v_idx[3];
		}

	    h_bonds.data[i].tag[2] = v_idx[0];
	    h_bonds.data[i].tag[3] = v_idx[1];

	    h_triangles.data[tr_idx[0]].tag[0] = v_idx[0];
	    h_triangles.data[tr_idx[0]].tag[1] = v_idx[3];
	    h_triangles.data[tr_idx[0]].tag[2] = v_idx[2];
    	    
	    h_triangles.data[tr_idx[1]].tag[0] = v_idx[1];
	    h_triangles.data[tr_idx[1]].tag[1] = v_idx[2];
	    h_triangles.data[tr_idx[1]].tag[2] = v_idx[3];

	    h_bonds.data[b_idx[0]].tag[2] = v_idx[3];
	    h_bonds.data[b_idx[0]].tag[3] = v_idx[4];

	    h_bonds.data[b_idx[1]].tag[2] = v_idx[3];
	    h_bonds.data[b_idx[1]].tag[3] = v_idx[5];

	    h_bonds.data[b_idx[2]].tag[2] = v_idx[2];
	    h_bonds.data[b_idx[2]].tag[3] = v_idx[6];

	    h_bonds.data[b_idx[3]].tag[2] = v_idx[2];
	    h_bonds.data[b_idx[3]].tag[3] = v_idx[7];

	    h_neigh_bonds.data[b_idx[0]].x = tr_idx[0];
	    h_neigh_bonds.data[b_idx[0]].y = tr_idx[2];

	    h_neigh_bonds.data[b_idx[2]].x = tr_idx[0];
	    h_neigh_bonds.data[b_idx[2]].y = tr_idx[4];

	    h_neigh_bonds.data[b_idx[1]].x = tr_idx[1];
	    h_neigh_bonds.data[b_idx[1]].y = tr_idx[3];

	    h_neigh_bonds.data[b_idx[3]].x = tr_idx[1];
	    h_neigh_bonds.data[b_idx[3]].y = tr_idx[5];

	    h_neigh_triags.data[tr_idx[0]].x = i;
	    h_neigh_triags.data[tr_idx[0]].y = b_idx[0];
	    h_neigh_triags.data[tr_idx[0]].z = b_idx[2];

	    h_neigh_triags.data[tr_idx[1]].x = i;
	    h_neigh_triags.data[tr_idx[1]].y = b_idx[1];
	    h_neigh_triags.data[tr_idx[1]].z = b_idx[3];

            //for (auto& force : m_forces)
            for (auto& force : forces)
                {
                force->postcomputeParameter(idx_a, idx_b, idx_cc, idx_dd, type_id);
                }
            m_mesh->getMeshBondData()->groupReorder();
            m_mesh->getMeshTriangleData()->groupReorder();
            }
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
        .def_property_readonly("forces", &MeshDynamicBondUpdater::getForces)
        .def_property("kT", &MeshDynamicBondUpdater::getT, &MeshDynamicBondUpdater::setT);
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
