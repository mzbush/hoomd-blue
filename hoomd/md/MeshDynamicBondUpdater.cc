// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file MeshDynamicBondUpdater.cc
    \brief Defines the MeshDynamicBondUpdater class
*/

#include "MeshDynamicBondUpdater.h"
#include "hoomd/RNGIdentifiers.h"
#include "hoomd/RandomNumbers.h"

#include <iostream>

#include <pybind11/stl_bind.h>
PYBIND11_MAKE_OPAQUE(std::vector<std::shared_ptr<hoomd::ForceCompute>>);

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

    m_update_order.resize(m_mesh->getMeshBondData()->getN());


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

    for (auto& force : m_forces)
        {
        force->precomputeParameter();
        //force->writeParameter();
        }

    {

    const Index2D& table_indexer = m_mesh->getMeshBondData()->getGPUTableIndexer();

    ArrayHandle<typename MeshBond::members_t> h_meshbondlist(m_mesh->getMeshBondData()->getGPUTable(),
                                                                 access_location::host,
                                                                 access_mode::read);

    ArrayHandle<unsigned int> h_n_meshbond(
        m_mesh->getMeshBondData()->getNGroupsArray(),
        access_location::host,
        access_mode::read);

    ArrayHandle<unsigned int> h_meshbond_pos_list(
        m_mesh->getMeshBondData()->getGPUPosTable(),
        access_location::host,
        access_mode::read);


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

    hoomd::RandomGenerator rng(
         hoomd::Seed(hoomd::RNGIdentifier::MeshDynamicBondUpdateOrder, timestep, seed),
         hoomd::Counter(m_exec_conf->getRank()));

// maybe too slow
    for (unsigned int i = 0; i < size; i++)
             m_update_order[i] = i;

    for (unsigned int i = size-1; i>0; i--)
	{
	unsigned int j = hoomd::UniformIntDistribution(i)(rng);
        std::swap(m_update_order[i], m_update_order[j]);
	}

    std::vector<uint2> changed;

    for (unsigned int cur_bond = 0; cur_bond < size; cur_bond++)
        {
	unsigned int i = m_update_order[cur_bond];
        typename MeshBond::members_t& bond = h_bonds.data[i];
        assert(bond.tag[0] < m_pdata->getMaximumTag() + 1);
        assert(bond.tag[1] < m_pdata->getMaximumTag() + 1);

        // transform a and b into indices into the particle data arrays
        // (MEM TRANSFER: 4 integers)
        unsigned int tag_a = bond.tag[0];
        unsigned int tag_b = bond.tag[1];

        unsigned int idx_a = h_rtag.data[tag_a];
        unsigned int idx_b = h_rtag.data[tag_b];

        unsigned int tag_c = bond.tag[2];
        unsigned int tag_d = bond.tag[3];

        if (tag_c == tag_d)
            continue;

	bool already_changed = false;
	for( unsigned int ii = 0; ii < changed.size(); ii++)
	   {
	   if( (changed[ii].x == tag_c && changed[ii].y == tag_d) || (changed[ii].x == tag_d && changed[ii].y == tag_c) )
	      {
              already_changed = true;
	      break;
	      }
	   }
	if(already_changed)
	   continue;

        unsigned int idx_c = h_rtag.data[tag_c];
        unsigned int idx_d = h_rtag.data[tag_d];

	unsigned int test_idx = idx_c;

	for( unsigned int ii = 0; ii < h_n_meshbond.data[idx_c]; ii++)
	   {
	   if( h_meshbond_pos_list.data[table_indexer(idx_c,ii)] > 1 )
	      continue;
	   test_idx = h_meshbondlist.data[table_indexer(idx_c,ii)].idx[0];
	   if(test_idx == idx_d)
	      break;
	   }

	if(test_idx == idx_d)
	   continue;

	unsigned int tr_idx1 = h_neigh_bonds.data[i].x;
	unsigned int tr_idx2 = h_neigh_bonds.data[i].y;

        typename Angle::members_t& triangleT = h_triangles.data[tr_idx1];

	for( unsigned int j = 0; j<3; j++)
	    {	
	    if(triangleT.tag[j] == tag_d)
	    	{
		tr_idx1 = h_neigh_bonds.data[i].y;
		tr_idx2 = h_neigh_bonds.data[i].x;
		break;
		}
	    }

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
        for (auto& force : m_forces)
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

	   uint2 bb;
	   bb.x = v_idx[2];
	   bb.y = v_idx[3];
	   changed.push_back(bb);

	   if (have_to_check_surrounding)
		   {
		   for (auto& force : m_forces)
			energyDifference += force->energyDiffSurrounding(h_rtag.data[v_idx[0]],h_rtag.data[v_idx[1]],h_rtag.data[v_idx[2]],h_rtag.data[v_idx[3]],h_rtag.data[v_idx[4]],h_rtag.data[v_idx[5]],h_rtag.data[v_idx[6]],h_rtag.data[v_idx[7]], type_id);
		   part_func = exp(-m_inv_T * energyDifference);
		   }
	   }

        if (part_func > rand_number)
            {
	    //Flip bond
	    if(v_idx[2] > v_idx[3])
		{
	    	h_bonds.data[i].tag[1] = v_idx[2];
	    	h_bonds.data[i].tag[0] = v_idx[3];
		}
	    else
	    	{
	    	h_bonds.data[i].tag[0] = v_idx[2];
	    	h_bonds.data[i].tag[1] = v_idx[3];
		}
	    h_bonds.data[i].tag[2] = v_idx[0];
	    h_bonds.data[i].tag[3] = v_idx[1];

	    //Update triagles asociated with the bond
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

	    //Update triangle neighbors list for the surrounding bonds 
	    h_neigh_bonds.data[b_idx[0]].x = tr_idx[0];
	    h_neigh_bonds.data[b_idx[0]].y = tr_idx[2];

	    h_neigh_bonds.data[b_idx[2]].x = tr_idx[0];
	    h_neigh_bonds.data[b_idx[2]].y = tr_idx[4];
	    
	    h_neigh_bonds.data[b_idx[1]].x = tr_idx[1];
	    h_neigh_bonds.data[b_idx[1]].y = tr_idx[3];

	    h_neigh_bonds.data[b_idx[3]].x = tr_idx[1];
	    h_neigh_bonds.data[b_idx[3]].y = tr_idx[5];

	    //Update bond neighbors list for the two triangles 
            h_neigh_triags.data[tr_idx[0]].x = i;
	    h_neigh_triags.data[tr_idx[0]].y = b_idx[0];
	    h_neigh_triags.data[tr_idx[0]].z = b_idx[2];
	    
	    h_neigh_triags.data[tr_idx[1]].x = i;
	    h_neigh_triags.data[tr_idx[1]].y = b_idx[1];
	    h_neigh_triags.data[tr_idx[1]].z = b_idx[3];

            for (auto& force : m_forces)
                {
                force->postcomputeParameter(idx_a, idx_b, idx_cc, idx_dd, type_id);
                }
            m_mesh->getMeshBondData()->groupReorder();
            m_mesh->getMeshTriangleData()->groupReorder();
            }
        }

    }
    //for (auto& force : forces)
    //    {
        //force->writeParameter();
    //    force->precomputeParameter();
        //force->writeParameter();
    //    }
    }

namespace detail
    {
void export_MeshDynamicBondUpdater(pybind11::module& m)
    {
    pybind11::bind_vector<std::vector<std::shared_ptr<ForceCompute>>>(m, "ForceComputeList");
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
