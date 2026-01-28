// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hoomd/mpcd/CellList.h"
#ifdef ENABLE_HIP
#include "hoomd/mpcd/CellListGPU.h"
#endif // ENABLE_HIP

#include "hoomd/Communicator.h"
#include "hoomd/SnapshotSystemData.h"
#include "hoomd/mpcd/Communicator.h"
#include "hoomd/test/upp11_config.h"

#include "utils.h"

HOOMD_UP_MAIN()

using namespace hoomd;

//! Test for correct calculation of MPCD grid dimensions
/*!
 * \param exec_conf Execution configuration
 * \param mpi_x Flag if expecting MPI in x dimension
 * \param mpi_y Flag if expecting MPI in y dimension
 * \param mpi_z Flag if expecting MPI in z dimension
 *
 * \tparam CL CellList class to use, should be consistent with \a exec_conf mode
 */
template<class CL>
void celllist_dimension_test(std::shared_ptr<ExecutionConfiguration> exec_conf,
                             bool mpi_x,
                             bool mpi_y,
                             bool mpi_z,
                             const Scalar3& tilt)
    {
    // only run tests on first partition
    if (exec_conf->getPartition() != 0)
        return;

    std::shared_ptr<SnapshotSystemData<Scalar>> snap(new SnapshotSystemData<Scalar>());
    snap->global_box = std::make_shared<BoxDim>(5.0);
    snap->global_box->setTiltFactors(tilt.x, tilt.y, tilt.z);
    snap->particle_data.type_mapping.push_back("A");
    snap->mpcd_data.resize(1);
    snap->mpcd_data.type_mapping.push_back("A");

    // configure domain decompsition
    std::vector<Scalar> fx, fy, fz;
    unsigned int n_req_ranks = 1;
    if (mpi_x)
        {
        n_req_ranks *= 2;
        fx.push_back(0.5);
        }
    if (mpi_y)
        {
        n_req_ranks *= 2;
        fy.push_back(0.45);
        }
    if (mpi_z)
        {
        n_req_ranks *= 2;
        fz.push_back(0.55);
        }
    UP_ASSERT_EQUAL(exec_conf->getNRanks(), n_req_ranks);
    std::shared_ptr<DomainDecomposition> decomposition(
        new DomainDecomposition(exec_conf, snap->global_box->getL(), fx, fy, fz));
    std::shared_ptr<SystemDefinition> sysdef(new SystemDefinition(snap, exec_conf, decomposition));
    std::shared_ptr<Communicator> pdata_comm(new Communicator(sysdef, decomposition));
    sysdef->setCommunicator(pdata_comm);

        {
        const Index3D& di = decomposition->getDomainIndexer();
        UP_ASSERT_EQUAL(di.getW(), (mpi_x) ? 2 : 1);
        UP_ASSERT_EQUAL(di.getH(), (mpi_y) ? 2 : 1);
        UP_ASSERT_EQUAL(di.getD(), (mpi_z) ? 2 : 1);
        }

    // initialize mpcd system
    auto pdata_1 = sysdef->getMPCDParticleData();
    std::shared_ptr<mpcd::CellList> cl(new CL(sysdef, make_uint3(5, 5, 5), false));

    // compute the cell list
    cl->computeDimensions();

    const bool is_orthorhombic = (tilt == make_scalar3(0, 0, 0));
    if (is_orthorhombic)
        {
        // check domain origins
        const int3 origin = cl->getOriginIndex();
        const uint3 pos = decomposition->getGridPos();
        if (mpi_x)
            {
            // right rank gets middle cell -> 0 and 3
            if (pos.x)
                {
                UP_ASSERT_EQUAL(origin.x, 3);
                }
            else
                {
                UP_ASSERT_EQUAL(origin.x, 0);
                }
            }
        else
            {
            UP_ASSERT_EQUAL(origin.x, 0);
            }

        if (mpi_y)
            {
            // hight rank gets middle cell due to domain decomp pushback -> 0 and 2
            // biased to lower edge -> upper domains need extra cell
            if (pos.y)
                {
                UP_ASSERT_EQUAL(origin.y, 2);
                }
            else
                {
                UP_ASSERT_EQUAL(origin.y, 0);
                }
            }
        else
            {
            UP_ASSERT_EQUAL(origin.y, 0);
            }

        if (mpi_z)
            {
            // lower rank gets middle cell -> 0 and 3
            if (pos.z)
                {
                UP_ASSERT_EQUAL(origin.z, 3);
                }
            else
                {
                UP_ASSERT_EQUAL(origin.z, 0);
                }
            }
        else
            {
            UP_ASSERT_EQUAL(origin.z, 0);
            }

        // check domain sizes
        const uint3 dim = cl->getDim();
        if (mpi_x)
            {
            // lower rank gets middle cell
            if (pos.x)
                {
                UP_ASSERT_EQUAL(dim.x, 2);
                }
            else
                {
                UP_ASSERT_EQUAL(dim.x, 3);
                }
            }
        else
            {
            UP_ASSERT_EQUAL(dim.x, 5);
            }

        if (mpi_y)
            {
            // higher rank gets middle cell due to domain decomp pushback
            // biased to lower edge -> upper domains need extra cell
            if (pos.y)
                {
                UP_ASSERT_EQUAL(dim.y, 3);
                }
            else
                {
                UP_ASSERT_EQUAL(dim.y, 2);
                }
            }
        else
            {
            UP_ASSERT_EQUAL(dim.y, 5);
            }

        if (mpi_z)
            {
            // lower rank gets middle cell
            if (pos.z)
                {
                UP_ASSERT_EQUAL(dim.z, 2);
                }
            else
                {
                UP_ASSERT_EQUAL(dim.z, 3);
                }
            }
        else
            {
            UP_ASSERT_EQUAL(dim.z, 5);
            }

        std::array<unsigned int, 6> num_comm = cl->getNComm();
        UP_ASSERT_EQUAL(num_comm[static_cast<unsigned int>(mpcd::detail::face::east)],
                        (mpi_x) ? ((pos.x) ? 0 : 1) : 0);
        UP_ASSERT_EQUAL(num_comm[static_cast<unsigned int>(mpcd::detail::face::west)], 0);
        UP_ASSERT_EQUAL(num_comm[static_cast<unsigned int>(mpcd::detail::face::north)], 0);
        UP_ASSERT_EQUAL(num_comm[static_cast<unsigned int>(mpcd::detail::face::south)],
                        (mpi_y) ? ((pos.y) ? 1 : 0) : 0);
        UP_ASSERT_EQUAL(num_comm[static_cast<unsigned int>(mpcd::detail::face::up)],
                        (mpi_z) ? ((pos.z) ? 0 : 1) : 0);
        UP_ASSERT_EQUAL(num_comm[static_cast<unsigned int>(mpcd::detail::face::down)], 0);

        // check coverage box
        const BoxDim coverage = cl->getCoverageBox();
        if (mpi_x)
            {
            if (pos.x)
                {
                UP_ASSERT_SMALL(coverage.getLo().x, tol);
                UP_ASSERT_CLOSE(coverage.getHi().x, 3.0, tol);
                }
            else
                {
                UP_ASSERT_CLOSE(coverage.getLo().x, -3.0, tol);
                UP_ASSERT_SMALL(coverage.getHi().x, tol);
                }
            }
        else
            {
            UP_ASSERT_CLOSE(coverage.getLo().x, -2.5, tol);
            UP_ASSERT_CLOSE(coverage.getHi().x, 2.5, tol);
            }

        if (mpi_y)
            {
            if (pos.y)
                {
                UP_ASSERT_CLOSE(coverage.getLo().y, -1.0, tol);
                UP_ASSERT_CLOSE(coverage.getHi().y, 3.0, tol);
                }
            else
                {
                UP_ASSERT_CLOSE(coverage.getLo().y, -3.0, tol);
                UP_ASSERT_SMALL(coverage.getHi().y, tol);
                }
            }
        else
            {
            UP_ASSERT_CLOSE(coverage.getLo().y, -2.5, tol);
            UP_ASSERT_CLOSE(coverage.getHi().y, 2.5, tol);
            }

        if (mpi_z)
            {
            if (pos.z)
                {
                UP_ASSERT_SMALL(coverage.getLo().z, tol);
                UP_ASSERT_CLOSE(coverage.getHi().z, 3.0, tol);
                }
            else
                {
                UP_ASSERT_CLOSE(coverage.getLo().z, -3.0, tol);
                UP_ASSERT_CLOSE(coverage.getHi().z, 1.0, tol);
                }
            }
        else
            {
            UP_ASSERT_CLOSE(coverage.getLo().z, -2.5, tol);
            UP_ASSERT_CLOSE(coverage.getHi().z, 2.5, tol);
            }
        }

    /*******************/
    // Change the cell size, and ensure everything stays up to date
    cl->setGlobalDim(make_uint3(10, 10, 10));
    cl->computeDimensions();
    if (is_orthorhombic)
        {
        // check domain origins
        const int3 origin = cl->getOriginIndex();
        const uint3 pos = decomposition->getGridPos();
        if (mpi_x)
            {
            // halfway is now exactly on a domain boundary
            if (pos.x)
                {
                UP_ASSERT_EQUAL(origin.x, 5);
                }
            else
                {
                UP_ASSERT_EQUAL(origin.x, 0);
                }
            }
        else
            {
            UP_ASSERT_EQUAL(origin.x, 0);
            }

        if (mpi_y)
            {
            // this edge falls halfway in the middle of cell 5 now
            if (pos.y)
                {
                UP_ASSERT_EQUAL(origin.y, 5);
                }
            else
                {
                UP_ASSERT_EQUAL(origin.y, 0);
                }
            }
        else
            {
            UP_ASSERT_EQUAL(origin.y, 0);
            }

        if (mpi_z)
            {
            // this edge falls halfway in the middle of cell 5 now
            if (pos.z)
                {
                UP_ASSERT_EQUAL(origin.z, 6);
                }
            else
                {
                UP_ASSERT_EQUAL(origin.z, 0);
                }
            }
        else
            {
            UP_ASSERT_EQUAL(origin.z, 0);
            }

        // check domain sizes
        const uint3 dim = cl->getDim();
        if (mpi_x)
            {
            // split evenly in x -> both domains are same size
            UP_ASSERT_EQUAL(dim.x, 5);
            }
        else
            {
            UP_ASSERT_EQUAL(dim.x, 10);
            }

        if (mpi_y)
            {
            // split evenly in y -> both domains are same size
            UP_ASSERT_EQUAL(dim.y, 5);
            }
        else
            {
            UP_ASSERT_EQUAL(dim.y, 10);
            }

        if (mpi_z)
            {
            // biased to upper edge -> lower domains need extra cell
            if (pos.z)
                {
                UP_ASSERT_EQUAL(dim.z, 4);
                }
            else
                {
                UP_ASSERT_EQUAL(dim.z, 6);
                }
            }
        else
            {
            UP_ASSERT_EQUAL(dim.z, 10);
            }

        std::array<unsigned int, 6> num_comm = cl->getNComm();
        UP_ASSERT_EQUAL(num_comm[static_cast<unsigned int>(mpcd::detail::face::east)], 0);
        UP_ASSERT_EQUAL(num_comm[static_cast<unsigned int>(mpcd::detail::face::west)], 0);
        // due to rounding error, gives 1 instead of 0
        UP_ASSERT_EQUAL(num_comm[static_cast<unsigned int>(mpcd::detail::face::north)],
                        (mpi_y) ? ((pos.y) ? 0 : 1) : 0);
        UP_ASSERT_EQUAL(num_comm[static_cast<unsigned int>(mpcd::detail::face::south)], 0);
        // bias requires communication
        UP_ASSERT_EQUAL(num_comm[static_cast<unsigned int>(mpcd::detail::face::up)],
                        (mpi_z) ? ((pos.z) ? 0 : 1) : 0);
        UP_ASSERT_EQUAL(num_comm[static_cast<unsigned int>(mpcd::detail::face::down)], 0);

        const BoxDim coverage = cl->getCoverageBox();
        if (mpi_x)
            {
            if (pos.x)
                {
                UP_ASSERT_CLOSE(coverage.getLo().x, -0.25, tol);
                UP_ASSERT_CLOSE(coverage.getHi().x, 2.75, tol);
                }
            else
                {
                UP_ASSERT_CLOSE(coverage.getLo().x, -2.75, tol);
                UP_ASSERT_CLOSE(coverage.getHi().x, 0.25, tol);
                }
            }
        else
            {
            UP_ASSERT_CLOSE(coverage.getLo().x, -2.5, tol);
            UP_ASSERT_CLOSE(coverage.getHi().x, 2.5, tol);
            }

        if (mpi_y)
            {
            if (pos.y)
                {
                UP_ASSERT_CLOSE(coverage.getLo().y, -0.25, tol);
                UP_ASSERT_CLOSE(coverage.getHi().y, 2.75, tol);
                }
            else
                {
                UP_ASSERT_CLOSE(coverage.getLo().y, -2.75, tol);
                UP_ASSERT_CLOSE(coverage.getHi().y, -0.25, tol);
                }
            }
        else
            {
            UP_ASSERT_CLOSE(coverage.getLo().y, -2.5, tol);
            UP_ASSERT_CLOSE(coverage.getHi().y, 2.5, tol);
            }

        if (mpi_z)
            {
            if (pos.z)
                {
                UP_ASSERT_CLOSE(coverage.getLo().z, 0.25, tol);
                UP_ASSERT_CLOSE(coverage.getHi().z, 2.75, tol);
                }
            else
                {
                UP_ASSERT_CLOSE(coverage.getLo().z, -2.75, tol);
                // floating point rounding makes this 0.75 not 0.25
                UP_ASSERT_CLOSE(coverage.getHi().z, 0.75, tol);
                }
            }
        else
            {
            UP_ASSERT_CLOSE(coverage.getLo().z, -2.5, tol);
            UP_ASSERT_CLOSE(coverage.getHi().z, 2.5, tol);
            }
        }
    }

//! Test for correct cell listing of a basic system
template<class CL>
void celllist_basic_test(std::shared_ptr<ExecutionConfiguration> exec_conf,
                         const Scalar3& L,
                         const Scalar3& tilt)
    {
    UP_ASSERT_EQUAL(exec_conf->getNRanks(), 8);

    auto ref_box = std::make_shared<BoxDim>(6.0);
    auto box = std::make_shared<BoxDim>(L);
    box->setTiltFactors(tilt.x, tilt.y, tilt.z);

    std::shared_ptr<SnapshotSystemData<Scalar>> snap(new SnapshotSystemData<Scalar>());
    snap->global_box = box;
    snap->particle_data.type_mapping.push_back("A");
    // place each particle in the same cell, but on different ranks
    /*
     * The +/- halves of the box owned by each domain are:
     *    x y z
     * 0: - - -
     * 1: + - -
     * 2: - + -
     * 3: + + -
     * 4: - - +
     * 5: + - +
     * 6: - + +
     * 7: + + +
     */
    snap->mpcd_data.resize(9);
    snap->mpcd_data.type_mapping.push_back("A");
    snap->mpcd_data.position[0] = scale(vec3<Scalar>(-0.1, -0.1, -0.1), ref_box, box);
    snap->mpcd_data.position[1] = scale(vec3<Scalar>(0.1, -0.1, -0.1), ref_box, box);
    snap->mpcd_data.position[2] = scale(vec3<Scalar>(-0.1, 0.1, -0.1), ref_box, box);
    snap->mpcd_data.position[3] = scale(vec3<Scalar>(0.1, 0.1, -0.1), ref_box, box);
    snap->mpcd_data.position[4] = scale(vec3<Scalar>(-0.1, -0.1, 0.1), ref_box, box);
    snap->mpcd_data.position[5] = scale(vec3<Scalar>(0.1, -0.1, 0.1), ref_box, box);
    snap->mpcd_data.position[6] = scale(vec3<Scalar>(-0.1, 0.1, 0.1), ref_box, box);
    snap->mpcd_data.position[7] = scale(vec3<Scalar>(0.1, 0.1, 0.1), ref_box, box);
    // put an extra particle on rank 0 so that at least one temp is defined
    snap->mpcd_data.position[8] = scale(vec3<Scalar>(-0.1, -0.1, -0.1), ref_box, box);

    snap->mpcd_data.velocity[0] = vec3<Scalar>(-1.0, -1.0, -1.0);
    snap->mpcd_data.velocity[1] = vec3<Scalar>(1.0, -1.0, -1.0);
    snap->mpcd_data.velocity[2] = vec3<Scalar>(-1.0, 1.0, -1.0);
    snap->mpcd_data.velocity[3] = vec3<Scalar>(1.0, 1.0, -1.0);
    snap->mpcd_data.velocity[4] = vec3<Scalar>(-1.0, -1.0, 1.0);
    snap->mpcd_data.velocity[5] = vec3<Scalar>(1.0, -1.0, 1.0);
    snap->mpcd_data.velocity[6] = vec3<Scalar>(-1.0, 1.0, 1.0);
    snap->mpcd_data.velocity[7] = vec3<Scalar>(1.0, 1.0, 1.0);
    snap->mpcd_data.velocity[8] = vec3<Scalar>(1.0, 1.0, 1.0);

    std::shared_ptr<DomainDecomposition> decomposition(
        new DomainDecomposition(exec_conf, snap->global_box->getL(), 2, 2, 2));
    std::shared_ptr<SystemDefinition> sysdef(new SystemDefinition(snap, exec_conf, decomposition));
    std::shared_ptr<Communicator> pdata_comm(new Communicator(sysdef, decomposition));
    sysdef->setCommunicator(pdata_comm);

    // initialize mpcd system
    std::shared_ptr<mpcd::ParticleData> pdata = sysdef->getMPCDParticleData();
    std::shared_ptr<mpcd::CellList> cl(new CL(sysdef, make_uint3(6, 6, 6), false));
    AllThermoRequest thermo_req(cl);
    cl->compute(0);
    const unsigned int my_rank = exec_conf->getRank();
        {
        ArrayHandle<unsigned int> h_cell_np(cl->getCellSizeArray(),
                                            access_location::host,
                                            access_mode::read);
        ArrayHandle<double4> h_cell_vel(cl->getCellVelocities(),
                                        access_location::host,
                                        access_mode::read);
        ArrayHandle<double> h_cell_energy(cl->getCellEnergies(),
                                          access_location::host,
                                          access_mode::read);
        ArrayHandle<double> h_cell_temp(cl->getCellTemperature(),
                                        access_location::host,
                                        access_mode::read);
        Index3D ci = cl->getCellIndexer();
        ArrayHandle<Scalar4> h_vel(pdata->getVelocities(),
                                   access_location::host,
                                   access_mode::read);

        switch (my_rank)
            {
        case 0:
            // global index is (2,2,2), with origin (0,0,0)
            UP_ASSERT_EQUAL(h_cell_np.data[ci(2, 2, 2)], 2);
            UP_ASSERT_EQUAL(__scalar_as_int(h_vel.data[0].w), ci(2, 2, 2));
            UP_ASSERT_EQUAL(__scalar_as_int(h_vel.data[1].w), ci(2, 2, 2));

            // check cell properties
            CHECK_CLOSE(h_cell_vel.data[ci(2, 2, 2)].x, 0.0, tol);
            CHECK_CLOSE(h_cell_vel.data[ci(2, 2, 2)].y, 0.0, tol);
            CHECK_CLOSE(h_cell_vel.data[ci(2, 2, 2)].z, 0.0, tol);
            CHECK_CLOSE(h_cell_vel.data[ci(2, 2, 2)].w, 2.0, tol);

            CHECK_CLOSE(h_cell_energy.data[ci(2, 2, 2)], 3.0, tol);
            CHECK_CLOSE(h_cell_temp.data[ci(2, 2, 2)], 2.0, tol);
            break;
        case 1:
            // global index is (3,2,2), with origin (3,0,0)
            UP_ASSERT_EQUAL(h_cell_np.data[ci(0, 2, 2)], 1);
            UP_ASSERT_EQUAL(__scalar_as_int(h_vel.data[0].w), ci(0, 2, 2));

            CHECK_CLOSE(h_cell_vel.data[ci(0, 2, 2)].x, 1.0, tol);
            CHECK_CLOSE(h_cell_vel.data[ci(0, 2, 2)].y, -1.0, tol);
            CHECK_CLOSE(h_cell_vel.data[ci(0, 2, 2)].z, -1.0, tol);
            CHECK_CLOSE(h_cell_vel.data[ci(0, 2, 2)].w, 1.0, tol);

            CHECK_CLOSE(h_cell_energy.data[ci(0, 2, 2)], 1.5, tol);
            CHECK_CLOSE(h_cell_temp.data[ci(0, 2, 2)], 0.0, tol);
            break;
        case 2:
            // global index is (2,3,2), with origin (0,3,0)
            UP_ASSERT_EQUAL(h_cell_np.data[ci(2, 0, 2)], 1);
            UP_ASSERT_EQUAL(__scalar_as_int(h_vel.data[0].w), ci(2, 0, 2));

            CHECK_CLOSE(h_cell_vel.data[ci(2, 0, 2)].x, -1.0, tol);
            CHECK_CLOSE(h_cell_vel.data[ci(2, 0, 2)].y, 1.0, tol);
            CHECK_CLOSE(h_cell_vel.data[ci(2, 0, 2)].z, -1.0, tol);
            CHECK_CLOSE(h_cell_vel.data[ci(2, 0, 2)].w, 1.0, tol);

            CHECK_CLOSE(h_cell_energy.data[ci(2, 0, 2)], 1.5, tol);
            CHECK_CLOSE(h_cell_temp.data[ci(2, 0, 2)], 0.0, tol);
            break;
        case 3:
            // global index is (3,3,2), with origin (3,3,0)
            UP_ASSERT_EQUAL(h_cell_np.data[ci(0, 0, 2)], 1);
            UP_ASSERT_EQUAL(__scalar_as_int(h_vel.data[0].w), ci(0, 0, 2));

            CHECK_CLOSE(h_cell_vel.data[ci(0, 0, 2)].x, 1.0, tol);
            CHECK_CLOSE(h_cell_vel.data[ci(0, 0, 2)].y, 1.0, tol);
            CHECK_CLOSE(h_cell_vel.data[ci(0, 0, 2)].z, -1.0, tol);
            CHECK_CLOSE(h_cell_vel.data[ci(0, 0, 2)].w, 1.0, tol);

            CHECK_CLOSE(h_cell_energy.data[ci(0, 0, 2)], 1.5, tol);
            CHECK_CLOSE(h_cell_temp.data[ci(0, 0, 2)], 0.0, tol);
            break;
        case 4:
            // global index is (2,2,3), with origin (0,0,3)
            UP_ASSERT_EQUAL(h_cell_np.data[ci(2, 2, 0)], 1);
            UP_ASSERT_EQUAL(__scalar_as_int(h_vel.data[0].w), ci(2, 2, 0));

            CHECK_CLOSE(h_cell_vel.data[ci(2, 2, 0)].x, -1.0, tol);
            CHECK_CLOSE(h_cell_vel.data[ci(2, 2, 0)].y, -1.0, tol);
            CHECK_CLOSE(h_cell_vel.data[ci(2, 2, 0)].z, 1.0, tol);
            CHECK_CLOSE(h_cell_vel.data[ci(2, 2, 0)].w, 1.0, tol);

            CHECK_CLOSE(h_cell_energy.data[ci(2, 2, 0)], 1.5, tol);
            CHECK_CLOSE(h_cell_temp.data[ci(2, 2, 0)], 0.0, tol);
            break;
        case 5:
            // global index is (3,2,3), with origin (3,0,3)
            UP_ASSERT_EQUAL(h_cell_np.data[ci(0, 2, 0)], 1);
            UP_ASSERT_EQUAL(__scalar_as_int(h_vel.data[0].w), ci(0, 2, 0));

            CHECK_CLOSE(h_cell_vel.data[ci(0, 2, 0)].x, 1.0, tol);
            CHECK_CLOSE(h_cell_vel.data[ci(0, 2, 0)].y, -1.0, tol);
            CHECK_CLOSE(h_cell_vel.data[ci(0, 2, 0)].z, 1.0, tol);
            CHECK_CLOSE(h_cell_vel.data[ci(0, 2, 0)].w, 1.0, tol);

            CHECK_CLOSE(h_cell_energy.data[ci(0, 2, 0)], 1.5, tol);
            CHECK_CLOSE(h_cell_temp.data[ci(0, 2, 0)], 0.0, tol);
            break;
        case 6:
            // global index is (2,3,3), with origin (0,3,3)
            UP_ASSERT_EQUAL(h_cell_np.data[ci(2, 0, 0)], 1);
            UP_ASSERT_EQUAL(__scalar_as_int(h_vel.data[0].w), ci(2, 0, 0));

            CHECK_CLOSE(h_cell_vel.data[ci(2, 0, 0)].x, -1.0, tol);
            CHECK_CLOSE(h_cell_vel.data[ci(2, 0, 0)].y, 1.0, tol);
            CHECK_CLOSE(h_cell_vel.data[ci(2, 0, 0)].z, 1.0, tol);
            CHECK_CLOSE(h_cell_vel.data[ci(2, 0, 0)].w, 1.0, tol);

            CHECK_CLOSE(h_cell_energy.data[ci(2, 0, 0)], 1.5, tol);
            CHECK_CLOSE(h_cell_temp.data[ci(2, 0, 0)], 0.0, tol);
            break;
        case 7:
            // global index is (3,3,3), with origin (3,3,3)
            UP_ASSERT_EQUAL(h_cell_np.data[ci(0, 0, 0)], 1);
            UP_ASSERT_EQUAL(__scalar_as_int(h_vel.data[0].w), ci(0, 0, 0));

            CHECK_CLOSE(h_cell_vel.data[ci(0, 0, 0)].x, 1.0, tol);
            CHECK_CLOSE(h_cell_vel.data[ci(0, 0, 0)].y, 1.0, tol);
            CHECK_CLOSE(h_cell_vel.data[ci(0, 0, 0)].z, 1.0, tol);
            CHECK_CLOSE(h_cell_vel.data[ci(0, 0, 0)].w, 1.0, tol);

            CHECK_CLOSE(h_cell_energy.data[ci(0, 0, 0)], 1.5, tol);
            CHECK_CLOSE(h_cell_temp.data[ci(0, 0, 0)], 0.0, tol);
            break;
            };
        }

    // Check the net stats of the system
    CHECK_CLOSE(cl->getNetMomentum().x, 1.0, tol);
    CHECK_CLOSE(cl->getNetMomentum().y, 1.0, tol);
    CHECK_CLOSE(cl->getNetMomentum().z, 1.0, tol);
    CHECK_CLOSE(cl->getNetEnergy(), 13.5, tol);
    CHECK_CLOSE(cl->getTemperature(), 2.0, tol);

    // apply a grid shift so that particles move into the same cell (3,3,3)
    // this will cause all the particles to be sent to the same rank as ghosts
    const Scalar3 shift = (Scalar(0.5) / 6) * make_scalar3(1, 1, 1);
    cl->setGridShift(-shift);
    cl->compute(1);
        {
        ArrayHandle<unsigned int> h_cell_np(cl->getCellSizeArray(),
                                            access_location::host,
                                            access_mode::read);
        ArrayHandle<double4> h_cell_vel(cl->getCellVelocities(),
                                        access_location::host,
                                        access_mode::read);
        ArrayHandle<double> h_cell_energy(cl->getCellEnergies(),
                                          access_location::host,
                                          access_mode::read);
        ArrayHandle<double> h_cell_temp(cl->getCellTemperature(),
                                        access_location::host,
                                        access_mode::read);
        Index3D ci = cl->getCellIndexer();
        ArrayHandle<Scalar4> h_vel(pdata->getVelocities(),
                                   access_location::host,
                                   access_mode::read);
        unsigned int num_ghosts = cl->getNGhosts();
        if (my_rank == 7)
            {
            // global index is (3,3,3), with origin (3,3,3)
            int3 global_cell = cl->getGlobalCell(make_int3(0, 0, 0));
            UP_ASSERT_EQUAL(global_cell.x, 3);
            UP_ASSERT_EQUAL(global_cell.y, 3);
            UP_ASSERT_EQUAL(global_cell.z, 3);
            UP_ASSERT_EQUAL(h_cell_np.data[ci(0, 0, 0)], 9);
            UP_ASSERT_EQUAL(num_ghosts, 8);
            UP_ASSERT_EQUAL(__scalar_as_int(h_vel.data[0].w), ci(0, 0, 0));

            CHECK_CLOSE(h_cell_vel.data[ci(0, 0, 0)].x, 0.111111, tol);
            CHECK_CLOSE(h_cell_vel.data[ci(0, 0, 0)].y, 0.111111, tol);
            CHECK_CLOSE(h_cell_vel.data[ci(0, 0, 0)].z, 0.111111, tol);
            CHECK_CLOSE(h_cell_vel.data[ci(0, 0, 0)].w, 9, tol);

            CHECK_CLOSE(h_cell_energy.data[ci(0, 0, 0)], 13.5, tol);
            CHECK_CLOSE(h_cell_temp.data[ci(0, 0, 0)], 1.111111, tol);
            }
        else
            {
            // other ranks no local particles are in cells
            UP_ASSERT_EQUAL(num_ghosts, 0);
            for (unsigned int c = 0; c < cl->getNCells(); ++c)
                {
                UP_ASSERT_EQUAL(h_cell_np.data[c], 0);

                CHECK_CLOSE(h_cell_vel.data[c].x, 0, tol);
                CHECK_CLOSE(h_cell_vel.data[c].y, 0, tol);
                CHECK_CLOSE(h_cell_vel.data[c].z, 0, tol);
                CHECK_CLOSE(h_cell_vel.data[c].w, 0, tol);

                CHECK_CLOSE(h_cell_energy.data[c], 0, tol);
                CHECK_CLOSE(h_cell_temp.data[c], 0, tol);
                }
            }
        }

    // apply a grid shift so that particles move into the same cell (2,2,2)
    // this will cause all the particles to be on the same rank
    cl->setGridShift(shift);
    cl->compute(2);
        {
        ArrayHandle<unsigned int> h_cell_np(cl->getCellSizeArray(),
                                            access_location::host,
                                            access_mode::read);
        ArrayHandle<double4> h_cell_vel(cl->getCellVelocities(),
                                        access_location::host,
                                        access_mode::read);
        ArrayHandle<double> h_cell_energy(cl->getCellEnergies(),
                                          access_location::host,
                                          access_mode::read);
        ArrayHandle<double> h_cell_temp(cl->getCellTemperature(),
                                        access_location::host,
                                        access_mode::read);
        Index3D ci = cl->getCellIndexer();
        ArrayHandle<Scalar4> h_vel(pdata->getVelocities(),
                                   access_location::host,
                                   access_mode::read);
        unsigned int num_ghosts = cl->getNGhosts();
        if (my_rank == 0)
            {
            // global index is (2,2,2), with origin (0,0,0)
            int3 global_cell = cl->getGlobalCell(make_int3(2, 2, 2));
            UP_ASSERT_EQUAL(global_cell.x, 2);
            UP_ASSERT_EQUAL(global_cell.y, 2);
            UP_ASSERT_EQUAL(global_cell.z, 2);
            UP_ASSERT_EQUAL(h_cell_np.data[ci(2, 2, 2)], 9);
            UP_ASSERT_EQUAL(num_ghosts, 7);
            UP_ASSERT_EQUAL(__scalar_as_int(h_vel.data[0].w), ci(2, 2, 2));

            CHECK_CLOSE(h_cell_vel.data[ci(2, 2, 2)].x, 0.111111, tol);
            CHECK_CLOSE(h_cell_vel.data[ci(2, 2, 2)].y, 0.111111, tol);
            CHECK_CLOSE(h_cell_vel.data[ci(2, 2, 2)].z, 0.111111, tol);
            CHECK_CLOSE(h_cell_vel.data[ci(2, 2, 2)].w, 9, tol);

            CHECK_CLOSE(h_cell_energy.data[ci(2, 2, 2)], 13.5, tol);
            CHECK_CLOSE(h_cell_temp.data[ci(2, 2, 2)], 1.111111, tol);
            }
        else
            {
            // other ranks no local particles are in cells
            UP_ASSERT_EQUAL(num_ghosts, 0);
            for (unsigned int c = 0; c < cl->getNCells(); ++c)
                {
                UP_ASSERT_EQUAL(h_cell_np.data[c], 0);

                CHECK_CLOSE(h_cell_vel.data[c].x, 0, tol);
                CHECK_CLOSE(h_cell_vel.data[c].y, 0, tol);
                CHECK_CLOSE(h_cell_vel.data[c].z, 0, tol);
                CHECK_CLOSE(h_cell_vel.data[c].w, 0, tol);

                CHECK_CLOSE(h_cell_energy.data[c], 0, tol);
                CHECK_CLOSE(h_cell_temp.data[c], 0, tol);
                }
            }
        }

    // Check the net stats of the system
    CHECK_CLOSE(cl->getNetMomentum().x, 1.0, tol);
    CHECK_CLOSE(cl->getNetMomentum().y, 1.0, tol);
    CHECK_CLOSE(cl->getNetMomentum().z, 1.0, tol);
    CHECK_CLOSE(cl->getNetEnergy(), 13.5, tol);
    CHECK_CLOSE(cl->getTemperature(), 1.111111, tol);
    }

//! Test for correct cell listing of a system with particles on the edges
template<class CL>
void celllist_edge_test(std::shared_ptr<ExecutionConfiguration> exec_conf,
                        const Scalar3& L,
                        const Scalar3& tilt)
    {
    UP_ASSERT_EQUAL(exec_conf->getNRanks(), 8);

    auto ref_box = std::make_shared<BoxDim>(5.0);
    auto box = std::make_shared<BoxDim>(L);
    box->setTiltFactors(tilt.x, tilt.y, tilt.z);

    std::shared_ptr<SnapshotSystemData<Scalar>> snap(new SnapshotSystemData<Scalar>());
    snap->global_box = box;
    snap->particle_data.type_mapping.push_back("A");
    // dummy initialize one particle to every domain, we will move them outside the domains for
    // the tests
    /*
     * The +/- halves of the box owned by each domain are:
     *    x y z
     * 0: - - -
     * 1: + - -
     * 2: - + -
     * 3: + + -
     * 4: - - +
     * 5: + - +
     * 6: - + +
     * 7: + + +
     */
    snap->mpcd_data.resize(8);
    snap->mpcd_data.type_mapping.push_back("A");
    snap->mpcd_data.position[0] = scale(vec3<Scalar>(-1.0, -1.0, -1.0), ref_box, box);
    snap->mpcd_data.position[1] = scale(vec3<Scalar>(1.0, -1.0, -1.0), ref_box, box);
    snap->mpcd_data.position[2] = scale(vec3<Scalar>(-1.0, 1.0, -1.0), ref_box, box);
    snap->mpcd_data.position[3] = scale(vec3<Scalar>(1.0, 1.0, -1.0), ref_box, box);
    snap->mpcd_data.position[4] = scale(vec3<Scalar>(-1.0, -1.0, 1.0), ref_box, box);
    snap->mpcd_data.position[5] = scale(vec3<Scalar>(1.0, -1.0, 1.0), ref_box, box);
    snap->mpcd_data.position[6] = scale(vec3<Scalar>(-1.0, 1.0, 1.0), ref_box, box);
    snap->mpcd_data.position[7] = scale(vec3<Scalar>(1.0, 1.0, 1.0), ref_box, box);
    std::vector<Scalar> fx {0.5};
    std::vector<Scalar> fy {0.45};
    std::vector<Scalar> fz {0.55};
    std::shared_ptr<DomainDecomposition> decomposition(
        new DomainDecomposition(exec_conf, snap->global_box->getL(), fx, fy, fz));
    std::shared_ptr<SystemDefinition> sysdef(new SystemDefinition(snap, exec_conf, decomposition));
    std::shared_ptr<Communicator> pdata_comm(new Communicator(sysdef, decomposition));
    sysdef->setCommunicator(pdata_comm);

    std::shared_ptr<mpcd::ParticleData> pdata = sysdef->getMPCDParticleData();
    std::shared_ptr<mpcd::CellList> cl(new CL(sysdef, make_uint3(5, 5, 5), false));

    // move particles to edges of domains for testing
    const unsigned int my_rank = exec_conf->getRank();
        {
        ArrayHandle<Scalar4> h_pos(pdata->getPositions(),
                                   access_location::host,
                                   access_mode::overwrite);
        switch (my_rank)
            {
        case 0:
            h_pos.data[0]
                = scale(make_scalar4(-0.01, -0.01, -0.01, __int_as_scalar(0)), ref_box, box);
            break;
        case 1:
            h_pos.data[0]
                = scale(make_scalar4(0.0, -0.01, -0.01, __int_as_scalar(0)), ref_box, box);
            break;
        case 2:
            h_pos.data[0]
                = scale(make_scalar4(-0.01, 0.0, -0.01, __int_as_scalar(0)), ref_box, box);
            break;
        case 3:
            h_pos.data[0] = scale(make_scalar4(0.0, 0.0, -0.01, __int_as_scalar(0)), ref_box, box);
            break;
        case 4:
            h_pos.data[0]
                = scale(make_scalar4(-0.01, -0.01, 0.0, __int_as_scalar(0)), ref_box, box);
            break;
        case 5:
            h_pos.data[0] = scale(make_scalar4(0.0, -0.01, 0.0, __int_as_scalar(0)), ref_box, box);
            break;
        case 6:
            h_pos.data[0] = scale(make_scalar4(-0.01, 0.0, 0.0, __int_as_scalar(0)), ref_box, box);
            break;
        case 7:
            h_pos.data[0] = scale(make_scalar4(0.0, 0.0, 0.0, __int_as_scalar(0)), ref_box, box);
            break;
            };
        }

    cl->compute(0);
        {
        ArrayHandle<unsigned int> h_cell_np(cl->getCellSizeArray(),
                                            access_location::host,
                                            access_mode::read);
        ArrayHandle<Scalar4> h_vel(pdata->getVelocities(),
                                   access_location::host,
                                   access_mode::read);
        ArrayHandle<unsigned int> h_ghost_cell_ids(cl->getGhostCellIds(),
                                                   access_location::host,
                                                   access_mode::read);
        unsigned int num_ghosts = cl->getNGhosts();
        if (cl->hasGlobalCell(make_int3(2, 2, 2)))
            {
            const unsigned int local_cell = make_local_cell(cl, 2, 2, 2);
            UP_ASSERT_EQUAL(h_cell_np.data[local_cell], 8);
            UP_ASSERT_EQUAL(__scalar_as_int(h_vel.data[0].w), local_cell);
            UP_ASSERT_EQUAL(num_ghosts, 7);
            for (unsigned int i = 0; i < num_ghosts; i++)
                {
                UP_ASSERT_EQUAL(h_ghost_cell_ids.data[i], local_cell);
                }
            }
        else
            {
            UP_ASSERT_EQUAL(num_ghosts, 0);
            for (unsigned int c = 0; c < cl->getNCells(); ++c)
                {
                UP_ASSERT_EQUAL(h_cell_np.data[c], 0);
                }
            }
        }

    // apply a grid shift, particles on left internal boundary will move up one cell
    // particles on right internal boundary will stay in the same cell
    const Scalar3 shift = (Scalar(0.5) / 5) * make_scalar3(1, 1, 1);
        {
        cl->setGridShift(-shift);
        cl->compute(1);
        ArrayHandle<unsigned int> h_cell_np(cl->getCellSizeArray(),
                                            access_location::host,
                                            access_mode::read);
        ArrayHandle<Scalar4> h_vel(pdata->getVelocities(),
                                   access_location::host,
                                   access_mode::read);
        ArrayHandle<unsigned int> h_ghost_cell_ids(cl->getGhostCellIds(),
                                                   access_location::host,
                                                   access_mode::read);
        unsigned int num_ghosts = cl->getNGhosts();

        std::array<int3, 8> cells_with_particles = {make_int3(2, 2, 2),
                                                    make_int3(3, 2, 2),
                                                    make_int3(2, 3, 2),
                                                    make_int3(3, 3, 2),
                                                    make_int3(2, 2, 3),
                                                    make_int3(3, 2, 3),
                                                    make_int3(2, 3, 3),
                                                    make_int3(3, 3, 3)};
        for (unsigned int i = 0; i < 8; i++)
            {
            int3 cell = cells_with_particles[i];
            if (cl->hasGlobalCell(cell))
                {
                const unsigned int local_cell = make_local_cell(cl, cell.x, cell.y, cell.z);
                UP_ASSERT_EQUAL(h_cell_np.data[local_cell], 1);
                if (num_ghosts)
                    {
                    // if there are ghosts, either the ghost or the local particle should be in the
                    // local cell
                    UP_ASSERT((h_ghost_cell_ids.data[0] == local_cell)
                              != (__scalar_as_int(h_vel.data[0].w) == int(local_cell)));
                    }
                else
                    {
                    UP_ASSERT_EQUAL(__scalar_as_int(h_vel.data[0].w), int(local_cell));
                    }
                }
            }
        }

        // apply a grid shift, particles on left internal boundary will stay in cell
        // particles on right internal boundary will move down one cell
        // if (*box == *ref_box)
        {
        cl->setGridShift(shift);
        cl->compute(2);

        ArrayHandle<unsigned int> h_cell_np(cl->getCellSizeArray(),
                                            access_location::host,
                                            access_mode::read);
        ArrayHandle<Scalar4> h_vel(pdata->getVelocities(),
                                   access_location::host,
                                   access_mode::read);
        ArrayHandle<unsigned int> h_ghost_cell_ids(cl->getGhostCellIds(),
                                                   access_location::host,
                                                   access_mode::read);
        unsigned int num_ghosts = cl->getNGhosts();
        std::array<int3, 8> cells_with_particles = {make_int3(1, 1, 1),
                                                    make_int3(2, 1, 1),
                                                    make_int3(1, 2, 1),
                                                    make_int3(2, 2, 1),
                                                    make_int3(1, 1, 2),
                                                    make_int3(2, 1, 2),
                                                    make_int3(1, 2, 2),
                                                    make_int3(2, 2, 2)};
        for (unsigned int i = 0; i < 8; i++)
            {
            int3 cell = cells_with_particles[i];
            if (cl->hasGlobalCell(cell))
                {
                const unsigned int local_cell = make_local_cell(cl, cell.x, cell.y, cell.z);
                UP_ASSERT_EQUAL(h_cell_np.data[local_cell], 1);
                if (num_ghosts)
                    {
                    // if there are ghosts, either the ghost or the local particle should be in the
                    // local cell
                    unsigned int particle_in_cell
                        = __scalar_as_int(h_vel.data[0].w) == int(local_cell);
                    for (unsigned int j = 0; j < num_ghosts; j++)
                        {
                        particle_in_cell += h_ghost_cell_ids.data[j] == local_cell;
                        }
                    UP_ASSERT_EQUAL(particle_in_cell, 1);
                    }
                else
                    {
                    UP_ASSERT_EQUAL(__scalar_as_int(h_vel.data[0].w), int(local_cell));
                    }
                }
            }
        }
    }

//! dimension test case for MPCD CellList class
UP_TEST(mpcd_cell_list_dimensions)
    {
        // mpi in 1d
        {
        std::shared_ptr<ExecutionConfiguration> exec_conf(
            new ExecutionConfiguration(ExecutionConfiguration::CPU));
        exec_conf->getMPIConfig()->splitPartitions(2);
        celllist_dimension_test<mpcd::CellList>(exec_conf,
                                                true,
                                                false,
                                                false,
                                                make_scalar3(0, 0, 0));
        celllist_dimension_test<mpcd::CellList>(exec_conf,
                                                false,
                                                true,
                                                false,
                                                make_scalar3(0, 0, 0));
        celllist_dimension_test<mpcd::CellList>(exec_conf,
                                                false,
                                                false,
                                                true,
                                                make_scalar3(0, 0, 0));
        }
        // mpi in 2d
        {
        std::shared_ptr<ExecutionConfiguration> exec_conf(
            new ExecutionConfiguration(ExecutionConfiguration::CPU));
        exec_conf->getMPIConfig()->splitPartitions(4);
        celllist_dimension_test<mpcd::CellList>(exec_conf,
                                                true,
                                                true,
                                                false,
                                                make_scalar3(0, 0, 0));
        celllist_dimension_test<mpcd::CellList>(exec_conf,
                                                true,
                                                false,
                                                true,
                                                make_scalar3(0, 0, 0));
        celllist_dimension_test<mpcd::CellList>(exec_conf,
                                                false,
                                                true,
                                                true,
                                                make_scalar3(0, 0, 0));
        }
        // mpi in 3d
        {
        std::shared_ptr<ExecutionConfiguration> exec_conf(
            new ExecutionConfiguration(ExecutionConfiguration::CPU));
        exec_conf->getMPIConfig()->splitPartitions(8);
        celllist_dimension_test<mpcd::CellList>(exec_conf, true, true, true, make_scalar3(0, 0, 0));
        }
    }

//! dimension test case for MPCD CellList class, triclinic
UP_TEST(mpcd_cell_list_dimensions_triclinic)
    {
        // mpi in 1d
        {
        auto exec_conf = std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::CPU);
        exec_conf->getMPIConfig()->splitPartitions(2);
        celllist_dimension_test<mpcd::CellList>(exec_conf,
                                                true,
                                                false,
                                                false,
                                                make_scalar3(0.5, -0.75, 1.0));
        celllist_dimension_test<mpcd::CellList>(exec_conf,
                                                false,
                                                true,
                                                false,
                                                make_scalar3(0.5, -0.75, 1.0));
        celllist_dimension_test<mpcd::CellList>(exec_conf,
                                                false,
                                                false,
                                                true,
                                                make_scalar3(0.5, -0.75, 1.0));
        }
        // mpi in 2d
        {
        auto exec_conf = std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::CPU);
        exec_conf->getMPIConfig()->splitPartitions(4);
        celllist_dimension_test<mpcd::CellList>(exec_conf,
                                                true,
                                                true,
                                                false,
                                                make_scalar3(0.5, -0.75, 1.0));
        celllist_dimension_test<mpcd::CellList>(exec_conf,
                                                true,
                                                false,
                                                true,
                                                make_scalar3(0.5, -0.75, 1.0));
        celllist_dimension_test<mpcd::CellList>(exec_conf,
                                                false,
                                                true,
                                                true,
                                                make_scalar3(0.5, -0.75, 1.0));
        }
        // mpi in 3d
        {
        auto exec_conf = std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::CPU);
        exec_conf->getMPIConfig()->splitPartitions(8);
        celllist_dimension_test<mpcd::CellList>(exec_conf,
                                                true,
                                                true,
                                                true,
                                                make_scalar3(0.5, -0.75, 1.0));
        }
    }

//! basic test case for MPCD CellList class
UP_TEST(mpcd_cell_list_basic_test)
    {
    celllist_basic_test<mpcd::CellList>(
        std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::CPU),
        make_scalar3(6.0, 6.0, 6.0),
        make_scalar3(0, 0, 0));
    }

//! basic test case for MPCD CellList class, noncubic
UP_TEST(mpcd_cell_list_basic_test_noncubic)
    {
    celllist_basic_test<mpcd::CellList>(
        std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::CPU),
        make_scalar3(6.5, 7.0, 7.5),
        make_scalar3(0, 0, 0));
    }

//! basic test case for MPCD CellList class, triclinic
UP_TEST(mpcd_cell_list_basic_test_triclinic)
    {
    celllist_basic_test<mpcd::CellList>(
        std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::CPU),
        make_scalar3(6.0, 6.0, 6.0),
        make_scalar3(0.5, -0.75, 1.0));
    }

//! edge test case for MPCD CellList class
UP_TEST(mpcd_cell_list_edge_test)
    {
    celllist_edge_test<mpcd::CellList>(
        std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::CPU),
        make_scalar3(5.0, 5.0, 5.0),
        make_scalar3(0, 0, 0));
    }

//! edge test case for MPCD CellList class, noncubic
UP_TEST(mpcd_cell_list_edge_test_noncubic)
    {
    celllist_edge_test<mpcd::CellList>(
        std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::CPU),
        make_scalar3(6.0, 6.5, 7.0),
        make_scalar3(0, 0, 0));
    }

//! edge test case for MPCD CellList class, triclinic
UP_TEST(mpcd_cell_list_edge_test_triclinic)
    {
    celllist_edge_test<mpcd::CellList>(
        std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::CPU),
        make_scalar3(5.0, 5.0, 5.0),
        make_scalar3(0.5, -0.75, 1.0));
    }

#ifdef ENABLE_HIP
//! dimension test case for MPCD CellListGPU class
UP_TEST(mpcd_cell_list_gpu_dimensions)
    {
        // mpi in 1d
        {
        std::shared_ptr<ExecutionConfiguration> exec_conf(
            new ExecutionConfiguration(ExecutionConfiguration::GPU));
        exec_conf->getMPIConfig()->splitPartitions(2);
        celllist_dimension_test<mpcd::CellListGPU>(exec_conf,
                                                   true,
                                                   false,
                                                   false,
                                                   make_scalar3(0, 0, 0));
        celllist_dimension_test<mpcd::CellListGPU>(exec_conf,
                                                   false,
                                                   true,
                                                   false,
                                                   make_scalar3(0, 0, 0));
        celllist_dimension_test<mpcd::CellListGPU>(exec_conf,
                                                   false,
                                                   false,
                                                   true,
                                                   make_scalar3(0, 0, 0));
        }
        // mpi in 2d
        {
        std::shared_ptr<ExecutionConfiguration> exec_conf(
            new ExecutionConfiguration(ExecutionConfiguration::GPU));
        exec_conf->getMPIConfig()->splitPartitions(4);
        celllist_dimension_test<mpcd::CellListGPU>(exec_conf,
                                                   true,
                                                   true,
                                                   false,
                                                   make_scalar3(0, 0, 0));
        celllist_dimension_test<mpcd::CellListGPU>(exec_conf,
                                                   true,
                                                   false,
                                                   true,
                                                   make_scalar3(0, 0, 0));
        celllist_dimension_test<mpcd::CellListGPU>(exec_conf,
                                                   false,
                                                   true,
                                                   true,
                                                   make_scalar3(0, 0, 0));
        }
        // mpi in 3d
        {
        std::shared_ptr<ExecutionConfiguration> exec_conf(
            new ExecutionConfiguration(ExecutionConfiguration::GPU));
        exec_conf->getMPIConfig()->splitPartitions(8);
        celllist_dimension_test<mpcd::CellListGPU>(exec_conf,
                                                   true,
                                                   true,
                                                   true,
                                                   make_scalar3(0, 0, 0));
        }
    }

//! dimension test case for MPCD CellListGPU class
UP_TEST(mpcd_cell_list_gpu_dimensions_triclinic)
    {
        // mpi in 1d
        {
        auto exec_conf = std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::GPU);
        exec_conf->getMPIConfig()->splitPartitions(2);
        celllist_dimension_test<mpcd::CellListGPU>(exec_conf,
                                                   true,
                                                   false,
                                                   false,
                                                   make_scalar3(0.5, -0.75, 1.0));
        celllist_dimension_test<mpcd::CellListGPU>(exec_conf,
                                                   false,
                                                   true,
                                                   false,
                                                   make_scalar3(0.5, -0.75, 1.0));
        celllist_dimension_test<mpcd::CellListGPU>(exec_conf,
                                                   false,
                                                   false,
                                                   true,
                                                   make_scalar3(0.5, -0.75, 1.0));
        }
        // mpi in 2d
        {
        auto exec_conf = std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::GPU);
        exec_conf->getMPIConfig()->splitPartitions(4);
        celllist_dimension_test<mpcd::CellListGPU>(exec_conf,
                                                   true,
                                                   true,
                                                   false,
                                                   make_scalar3(0.5, -0.75, 1.0));
        celllist_dimension_test<mpcd::CellListGPU>(exec_conf,
                                                   true,
                                                   false,
                                                   true,
                                                   make_scalar3(0.5, -0.75, 1.0));
        celllist_dimension_test<mpcd::CellListGPU>(exec_conf,
                                                   false,
                                                   true,
                                                   true,
                                                   make_scalar3(0.5, -0.75, 1.0));
        }
        // mpi in 3d
        {
        auto exec_conf = std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::GPU);
        exec_conf->getMPIConfig()->splitPartitions(8);
        celllist_dimension_test<mpcd::CellListGPU>(exec_conf,
                                                   true,
                                                   true,
                                                   true,
                                                   make_scalar3(0.5, -0.75, 1.0));
        }
    }

//! basic test case for MPCD CellListGPU class
UP_TEST(mpcd_cell_list_gpu_basic_test)
    {
    celllist_basic_test<mpcd::CellListGPU>(
        std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::GPU),
        make_scalar3(6.0, 6.0, 6.0),
        make_scalar3(0, 0, 0));
    }

//! basic test case for MPCD CellListGPU class, noncubic
UP_TEST(mpcd_cell_list_gpu_basic_test_noncubic)
    {
    celllist_basic_test<mpcd::CellListGPU>(
        std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::GPU),
        make_scalar3(6.5, 7.0, 7.5),
        make_scalar3(0, 0, 0));
    }

//! basic test case for MPCD CellListGPU class, triclinic
UP_TEST(mpcd_cell_list_gpu_basic_test_triclinic)
    {
    celllist_basic_test<mpcd::CellListGPU>(
        std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::GPU),
        make_scalar3(6.0, 6.0, 6.0),
        make_scalar3(0.5, -0.75, 1.0));
    }

//! edge test case for MPCD CellListGPU class
UP_TEST(mpcd_cell_list_gpu_edge_test)
    {
    celllist_edge_test<mpcd::CellListGPU>(
        std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::GPU),
        make_scalar3(5.0, 5.0, 5.0),
        make_scalar3(0, 0, 0));
    }

//! edge test case for MPCD CellListGPU class, noncubic
UP_TEST(mpcd_cell_list_gpu_edge_test_noncubic)
    {
    celllist_edge_test<mpcd::CellListGPU>(
        std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::GPU),
        make_scalar3(6.0, 6.5, 7.0),
        make_scalar3(0, 0, 0));
    }

//! edge test case for MPCD CellListGPU class, triclinic
UP_TEST(mpcd_cell_list_gpu_edge_test_triclinic)
    {
    celllist_edge_test<mpcd::CellListGPU>(
        std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::GPU),
        make_scalar3(5.0, 5.0, 5.0),
        make_scalar3(0.5, -0.75, 1.0));
    }
#endif // ENABLE_HIP
