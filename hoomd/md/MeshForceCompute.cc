// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "MeshForceCompute.h"

#include <iostream>

using namespace std;

/*! \file MeshForceCompute.cc
    \brief Contains code for the MeshForceCompute class
*/

namespace hoomd
    {
namespace md
    {
/*! \param sysdef System to compute forces on
    \param meshdef Mesh triangulation
    \post Memory is allocated, and forces are zeroed.
*/
MeshForceCompute::MeshForceCompute(std::shared_ptr<SystemDefinition> sysdef,
                                                   std::shared_ptr<MeshDefinition> meshdef)
    : ForceCompute(sysdef), m_mesh_data(meshdef){}

namespace detail
    {
void export_MeshForceCompute(pybind11::module& m)
    {
    pybind11::class_<MeshForceCompute,
                     ForceCompute,
                     std::shared_ptr<MeshForceCompute>>(m, "MeshForceCompute")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<MeshDefinition>>());
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
