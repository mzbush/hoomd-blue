// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hoomd/ForceCompute.h"
#include "hoomd/MeshDefinition.h"

#include <memory>

/*! \file MeshForceCompute.h
    \brief Declares the MeshForceCompute class
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

#ifndef __MeshFORCECOMPUTE_H__
#define __MeshFORCECOMPUTE_H__

namespace hoomd
    {
namespace md
    {
//! Structure for passing the meshforce arrays around

class PYBIND11_EXPORT MeshForceCompute : public ForceCompute
    {
    public:
    //! Constructs the compute
    MeshForceCompute(std::shared_ptr<SystemDefinition> sysdef,
                             std::shared_ptr<MeshDefinition> meshdef);

    //! compute areas
    virtual void precomputeParameter(){ };

    protected:
    std::shared_ptr<MeshDefinition> m_mesh_data; //!< Mesh data to use in computing helfich energy
    };

namespace detail
    {

//! Exports the MeshForceCompute class to python
#ifndef __HIPCC__
void export_MeshForceCompute(pybind11::module& m);
#endif

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd

#endif // __MESHFORCECOMPUTE_H__
