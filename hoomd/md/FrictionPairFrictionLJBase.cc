// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "FrictionPair.h"
#include "EvaluatorPairFrictionLJBase.h"
#include "EvaluatorPairFrictionLJVariants.h"


namespace hoomd
    {
namespace md
    {
namespace detail
    {

void export_FrictionPairFrictionLJLinear(pybind11::module& m)
    {
    export_FrictionPair<EvaluatorPairFrictionLJLinear>(m, "FrictionPairFrictionLJLinear");
    }

void export_FrictionPairFrictionLJConstant(pybind11::module& m)
    {
    export_FrictionPair<EvaluatorPairFrictionLJConstant>(m, "FrictionPairFrictionLJConstant");
    }

void export_FrictionPairFrictionLJCoulombNewton(pybind11::module& m)
    {
    export_FrictionPair<EvaluatorPairFrictionLJCoulombNewton>(m, "FrictionPairFrictionLJCoulombNewton");
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
