// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "FrictionPairGPU.cuh"
#include "EvaluatorPairFrictionLJBase.h"
#include "EvaluatorPairFrictionLJVariants.h"

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
        template hipError_t __attribute__((visibility("default")))
        gpu_compute_pair_friction_forces<EvaluatorPairFrictionLJLinear>(
            const a_pair_args_t& pair_args,
            const EvaluatorPairFrictionLJLinear::param_type* d_param,
            const EvaluatorPairFrictionLJLinear::shape_type* d_shape_param);
        
        template hipError_t __attribute__((visibility("default")))
        gpu_compute_pair_friction_forces<EvaluatorPairFrictionLJConstant>(
            const a_pair_args_t& pair_args,
            const EvaluatorPairFrictionLJConstant::param_type* d_param,
            const EvaluatorPairFrictionLJConstant::shape_type* d_shape_param);
        
        template hipError_t __attribute__((visibility("default")))
        gpu_compute_pair_friction_forces<EvaluatorPairFrictionLJCoulombNewton>(
            const a_pair_args_t& pair_args,
            const EvaluatorPairFrictionLJCoulombNewton::param_type* d_param,
            const EvaluatorPairFrictionLJCoulombNewton::shape_type* d_shape_param);
            } // end namespace kernel
            } // end namespace md
            } // end namespace hoomd
