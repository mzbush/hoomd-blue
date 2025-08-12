# Copyright (c) 2009-2025 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

r"""Frictional pair force classes apply a force, and torque on every
particle in the simulation state.

`FrictionPair` applies cuttoffs, exclusions, and assigns per particle
energies and virials in the same manner as `hoomd.md.pair.Pair` (Energies and virials not adapted!)

`FrictionPair` does not support the ``'xplor'`` shifting mode or the ``r_on``
parameter.

"""

from collections.abc import Sequence
import json
from numbers import Number

from hoomd.md.pair.pair import Pair
from hoomd.logging import log
from hoomd.data.parameterdicts import TypeParameterDict
from hoomd.data.typeparam import TypeParameter
from hoomd.data.typeconverter import OnlyTypes, OnlyIf, to_type_converter
import numpy as np

class FrictionalPair(Pair):
    r"""Base class friction pair force.

    `FrictionPair` is the base class for all frictional pair forces.

    Warning:
        This class should not be instantiated by users. The class can be used
        for `isinstance` or `issubclass` checks.
    """

    __doc__ += Pair._doc_inherited
    _accepted_modes = ("none", "shift")

    def __init__(self, nlist, default_r_cut=None, mode="none"):
        super().__init__(nlist, default_r_cut, 0.0, mode)

    def _return_type_shapes(self):
        type_shapes = self._cpp_obj.getTypeShapesPy()
        ret = [json.loads(json_string) for json_string in type_shapes]
        return ret

class FrictionLJLinear(FrictionalPair):
    r"""Linear frictional model pair force with the LJ conservative force. 

    Args:
        nlist (hoomd.md.nlist.NeighborList): Neighbor list
        default_r_cut (float): Default cutoff radius: math:`[\mathrm{length}]`.
    
    `FrictionLJLinear` computes the frictional interaction between pairs of particles.   
    """

    _cpp_class_name = "FrictionPairFrictionLJLinear"
    __doc__ = __doc__.replace("{inherited}", FrictionalPair._doc_inherited)

    def __init__(self, nlist, default_r_cut=None):
        super().__init__(nlist, default_r_cut, 'none')
        params = TypeParameter(
            'params', 'particle_types',
            TypeParameterDict(epsilon=float, sigma=float, gamma_f=float, kT=float, len_keys=2))
        self._add_typeparam(params)

class FrictionLJConstant(FrictionalPair):
    r"""Constant frictional model pair force with the LJ conservative force. 

    Args:
        nlist (hoomd.md.nlist.NeighborList): Neighbor list
        default_r_cut (float): Default cutoff radius: math:`[\mathrm{length}]`.
    
    `FrictionLJConstant` computes the frictional interaction between pairs of particles.   
    """

    _cpp_class_name = "FrictionPairFrictionLJConstant"
    __doc__ = __doc__.replace("{inherited}", FrictionalPair._doc_inherited)

    def __init__(self, nlist, default_r_cut=None):
        super().__init__(nlist, default_r_cut, 'none')
        params = TypeParameter(
            'params', 'particle_types',
            TypeParameterDict(epsilon=float, sigma=float, kappa_f=float, kT=float, len_keys=2))
        self._add_typeparam(params)

class FrictionLJCoulombNewton(FrictionalPair):
    r"""Coulomb-Newton frictional model pair force with the LJ conservative force. 

    Args:
        nlist (hoomd.md.nlist.NeighborList): Neighbor list
        default_r_cut (float): Default cutoff radius: math:`[\mathrm{length}]`.
    
    `FrictionLJCoulombNewton` computes the frictional interaction between pairs of particles.   
    """

    _cpp_class_name = "FrictionPairFrictionLJCoulombNewton"
    __doc__ = __doc__.replace("{inherited}", FrictionalPair._doc_inherited)

    def __init__(self, nlist, default_r_cut=None):
        super().__init__(nlist, default_r_cut, 'none')
        params = TypeParameter(
            'params', 'particle_types',
            TypeParameterDict(epsilon=float, sigma=float, gamma_f=float, kappa_f=float, kT=float, len_keys=2))
        self._add_typeparam(params)
        
__all__ = [
    "FrictionPair",
    "FrictionLJLinear",
    "FrictionLJConstant",
    "FrictionLJCoulombNewton",
]
 