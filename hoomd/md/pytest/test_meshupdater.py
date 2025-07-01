# Copyright (c) 2009-2025 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import hoomd
import pytest
import numpy as np

def test_before_attaching():
    mesh = hoomd.mesh.Mesh()
    mesh_potential1 = hoomd.md.mesh.bond.Harmonic(mesh)
    mesh_potential1.params.default = dict(k=100,r0=1)

    mesh_potential2 = hoomd.md.mesh.conservation.TriangleArea(mesh)
    mesh_potential2.params.default = dict(k=100,A0=1)

    mdb = hoomd.md.update.MeshDynamicalBonding(10,mesh, 1, forces=[mesh_potential1])

    assert mdb.trigger == hoomd.trigger.Periodic(10)
    assert mdb.kT == 1
    assert mdb.mesh == mesh
    assert mdb.forces == [mesh_potential1]

    mdb.trigger = hoomd.trigger.Periodic(100)
    mdb.kT = 0.5
    mdb.forces.append(mesh_potential2)

    tree = hoomd.md.nlist.Cell(buffer=0.5)

    lj = hoomd.md.pair.LJ(tree, default_r_cut=2)
    lj.params.default = dict(epsilon=1, sigma=1)

    with pytest.raises(ValueError):
        mdb.forces.append(lj)

    assert mdb.trigger == hoomd.trigger.Periodic(100)
    assert mdb.kT == 0.5
    assert mdb.forces == [mesh_potential1,mesh_potential2]

@pytest.fixture(scope="session")
def local_snapshot_factory(device):
    def make_snapshot(d=0.0, particle_types=["A"], L=20):
        s = hoomd.Snapshot(device.communicator)
        N = 16
        if s.communicator.rank == 0:
            box = [4, 2*np.sqrt(3), L, 0, 0, 0]
            s.configuration.box = box
            s.particles.N = N

            base_positions = np.array(
                [
                    [-2.0,-1.73205081,0.0],
                    [-1.0,-1.73205081,0.0],
                    [0.0,-1.73205081,0.0],
                    [1.0,-1.73205081,0.0],
                    [-1.5,-0.8660254,0.0],
                    [-0.5,-0.8660254,d],
                    [0.5,-0.8660254,d],
                    [1.5,-0.8660254,0.0],
                    [-2.0,0.,0.0],
                    [-1.0,0.,0.0],
                    [0.0,0.,0.0],
                    [1.0,0.,0.0],
                    [-1.5,0.8660254,0.0],
                    [-0.5,0.8660254,0.0],
                    [0.5,0.8660254,0.0],
                    [1.5,0.8660254,0.0]
                ]
            )
            # move particles slightly in direction of MPI decomposition which
            # varies by simulation dimension
            s.particles.position[:] = base_positions
            s.particles.types = particle_types
        return s

    return make_snapshot


def test_after_attaching(local_snapshot_factory, simulation_factory):
    snap = local_snapshot_factory(d=1, L=20)
    sim = simulation_factory(snap)

    mesh = hoomd.mesh.Mesh()

    mesh_potential1 = hoomd.md.mesh.bond.Harmonic(mesh)
    mesh_potential1.params.default = dict(k=100,r0=1)

    mesh_potential2 = hoomd.md.mesh.conservation.TriangleArea(mesh)
    mesh_potential2.params.default = dict(k=100,A0=1)

    mdb = hoomd.md.update.MeshDynamicalBonding(10,mesh, 1, forces=[mesh_potential1])
    sim.operations.updaters.append(mdb)

    sim.operations.integrator = hoomd.md.Integrator(dt=0.005)

    sim.run(0)

    assert mdb.trigger == hoomd.trigger.Periodic(10)
    assert mdb.kT == 1
    assert mdb.mesh == mesh
    assert mdb.forces == [mesh_potential1]

    mdb.trigger = hoomd.trigger.Periodic(100)
    mdb.kT = 0.5
    mdb.forces.append(mesh_potential2)

    with pytest.raises(hoomd.error.MutabilityError):
        mdb.mesh = hoomd.mesh.Mesh()

    assert mdb.trigger == hoomd.trigger.Periodic(100)
    assert mdb.kT == 0.5
    assert mdb.forces == [mesh_potential1,mesh_potential2]

mesh_triangle_start=[
            [ 0,  1,  4],
            [ 0, 12,  1],
            [ 1,  2,  5],
            [ 1, 13,  2],
            [ 2,  3,  6],
            [ 2, 14,  3],
            [ 3,  0,  7],
            [ 3, 15,  0],
            [ 4,  5,  9],
            [ 4,  1,  5],
            [ 5,  2, 10],
            [ 6, 10,  2],
            [ 6,  7, 11],
            [ 6,  3,  7],
            [ 7,  4,  8],
            [ 7,  0,  4],
            [ 8,  9, 12],
            [ 8,  4,  9],
            [ 9, 10, 13],
            [ 9,  5, 10],
            [10, 11, 14],
            [10,  6, 11],
            [11,  8, 15],
            [11,  7,  8],
            [12, 13,  1],
            [12,  9, 13],
            [13, 14,  2],
            [13, 10, 14],
            [14, 15,  3],
            [14, 11, 15],
            [15, 12,  0],
            [15,  8, 12]
        ]

mesh_triangle_end=[
            [ 0,  1,  4],
            [ 0, 12,  1],
            [ 1,  2,  5],
            [ 1, 13,  2],
            [ 2,  3,  6],
            [ 2, 14,  3],
            [ 3,  0,  7],
            [ 3, 15,  0],
            [ 4,  5,  9],
            [ 4,  1,  5],
            [ 2,  6,  5],
            [10,  5,  6],
            [ 6,  7, 11],
            [ 6,  3,  7],
            [ 7,  4,  8],
            [ 7,  0,  4],
            [ 8,  9, 12],
            [ 8,  4,  9],
            [ 9, 10, 13],
            [ 9,  5, 10],
            [10, 11, 14],
            [10,  6, 11],
            [11,  8, 15],
            [11,  7,  8],
            [12, 13,  1],
            [12,  9, 13],
            [13, 14,  2],
            [13, 10, 14],
            [14, 15,  3],
            [14, 11, 15],
            [15, 12,  0],
            [15,  8, 12]
        ]


def test_updating(local_snapshot_factory, simulation_factory):
    snap = local_snapshot_factory(d=0.0, L=20)
    sim = simulation_factory(snap)

    mesh = hoomd.mesh.Mesh()
    mesh.types = ["mesh"]
    mesh.triangulation = dict(type_ids=[0] * len(mesh_triangle_start), triangles=mesh_triangle_start)

    mesh_potential = hoomd.md.mesh.bond.Harmonic(mesh)
    mesh_potential.params.default = dict(k=1,r0=1)

    integrator = hoomd.md.Integrator(dt=0.01)
    integrator.forces.append(mesh_potential)
    sim.operations.integrator = integrator

    sim.run(0)

    assert not np.isclose(mesh_potential.energy, 0)

    mdb = hoomd.md.update.MeshDynamicalBonding(hoomd.trigger.Periodic(1),mesh, kT = 0.001, forces=[mesh_potential])

    sim.operations += mdb

    sim.run(1)

    assert np.isclose(mesh_potential.energy, 0)

    assert np.array_equal(mesh.triangles, mesh_triangle_end)

_harmonic_arg_list = [
    (hoomd.md.mesh.bond.Harmonic, dict(k=30.0,r0=1.0))
]
_FENE_arg_list = [
    (hoomd.md.mesh.bond.FENEWCA, dict(k=30.0,r0=2.0,epsilon=1.0,sigma=1.0,delta=0.0))
]
_Tether_arg_list = [
    (hoomd.md.mesh.bond.Tether, dict(k_b=10,l_min=0.5,l_c1=0.9,l_c0=1.1,l_max=2.0))
]
_BendingRigidity_arg_list = [
    (hoomd.md.mesh.bending.BendingRigidity, dict(k=300.0))
]
_Helfrich_arg_list = [
    (hoomd.md.mesh.bending.Helfrich, dict(k=100.0))
]
_AreaConservation_arg_list = [
    (hoomd.md.mesh.conservation.Area, dict(k=100.0,A0=8*np.sqrt(3)))
]
_TriangleAreaConservation_arg_list = [
    (
        hoomd.md.mesh.conservation.TriangleArea,
        dict(k=100.0,A0=np.sqrt(3) /4)
    )
]

def get_mesh_potential_and_args():
    return (
        _harmonic_arg_list
        + _FENE_arg_list
        + _Tether_arg_list
        + _AreaConservation_arg_list
        + _TriangleAreaConservation_arg_list
        + _BendingRigidity_arg_list
        + _Helfrich_arg_list
    )

@pytest.mark.parametrize(
    "mesh_potential_cls, potential_kwargs", get_mesh_potential_and_args()
)
def test_reduce_energy(local_snapshot_factory, simulation_factory,mesh_potential_cls, potential_kwargs):
    snap = local_snapshot_factory(d=0.5, L=20)
    sim = simulation_factory(snap)

    mesh = hoomd.mesh.Mesh()
    mesh.types = ["mesh"]
    mesh.triangulation = dict(type_ids=[0] * len(mesh_triangle_start), triangles=mesh_triangle_start)

    mesh_potential = mesh_potential_cls(mesh)
    mesh_potential.params["mesh"] = potential_kwargs

    integrator = hoomd.md.Integrator(dt=0.01)
    integrator.forces.append(mesh_potential)
    sim.operations.integrator = integrator

    sim.run(0)

    energy = mesh_potential.energy

    mdb = hoomd.md.update.MeshDynamicalBonding(hoomd.trigger.Periodic(1),mesh, kT = 0.001, forces=[mesh_potential])

    sim.operations += mdb

    sim.run(1)

    assert mesh_potential.energy <= energy
