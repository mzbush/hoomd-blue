# Copyright (c) 2009-2025 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import numpy as np
import pytest

import hoomd
from hoomd.conftest import pickling_check


@pytest.fixture
def small_snap():
    snap = hoomd.Snapshot()
    if snap.communicator.rank == 0:
        snap.configuration.box = [20, 20, 20, 0, 0, 0]
        snap.particles.N = 1
        snap.particles.types = ["A"]
        snap.mpcd.N = 1
        snap.mpcd.types = ["A"]
    return snap


@pytest.fixture
def dimer_rigid_body():
    return {
        "constituent_types": ["B", "B"],
        "positions": [[-1.2, 0, 0], [1.2, 0, 0]],
        "orientations": [[1, 0, 0, 0], [1, 0, 0, 0]],
    }


@pytest.fixture
def dimer_properties():
    return {"inertia": [0.0, 2.88, 2.88], "mass": np.array([2, 1, 1])}


def test_cell_list(small_snap, simulation_factory):
    if small_snap.communicator.rank == 0:
        small_snap.configuration.box = [20, 30, 40, 0, 0, 0]
    sim = simulation_factory(small_snap)

    cl = hoomd.mpcd.collide.CellList()

    cl._attach(sim)
    assert cl.num_cells == (20, 30, 40)


@pytest.mark.parametrize(
    "cls, init_args",
    [
        (
            hoomd.mpcd.collide.AndersenThermostat,
            {
                "kT": 1.0,
            },
        ),
        (
            hoomd.mpcd.collide.StochasticRotationDynamics,
            {
                "angle": 90,
            },
        ),
    ],
    ids=["AndersenThermostat", "StochasticRotationDynamics"],
)
class TestCollisionMethod:
    def test_create(self, small_snap, simulation_factory, cls, init_args):
        sim = simulation_factory(small_snap)
        cm = cls(period=5, **init_args)
        ig = hoomd.mpcd.Integrator(dt=0.02, collision_method=cm)
        sim.operations.integrator = ig

        assert ig.collision_method is cm
        assert cm.embedded_particles is None
        assert cm.period == 5
        if "kT" in init_args:
            assert isinstance(cm.kT, hoomd.variant.Constant)
            assert cm.kT(0) == init_args["kT"]

        sim.run(0)
        assert ig.collision_method is cm
        assert cm.embedded_particles is None
        assert cm.period == 5
        if "kT" in init_args:
            assert isinstance(cm.kT, hoomd.variant.Constant)
            assert cm.kT(0) == init_args["kT"]

    def test_pickling(self, small_snap, simulation_factory, cls, init_args):
        cm = cls(period=1, **init_args)
        pickling_check(cm)

        sim = simulation_factory(small_snap)
        sim.operations.integrator = hoomd.mpcd.Integrator(dt=0.02, collision_method=cm)
        sim.run(0)
        pickling_check(cm)

    def test_embed(self, small_snap, simulation_factory, cls, init_args):
        sim = simulation_factory(small_snap)
        cm = cls(period=1, embedded_particles=hoomd.filter.All(), **init_args)
        sim.operations.integrator = hoomd.mpcd.Integrator(dt=0.02, collision_method=cm)

        assert isinstance(cm.embedded_particles, hoomd.filter.All)
        sim.run(0)
        assert isinstance(cm.embedded_particles, hoomd.filter.All)

    def test_temperature(self, small_snap, simulation_factory, cls, init_args):
        sim = simulation_factory(small_snap)
        if "kT" not in init_args:
            init_args["kT"] = 1.0
            kT_required = False
        else:
            kT_required = True
        cm = cls(period=1, **init_args)
        sim.operations.integrator = hoomd.mpcd.Integrator(dt=0.02, collision_method=cm)

        assert isinstance(cm.kT, hoomd.variant.Constant)
        assert cm.kT(0) == 1.0
        sim.run(0)
        assert isinstance(cm.kT, hoomd.variant.Constant)

        ramp = hoomd.variant.Ramp(1.0, 2.0, 0, 10)
        cm.kT = ramp
        assert cm.kT is ramp
        sim.run(0)
        assert cm.kT is ramp

        if not kT_required:
            cm.kT = None
            assert cm.kT is None
            sim.run(0)
            assert cm.kT is None

    def test_run(self, small_snap, simulation_factory, cls, init_args):
        sim = simulation_factory(small_snap)
        cm = cls(period=1, **init_args)
        sim.operations.integrator = hoomd.mpcd.Integrator(dt=0.02, collision_method=cm)

        # test that one step can run without error with only solvent
        sim.run(1)

        # test that one step can run without error with embedded particles
        if "kT" not in init_args:
            init_args["kT"] = 1.0
        sim.operations.integrator.collision_method = cls(
            period=1, embedded_particles=hoomd.filter.All(), **init_args
        )
        sim.run(1)


def test_rigid_collide(
    one_particle_snapshot_factory,
    simulation_factory,
    dimer_rigid_body,
    dimer_properties,
):
    snap = one_particle_snapshot_factory(particle_types=["A", "B"])
    if snap.communicator.rank == 0:
        snap.particles.moment_inertia[:] = [dimer_properties["inertia"]]
        snap.particles.mass[:] = [dimer_properties["mass"][0]]
        snap.particles.velocity[:] = [[0, 0, 1]]
        # snap.particles.angmom[:] = [[0,0,5,0]]
        snap.mpcd.N = 2
        snap.mpcd.types = ["C"]
        snap.mpcd.position[:] = [[1.2, 0, 0], [-1.2, 0, 0]]
        snap.mpcd.velocity[:] = [[0.6, 0.7, 0.8], [0.6, -0.7, 0.8]]

    sim = simulation_factory(snap)
    sim.seed = 5

    rigid = hoomd.md.constrain.Rigid()
    rigid.body["A"] = dimer_rigid_body
    rigid.create_bodies(sim.state)

    intermed_snapshot = sim.state.get_snapshot()
    if intermed_snapshot.communicator.rank == 0:
        flags = (
            intermed_snapshot.particles.typeid
            == intermed_snapshot.particles.types.index("B")
        )
        intermed_snapshot.particles.mass[flags] = dimer_properties["mass"][flags]
    sim.state.set_snapshot(intermed_snapshot)

    cm = hoomd.mpcd.collide.StochasticRotationDynamics(
        period=1,
        angle=90,
        embedded_particles=hoomd.filter.Rigid(flags=("constituent",)),
    )
    sim.operations.integrator = hoomd.mpcd.Integrator(
        dt=0, collision_method=cm, integrate_rotational_dof=True, rigid=rigid
    )
    sim.run(1)
    new_snapshot = sim.state.get_snapshot()
    if new_snapshot.communicator.rank == 0:
        assert np.array_equal(dimer_properties["mass"], new_snapshot.particles.mass)
        new_velo = new_snapshot.particles.velocity
        # the central particle speed should change despite not being in the filter
        assert not np.any(np.isclose(new_velo[0], [0, 0, 1]))
        # for a spinning dimer, constituent particle velocities could cancel
        calculated_central_speed = (new_velo[2] - new_velo[1]) / 2 + new_velo[1]
        assert np.allclose(new_velo[0], calculated_central_speed)
