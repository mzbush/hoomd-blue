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

    @pytest.mark.parametrize(
        "rigid_velo", [[0, 0, 0], [1, 2, 3]], ids=["Stationary", "Moving"]
    )
    @pytest.mark.parametrize("rigid_angmom", [[0, 0, 0, 0]], ids=["Nonrotating"])
    @pytest.mark.parametrize(
        "rigid_pos", [[0, 0, 0], [5, 5, 5]], ids=["center", "edge"]
    )
    @pytest.mark.parametrize(
        "rigid_def,rigid_properties",
        [
            (
                {
                    "constituent_types": ["B", "B"],
                    "positions": [[-2.4, 0, 0], [2.4, 0, 0]],
                    "orientations": [[1, 0, 0, 0], [1, 0, 0, 0]],
                },
                {"inertia": [0.0, 11.52, 11.52], "mass": np.array([2, 1, 1])},
            ),
            (
                {
                    "constituent_types": ["B", "B", "B", "B"],
                    "positions": np.array(
                        [
                            [1, 0, -1 / (2 ** (1.0 / 2.0))],
                            [-1, 0, -1 / (2 ** (1.0 / 2.0))],
                            [0, -1, 1 / (2 ** (1.0 / 2.0))],
                            [0, 1, 1 / (2 ** (1.0 / 2.0))],
                        ]
                    )
                    * 2,
                    "orientations": [(1.0, 0.0, 0.0, 0.0)] * 4,
                },
                {
                    "inertia": [1.0, 1.0, 1.0],
                    "mass": np.array([1 / 4, 1 / 16, 1 / 16, 1 / 16, 1 / 16]),
                },
            ),
            (
                {
                    "constituent_types": ["B", "B", "B", "B"],
                    "positions": np.array(
                        [
                            [1, 0, -1 / (2 ** (1.0 / 2.0))],
                            [-1, 0, -1 / (2 ** (1.0 / 2.0))],
                            [0, -1, 1 / (2 ** (1.0 / 2.0))],
                            [0, 1, 1 / (2 ** (1.0 / 2.0))],
                        ]
                    )
                    * 2,
                    "orientations": [(1.0, 0.0, 0.0, 0.0)] * 4,
                },
                {
                    "inertia": [7.0, 5.0, 6.0],
                    "mass": np.array([1.5, 1 / 4, 1 / 4, 1 / 2, 1 / 2]),
                },
            ),
        ],
        ids=["dimer", "cross", "uneven-mass"],
    )
    def test_rigid_collide(
        self,
        one_particle_snapshot_factory,
        simulation_factory,
        cls,
        init_args,
        rigid_velo,
        rigid_angmom,
        rigid_pos,
        rigid_def,
        rigid_properties,
    ):
        if "kT" not in init_args:
            init_args["kT"] = 1.0
        mpcd_N = len(rigid_def["constituent_types"])
        rng = np.random.default_rng(seed=42)
        mpcd_velo = rng.normal(0.0, np.sqrt(init_args["kT"]), (mpcd_N, 3))
        mpcd_velo -= np.mean(mpcd_velo, axis=0)

        # create simulation
        initial_snap = one_particle_snapshot_factory(
            particle_types=["A", "B"], position=rigid_pos, L=11
        )
        total_mass = rigid_properties["mass"][0]
        if initial_snap.communicator.rank == 0:
            initial_snap.particles.moment_inertia[:] = [rigid_properties["inertia"]]
            initial_snap.particles.mass[:] = [total_mass]
            initial_snap.particles.velocity[:] = [rigid_velo]
            initial_snap.particles.angmom[:] = [rigid_angmom]

        sim = simulation_factory(initial_snap)
        sim.seed = 5

        rigid = hoomd.md.constrain.Rigid()
        rigid.body["A"] = rigid_def
        rigid.create_bodies(sim.state)

        intermed_snap = sim.state.get_snapshot()
        if intermed_snap.communicator.rank == 0:
            # add mass of constituents
            flags = (
                intermed_snap.particles.typeid
                == intermed_snap.particles.types.index("B")
            )
            intermed_snap.particles.mass[flags] = rigid_properties["mass"][flags]
            intermed_snap.wrap()

            # place the mpcd particles on top of constituents
            intermed_snap.mpcd.N = mpcd_N
            intermed_snap.mpcd.types = ["C"]
            intermed_snap.mpcd.position[:] = intermed_snap.particles.position[flags]
            intermed_snap.mpcd.velocity[:] = mpcd_velo
        sim.state.set_snapshot(intermed_snap)

        sim.operations.integrator = hoomd.mpcd.Integrator(
            dt=0, integrate_rotational_dof=True, rigid=rigid
        )
        sim.operations.integrator.cell_list.shift = False
        sim.operations.integrator.collision_method = cls(
            period=1,
            embedded_particles=hoomd.filter.Rigid(flags=("constituent",)),
            **init_args,
        )

        # run simulation
        sim.run(1)
        new_snap = sim.state.get_snapshot()
        if new_snap.communicator.rank == 0:
            assert np.array_equal(rigid_properties["mass"], new_snap.particles.mass)
            central_flag = new_snap.particles.typeid == new_snap.particles.types.index(
                "A"
            )
            constit_flag = new_snap.particles.typeid == new_snap.particles.types.index(
                "B"
            )
            new_velo_central = new_snap.particles.velocity[central_flag]
            new_velo_constituent = new_snap.particles.velocity[constit_flag]
            # solve for expected central particle velocity based on linear momentum
            initial_mpcd_momentum = np.sum(mpcd_velo, axis=0)
            initial_md_momentum = np.array(rigid_velo) * total_mass
            initial_momentum = initial_md_momentum + initial_mpcd_momentum

            final_mpcd_momentum = np.sum(
                new_snap.mpcd.velocity * new_snap.mpcd.mass, axis=0
            )
            final_md_linmom = initial_momentum - final_mpcd_momentum
            final_md_velocity = final_md_linmom / total_mass
            assert np.allclose(final_md_velocity, new_velo_central)

            # solve for expected angular momentum change based on solvent
            # multiply expected change by 2 to get quaternion
            change_mpcd_momentum = (
                new_snap.mpcd.velocity - mpcd_velo
            ) * new_snap.mpcd.mass
            expected_change_md_angmom = np.zeros(4)
            expected_change_md_angmom[1:] = (
                np.sum(
                    np.cross(rigid_def["positions"], -1 * change_mpcd_momentum), axis=0
                )
                * 2
            )
            change_md_angmom = new_snap.particles.angmom[0] - rigid_angmom
            assert np.allclose(expected_change_md_angmom, change_md_angmom)

            # check the constituent velocities match the central particle
            # since orientation is stuck at [1, 0, 0, 0], angular velocity
            # is 0.5 * real part of angmom / inertia.
            new_angmom = new_snap.particles.angmom[0]
            omega = [
                0 if i == 0 else 0.5 * a / i
                for i, a in zip(rigid_properties["inertia"], new_angmom[1:])
            ]
            expected_tangential_velocity = np.cross(omega, rigid_def["positions"])
            expected_velocity = np.add(expected_tangential_velocity, new_velo_central)
            assert np.allclose(new_velo_constituent, expected_velocity)
