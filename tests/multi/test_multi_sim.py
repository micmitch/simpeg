import numpy as np
from SimPEG.potential_fields import gravity
from SimPEG.electromagnetics.static import resistivity as dc
from SimPEG import maps
from discretize import TensorMesh
import scipy.sparse as sp

from SimPEG.stitching import MultiSimulation, SumMultiSimulation, RepeatedSimulation


def test_multi_sim_correctness():
    mesh = TensorMesh([16, 16, 16], origin="CCN")

    rx_locs = np.mgrid[-0.25:0.25:5j, -0.25:0.25:5j, 0:1:1j]
    rx_locs = rx_locs.reshape(3, -1).T
    rxs = dc.receivers.Pole(rx_locs)
    source_locs = np.mgrid[-0.5:0.5:10j, 0:1:1j, 0:1:1j].reshape(3, -1).T
    src_list = [
        dc.sources.Pole(
            [
                rxs,
            ],
            location=loc,
        )
        for loc in source_locs
    ]
    survey_full = dc.Survey(src_list)
    full_sim = dc.Simulation3DNodal(
        mesh, survey=survey_full, sigmaMap=maps.IdentityMap()
    )

    m_test = np.arange(mesh.n_cells) / mesh.n_cells + 0.1

    # split by chunks of sources
    chunk_size = 3
    sims = []
    mappings = []
    for i in range(0, len(src_list) + 1, chunk_size):
        end = min(i + chunk_size, len(src_list))
        if i == end:
            break
        survey_chunk = dc.Survey(src_list[i:end])
        sims.append(
            dc.Simulation3DNodal(mesh, survey=survey_chunk, sigmaMap=maps.IdentityMap())
        )
        mappings.append(maps.IdentityMap())

    multi_sim = MultiSimulation(sims, mappings)

    # test fields objects
    f_full = full_sim.fields(m_test)
    f_mult = multi_sim.fields(m_test)
    sol_full = f_full[:, "phiSolution"]
    sol_mult = np.concatenate([f[:, "phiSolution"] for f in f_mult], axis=1)
    np.testing.assert_allclose(sol_full, sol_mult)

    # test data output
    d_full = full_sim.dpred(m_test, f=f_full)
    d_mult = multi_sim.dpred(m_test, f=f_mult)
    np.testing.assert_allclose(d_full, d_mult)

    # test Jvec
    u = np.random.rand(mesh.n_cells)
    jvec_full = full_sim.Jvec(m_test, u, f=f_full)
    jvec_mult = multi_sim.Jvec(m_test, u, f=f_mult)

    np.testing.assert_allclose(jvec_full, jvec_mult)

    # test Jtvec
    v = np.random.rand(survey_full.nD)
    jtvec_full = full_sim.Jtvec(m_test, v, f=f_full)
    jtvec_mult = multi_sim.Jtvec(m_test, v, f=f_mult)

    np.testing.assert_allclose(jtvec_full, jtvec_mult)

    # test get diag
    diag_full = full_sim.getJtJdiag(m_test, f=f_full)
    diag_mult = multi_sim.getJtJdiag(m_test, f=f_mult)

    np.testing.assert_allclose(diag_full, diag_mult)


def test_sum_sim_correctness():
    mesh = TensorMesh([16, 16, 16], origin="CCN")

    rx_locs = np.mgrid[-0.25:0.25:5j, -0.25:0.25:5j, 0:1:1j].reshape(3, -1).T
    rx = gravity.Point(rx_locs, components=["gz"])
    survey = gravity.Survey(gravity.SourceField(rx))
    full_sim = gravity.Simulation3DIntegral(
        mesh, survey=survey, rhoMap=maps.IdentityMap(), n_processes=1
    )

    mesh_bot = TensorMesh([mesh.h[0], mesh.h[1], mesh.h[2][:8]], origin=mesh.origin)
    mesh_top = TensorMesh(
        [mesh.h[0], mesh.h[1], mesh.h[2][8:]], origin=["C", "C", mesh.nodes_z[8]]
    )

    mappings = [
        maps.Mesh2Mesh((mesh_bot, mesh)),
        maps.Mesh2Mesh((mesh_top, mesh)),
    ]
    sims = [
        gravity.Simulation3DIntegral(
            mesh_bot, survey=survey, rhoMap=maps.IdentityMap(), n_processes=1
        ),
        gravity.Simulation3DIntegral(
            mesh_top, survey=survey, rhoMap=maps.IdentityMap(), n_processes=1
        ),
    ]

    sum_sim = SumMultiSimulation(sims, mappings)

    m_test = np.arange(mesh.n_cells) / mesh.n_cells + 0.1

    # test fields objects
    f_full = full_sim.fields(m_test)
    f_mult = sum_sim.fields(m_test)
    np.testing.assert_allclose(f_full, sum(f_mult))

    # test data output
    d_full = full_sim.dpred(m_test, f=f_full)
    d_mult = sum_sim.dpred(m_test, f=f_mult)
    np.testing.assert_allclose(d_full, d_mult)

    # test Jvec
    u = np.random.rand(mesh.n_cells)
    jvec_full = full_sim.Jvec(m_test, u, f=f_full)
    jvec_mult = sum_sim.Jvec(m_test, u, f=f_mult)

    np.testing.assert_allclose(jvec_full, jvec_mult)

    # test Jtvec
    v = np.random.rand(survey.nD)
    jtvec_full = full_sim.Jtvec(m_test, v, f=f_full)
    jtvec_mult = sum_sim.Jtvec(m_test, v, f=f_mult)

    np.testing.assert_allclose(jtvec_full, jtvec_mult)

    # test get diag
    diag_full = full_sim.getJtJdiag(m_test, f=f_full)
    diag_mult = sum_sim.getJtJdiag(m_test, f=f_mult)

    np.testing.assert_allclose(diag_full, diag_mult)


def test_repeat_sim_correctness():
    # multi sim is tested for correctness
    # so can test the repeat against the multi sim
    mesh = TensorMesh([8, 8, 8], origin="CCN")

    rx_locs = np.mgrid[-0.25:0.25:5j, -0.25:0.25:5j, 0:1:1j].reshape(3, -1).T
    rx = gravity.Point(rx_locs, components=["gz"])
    survey = gravity.Survey(gravity.SourceField(rx))
    sim = gravity.Simulation3DIntegral(
        mesh, survey=survey, rhoMap=maps.IdentityMap(), n_processes=1
    )

    time_mesh = TensorMesh(
        [
            8,
        ],
        origin=[
            0,
        ],
    )
    sim_ts = np.linspace(0, 1, 6)

    mappings = []
    simulations = []
    eye = sp.eye(mesh.n_cells, mesh.n_cells)
    for t in sim_ts:
        ave_time = time_mesh.get_interpolation_matrix(
            [
                t,
            ]
        )
        ave_full = sp.kron(ave_time, eye, format="csr")
        mappings.append(maps.LinearMap(ave_full))
        simulations.append(
            gravity.Simulation3DIntegral(
                mesh, survey=survey, rhoMap=maps.IdentityMap(), n_processes=1
            )
        )

    multi_sim = MultiSimulation(simulations, mappings)
    repeat_sim = RepeatedSimulation(sim, mappings)

    model = np.random.rand(time_mesh.n_cells, mesh.n_cells).reshape(-1)

    # test field things
    f_full = multi_sim.fields(model)
    f_mult = repeat_sim.fields(model)
    np.testing.assert_equal(np.c_[f_full], np.c_[f_mult])

    d_full = multi_sim.dpred(model, f_full)
    d_repeat = repeat_sim.dpred(model, f_mult)
    np.testing.assert_equal(d_full, d_repeat)

    # test Jvec
    u = np.random.rand(len(model))
    jvec_full = multi_sim.Jvec(model, u, f=f_full)
    jvec_mult = repeat_sim.Jvec(model, u, f=f_mult)
    np.testing.assert_allclose(jvec_full, jvec_mult)

    # test Jtvec
    v = np.random.rand(len(sim_ts) * survey.nD)
    jtvec_full = multi_sim.Jtvec(model, v, f=f_full)
    jtvec_mult = repeat_sim.Jtvec(model, v, f=f_mult)
    np.testing.assert_allclose(jtvec_full, jtvec_mult)

    # test get diag
    diag_full = multi_sim.getJtJdiag(model, f=f_full)
    diag_mult = repeat_sim.getJtJdiag(model, f=f_mult)
    np.testing.assert_allclose(diag_full, diag_mult)
