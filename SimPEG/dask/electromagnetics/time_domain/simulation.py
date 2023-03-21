import dask
import dask.array

from ....electromagnetics.time_domain.simulation import BaseTDEMSimulation as Sim
from ....utils import Zero
from multiprocessing import cpu_count
import numpy as np
import scipy.sparse as sp
from time import time
from dask import array, compute, delayed
from SimPEG.dask.simulation import dask_Jvec, dask_Jtvec, dask_getJtJdiag
import zarr
from SimPEG.utils import mkvc
from tqdm import tqdm
Sim.sensitivity_path = './sensitivity/'
Sim.gtgdiag = None
Sim.store_sensitivities = True

Sim.getJtJdiag = dask_getJtJdiag
Sim.Jvec = dask_Jvec
Sim.Jtvec = dask_Jtvec


def fields(self, m=None, return_Ainv=False):
    if m is not None:
        self.model = m

    f = self.fieldsPair(self)

    # set initial fields
    f[:, self._fieldType + "Solution", 0] = self.getInitialFields()

    Ainv = {}
    ATinv = {}
    for tInd, dt in enumerate(self.time_steps):

        if dt not in Ainv:
            A = self.getAdiag(tInd)
            Ainv[dt] = self.solver(sp.csr_matrix(A), **self.solver_opts)
            if return_Ainv:
                ATinv[dt] = self.solver(sp.csr_matrix(A.T), **self.solver_opts)

        rhs = self.getRHS(tInd + 1)
        Asubdiag = self.getAsubdiag(tInd)
        sol = Ainv[dt] * (rhs - Asubdiag * f[:, (self._fieldType + "Solution"), tInd])
        f[:, self._fieldType + "Solution", tInd + 1] = sol

    for A in Ainv.values():
        A.clean()

    if return_Ainv:
        return f, ATinv
    else:
        return f, None


Sim.fields = fields


def dask_dpred(self, m=None, f=None, compute_J=False):
    """
    dpred(m, f=None)
    Create the projected data from a model.
    The fields, f, (if provided) will be used for the predicted data
    instead of recalculating the fields (which may be expensive!).

    .. math::

        d_\\text{pred} = P(f(m))

    Where P is a projection of the fields onto the data space.
    """
    if self.survey is None:
        raise AttributeError(
            "The survey has not yet been set and is required to compute "
            "data. Please set the survey for the simulation: "
            "simulation.survey = survey"
        )

    if f is None:
        if m is None:
            m = self.model
        f, Ainv = self.fields(m, return_Ainv=compute_J)

    def evaluate_receiver(source, receiver, mesh, time_mesh, fields):
        return receiver.eval(source, mesh, time_mesh, fields).flatten()

    row = delayed(evaluate_receiver, pure=True)
    rows = []
    for src in self.survey.source_list:
        for rx in src.receiver_list:
            rows.append(array.from_delayed(
                row(src, rx, self.mesh, self.time_mesh, f),
                dtype=np.float32,
                shape=(rx.nD,),
            ))

    data = array.hstack(rows).compute()

    if compute_J and self._Jmatrix is None:
        Jmatrix = self.compute_J(f=f, Ainv=Ainv)
        return data, Jmatrix

    return data


Sim.dpred = dask_dpred
Sim.field_derivs = None



def compute_J(self, f=None, Ainv=None):

    if f is None:
        f, Ainv = self.fields(self.model, return_Ainv=True)

    m_size = self.model.size
    row_chunks = int(np.ceil(
        float(self.survey.nD) / np.ceil(float(m_size) * self.survey.nD * 8. * 1e-6 / self.max_chunk_size)
    ))

    solution_type = self._fieldType + "Solution"  # the thing we solved for

    if self.store_sensitivities == "disk":
        Jmatrix = zarr.open(
            self.sensitivity_path + f"J.zarr",
            mode='w',
            shape=(self.survey.nD, m_size),
            chunks=(row_chunks, m_size)
        )
        partial_derivs = zarr.open(
            self.sensitivity_path + f"partials.zarr",
            mode='w',
            shape=(self.getAsubdiag(0).shape[0], self.survey.nD),
            chunks=(self.getAsubdiag(0).shape[0], row_chunks)
        )
    else:
        Jmatrix = np.zeros((self.survey.nD, m_size), dtype=np.float32)
        partial_derivs = np.zeros((self.getAsubdiag(0).shape[0], self.survey.nD), dtype=np.float32)

    if self.field_derivs is None:
        block_size = len(f[self.survey.source_list[0], solution_type, 0])
        field_derivs = []

        for tInd in range(self.nT + 1):
            d_count = 0
            df_duT_v = []
            for i_s, src in enumerate(self.survey.source_list):
                src_field_derivs = delayed(block_deriv, pure=True)(self, src, tInd, f, block_size)
                df_duT_v += [src_field_derivs]
                d_count += np.sum([rx.nD for rx in src.receiver_list])

            field_derivs += [df_duT_v]
        self.field_derivs = dask.compute(field_derivs)[0]

    f = dask.delayed(f)
    field_derivatives = None
    batch_map = {}

    for tInd, dt in tqdm(zip(reversed(range(self.nT)), reversed(self.time_steps))):

        AdiagTinv = Ainv[dt]
        Asubdiag = self.getAsubdiag(tInd)
        d_count = 0
        block_count = 0
        field_deriv_blocks = []
        j_row_blocks = []
        count = 0
        batch_block = []
        batch_indices = []
        batch_count = 0
        for isrc, src in enumerate(self.survey.source_list):
            field_blocks = []
            n_data = self.field_derivs[tInd+1][isrc][0].shape[1]
            n_blocks = int(np.ceil((m_size * n_data) * 8. * 1e-6 / 128.))
            sub_blocks = np.array_split(np.arange(n_data), n_blocks)

            for i_block, block_ind in enumerate(sub_blocks):

                if field_derivatives is None:
                    batch_block.append(self.field_derivs[tInd + 1][isrc][0][:, block_ind].toarray())
                    batch_map[isrc, i_block] = (batch_count, count)
                else:
                    i_file, i_block = batch_map[isrc, i_block]
                    batch_block.append(field_derivatives[i_file][:, i_block:i_block + len(block_ind)])

                batch_indices.append((isrc, block_ind))
                block_count += 1

                if block_count >= cpu_count():
                    f_blocks, j_blocks = process_blocks(
                        self, AdiagTinv, d_count, batch_block, batch_indices, Asubdiag, f, tInd,
                        solution_type, Jmatrix
                    )
                    field_deriv_blocks.append(dask.array.hstack(f_blocks))
                    j_row_blocks.append(j_blocks)

                    batch_block, batch_indices = [], []
                    block_count = 0
                    batch_count += 1
                    d_count += count
                    count = 0
                    
                count += len(block_ind)
                # if isrc not in field_derivatives:
                #     ATinv_df_duT_v = (
                #         AdiagTinv * self.field_derivs[tInd + 1][isrc][0][:, block_ind].toarray()
                #     )
                # else:
                #     ATinv_df_duT_v = AdiagTinv * np.asarray(field_derivatives[isrc][:, block_ind])

        f_blocks, j_blocks = process_blocks(
            self, AdiagTinv, d_count, batch_block, batch_indices, Asubdiag, f, tInd,
            solution_type, Jmatrix
        )
        field_deriv_blocks.append(dask.array.hstack(f_blocks))
        j_row_blocks.append(j_blocks)
        del field_derivatives

        if self.store_sensitivities == "disk":
            Jmatrix.set_orthogonal_selection(
                (np.arange(self.survey.nD), slice(None)),
                Jmatrix + dask.array.vstack(j_row_blocks).astype(np.float32)
            )
            field_derivatives = [
                dask.array.to_zarr(
                    field_deriv_blocks[i], self.sensitivity_path + f"field_derivs_{i}.zarr",
                    overwrite=True,
                    return_stored = True,
                ) for i in range(len(field_deriv_blocks))
            ]
        else:
            dask.compute(j_row_blocks)
            field_derivatives = dask.compute(field_deriv_blocks)[0]

    for A in Ainv.values():
        A.clean()

    if self.store_sensitivities == "disk":
        del Jmatrix
        return array.from_zarr(self.sensitivity_path + f"J.zarr")

    return Jmatrix

Sim.compute_J = compute_J


def process_blocks(
        self, AdiagTinv, d_count, batch_block, batch_indices, Asubdiag, f, tInd,
        solution_type, Jmatrix
    ):
    ATinv_df_duT_v = AdiagTinv * np.hstack(batch_block)
    field_blocks = []
    j_row_blocks = []
    count = 0
    for block, indices in zip(batch_block, batch_indices):
        block_size = block.shape[1]
        field_blocks.append(
            dask.array.from_delayed(
                delayed(parallel_field_deriv, pure=True)(
                    ATinv_df_duT_v[:, count: count + block_size], Asubdiag,
                    self.field_derivs[tInd][indices[0]][0][:, indices[1]]
                ),
                shape=(Asubdiag.shape[0], block_size),
                dtype=np.float64
            )
        )
        j_row_blocks.append(dask.array.from_delayed(
            delayed(parallel_block_compute, pure=True)(
                self, f,
                self.survey.source_list[indices[0]],
                ATinv_df_duT_v[:, count: count + block_size],
                tInd,
                solution_type,
                d_count,
                Jmatrix,
                self.field_derivs[tInd + 1][indices[0]][1][indices[1], :]
            ),
            shape=(block_size, Jmatrix.shape[1]),
            dtype=np.float32
        ))
        count += block_size
        d_count += block_size

    return field_blocks, j_row_blocks


def block_deriv(simulation, src, tInd, f, block_size):
    src_field_derivs = []
    j_initial = []
    for rx in src.receiver_list:

        v = sp.eye(rx.nD, dtype=float)
        PT_v = rx.evalDeriv(
            src, simulation.mesh, simulation.time_mesh, f, v, adjoint=True
        )
        df_duTFun = getattr(f, "_{}Deriv".format(rx.projField), None)

        cur = df_duTFun(
            simulation.nT,
            src,
            None,
            PT_v[tInd * block_size:(tInd + 1) * block_size, :],
            adjoint=True,
        )
        src_field_derivs.append(cur[0])
        j_initial.append(cur[1].T)

    # n_blocks = int(np.ceil(np.prod(src_field_derivs.shape) * 8. * 1e-6 / 128.))
    # ind_col = np.array_split(np.arange(src_field_derivs.shape[1]), col_blocks)
    # return [src_field_derivs[:, ind] for ind in ind_col]
    return sp.hstack(src_field_derivs), sp.vstack(j_initial)


def parallel_field_deriv(ATinv_df_duT_v, Asubdiag, field_derivs):
    return field_derivs - Asubdiag.T * ATinv_df_duT_v


def parallel_block_compute(simulation, f, src, ATinv_df_duT_v, tInd, solution_type, d_count, Jmatrix, j_initial):
    dAsubdiagT_dm_v = simulation.getAsubdiagDeriv(
        tInd, f[src, solution_type, tInd], ATinv_df_duT_v, adjoint=True
    )

    dRHST_dm_v = simulation.getRHSDeriv(
        tInd + 1, src, ATinv_df_duT_v, adjoint=True
    )
    un_src = f[src, solution_type, tInd + 1]
    dAT_dm_v = simulation.getAdiagDeriv(
        tInd, un_src, ATinv_df_duT_v, adjoint=True
    )

    if simulation.store_sensitivities == "disk":
        return (-dAT_dm_v - dAsubdiagT_dm_v + dRHST_dm_v).T + j_initial

    Jmatrix[d_count:d_count + dAT_dm_v.shape[1], :] += (-dAT_dm_v - dAsubdiagT_dm_v + dRHST_dm_v).T + j_initial
