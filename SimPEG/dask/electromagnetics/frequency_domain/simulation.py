from ....electromagnetics.frequency_domain.simulation import BaseFDEMSimulation as Sim
from ....utils import Zero, mkvc
from  time import time
import numpy as np
import scipy.sparse as sp
import multiprocessing
from dask import array, compute, delayed
from dask.distributed import Future
import zarr
from time import time

Sim.sensitivity_path = './sensitivity/'
Sim.gtgdiag = None
Sim.store_sensitivities = True


def fields(self, m=None, return_Ainv=False):
    if m is not None:
        self.model = m

    f = self.fieldsPair(self)

    Ainv = []
    for freq in self.survey.frequencies:
        A = self.getA(freq)
        rhs = self.getRHS(freq)

        if return_Ainv:
            Ainv += [self.solver(sp.csr_matrix(A.T), **self.solver_opts)]

        Ainv_solve = self.solver(sp.csr_matrix(A), **self.solver_opts)
        u = Ainv_solve * rhs
        Srcs = self.survey.get_sources_by_frequency(freq)
        f[Srcs, self._solutionType] = u

        Ainv_solve.clean()

    if return_Ainv:
        return f, Ainv
    else:
        return f, None


Sim.fields = fields


def dask_getJtJdiag(self, m, W=None):
    """
        Return the diagonal of JtJ
    """
    self.model = m
    if self.gtgdiag is None:
        if isinstance(self.Jmatrix, Future):
            self.Jmatrix  # Wait to finish

        if W is None:
            W = np.ones(self.nD)
        else:
            W = W.diagonal()

        diag = array.einsum('i,ij,ij->j', W, self.Jmatrix, self.Jmatrix)

        if isinstance(diag, array.Array):
            diag = np.asarray(diag.compute())

        self.gtgdiag = diag
    return self.gtgdiag


Sim.getJtJdiag = dask_getJtJdiag


def dask_Jvec(self, m, v):
    """
        Compute sensitivity matrix (J) and vector (v) product.
    """
    self.model = m

    if isinstance(self.Jmatrix, np.ndarray):
        return self.Jmatrix @ v.astype(np.float32)

    if isinstance(self.Jmatrix, Future):
        self.Jmatrix  # Wait to finish

    return array.dot(self.Jmatrix, v)


Sim.Jvec = dask_Jvec


def dask_Jtvec(self, m, v):
    """
        Compute adjoint sensitivity matrix (J^T) and vector (v) product.
    """
    self.model = m

    if isinstance(self.Jmatrix, np.ndarray):
        return self.Jmatrix.T @ v.astype(np.float32)

    if isinstance(self.Jmatrix, Future):
        self.Jmatrix  # Wait to finish

    return array.dot(v, self.Jmatrix)


Sim.Jtvec = dask_Jtvec


def compute_J(self, f=None, Ainv=None):

    if f is None:
        f, Ainv = self.fields(self.model, return_Ainv=True)

    m_size = self.model.size
    row_chunks = int(np.ceil(
        float(self.survey.nD) / np.ceil(float(m_size) * self.survey.nD * 8. * 1e-6 / self.max_chunk_size)
    ))
    sub_threads = int(multiprocessing.cpu_count() / 2)
    if self.store_sensitivities == "disk":
        Jmatrix = zarr.open(
            self.sensitivity_path + f"J.zarr",
            mode='w',
            shape=(self.survey.nD, m_size),
            chunks=(row_chunks, m_size)
        )
    else:
        Jmatrix = np.zeros((self.survey.nD, m_size), dtype=np.float32)

    count = 0
    block_count = 0

    for A_i, freq in zip(Ainv, self.survey.frequencies):

        for ss, src in enumerate(self.survey.get_sources_by_frequency(freq)):
            df_duT, df_dmT = [], []
            blocks_dfduT = []
            blocks_dfdmT = []
            u_src = f[src, self._solutionType]
            ct = time()
            print("In loop over receivers")
            for rx in src.receiver_list:
                v = np.eye(rx.nD, dtype=float)
                n_blocs = np.ceil(2 * rx.nD / row_chunks * sub_threads)
                print("In loop over blocks")
                for block in np.array_split(v, n_blocs, axis=1):

                    block_count += block.shape[1] * 2
                    blocks_dfduT.append(
                        array.from_delayed(
                            delayed(dfduT, pure=True)(src, rx, self.mesh, f, block),
                            dtype=np.float32,
                            shape=(u_src.shape[0], block.shape[1]*2)
                        )
                    )
                    blocks_dfdmT.append(
                            delayed(dfdmT, pure=True)(src, rx, self.mesh, f, block),
                    )

                    if block_count >= (row_chunks * sub_threads):
                        print(f"{ss}: Block {count}: {time()-ct}")
                        count = parallel_block_compute(self, Jmatrix, src, A_i, freq, blocks_dfduT, blocks_dfdmT, u_src,sub_threads, count)
                        blocks_dfduT = []
                        blocks_dfdmT = []
                        block_count = 0
                        # blocks_dfduT, count = store_block(blocks_dfduT, count)

            if blocks_dfduT:
                count = parallel_block_compute(self, Jmatrix, src, A_i, freq, blocks_dfduT, blocks_dfdmT, u_src, sub_threads, count)
                block_count = 0

    # if len(blocks) != 0:
    #     if self.store_sensitivities == "disk":
    #         Jmatrix.set_orthogonal_selection(
    #             (np.arange(count, self.survey.nD), slice(None)),
    #             blocks.astype(np.float32)
    #         )
    #     else:
    #         Jmatrix[count: self.survey.nD, :] = (
    #             blocks.astype(np.float32)
    #         )

    for A in Ainv:
        A.clean()

    if self.store_sensitivities == "disk":
        del Jmatrix
        return array.from_zarr(self.sensitivity_path + f"J.zarr")
    else:
        return Jmatrix


Sim.compute_J = compute_J


def eval_store_block(simulation, Jmatrix, ATinvdf_duT, freq, df_dmT, u_src, src, row_count):
    """
    Evaluate the sensitivities for the block or data and store to zarr
    """
    print("Line 132")
    dA_dmT = simulation.getADeriv(freq, u_src, ATinvdf_duT, adjoint=True)
    print("Line 136")
    dRHS_dmT = simulation.getRHSDeriv(freq, src, ATinvdf_duT, adjoint=True)
    print("Line 138")
    du_dmT = -dA_dmT
    if not isinstance(dRHS_dmT, Zero):
        du_dmT += dRHS_dmT
    if not isinstance(df_dmT, Zero):
        du_dmT += df_dmT

    block = np.array(du_dmT, dtype=complex).real.T
    print("Line 146")
    if simulation.store_sensitivities == "disk":
        Jmatrix.set_orthogonal_selection(
            (np.arange(row_count, row_count + block.shape[0]), slice(None)),
            block.astype(np.float32)
        )
    else:
        Jmatrix[row_count: row_count + block.shape[0], :] = (
            block.astype(np.float32)
        )

    # row_count += block.shape[0]
    # return row_count


def dfduT(source, receiver, mesh, fields, block):
    dfduT, _ = receiver.evalDeriv(
        source, mesh, fields, v=block, adjoint=True
    )

    return dfduT


def dfdmT(source, receiver, mesh, fields, block):
    _, dfdmT = receiver.evalDeriv(
        source, mesh, fields, v=block, adjoint=True
    )

    return dfdmT


def parallel_block_compute(simulation, Jmatrix, src, A_i, freq, blocks_dfduT, blocks_dfdmT, u_src, sub_threads, count):
    field_derivs = array.hstack(blocks_dfduT).compute()

    # Direct-solver call
    ATinvdf_duT = A_i * field_derivs

    # Even split
    split = np.linspace(0, (ATinvdf_duT.shape[1]) / 2, sub_threads)[1:-1].astype(int) * 2
    sub_blocks_dfduT = np.array_split(ATinvdf_duT, split, axis=1)

    if isinstance(compute(blocks_dfdmT[0])[0], Zero):
        sub_blocks_dfdmt = [Zero()] * len(sub_blocks_dfduT)
    else:
        compute_blocks_dfdmT = array.hstack([
            array.from_delayed(
                dfdmT_block,
                dtype=np.float32,
                shape=(u_src.shape[0], dfdmT_block.shape[1] * 2))
            for dfdmT_block in blocks_dfdmT
        ]).compute()
        sub_blocks_dfdmt = np.array_split(compute_blocks_dfdmT, split, axis=1)

    sub_process = []
    for sub_block_dfduT, sub_block_dfdmT in zip(sub_blocks_dfduT, sub_blocks_dfdmt):
        print("Computing derivs")
        sub_process.append(
            delayed(eval_store_block, pure=True)(
                simulation,
                Jmatrix,
                sub_block_dfduT,
                freq,
                sub_block_dfdmT,
                u_src,
                src,
                count
            )
        )
        count += int(sub_block_dfduT.shape[1] / 2)

    compute(sub_process)

    return count