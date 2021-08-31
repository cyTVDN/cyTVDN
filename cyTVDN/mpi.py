#!/usr/bin/env python
import cyTVDN as tv
from mpi4py import MPI
import numpy as np
from tqdm import tqdm
from time import time
import h5py
from hurry import filesize
import psutil

from loguru import logging
import sys

# old logging setup using the stdlib Logger
# logger = logging.getLogger("cyTVDN")
# logger.setLevel(logging.DEBUG)
# logger.handlers = []

# handler = logging.StreamHandler(sys.stdout)
# handler.setLevel(logging.DEBUG)
# formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
# handler.setFormatter(formatter)
# logger.addHandler(handler)

# new loguru log setup:
logger.add(sys.stdout)

try:
    import py4DSTEM
except Exception:
    logger.info("Failed to import py4DSTEM. Cannot read 4D-STEM Data...")

try:
    from ncempy.io.dm import fileDM
except Exception:
    logger.info("Failed to import ncempy. Cannot read EELS data...")


def run_MPI():
    import argparse
    import os

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    Nworkers = comm.Get_size()

    HEAD_WORKER = rank == 0

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(description="Launch TV Denoising using MPI.")
    parser.add_argument(
        "-i", "--input", type=os.path.abspath, nargs=1, help="input file"
    )
    parser.add_argument(
        "-o", "--output", type=os.path.abspath, nargs=1, help="output file"
    )
    parser.add_argument(
        "-d", "--dimensions", type=int, nargs=1, help="Number of Dimensions (3 or 4)"
    )
    parser.add_argument(
        "-f",
        "--fista",
        type=str2bool,
        nargs=1,
        help="Use acceleration? 0 or 1.",
        default=False,
    )
    parser.add_argument(
        "-n",
        "--niterations",
        type=int,
        nargs="+",
        help="Number of iterations (Specify 2 values for hybrid.)",
    )
    parser.add_argument("-L", "--lambda", type=float, nargs="+")
    parser.add_argument("-m", "--mu", type=float, nargs="+")
    parser.add_argument("-v", "--verbose", type=str2bool, default=False)

    args = vars(parser.parse_args())

    # VERBOSE = args["verbose"]
    VERBOSE = True

    ndim = args["dimensions"][0]
    FISTA = args["fista"][0]
    niter = args["niterations"]
    BC_mode = 2
    lam = np.array(args["lambda"])
    mu = np.array(args["mu"])
    outfile = args["output"][0]

    if HEAD_WORKER:
        logger.info(f"Running MPI denoising with arguments: {args}")
        logger.info(f"Python sees OMP_NUM_THREADS as {os.environ['OMP_NUM_THREADS']}")

    # each worker must load a memory map into the data:
    t_read_start = time()
    if ndim == 3:
        # load EELS SI data using ncempy
        dmf = fileDM(args["input"][0])
        data = dmf.getMemmap(2)
        # squeeze while retaining memmap (native numpy squeeze) tries to load the array in RAM
        while data.shape[0] == 1:
            data = data.reshape(data.shape[1:])
        size = data.shape[:2]
    elif ndim == 4:
        # load 4D data using py4DSTEM

        # load DM data:
        if "dm" in args["input"][0][-3:]:
            data = py4DSTEM.file.io.read(args["input"][0], load="dmmmap")
            size = data.shape[:2]
        # load EMD data:
        elif any(ftype in args["input"][0].split(".")[-1] for ftype in ("h5", "emd")):
            fb = py4DSTEM.file.io.FileBrowser(args["input"][0])

            # hack to swap out the HDF driver:
            # fb.file.close()
            fb.file = h5py.File(
                args["input"][0], "r", driver="mpio", comm=MPI.COMM_WORLD
            )

            dc = fb.get_dataobject(0, memory_map=True)
            data = dc.data
            size = data.shape[:2]
        else:
            if HEAD_WORKER:
                raise (NotImplementedError("Incompatible File type..."))
    else:
        if HEAD_WORKER:
            raise (AssertionError("Bad number of dimensions..."))

    if HEAD_WORKER:
        logger.info(f"Loaded memory map. Data size is: {data.shape}")
        logger.info(f"Loading memory map took {time()-t_read_start} seconds.")

    # calculate the best division of labor:
    edges = np.zeros((Nworkers,))
    for i in range(1, Nworkers + 1):
        if Nworkers % i == 0:
            # this is a factor of the number of workers, so a valid size
            wx = i
            wy = Nworkers / i

            # x and y sizes of the chunks, not including overlap
            sx = np.ceil(size[0] / wx)
            sy = np.ceil(size[1] / wy)

            edges[i - 1] = (Nworkers - 1) * (2 * sx + 2 * sy)

        else:
            # this is not a valid tiling shape
            edges[i - 1] = np.nan

    # Get the number of tiles in X and Y for the grid of workers:
    wx = int(np.nanargmin(edges) + 1)
    wy = int(Nworkers / wx)

    if HEAD_WORKER:
        logger.info(f"Dividing work over a {wx} by {wy} grid...")

    # Figure out the slices that this worker is responsible for:
    tile_x, tile_y = np.unravel_index(rank, (wx, wy))

    logger.debug(f"Worker {rank} is doing tile {tile_x},{tile_y}.")

    # first get the size in each direction
    nx = int(np.ceil(size[0] / wx))
    ny = int(np.ceil(size[1] / wy))

    # get the slices for which this worker's data is valid (i.e. the slice before adding overlaps)
    valid_slice_x = slice(
        tile_x * nx, (tile_x + 1) * nx if (tile_x + 1) * nx <= size[0] else size[0]
    )
    valid_slice_y = slice(
        tile_y * ny, (tile_y + 1) * ny if (tile_y + 1) * ny <= size[1] else size[1]
    )

    # now get the slices to actually read
    read_slice_x = slice(
        valid_slice_x.start - 1 if valid_slice_x.start > 0 else 0,
        valid_slice_x.stop + 1 if valid_slice_x.stop + 1 <= size[0] else size[0],
    )
    read_slice_y = slice(
        valid_slice_y.start - 1 if valid_slice_y.start > 0 else 0,
        valid_slice_y.stop + 1 if valid_slice_y.stop + 1 <= size[1] else size[1],
    )

    logger.debug(
        f"Worker {rank} at tile {tile_x},{tile_y} is reading slice {read_slice_x},{read_slice_y}..."
    )

    # set some flags for determining if this worker should shift data at each step:
    SHIFT_X_POS = tile_x < (wx - 1)
    SHIFT_X_NEG = tile_x > 0

    SHIFT_Y_POS = tile_y < (wy - 1)
    SHIFT_Y_NEG = tile_y > 0

    # get the slice *relative to the local chunk* that represents valid data
    # (this is used later for deciding what data from the local chunk is saved)
    local_valid_slice_x = slice(1 if SHIFT_X_NEG else 0, -1 if SHIFT_X_POS else None)
    local_valid_slice_y = slice(1 if SHIFT_Y_NEG else 0, -1 if SHIFT_Y_POS else None)

    # figure out the sources and destinations for each shift
    RANK_X_POS = (
        np.ravel_multi_index((tile_x + 1, tile_y), (wx, wy)) if SHIFT_X_POS else None
    )
    RANK_X_NEG = (
        np.ravel_multi_index((tile_x - 1, tile_y), (wx, wy)) if SHIFT_X_NEG else None
    )
    RANK_Y_POS = (
        np.ravel_multi_index((tile_x, tile_y + 1), (wx, wy)) if SHIFT_Y_POS else None
    )
    RANK_Y_NEG = (
        np.ravel_multi_index((tile_x, tile_y - 1), (wx, wy)) if SHIFT_Y_NEG else None
    )

    logger.debug(
        f"Rank {rank} has neighbors: +x {RANK_X_POS} \t -x: {RANK_X_NEG} \t +y: {RANK_Y_POS} \t -y: {RANK_Y_NEG}"
    )

    # load in the data and make it contiguous
    t_load_start = time()
    if ndim == 3:
        raw = np.ascontiguousarray(data[read_slice_x, read_slice_x, :]).astype(
            np.float32
        )
    elif args["dimensions"][0] == 4:
        # TODO: fix this for non-py4DSTEM files!!!
        raw = np.zeros(
            (
                read_slice_x.stop - read_slice_x.start,
                read_slice_y.stop - read_slice_y.start,
                data.shape[2],
                data.shape[3],
            ),
            dtype=np.float32,
        )
        logger.debug(f"Raw is shape {raw.shape}")
        data.read_direct(raw, source_sel=np.s_[read_slice_x, read_slice_y, :, :])
        # TODO: make dtype a flag

    if HEAD_WORKER:
        logger.info(f"Head worker finished reading raw data...")
        logger.info(
            f"Reading raw data took {time()-t_load_start} seconds. Data size is {filesize.size(raw.nbytes,system=filesize.alternative)}"
        )

    recon = raw.copy()

    lambdaInv = (1.0 / lam).astype(recon.dtype)
    lam_mu = (lam / mu).astype(recon.dtype)

    if ndim == 3:
        # 3D is boring, I'll implement it later...
        if HEAD_WORKER:
            logger.error("Oops... Haven't implemented 3D yet. Sorry")

    elif ndim == 4:
        # allocate accumulators
        t_accum_start = time()
        acc0 = np.zeros_like(recon)
        acc1 = np.zeros_like(recon)
        acc2 = np.zeros_like(recon)
        acc3 = np.zeros_like(recon)

        # allocate MPI sync buffers
        x_pos_buffer = np.zeros(
            (raw.shape[1], raw.shape[2], raw.shape[3]), dtype=np.float32
        )
        x_neg_buffer = np.zeros_like(x_pos_buffer)
        y_pos_buffer = np.zeros(
            (raw.shape[0], raw.shape[2], raw.shape[3]), dtype=np.float32
        )
        y_neg_buffer = np.zeros_like(y_pos_buffer)

        if HEAD_WORKER:
            logger.info(
                f"Allocating the main accumulators and buffers took {time() - t_accum_start} seconds"
            )

        if FISTA:
            d1 = np.zeros_like(recon)
            d2 = np.zeros_like(recon)
            d3 = np.zeros_like(recon)
            d4 = np.zeros_like(recon)

            # allocate MPI sync buffers
            x_pos_buffer_FISTA = np.zeros(
                (raw.shape[1], raw.shape[2], raw.shape[3]), dtype=np.float32
            )
            x_neg_buffer_FISTA = np.zeros_like(x_pos_buffer)
            y_pos_buffer_FISTA = np.zeros(
                (raw.shape[0], raw.shape[2], raw.shape[3]), dtype=np.float32
            )
            y_neg_buffer_FISTA = np.zeros_like(y_pos_buffer)

            tk = 1.0

        if HEAD_WORKER:
            logger.info(
                f"With all accumulators allocated, free RAM is {filesize.size(psutil.virtual_memory().available,system=filesize.alternative)}."
            )
        else:
            logger.debug(
                f"With all accumulators allocated, free RAM on rank {rank} is {filesize.size(psutil.virtual_memory().available,system=filesize.alternative)}."
            )

        # create the iterators (so that only the head spits out tqdm stuff)
        iterator = tqdm(range(niter[0])) if HEAD_WORKER else range(niter[0])

        if FISTA:
            logger.error("Oops, haven't done FISTA yet...")

        else:
            for i in iterator:
                # perform an update step along dim 0
                t0 = time()
                tv.accumulator_update_4D(recon, acc0, 0, lambdaInv[0], BC_mode=BC_mode)
                logger.debug(
                    f"X update step : rank {rank} : iteration {i} : took {time()-t0} sec"
                )

                t0 = time()
                # start comms to send data right, receive data left:
                if SHIFT_X_POS:
                    x_pos_buffer[:] = np.squeeze(acc0[-1, :, :, :])
                    mpi_send_x_pos = comm.Isend(x_pos_buffer, dest=RANK_X_POS,)
                if SHIFT_X_NEG:  # shift x left <=> recieve data x left
                    x_neg_buffer[:] = 0
                    mpi_recv_x_neg = comm.Irecv(x_neg_buffer, source=RANK_X_NEG,)
                logger.debug(
                    f"X MPI sync step : rank {rank} : iteration {i} : took {time()-t0} sec"
                )

                # perform an update step along dim 1
                t0 = time()
                tv.accumulator_update_4D(recon, acc1, 1, lambdaInv[1], BC_mode=BC_mode)
                logger.debug(
                    f"Y update step : rank {rank} : iteration {i} : took {time()-t0} sec"
                )

                t0 = time()
                # start comms to send data right, receive data left:
                if SHIFT_Y_POS:
                    y_pos_buffer[:] = np.squeeze(acc1[:, -1, :, :])
                    mpi_send_y_pos = comm.Isend(y_pos_buffer, dest=RANK_Y_POS,)
                if SHIFT_Y_NEG:  # shift y left <=> recieve data y left
                    y_neg_buffer[:] = 0
                    mpi_recv_y_neg = comm.Irecv(y_neg_buffer, source=RANK_Y_NEG,)
                logger.debug(
                    f"X MPI sync step : rank {rank} : iteration {i} : took {time()-t0} sec"
                )

                # perform update steps on the non-communicating directions
                if VERBOSE and HEAD_WORKER:
                    logger.info("Starting Qx/Qy acc update")
                t0 = time()
                tv.accumulator_update_4D(recon, acc2, 2, lambdaInv[2], BC_mode=BC_mode)
                tv.accumulator_update_4D(recon, acc3, 3, lambdaInv[3], BC_mode=BC_mode)
                logger.debug(
                    f"Qx/Qy update step : rank {rank} : iteration {i} : took {time()-t0} sec"
                )

                comm.Barrier()
                # block until communication finishes. copy buffered data.
                if HEAD_WORKER:
                    logger.info(
                        f"Passed accumulator barrier on iteration {i} and entering sync block."
                    )
                else:
                    logger.debug(
                        f"Rank {rank} passed accumulator barrier and entering sync block."
                    )
                t_comm_wait = time()
                if SHIFT_X_NEG:
                    mpi_recv_x_neg.Wait()
                    acc0[0, :, :, :] = x_neg_buffer
                if SHIFT_Y_NEG:
                    mpi_recv_y_neg.Wait()
                    acc1[:, 0, :, :] = y_neg_buffer
                if SHIFT_X_POS:
                    mpi_send_x_pos.Wait()
                if SHIFT_Y_POS:
                    mpi_send_y_pos.Wait()

                if HEAD_WORKER:
                    logger.info(
                        f"Rank {rank} at iteration {i} spent {time()-t_comm_wait} seconds waiting for accumulator communication"
                    )
                else:
                    logger.debug(
                        f"Rank {rank} at iteration {i} spent {time()-t_comm_wait} seconds waiting for accumulator communication"
                    )

                # perform a datacube update step:
                if VERBOSE and HEAD_WORKER:
                    logger.info("Starting datacube update")
                t0 = time()
                tv.datacube_update_4D(
                    raw, recon, acc0, acc1, acc2, acc3, lam_mu, BC_mode=BC_mode
                )
                logger.debug(
                    f"Datacube update step : rank {rank} : iteration {i} : took {time()-t0} sec"
                )

                t_comm_wait = time()
                # start comms to send data left, receive data right
                if SHIFT_X_NEG:
                    x_neg_buffer[:] = np.squeeze(recon[0, :, :, :])
                    mpi_send_x_neg = comm.Isend(x_neg_buffer, dest=RANK_X_NEG,)
                if SHIFT_X_POS:
                    x_pos_buffer[:] = 0
                    mpi_recv_x_pos = comm.Irecv(x_pos_buffer, source=RANK_X_POS,)
                if SHIFT_Y_NEG:
                    y_neg_buffer[:] = np.squeeze(recon[:, 0, :, :])
                    mpi_send_y_neg = comm.Isend(y_neg_buffer, dest=RANK_Y_NEG,)
                if SHIFT_Y_POS:
                    y_pos_buffer[:] = 0
                    mpi_recv_y_pos = comm.Irecv(y_pos_buffer, source=RANK_Y_POS)

                # Block until communication finishes
                comm.Barrier()
                if VERBOSE and HEAD_WORKER:
                    logger.info("Passed second barrier and entering sync block.")
                t_comm_wait = time()
                if SHIFT_X_POS:
                    mpi_recv_x_pos.Wait()
                    recon[-1, :, :, :] = x_pos_buffer
                if SHIFT_Y_POS:
                    mpi_recv_y_pos.Wait()
                    recon[:, -1, :, :] = y_pos_buffer
                if SHIFT_X_NEG:
                    mpi_send_x_neg.Wait()
                if SHIFT_Y_NEG:
                    mpi_send_y_neg.Wait()
                if VERBOSE and HEAD_WORKER:
                    logger.info(
                        f"Rank {rank} at iteration {i} spent {time()-t_comm_wait} seconds waiting for reconstruction communication"
                    )

    # temporary kludge for writing output files
    t_save_start = time()
    logger.info(f"Rank {rank} is saving data...")
    fout = h5py.File(
        outfile.split(".")[-2] + ".emd", "w", driver="mpio", comm=MPI.COMM_WORLD
    )
    group_toplevel = fout.create_group("4DSTEM_experiment")
    group_toplevel.attrs.create("emd_group_type", 2)
    group_toplevel.attrs.create("version_major", 0)
    group_toplevel.attrs.create("version_minor", 7)

    # Write data groups
    group_toplevel.create_group("metadata")
    group_data = group_toplevel.create_group("data")
    group_datacubes = group_data.create_group("datacubes")
    group_data.create_group("counted_datacubes")
    group_data.create_group("diffractionslices")
    group_data.create_group("realslices")
    group_data.create_group("pointlists")
    group_data.create_group("pointlistarrays")

    grp_dc = group_datacubes.create_group("datacube_0")
    dset = grp_dc.create_dataset("data", data.shape)

    grp_dc.attrs.create("emd_group_type", 1)
    grp_dc.attrs.create("metadata", -1)

    data_datacube = grp_dc["data"]

    R_Nx, R_Ny, Q_Nx, Q_Ny = data_datacube.shape
    data_R_Nx = grp_dc.create_dataset("dim1", (R_Nx,))
    data_R_Ny = grp_dc.create_dataset("dim2", (R_Ny,))
    data_Q_Nx = grp_dc.create_dataset("dim3", (Q_Nx,))
    data_Q_Ny = grp_dc.create_dataset("dim4", (Q_Ny,))

    if rank == 0:
        # Populate uncalibrated dimensional axes
        data_R_Nx[...] = np.arange(0, R_Nx)
        data_R_Nx.attrs.create("name", np.string_("R_x"))
        data_R_Nx.attrs.create("units", np.string_("[pix]"))
        data_R_Ny[...] = np.arange(0, R_Ny)
        data_R_Ny.attrs.create("name", np.string_("R_y"))
        data_R_Ny.attrs.create("units", np.string_("[pix]"))
        data_Q_Nx[...] = np.arange(0, Q_Nx)
        data_Q_Nx.attrs.create("name", np.string_("Q_x"))
        data_Q_Nx.attrs.create("units", np.string_("[pix]"))
        data_Q_Ny[...] = np.arange(0, Q_Ny)
        data_Q_Ny.attrs.create("name", np.string_("Q_y"))
        data_Q_Ny.attrs.create("units", np.string_("[pix]"))

    dset.write_direct(
        recon,
        source_sel=np.s_[local_valid_slice_x, local_valid_slice_y, :, :],
        dest_sel=np.s_[valid_slice_x, valid_slice_y, :, :],
    )
    # dset[valid_slice_x, valid_slice_y, :, :] = recon[
    #     local_valid_slice_x, local_valid_slice_y, :, :
    # ]
    fout.close()

    logger.info(f"Rank {rank} is done! Writing data took {time()-t_save_start} seconds")
