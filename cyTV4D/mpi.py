#!/usr/bin/env python
import cyTV4D as tv
from mpi4py import MPI
import numpy as np

try:
    import py4DSTEM
except Exception:
    print("Failed to import py4DSTEM. Cannot read 4D-STEM Data...")

try:
    from ncempy.io.dm import fileDM
except Exception:
    print("Failed to import ncempy. Cannot read EELS data...")


def run_MPI():
    import argparse
    import os

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    Nworkers = comm.Get_size()

    HEAD_WORKER = rank == 0

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
        "-f", "-fista", type=bool, nargs=1, help="Use acceleration? 0 or 1."
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
    parser.add_argument("-v", "--verbose", type=bool, default=False)

    args = vars(parser.parse_args())

    VERBOSE = args["verbose"]

    if HEAD_WORKER:
        print(f"Running MPI denoising with arguments: {args}")

    # each worker must load a memory map into the data:
    if args["dimensions"][0] == 3:
        # load EELS SI data using ncempy
        dmf = fileDM(args["input"][0])
        data = dmf.getMemmap(2)
        # squeeze while retaining memmap (native numpy squeeze) tries to load the array in RAM
        while data.shape[0] == 1:
            data = data.reshape(data.shape[1:])
        size = data.shape[:2]
    elif args["dimensions"][0] == 4:
        # load 4D data using py4DSTEM

        # load DM data:
        if "dm" in args["input"][0][-3:]:
            data = py4DSTEM.file.io.read(args["input"][0], load="dmmmap")
            size = data.shape[:2]
        # load EMD data:
        elif "h5" in args["input"][0][-2:]:
            fb = py4DSTEM.file.io.FileBrowser(args["input"])
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
        print(f"Loaded memory map. Data size is: {data.shape}")

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
        print(f"Dividing work over a {wx} by {wy} grid...")

    # Figure out the slices that this worker is responsible for:
    tile_x, tile_y = np.unravel_index(rank, (wx, wy))

    if VERBOSE:
        print(f"Worker {rank} is doing tile {tile_x},{tile_y}.")

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

    if VERBOSE:
        print(f"Worker {rank} at tile {tile_x},{tile_y} is reading slice {read_slice_x},{read_slice_y}...")

    # set some flags for determining if this worker should shift data at each step:
    SHIFT_X_POS = tile_x < (wx - 1)
    SHIFT_X_NEG = tile_x > 0

    SHIFT_Y_POS = tile_y < (wy - 1)
    SHIFT_Y_NEG = tile_y > 0

    # load in the data and make it contiguous
    if args["dimensions"][0] == 3:
        raw = np.ascontiguousarray(data[read_slice_x, read_slice_x, :]).astype(
            np.float32
        )
    elif args["dimensions"][0] == 4:
        raw = np.ascontiguousarray(data[read_slice_x, read_slice_y, :, :]).astype(
            np.float32
        )  # TODO: make dtype a flag

    recon = np.zeros_like(raw)
