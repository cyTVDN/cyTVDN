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

    args = vars(parser.parse_args())

    if HEAD_WORKER:
        print(f"Running MPI denoising with arguments: {args}")

    # each worker must load a memory map into the data:
    if args["dimensions"][0] == 3:
        # load EELS SI data using ncempy
        dmf = fileDM(args["input"][0])
        data = dmf.getMemmap(2)
        size = data.shape[2:]
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

    # first get the size in each direction
    nx = np.ceil(size[0] / wx)
    ny = np.ceil(size[1] / wy)
