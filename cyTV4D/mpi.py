#!/usr/bin/env python
import cyTV4D as tv
import mpi4py as MPI
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

    args = parser.parse_args()
    print(args)
