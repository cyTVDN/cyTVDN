# cyTV4D
Fast multi-threaded multi-dimensional total variational denoising

## Algorithm

Describe algo here...

## Installation
Naturally, you will need a C compiler in order for Cython to compile the module. On Macs this is somewhat complicated because the Apple-provided versions of `gcc` and `clang` do not support the `-fopenmp` option, which is used to enable multithreaded execution. To build with multithreading on a Mac you will need to use Homebrew to install `llvm`, and `libomp`. 

To install, clone the git repo, navigate to the directory and run:
```bash
python setup.py build_ext
python setup.py install
```

### High-performance computing
When building on HPC, note that `gcc` and Cray compilers use `-fopenmp` while Intel compilers require `-qopenmp` to enable threading.

On NERSC, you must export the correct compiler as part of the `build_ext` command:

For Cray compilers:
```bash
module swap PrgEnv-Intel PrgEnv-Cray
CC=cc LDSHARED="cc -shared" python setup.py build_ext
```
For Intel compilers, first edit `setup.py` to replace `-fopenmp` with `-qopenmp`, then run:
```bash
module load PrgEnv-Intel # this is loaded by default
CC=icc LDSHARED="icc -shared" python setup.py build_ext
```

#### MPI
For datasets that are too large to fit in RAM on a single machine, an MPI implementation is provided. The implementation is roughly as follows:
* Divide the work across a 2D grid along the real-space axes. (*This is not necessarily the most efficient division of labor, but it makes I/O and communication easier.*)
* Each worker loads a chunk of the data, with one unit of overlap in scan position in each direction. At the edges of the scan there is no "overlap." 
* Each worker performs an acuumulator update step using only its local hunk. *Some of the computed values at the boundaries are invalid because they do not respect the global boundary conditions of the problem.*
* The overlap regions are synchronized by inter-worker communication over MPI. The accumulators (which are backward differences) shift data "right". This fixes the boundary condition errors from the previous step.
* Each worker performs a reconstruction update step. *Again, the global boundary condition is not respected, so synchronization is needed.*
* The boundary regions of the reconstruction are propagated via MPI. The reconstruction (which uses a forward difference) shifts data "left".

> It's probably best to have one MPI worker per node, and to set OpenMP to use all (or maybe all-1 to leave some resources for async communication?) of the node's threads. I assert this on the basis that all the computations are bound by memory bandwidth, so there's probably no benefit to have more workers per node. Having more workers increases the amount of MPI communication needed, and having fewer but larger workers reduces the total communication load.

>This implementation currently only allows for the J-Z boundary condition. Other BCs change what data has to be synchronized, and it gets complicated quickly. J-Z seems to work best for all data we've tested, and has half the inter-worker communication needs versus mirror or periodic. 

## Usage
Example usage for EELS:
```python
import cyTV4D as tv
import numpy as np
# eels_data should be a 3D numpy array, usually with
# scan axes on axes 0 and 1, and eels spetra along axis 2
mu = np.array([1, 1, 0.5], dtype=eels_data.dtype)
lam = mu / 16.
recon, convergence = tv.denoise3D(eels_data, lam, mu, iterations=1_000, FISTA=False)
```

Example usage for 4D-STEM:
```python
import cyTV4D as tv
import numpy as np
import py4DSTEM

datacube = py4DSTEM.file.io.read('/Path/To/Data.dm4')
mu = np.array([1, 1, 0.5, 0.5], dtype=datacube.data.dtype)
lam = mu / 32.
recon, convergence = tv.denoise4D(datacube.data, lam, mu, iterations=1_000, FISTA=False)
```

### Notes

**Memory Layout** Because of the way that Cython converts numpy arrays to C, it is required that all the numpy arrays used in the computation are C-contiguous. This is normally not a problem, as numpy arrays are created in C-contiguous mode by default. However, if you slice a numpy array, this produces a view which will usually *not* be C-contiguous. Thus, if you want to split a large TV denoising problem into chunks it will be necessary to create copies of the chunks with `datacube[chunk_x:chunk_y,:,:].copy()`, and to copy the denoised data back into the larger array, or to use `np.ascontiguousarray()`.