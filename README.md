# cyTV4D
Fast multi-threaded four-dimensional anisotropic total variational denoising

## Algorithm

Describe algo here...

## Installation
Naturally, you will need a C compiler in order for Cython to compile the module. On Macs this is somewhat complicated because the Apple-provided versions of `gcc` and `clang` do not support the `-fopenmp` option, which is used to enable multithreaded execution. To build with multithreading on a Mac you will need to use Homebrew to install `gcc`, `llvm`, and `libomp`. 


## Usage

### Notes

**Memory Layout** It is required that all the numpy arrays used in the computation are C-contiguous. This is normally not a problem, as numpy arrays are created in C-contiguous mode by default. However, if you slice a numpy array, this produces a view which will usually *not* be C-contiguous. Thus, if you want to split a large TV denoising problem into chunks it will be necessary to create copies of the chunks with `datacube[chunk_x:chunk_y,:,:].copy()`, and to copy the denoised data back into the larger array.