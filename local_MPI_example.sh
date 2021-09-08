# this is an example script for testing the MPI script on a local machine

# if you want to test locally you may run into problems with the parallel HDF5
# installation. this seems to work on my iMac:
# pip uninstall h5py
# brew install hdf5-mpi
# CC="mpicc" HDF5_MPI="ON" HDF5_DIR="***PATH/TO/HDF5/INSTALL" pip install --no-binary=h5py h5py

OMP_NUM_THREADS=1 \
mpirun -n 4 \
    cyTVMPI -i Sample_LFP_v0,12.h5 \
            -o tvtest.emd \
            -d 4 \
            -f 0 \
            -n 3 \
            -L 0.1 0.1 0.1 0.1 \
            -m 0.3 \
            -v 1 > mpitest.log