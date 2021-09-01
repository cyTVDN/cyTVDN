# this is an example script for testing the MPI script on a local machine

mpirun -n 4 OMP_NUM_THREADS=1 \
    cyTVMPI -i Sample_LFP_datacube.h5 \
            -o tvtest.emd \
            -d 4 \
            -f 0 \
            -n 3 \
            -L 0.1 0.1 0.1 0.1 \
            -m 0.3 \
            -v 1 > mpitest.out