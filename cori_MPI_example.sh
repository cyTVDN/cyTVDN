#!/bin/bash
#SBATCH --constraint=knl
#SBATCH --nodes=12
#SBATCH --time=00:30:00
#SBATCH --qos=regular
#SBATCH --ntasks=12
#SBATCH --cpus-per-task=272

# Enter your email to be notified about job status!
# (Uncomment these lines to enable. It should have one '#'
# #SBATCH --mail-type=begin,end,fail
# #SBATCH --mail-user=your.email@here.com

# INPUT AND OUTPUT FILES
INFILE=$SCRATCH/denoising/HSITestDataForColin-298Frames-77x226px-py4DSTEM.h5
OUTFILE=$SCRATCH/denoising/HSI_testrun.emd

LOGFILE=$SCRATCH/denoising/mpi-$SLURM_JOB_ID.out

# SET THESE VARIABLES TO CHOOSE OPTIONS
NDIMS=4 # Number of dimensions (3 or 4)
FISTA=0 # Use FISTA? 0 for no, 1 for yes
N_ITERATIONS=40 # Number of iterations
# Lambda and mu can each can be specified as a single value or one for each dimension.
# When setting multiple values, they must be placed in quotes and separated by spaces.
LAMBDA="0.01 0.01 0.01 0.01" # Lambda 
MU="1 1 1 1" # Mu 

#############################################################

module load python
module load h5py-parallel
source activate tv

export OMP_NUM_THREADS=204
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

export OMP_DISPLAY_AFFINITY=TRUE
export OMP_WAIT_POLICY=passive

export KMP_AFFINITY=TRUE

srun --cpu_bind=cores cyTVMPI \
	-i $INFILE \
	-o $OUTFILE \
	-d $NDIMS -f $FISTA -n $N_ITERATIONS \
	-L $LAMBDA -m $MU -v 1 \
	> $LOGFILE

# In my experimentation, these MPI/OpenMP settings worked best
# but they may not be quite right?
