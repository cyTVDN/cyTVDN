cimport cython
from cython.parallel import prange
import numpy as np

# define a fused floating point type so that we can use single or double precision floats:
ctypedef fused _float:
    float
    double

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def sum_square_error(_float[:,:,:,::] a, _float[:,:,:,::] b):
    cdef int i,j,k,l
    
    if _float is float:
        dtype = np.float32
    if _float is double:
        dtype = np.double
    mserr_np = np.zeros((a.shape[0],),dtype=dtype)
    cdef _float[:] mserr = mserr_np
    
    cdef _float tmp
    
    for i in prange(a.shape[0],nogil=True):
        for j in range(a.shape[1]):
            for k in range(a.shape[2]):
                for l in range(a.shape[3]):
                    tmp = a[i,j,k,l] - b[i,j,k,l]
                    mserr[i] = mserr[i] + (tmp*tmp)
                    
    return mserr_np.sum()


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def datacube_update(_float[:,:,:,::] orig, _float[:,:,:,::] recon, _float[:,:,:,::] b1, 
               _float[:,:,:,::] b2, _float[:,:,:,::] b3, _float[:,:,:,::] b4, _float[:] lambda_mu, bint PBC=False):
    '''
    perform the TV update step:
    recon = orig - ( b1 - roll(b1,-1,1) )*lambda_mu - ...
    '''
    # shape of the 4-D array
    cdef int shape[4]
    shape[:] = [orig.shape[0],orig.shape[1],orig.shape[2],orig.shape[3]]
    
    # index variables for the main 4D loop
    cdef int i,j,k,l

    # index limits for the mirror boundary condition
    # (These can't be defined inside the conditional)
    cdef int MBCend[4]
    MBCend = shape
    for q in range(4):
        MBCend[q] -= 1
    
    if PBC == True:
        # in the case of periodic boundary conditions we can
        # perform the entire loop at once, correctly wrapping indices.
        # this is gonna be somewhat slower than ideal but probably still
        # faster than if we left wraparound on...
        for i in prange(shape[0],nogil=True):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    for l in range(shape[3]):
                        recon[i,j,k,l] = orig[i,j,k,l] - \
                            (
                                (lambda_mu[0] * ( b1[i,j,k,l] - b1[(i+1)%shape[0],j,k,l])) +
                                (lambda_mu[1] * ( b2[i,j,k,l] - b2[i,(j+1)%shape[1],k,l])) +
                                (lambda_mu[2] * ( b3[i,j,k,l] - b3[i,j,(k+1)%shape[2],l])) +
                                (lambda_mu[3] * ( b4[i,j,k,l] - b4[i,j,k,(l+1)%shape[3]]))
                            )
    else:
        # handling the mirror boundary condition 
        # simply requires different math for computing the indices.
        # The new indexing is max(i+1,shape-1), so precompute shape[:]-1
        for i in prange(shape[0],nogil=True):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    for l in range(shape[3]):
                        recon[i,j,k,l] = orig[i,j,k,l] - \
                            (
                                (lambda_mu[0] * ( b1[i,j,k,l] - b1[max(i+1,MBCend[0]),j,k,l])) +
                                (lambda_mu[1] * ( b2[i,j,k,l] - b2[i,max(i+1,MBCend[1]),k,l])) +
                                (lambda_mu[2] * ( b3[i,j,k,l] - b3[i,j,max(i+1,MBCend[2]),l])) +
                                (lambda_mu[3] * ( b4[i,j,k,l] - b4[i,j,k,max(i+1,MBCend[3])]))
                            )