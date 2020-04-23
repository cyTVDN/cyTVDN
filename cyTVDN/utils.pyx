cimport cython
from cython.parallel import prange
import numpy as np
from libc.math cimport fabs

# define a fused floating point type so that we can use single or double precision floats:
ctypedef fused _float:
    float
    double

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def sum_square_error_4D(_float[:,:,:,::] a, _float[:,:,:,::] b):
    cdef Py_ssize_t i,j,k,l, ij
    
    cdef _float mserr = 0.0
    
    cdef _float tmp
    
    for ij in prange(a.shape[0]*a.shape[1],nogil=True):
        i = ij / a.shape[1]
        j = ij % a.shape[1]
        for k in range(a.shape[2]):
            for l in range(a.shape[3]):
                tmp = a[i,j,k,l] - b[i,j,k,l]
                mserr += (tmp*tmp)
                    
    #return mserr_np.sum()
    return mserr

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def sum_square_error_3D(_float[:,:,::] a, _float[:,:,::] b):
    cdef Py_ssize_t i,j,k, ij
    
    cdef _float mserr = 0.0
    
    cdef _float tmp
    
    for ij in prange(a.shape[0]*a.shape[1],nogil=True):
        i = ij / a.shape[1]
        j = ij % a.shape[1]
        for k in range(a.shape[2]):
                tmp = a[i,j,k] - b[i,j,k]
                mserr += (tmp*tmp)
                    
    return mserr

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def datacube_update_4D(_float[:,:,:,::] orig, _float[:,:,:,::] recon, _float[:,:,:,::] b1, 
               _float[:,:,:,::] b2, _float[:,:,:,::] b3, _float[:,:,:,::] b4, _float[:] lambda_mu, int BC_mode=2):
    '''
    perform the TV update step:
    recon = orig - ( b1 - roll(b1,-1,1) )*lambda_mu - ...
    Boundary conditions for the gradient operator can be:
        0: periodic
        1: mirror
        2: Jia-Zhao Adv Comp Math 2010 33:231-241
            NOTE: We implememnt the JZ boundary conditions identically to periodic
            under the assumption that the accumulators are also computed with JZ
            boundary conditions, such that there are zeros in all the right places
    '''
    # shape of the 4-D array
    cdef int shape[4]
    shape[:] = [orig.shape[0],orig.shape[1],orig.shape[2],orig.shape[3]]
    
    # index variables for the main 4D loop
    cdef Py_ssize_t i,j,k,l, ij 

    # index limits for the mirror boundary condition
    # (In Cython these can't be defined inside the conditional)
    cdef int MBCend[4]
    MBCend = shape
    for q in range(4):
        MBCend[q] -= 1

    cdef _float delta = 0.0
    cdef _float recon_norm = 0.0
    cdef _float oldVal
    
    if (BC_mode == 0) | (BC_mode == 2):
        # in the case of periodic boundary conditions we can
        # perform the entire loop at once, correctly wrapping indices.
        # this is gonna be somewhat slower than ideal but probably still
        # faster than if we left wraparound on...
        for ij in prange(shape[0]*shape[1],nogil=True):
            i = ij / shape[1]
            j = ij % shape[1]
            for k in range(shape[2]):
                for l in range(shape[3]):
                    oldVal = recon[i,j,k,l]
                    recon[i,j,k,l] = orig[i,j,k,l] - \
                        (
                            (lambda_mu[0] * ( b1[i,j,k,l] - b1[(i+1)%shape[0],j,k,l])) +
                            (lambda_mu[1] * ( b2[i,j,k,l] - b2[i,(j+1)%shape[1],k,l])) +
                            (lambda_mu[2] * ( b3[i,j,k,l] - b3[i,j,(k+1)%shape[2],l])) +
                            (lambda_mu[3] * ( b4[i,j,k,l] - b4[i,j,k,(l+1)%shape[3]]))
                        )
                    delta += fabs(recon[i,j,k,l] - oldVal)
                    recon_norm += fabs(oldVal)
    elif BC_mode == 1:
        # handling the mirror boundary condition 
        # simply requires different math for computing the indices.
        # The new indexing is max(i+1,shape-1), so precompute shape[:]-1 as MBCend
        for ij in prange(shape[0]*shape[1],nogil=True):
            i = ij / shape[1]
            j = ij % shape[1]
            for k in range(shape[2]):
                for l in range(shape[3]):
                    oldVal = recon[i,j,k,l]
                    recon[i,j,k,l] = orig[i,j,k,l] - \
                        (
                            (lambda_mu[0] * ( b1[i,j,k,l] - b1[max(i+1,MBCend[0]),j,k,l])) +
                            (lambda_mu[1] * ( b2[i,j,k,l] - b2[i,max(j+1,MBCend[1]),k,l])) +
                            (lambda_mu[2] * ( b3[i,j,k,l] - b3[i,j,max(k+1,MBCend[2]),l])) +
                            (lambda_mu[3] * ( b4[i,j,k,l] - b4[i,j,k,max(l+1,MBCend[3])]))
                        )
                    delta += fabs(recon[i,j,k,l] - oldVal)
                    recon_norm += fabs(oldVal)

    return delta / recon_norm


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def datacube_update_3D(_float[:,:,::] orig, _float[:,:,::] recon, _float[:,:,::] b1, 
               _float[:,:,::] b2, _float[:,:,::] b3, _float[:] lambda_mu, int BC_mode=2):
    '''
    perform the TV update step:
    recon = orig - ( b1 - roll(b1,-1,1) )*lambda_mu - ...
    Boundary conditions for the gradient operator can be:
        0: periodic
        1: mirror
        2: Jia-Zhao Adv Comp Math 2010 33:231-241
            NOTE: We implememnt the JZ boundary conditions identically to periodic
            under the assumption that the accumulators are also computed with JZ
            boundary conditions, such that there are zeros in all the right places
    '''
    # shape of the 3-D array
    cdef Py_ssize_t shape[3]
    shape[:] = [orig.shape[0],orig.shape[1],orig.shape[2]]
    #cdef Py_ssize_t outer_iterator = shape[0] * shape[1]
    
    # index variables for the main 3D loop
    cdef Py_ssize_t i,j,k, ij

    # index limits for the mirror boundary condition
    # (In Cython these can't be defined inside the conditional)
    cdef int MBCend[3]
    MBCend = shape
    for q in range(3):
        MBCend[q] -= 1

    cdef _float delta = 0.0
    cdef _float recon_norm = 0.0
    cdef _float oldVal
    
    if (BC_mode == 0) | (BC_mode == 2):
        # in the case of periodic boundary conditions we can
        # perform the entire loop at once, correctly wrapping indices.
        # this is gonna be somewhat slower than ideal but probably still
        # faster than if we left wraparound on...
        for ij in prange(shape[0]*shape[1],nogil=True):
            i = ij / shape[1]
            j = ij % shape[1]
            for k in range(shape[2]):
                oldVal = recon[i,j,k]
                recon[i,j,k] = orig[i,j,k] - \
                    (
                        (lambda_mu[0] * ( b1[i,j,k] - b1[(i+1)%shape[0],j,k])) +
                        (lambda_mu[1] * ( b2[i,j,k] - b2[i,(j+1)%shape[1],k])) +
                        (lambda_mu[2] * ( b3[i,j,k] - b3[i,j,(k+1)%shape[2]]))
                    )
                delta += fabs(recon[i,j,k] - oldVal)
                recon_norm += fabs(oldVal)
    elif BC_mode == 1:
        # handling the mirror boundary condition 
        # simply requires different math for computing the indices.
        # The new indexing is max(i+1,shape-1), so precompute shape[:]-1 as MBCend
        for ij in prange(shape[0]*shape[1],nogil=True):
            i = ij / shape[1]
            j = ij % shape[1]
            for k in range(shape[2]):
                oldVal = recon[i,j,k]
                recon[i,j,k] = orig[i,j,k] - \
                    (
                        (lambda_mu[0] * ( b1[i,j,k] - b1[max(i+1,MBCend[0]),j,k])) +
                        (lambda_mu[1] * ( b2[i,j,k] - b2[i,max(i+1,MBCend[1]),k])) +
                        (lambda_mu[2] * ( b3[i,j,k] - b3[i,j,max(i+1,MBCend[2])]))
                    )
                delta += fabs(recon[i,j,k] - oldVal)
                recon_norm += fabs(recon_norm)

    return delta / recon_norm