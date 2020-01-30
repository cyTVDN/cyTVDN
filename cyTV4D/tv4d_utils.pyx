cimport cython
from cython.parallel import prange
import numpy as np

# define a fused floating point type so that we can use single or double precision floats:
ctypedef fused _float:
    float
    double

cdef _float clipval(_float a, _float val) nogil:
    return min(max(a,-val),val)
    
@cython.boundscheck(False)
@cython.wraparound(False)
def accumulator_update(_float[:,:,:,::] a,_float[:,:,:,::] b,int ax, _float clip):
    '''
    computes b = clip( a - roll(a,x,axis=ax) + b, -clip,+clip ) in place
    '''
    
    # shape of the 4-D array
    cdef int shape[4]
    shape[:] = [a.shape[0],a.shape[1],a.shape[2],a.shape[3]]
    
    # start point on each axis. this is zero for all axes but the rolling direction
    cdef int start[4]
    start[:] = [0,0,0,0]
    start[ax] += 1

    # index variables for the 4D loop
    cdef int i,j,k,l
    
    # perform the main loop
    for i in prange(start[0],shape[0],nogil=True):
        for j in range(start[1],shape[1]):
            for k in range(start[2],shape[2]):
                for l in range(start[3],shape[3]):
                    b[i,j,k,l] = clipval( a[i,j,k,l] - a[i-start[0],j-start[1],k-start[2],l-start[3]]
                                     + b[i,j,k,l], clip)
                    
    # perform the final hyperslab
    # in principle we can use index wraparound to make this easier
    # but that would have a speed penalty on every iteration in the 
    # main loop, so we do this explicitly for zero overhead
    cdef int m,n,o,p
    cdef int stop[4]
    stop = shape
    stop[ax] = 1
    
    cdef int delta[4]
    delta[:] = [0,0,0,0]
    delta[ax] = shape[ax] - 1
    
    for m in range(stop[0]):
        for n in range(stop[1]):
            for o in range(stop[2]):
                for p in range(stop[3]):
                    b[m,n,o,p] = clipval( a[m,n,o,p] - a[m+delta[0],n+delta[1],o+delta[2],p+delta[3]]
                                        + b[m,n,o,p], clip)
    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def datacube_update(_float[:,:,:,::] orig, _float[:,:,:,::] recon, _float[:,:,:,::] b1, 
               _float[:,:,:,::] b2, _float[:,:,:,::] b3, _float[:,:,:,::] b4, _float[:] lambda_mu):
    '''
    perform the TV update step:
    recon = orig - ( b1 - roll(b1,-1,1) )*lambda_mu - ...
    '''
    # shape of the 4-D array
    cdef int shape[4]
    shape[:] = [orig.shape[0],orig.shape[1],orig.shape[2],orig.shape[3]]
    
    # index variables for the main 4D loop
    cdef int i,j,k,l
    
    # perform the entire loop at once, correctly wrapping indices
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
    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def MSE(_float[:,:,:,::] a, _float[:,:,:,::] b):
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