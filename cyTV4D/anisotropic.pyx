cimport cython
from cython.parallel import prange
import numpy as np
from libc.math cimport fabs

# define a fused floating point type so that we can use single or double precision floats:
ctypedef fused _float:
    float
    double

cdef _float clipval(_float a, _float val) nogil:
    return min(max(a,-val),val)
    
@cython.boundscheck(False)
@cython.wraparound(False)
def accumulator_update_4D(_float[:,:,:,::] a,_float[:,:,:,::] b,int ax, _float clip, int BC_mode=2):
    '''
    computes b = clip( a - roll(a,x,axis=ax) + b, -clip,+clip ) in place
    Boundary conditions for the gradient operator can be:
        0: periodic
        1: mirror
        2: Jia-Zhao Adv Comp Math 2010 33:231-241
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

    # keep track of the norm of the accumulator
    if _float is float:
        dtype = np.float32
    if _float is double:
        dtype = np.double
    norm_np = np.zeros((a.shape[0],),dtype=dtype)
    cdef _float[:] norm = norm_np

    cdef _float b_new
    
    # perform the main loop
    for i in prange(start[0],shape[0],nogil=True):
        for j in range(start[1],shape[1]):
            for k in range(start[2],shape[2]):
                for l in range(start[3],shape[3]):
                    b_new = clipval( a[i,j,k,l] - a[i-start[0],j-start[1],k-start[2],l-start[3]]
                                                         + b[i,j,k,l], clip)
                    norm[i] += fabs(b_new)
                    b[i,j,k,l] = b_new
                    
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
    if BC_mode == 0:
        delta[ax] = shape[ax] - 1
    elif BC_mode == 1:
        delta[ax] = 1
    elif BC_mode == 2:
        # keep all deltas at zero to make each entry on the hyperslab zero!
        delta[ax] = 0
    
    for m in range(stop[0]):
        for n in range(stop[1]):
            for o in range(stop[2]):
                for p in range(stop[3]):
                    b_new = clipval( a[m,n,o,p] - a[m+delta[0],n+delta[1],o+delta[2],p+delta[3]]
                                        + b[m,n,o,p], clip)
                    norm[m] += fabs(b_new)
                    b[m,n,o,p] = b_new

    return norm_np.sum()

@cython.boundscheck(False)
@cython.wraparound(False)
def accumulator_update_4D_FISTA(_float[:,:,:,::] a, _float[:,:,:,::] b, _float[:,:,:,::] d,
    _float tk, int ax, _float clip, int BC_mode=2):
    '''
    computes b = clip( a - roll(a,x,axis=ax) + b, -clip,+clip ) in place
    Boundary conditions for the gradient operator can be:
        0: periodic
        1: mirror
        2: Jia-Zhao Adv Comp Math 2010 33:231-241
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

    # temporary holder for updated d value
    cdef _float d_new = 0.0

    # keep track of the norm of the accumulator
    if _float is float:
        dtype = np.float32
    if _float is double:
        dtype = np.double
    norm_np = np.zeros((a.shape[0],),dtype=dtype)
    cdef _float[:] norm = norm_np

    cdef _float b_new
    
    # perform the main loop
    for i in prange(start[0],shape[0],nogil=True):
        for j in range(start[1],shape[1]):
            for k in range(start[2],shape[2]):
                for l in range(start[3],shape[3]):
                    d_new = clipval( a[i,j,k,l] - a[i-start[0],j-start[1],k-start[2],l-start[3]]
                                     + b[i,j,k,l], clip)
                    b_new = d_new + tk*(d_new - d[i,j,k,l])
                    b[i,j,k,l] = b_new
                    norm[i] += fabs(b_new)
                    d[i,j,k,l] = d_new
                    
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
    if BC_mode == 0:
        delta[ax] = shape[ax] - 1
    elif BC_mode == 1:
        delta[ax] = 1
    elif BC_mode == 2:
        # keep all deltas at zero to make each entry on the hyperslab zero!
        delta[ax] = 0
    
    for m in range(stop[0]):
        for n in range(stop[1]):
            for o in range(stop[2]):
                for p in range(stop[3]):
                    d_new = clipval( a[m,n,o,p] - a[m+delta[0],n+delta[1],o+delta[2],p+delta[3]]
                                    + b[m,n,o,p], clip)
                    b_new = d_new + tk*(d_new - d[m,n,o,p])
                    b[m,n,o,p] = b_new
                    norm[m] += fabs(b_new)
                    d[m,n,o,p] = d_new

    return norm_np.sum()

@cython.boundscheck(False)
@cython.wraparound(False)
def accumulator_update_3D(_float[:,:,::] a,_float[:,:,::] b,int ax, _float clip, int BC_mode=2):
    '''
    computes b = clip( a - roll(a,x,axis=ax) + b, -clip,+clip ) in place
    Boundary conditions for the gradient operator can be:
        0: periodic
        1: mirror
        2: Jia-Zhao Adv Comp Math 2010 33:231-241
    '''
    
    # shape of the 4-D array
    cdef int shape[3]
    shape[:] = [a.shape[0],a.shape[1],a.shape[2]]
    
    # start point on each axis. this is zero for all axes but the rolling direction
    cdef int start[3]
    start[:] = [0,0,0]
    start[ax] += 1

    # index variables for the 4D loop
    cdef int i,j,k
    
    # perform the main loop
    for i in prange(start[0],shape[0],nogil=True):
        for j in range(start[1],shape[1]):
            for k in range(start[2],shape[2]):
                    b[i,j,k] = clipval( a[i,j,k] - a[i-start[0],j-start[1],k-start[2]]
                                     + b[i,j,k], clip)
                    
    # perform the final hyperslab
    # in principle we can use index wraparound to make this easier
    # but that would have a speed penalty on every iteration in the 
    # main loop, so we do this explicitly for zero overhead
    cdef int m,n,o
    cdef int stop[3]
    stop = shape
    stop[ax] = 1
    
    cdef int delta[3]
    delta[:] = [0,0,0]
    if BC_mode == 0:
        delta[ax] = shape[ax] - 1
    elif BC_mode == 1:
        delta[ax] = 1
    elif BC_mode == 2:
        # keep all deltas at zero to make each entry on the hyperslab zero!
        delta[ax] = 0
    
    for m in range(stop[0]):
        for n in range(stop[1]):
            for o in range(stop[2]):
                    b[m,n,o] = clipval( a[m,n,o] - a[m+delta[0],n+delta[1],o+delta[2]]
                                        + b[m,n,o], clip)
    

@cython.boundscheck(False)
@cython.wraparound(False)
def accumulator_update_3D_FISTA(_float[:,:,::] a, _float[:,:,::] b, _float[:,:,::] d,
    _float tk, int ax, _float clip, int BC_mode=2):
    '''
    computes b = clip( a - roll(a,x,axis=ax) + b, -clip,+clip ) in place
    Boundary conditions for the gradient operator can be:
        0: periodic
        1: mirror
        2: Jia-Zhao Adv Comp Math 2010 33:231-241
    '''
    
    # shape of the 4-D array
    cdef int shape[3]
    shape[:] = [a.shape[0],a.shape[1],a.shape[2]]
    
    # start point on each axis. this is zero for all axes but the rolling direction
    cdef int start[3]
    start[:] = [0,0,0]
    start[ax] += 1

    # index variables for the 4D loop
    cdef int i,j,k

    # temporary holder for updated d value
    cdef _float d_new = 0.0
    
    # perform the main loop
    for i in prange(start[0],shape[0],nogil=True):
        for j in range(start[1],shape[1]):
            for k in range(start[2],shape[2]):
                    d_new = clipval( a[i,j,k] - a[i-start[0],j-start[1],k-start[2]]
                                     + b[i,j,k], clip)
                    b[i,j,k] = d_new + tk*(d_new - d[i,j,k])
                    d[i,j,k] = d_new
                    
    # perform the final hyperslab
    # in principle we can use index wraparound to make this easier
    # but that would have a speed penalty on every iteration in the 
    # main loop, so we do this explicitly for zero overhead
    cdef int m,n,o
    cdef int stop[3]
    stop = shape
    stop[ax] = 1
    
    cdef int delta[3]
    delta[:] = [0,0,0]
    if BC_mode == 0:
        delta[ax] = shape[ax] - 1
    elif BC_mode == 1:
        delta[ax] = 1
    elif BC_mode == 2:
        # keep all deltas at zero to make each entry on the hyperslab zero!
        delta[ax] = 0
    
    for m in range(stop[0]):
        for n in range(stop[1]):
            for o in range(stop[2]):
                    d_new = clipval( a[m,n,o] - a[m+delta[0],n+delta[1],o+delta[2]]
                                    + b[m,n,o], clip)
                    b[m,n,o] = d_new + tk*(d_new - d[m,n,o])
                    d[m,n,o] = d_new
