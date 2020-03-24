cimport cython
from cython.parallel import prange
import numpy as np
from libc.math cimport fabs, hypot

# define a fused floating point type so that we can use single or double precision floats:
ctypedef fused _float:
    float
    double

cdef _float clipval(_float a, _float val) nogil:
    return min(max(a,-val),val)
    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def iso_accumulator_update_4D(_float[:,:,:,::] a,_float[:,:,:,::] b1, _float[:,:,:,::] b2,
    int ax1, int ax2, _float clip):
    '''
    computes b = clip( a - roll(a,x,axis=ax) + b, -clip,+clip ) in place
    Boundary conditions for the gradient operator can be
    only those given by Jia-Zhao Adv Comp Math 2010 33:231-241
    '''
    
    # shape of the 4-D array
    cdef Py_ssize_t shape[4]
    shape[:] = [a.shape[0],a.shape[1],a.shape[2],a.shape[3]]
    
    # start point on each axis. this is zero for all axes but the rolling direction
    cdef Py_ssize_t start[4]
    start[:] = [0,0,0,0]
    # with the new loop scheme, we can cover all elements in one pass
    # start[ax1] += 1
    # start[ax2] += 1

    # strides for taking each separate gradient
    cdef Py_ssize_t master_stride1[4]
    master_stride1[:] = [0,0,0,0]
    master_stride1[ax1] += 1
    cdef Py_ssize_t master_stride2[4]
    master_stride2[:] = [0,0,0,0]
    master_stride2[ax2] += 1

    # local strides that will be updated based on boundary conditions
    cdef Py_ssize_t stride1[4]
    cdef Py_ssize_t stride2[4]
    stride1[:] = [0,0,0,0]
    stride2[:] = [0,0,0,0]

    # index variables for the 4D loop
    cdef Py_ssize_t i,j,k,l, ij

    # stride

    cdef _float norm = 0.0
    cdef _float delta1, delta2, b_mag
    
    # for better division of labor on systems with a lot of threads, 
    # we roll the two outer loops together into one prange
    cdef Py_ssize_t outer_iterator = (shape[0] - start[0]) * (shape[1] - start[1])

    # perform the main loop (where all the entries have left and top neighbors)
    for ij in prange(outer_iterator,nogil=True):
        i = start[0] + (ij / (shape[1] - start[1]))
        j = start[1] + (ij % (shape[1] - start[1]))

        # dynamically figure out strides to avoid multiple loops!
        # set stride to 1 if (a) it is a gradient axis and 
        # (b) we are not at the zero element
        stride1[0] = 1 if (master_stride1[0] & (i > 0)) else 0
        stride2[0] = 1 if (master_stride2[0] & (i > 0)) else 0

        stride1[1] = 1 if (master_stride1[1] & (j > 0)) else 0
        stride2[1] = 1 if (master_stride2[1] & (j > 0)) else 0

        for k in range(start[2],shape[2]):
            stride1[2] = 1 if (master_stride1[2] & (k > 0)) else 0
            stride2[2] = 1 if (master_stride2[2] & (k > 0)) else 0

            for l in range(start[3],shape[3]):
                stride1[3] = 1 if (master_stride1[3] & (l > 0)) else 0
                stride2[3] = 1 if (master_stride2[3] & (l > 0)) else 0

                delta1 = a[i,j,k,l] - a[i-stride1[0],j-stride1[1],k-stride1[2],l-stride1[3]] + b1[i,j,k,l]
                delta2 = a[i,j,k,l] - a[i-stride2[0],j-stride2[1],k-stride2[2],l-stride2[3]] + b2[i,j,k,l]

                b_mag = hypot(delta1,delta2)

                if b_mag > clip:
                    delta1 = delta1 / ( b_mag / clip )
                    delta2 = delta2 / ( b_mag / clip )

                norm += fabs(delta1) + fabs(delta2)
                b1[i,j,k,l] = delta1
                b2[i,j,k,l] = delta2

    return norm

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def iso_accumulator_update_4D_FISTA(_float[:,:,:,::] a, _float[:,:,:,::] b1,
        _float[:,:,:,::] b2, _float[:,:,:,::] d1, _float[:,:,:,::] d2,
        _float tk, int ax1, int ax2, _float clip):
    '''
    computes b = clip( a - roll(a,x,axis=ax) + b, -clip,+clip ) in place
    Boundary conditions for the gradient operator can be:
        0: periodic
        1: mirror
        2: Jia-Zhao Adv Comp Math 2010 33:231-241
    '''
    
    # shape of the 4-D array
    cdef Py_ssize_t shape[4]
    shape[:] = [a.shape[0],a.shape[1],a.shape[2],a.shape[3]]
    
    # start point on each axis. this is zero for all axes but the rolling direction
    cdef Py_ssize_t start[4]
    start[:] = [0,0,0,0]

    # strides for taking each separate gradient
    cdef Py_ssize_t master_stride1[4]
    master_stride1[:] = [0,0,0,0]
    master_stride1[ax1] += 1
    cdef Py_ssize_t master_stride2[4]
    master_stride2[:] = [0,0,0,0]
    master_stride2[ax2] += 1

    # local strides that will be updated based on boundary conditions
    cdef Py_ssize_t stride1[4]
    cdef Py_ssize_t stride2[4]
    stride1[:] = [0,0,0,0]
    stride2[:] = [0,0,0,0]

    # index variables for the 4D loop
    cdef Py_ssize_t i,j,k,l, ij

    cdef _float norm = 0.0
    cdef _float b1_new, b2_new, delta1, delta2, b_mag
    
    # for better division of labor on systems with a lot of threads, 
    # we roll the two outer loops together into one prange
    cdef Py_ssize_t outer_iterator = (shape[0] - start[0]) * (shape[1] - start[1])

    # perform the main loop
    for ij in prange(outer_iterator,nogil=True):
        i = start[0] + (ij / (shape[1] - start[1]))
        j = start[1] + (ij % (shape[1] - start[1]))

        # dynamically figure out strides to avoid multiple loops!
        # set stride to 1 if (a) it is a gradient axis and 
        # (b) we are not at the zero element
        stride1[0] = 1 if (master_stride1[0] & (i > 0)) else 0
        stride2[0] = 1 if (master_stride2[0] & (i > 0)) else 0

        stride1[1] = 1 if (master_stride1[1] & (j > 0)) else 0
        stride2[1] = 1 if (master_stride2[1] & (j > 0)) else 0


        for k in range(start[2],shape[2]):
            stride1[2] = 1 if (master_stride1[2] & (k > 0)) else 0
            stride2[2] = 1 if (master_stride2[2] & (k > 0)) else 0

            for l in range(start[3],shape[3]):
                stride1[3] = 1 if (master_stride1[3] & (l > 0)) else 0
                stride2[3] = 1 if (master_stride2[3] & (l > 0)) else 0

                delta1 = a[i,j,k,l] - a[i-stride1[0],j-stride1[1],k-stride1[2],l-stride1[3]] + b1[i,j,k,l]
                delta2 = a[i,j,k,l] - a[i-stride2[0],j-stride2[1],k-stride2[2],l-stride2[3]] + b2[i,j,k,l]

                b_mag = hypot(delta1,delta2)

                if b_mag > clip:
                    delta1 = delta1 / ( b_mag / clip )
                    delta2 = delta2 / ( b_mag / clip )

                b1_new = delta1 + tk*(delta1 - d1[i,j,k,l])
                b2_new = delta2 + tk*(delta2 - d2[i,j,k,l])

                b1[i,j,k,l] = b1_new
                b2[i,j,k,l] = b2_new

                norm += fabs(b1_new) + fabs(b2_new)

                d1[i,j,k,l] = delta1
                d2[i,j,k,l] = delta2

    return norm

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def accumulator_update_3D(_float[:,:,::] a,_float[:,:,::] b,int ax, _float clip, int BC_mode=2):
    '''
    computes b = clip( a - roll(a,x,axis=ax) + b, -clip,+clip ) in place
    Boundary conditions for the gradient operator can be:
        0: periodic
        1: mirror
        2: Jia-Zhao Adv Comp Math 2010 33:231-241
    '''
    
    # shape of the 4-D array
    cdef Py_ssize_t shape[3]
    shape[:] = [a.shape[0],a.shape[1],a.shape[2]]
    
    # start point on each axis. this is zero for all axes but the rolling direction
    cdef Py_ssize_t start[3]
    start[:] = [0,0,0]
    start[ax] += 1

    # keep track of the norm of the accumulators

    cdef _float norm = 0.0
    cdef _float b_new

    # index variables for the 4D loop
    cdef Py_ssize_t i,j,k, ij
    
    # for better division of labor on systems with a lot of threads, 
    # we roll the two outer loops together into one prange
    cdef Py_ssize_t outer_iterator = (shape[0] - start[0]) * (shape[1] - start[1])

    # perform the main loop
    for ij in prange(outer_iterator,nogil=True):
        i = start[0] + (ij / (shape[1] - start[1]))
        j = start[1] + (ij % (shape[1] - start[1]))
        for k in range(start[2],shape[2]):
            b_new = clipval( a[i,j,k] - a[i-start[0],j-start[1],k-start[2]]
                                                 + b[i,j,k], clip)
            norm += fabs(b_new)
            b[i,j,k] = b_new

                    
    # perform the final hyperslab
    # in principle we can use index wraparound to make this easier
    # but that would have a speed penalty on every iteration in the 
    # main loop, so we do this explicitly for zero overhead
    cdef Py_ssize_t m,n,o
    cdef Py_ssize_t stop[3]
    stop = shape
    stop[ax] = 1
    
    cdef Py_ssize_t delta[3]
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
                b_new = clipval( a[m,n,o] - a[m+delta[0],n+delta[1],o+delta[2]]
                                    + b[m,n,o], clip)
                norm += fabs(b_new)
                b[m,n,o] = b_new

    return norm
    

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
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
    cdef Py_ssize_t shape[3]
    shape[:] = [a.shape[0],a.shape[1],a.shape[2]]
    
    # start point on each axis. this is zero for all axes but the rolling direction
    cdef Py_ssize_t start[3]
    start[:] = [0,0,0]
    start[ax] += 1

    cdef _float norm = 0.0
    cdef _float b_new

    # index variables for the 4D loop
    cdef Py_ssize_t i,j,k, ij

    # temporary holder for updated d value
    cdef _float d_new = 0.0
    
    # for better division of labor on systems with a lot of threads, 
    # we roll the two outer loops together into one prange
    cdef Py_ssize_t outer_iterator = (shape[0] - start[0]) * (shape[1] - start[1])

    # perform the main loop
    for ij in prange(outer_iterator,nogil=True):
        i = start[0] + (ij / (shape[1] - start[1]))
        j = start[1] + (ij % (shape[1] - start[1]))
        for k in range(start[2],shape[2]):
            d_new = clipval( a[i,j,k] - a[i-start[0],j-start[1],k-start[2]]
                             + b[i,j,k], clip)
            b_new = d_new + tk*(d_new - d[i,j,k])
            b[i,j,k] = b_new
            norm += fabs(b_new)
            d[i,j,k] = d_new

                    
    # perform the final hyperslab
    # in principle we can use index wraparound to make this easier
    # but that would have a speed penalty on every iteration in the 
    # main loop, so we do this explicitly for zero overhead
    cdef Py_ssize_t m,n,o
    cdef Py_ssize_t stop[3]
    stop = shape
    stop[ax] = 1
    
    cdef Py_ssize_t delta[3]
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
                b_new = d_new + tk*(d_new - d[m,n,o])
                b[m,n,o] = b_new
                norm += fabs(b_new)
                d[m,n,o] = d_new

    return norm
