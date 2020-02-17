from cyTV4D.utils import datacube_update_4D, sum_square_error
from cyTV4D.anisotropic import accumulator_update_4D, accumulator_update_4D_FISTA

import numpy as np
from tqdm import tqdm
from hurry.filesize import size, alternative
import psutil
from tabulate import tabulate


def denoise4D(datacube, lam, mu, iterations=75, BC_mode=2, FISTA=False, reference_data=None):
    '''
    Perform Proximal Anisotropic Total Variational denoising on a 4D datacube

    Arguments:
    datacube        (np.array) a C-contiguous 4D numpy array. Must be float32 or float64 dtype
    lam             (np.array) TV weights for each dimension. Must be same dtype as datacube
    mu              (float) TV weighting parameter
    iterations      (int) number of iterations to perform TV update step
    BC_mode         (int) boundary conditions for evaluating difference operators:
                        0: Periodic
                        1: Mirror
                        2: Jia-Zhao Adv Comp Math 2010 33:231-241
    FISTA           (bool) whether to use FISA Acceleration. Converges much faster,
                    but involves much more memory use
    reference_data  (np.array) For testing convergence, pass an infinite signal dataset
                    and the mean square error will be calculated for each iteration

    The algorithm used is an extension of that shown in this paper:
    Jia, Rong-Qing, and Hanqing Zhao. "A fast algorithm for the total variation model of image denoising."
    Advances in Computational Mathematics 33.2 (2010): 231-241.
    '''

    assert datacube.dtype in (np.float32, np.float64), "datacube must be floating point datatype."

    assert lam.dtype == datacube.dtype, "Lambda must have same dtype as datacube."

    assert datacube.flags['C_CONTIGUOUS'], "datacube must be C-contiguous. Try np.ascontiguousarray(datacube) on the array"

    lambdaInv = 1. / lam
    lam_mu = (lam / mu).astype(datacube.dtype)

    assert np.all(lam_mu < (1. / 8.)) & np.all(lam_mu > 0), "Parameters must satisfy 0 < λ/μ < 1/8"
    try:
        print(f"λ/μ ≈ [1/{mu/lam[0]:.0f}, 1/{mu/lam[1]:.0f}, 1/{mu/lam[2]:.0f}, 1/{mu/lam[3]:.0f}]")
    except Exception:
        print("I tried to print the memory requirements but your system doesn't like Unicode...")
    # warn about memory requirements
    print(f"Available RAM: {size(psutil.virtual_memory().available,system=alternative)}", flush=True)
    if FISTA:
        print(f"Unaccelerated TV denoising will require {size(datacube.nbytes*13,system=alternative)} of RAM...", flush=True)
    else:
        print(f"Unaccelerated TV denoising will require {size(datacube.nbytes*5,system=alternative)} of RAM...", flush=True)

    calculate_error = reference_data is not None
    if calculate_error:
        MSE = np.zeros((iterations+1,), dtype=datacube.dtype)
        MSE[0] = sum_square_error(datacube, reference_data)

    # allocate memory for the accumulators and the output datacube
    acc1 = np.zeros_like(datacube)
    acc2 = np.zeros_like(datacube)
    acc3 = np.zeros_like(datacube)
    acc4 = np.zeros_like(datacube)

    # if using FISTA, allocate the extra auxiliary arrays
    if FISTA:
        d1 = np.zeros_like(datacube)
        d2 = np.zeros_like(datacube)
        d3 = np.zeros_like(datacube)
        d4 = np.zeros_like(datacube)

        tk = 1.

    recon = datacube.copy()

    if FISTA:
        for i in tqdm(range(int(iterations)), desc='FISTA Accelerated Denoising'):
            # update the tk factor
            tk_new = (1 + np.sqrt(1 + 4*tk**2)) / 2
            tk_ratio = tk / tk_new
            tk = tk_new

            # update accumulators
            accumulator_update_4D_FISTA(recon, acc1, d1, tk_ratio, 0, lambdaInv[0], BC_mode=BC_mode)
            accumulator_update_4D_FISTA(recon, acc2, d2, tk_ratio, 1, lambdaInv[1], BC_mode=BC_mode)
            accumulator_update_4D_FISTA(recon, acc3, d3, tk_ratio, 2, lambdaInv[2], BC_mode=BC_mode)
            accumulator_update_4D_FISTA(recon, acc4, d4, tk_ratio, 3, lambdaInv[3], BC_mode=BC_mode)

            datacube_update_4D(datacube, recon, acc1, acc2, acc3, acc4, lam_mu, BC_mode=BC_mode)

            if calculate_error:
                MSE[i+1] = sum_square_error(reference_data,recon)
    else:
        for i in tqdm(range(int(iterations)), desc='Unaccelerated TV Denoising'):
            # update accumulators
            accumulator_update_4D(recon, acc1, 0, lambdaInv[0], BC_mode=BC_mode)
            accumulator_update_4D(recon, acc2, 1, lambdaInv[1], BC_mode=BC_mode)
            accumulator_update_4D(recon, acc3, 2, lambdaInv[2], BC_mode=BC_mode)
            accumulator_update_4D(recon, acc4, 3, lambdaInv[3], BC_mode=BC_mode)

            # update reconstruction
            datacube_update_4D(datacube, recon, acc1, acc2, acc3, acc4, lam_mu, BC_mode=BC_mode)

            if calculate_error:
                MSE[i+1] = sum_square_error(reference_data,recon)

    if calculate_error:
        return recon, MSE
    else:
        return recon


def check_memory(datacube):
    '''
    Determine if there is sufficient RAM to perform different TV denoising algorithms
    '''
    avail = psutil.virtual_memory().available
    dcsize = datacube.nbytes

    def fmt(x):
        return size(x, system=alternative)

    def checkmark(b):
        return "✅" if b < avail else "❌"

    headers = ["Algorithm", "RAM Needed", "OK?"]

    if len(datacube.shape) == 4:
        algos = [
            ["Anisotropic Unaccelerated", fmt(dcsize * 5), checkmark(dcsize * 5)],
            ["Anisotropic FISTA", fmt(dcsize * 13), checkmark(dcsize * 13)],
            ["(Half-)Isotropic Unaccelerated", fmt(dcsize * 5), checkmark(dcsize * 5)]
        ]
    elif len(datacube.shape) == 3:
        algos = [
            ["Anisotropic Unaccelerated", fmt(dcsize * 4), checkmark(dcsize * 4)],
            ["Anisotropic FISTA", fmt(dcsize * 11), checkmark(dcsize * 11)],
            ["(Half-)Isotropic Unaccelerated", fmt(dcsize * 5), checkmark(dcsize * 5)]
        ]

    print(tabulate(algos, headers))
