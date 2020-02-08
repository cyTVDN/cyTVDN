from cyTV4D.utils import datacube_update, MSE
from cyTV4D.anisotropic import accumulator_update

import numpy as np
from tqdm import tqdm
from hurry.filesize import size, alternative
import psutil
from tabulate import tabulate

def denoise4D(datacube, lam, mu, iterations=75, PBC=False):
    '''
    Perform Proximal Anisotropic Total Variational denoising on a 4D datacube

    Arguments:
    datacube        (np.array) a C-contiguous 4D numpy array. Must be float32 or float64 dtype
    lam             (np.array) TV weights for each dimension. Must be same dtype as datacube
    mu              (float) TV weighting parameter
    iterations      (int) number of iterations to perform TV update step
    PBC             (bool) whether to use periodic boundary conditions (True) or
                    mirror boundary conditions (False, default)

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

    # warn about memory requirements
    print(f"Available RAM: {size(psutil.virtual_memory().available,system=alternative)}", flush=True)
    print(f"Unaccelerated TV denoising will require {size(datacube.nbytes*5,system=alternative)} of RAM...", flush=True)

    # allocate memory for the accumulators and the output datacube
    acc1 = np.zeros_like(datacube)
    acc2 = np.zeros_like(datacube)
    acc3 = np.zeros_like(datacube)
    acc4 = np.zeros_like(datacube)

    recon = np.zeros_like(datacube)

    for i in tqdm(range(int(iterations))):
        # update accumulators
        accumulator_update(recon, acc1, 0, lambdaInv[0])
        accumulator_update(recon, acc2, 1, lambdaInv[1])
        accumulator_update(recon, acc3, 2, lambdaInv[2])
        accumulator_update(recon, acc4, 3, lambdaInv[3])

        # update reconstruction
        datacube_update(datacube, recon, acc1, acc2, acc3, acc4, lam_mu)

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

    algos = [
        ["Anisotropic Unaccelerated", fmt(dcsize * 5), checkmark(dcsize * 5)],
        ["Anisotropic FISTA", fmt(dcsize * 13), checkmark(dcsize * 13)],
        ["(Half-)Isotropic Unaccelerated", fmt(dcsize * 5), checkmark(dcsize * 5)]
    ]

    print(tabulate(algos, headers))
