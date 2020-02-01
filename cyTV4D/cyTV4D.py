from cyTV4D.tv4d_utils import accumulator_update, datacube_update, MSE

import numpy as np
from tqdm import tqdm

def denoise4D(datacube, lam, mu, iterations=75):
    '''
    Perform Proximal Anisotropic Total Variational denoising on a 4D datacube

    Arguments:
    datacube        (np.array) a C-contiguous 4D numpy array. Must be float32 or float64 dtype
    lam             (np.array) TV weights for each dimension. Must be same dtype as datacube
    mu              (float) TV weighting parameter
    iterations      (int) number of iterations to perform TV update step

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
