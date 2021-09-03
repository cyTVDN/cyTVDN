from cyTVDN.utils import datacube_update_4D, datacube_update_3D
from cyTVDN.utils import sum_square_error_4D, sum_square_error_3D
from cyTVDN.anisotropic import accumulator_update_4D, accumulator_update_4D_FISTA
from cyTVDN.anisotropic import accumulator_update_3D, accumulator_update_3D_FISTA

from cyTVDN.halfisotropic import (
    iso_accumulator_update_4D,
    iso_accumulator_update_4D_FISTA,
)

import numpy as np
from tqdm import tqdm
from hurry.filesize import size, alternative
import psutil
from tabulate import tabulate
from typing import Optional, Tuple


def denoise4D(
    datacube: np.ndarray,
    mu: np.ndarray,
    iterations: int = 10,
    FISTA: bool = True,
    stopping_relative_change: Optional[float] = None,
    isotropic_R: bool = False,
    isotropic_Q: bool = False,
    reference_data: Optional[np.ndarray] = None,
    BC_mode: int = 2,
    lam: Optional[np.ndarray] = None,
    quiet: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Perform Proximal Anisotropic Total Variational denoising on a 4D datacube

    Arguments:
    datacube        (np.array) a C-contiguous 4D numpy array. Must be float32 or float64 dtype
    mu              (4-element np.array) TV weighting parameter
    iterations      (int) number of iterations to perform TV update step
    FISTA           (bool) whether to use FISTA Acceleration. Converges much faster,
                    but involves much more memory use
    stopping_relative_change
                    (float) stopping criterion for relative change in reconstruction at each update step
                    if None, do iterations. if specified, stop when the relative change is below this value
                    A value of 0.05 seems to work best. 
    isotropic_R     (bool) Use half-isotropic algorithm on axes 0 and 1 (real space in py4DSTEM convention)
    isotropic_Q     (bool) Use half-isotropic algorithm on axies 2 and 3 (reciprocal space in py4DSTEM convention)
    reference_data  (np.array) For testing convergence, pass an infinite signal dataset
                    and the mean square error will be calculated for each iteration
    BC_mode         (int) boundary conditions for evaluating difference operators:
                                    0: Periodic
                                    1: Mirror
                        (default)   2: Jia-Zhao Adv Comp Math 2010 33:231-241
    lam             (np.array) TV weights for each dimension. Must be same dtype as datacube
    quiet           (bool) Suppress informational messages and clear the progress bar after running.


    The algorithm used is an extension of the one shown in this paper:
    Jia, Rong-Qing, and Hanqing Zhao. "A fast algorithm for the total variation model of image denoising."
    Advances in Computational Mathematics 33.2 (2010): 231-241.
    """

    assert datacube.dtype in (
        np.float32,
        np.float64,
    ), "datacube must be floating point datatype."

    if lam is None:
        lam = mu * 1.0 / 32.0

    assert lam.dtype == datacube.dtype, "Lambda must have same dtype as datacube."
    assert mu.dtype == datacube.dtype, "Mu must have same dtype as datacube."

    assert datacube.flags[
        "C_CONTIGUOUS"
    ], "datacube must be C-contiguous. Try np.ascontiguousarray(datacube) on the array."

    lambdaInv = 1.0 / lam
    lam_mu = (lam / mu).astype(datacube.dtype)

    if not quiet:
        try:
            print(
                f"λ/μ ≈ [1/{mu[0]/lam[0]:.0f}, 1/{mu[1]/lam[1]:.0f}, 1/{mu[2]/lam[2]:.0f}, 1/{mu[3]/lam[3]:.0f}]"
            )
        except Exception:
            print(
                "I tried to print with pretty characters but your system doesn't like Unicode..."
            )
    if (np.any(lam_mu > (1.0 / 32.0)) or np.any(lam_mu <= 0)) and not quiet:
        print("WARNING: Parameters must satisfy 0 < λ/μ <= 1/32 or result may diverge!")

    # warn about memory requirements
    if not quiet:
        print(
            f"Available RAM: {size(psutil.virtual_memory().available,system=alternative)}",
            flush=True,
        )

    unaccelerated = not FISTA
    if type(iterations) in (list, tuple):
        FISTA = True
        unaccelerated = True

        iterations_FISTA = iterations[0]
        iterations_unacc = iterations[1]
    else:
        iterations_FISTA = iterations * FISTA
        iterations_unacc = iterations * (not FISTA)

    if not quiet:
        if FISTA:
            print(
                f"FISTA Accelerated TV denoising will require {size(datacube.nbytes*9,system=alternative)} of RAM...",
                flush=True,
            )
        else:
            print(
                f"Unaccelerated TV denoising will require {size(datacube.nbytes*5,system=alternative)} of RAM...",
                flush=True,
            )

    calculate_MSE = reference_data is not None
    if calculate_MSE:
        MSE = np.zeros((iterations_FISTA + iterations_unacc + 1,), dtype=datacube.dtype)
        MSE[0] = sum_square_error_4D(datacube, reference_data)

    b_norm = np.zeros((iterations_FISTA + iterations_unacc), dtype=datacube.dtype)
    delta_recon = np.zeros_like(b_norm)

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

        tk = 1.0

    recon = datacube.copy()

    if FISTA:
        for i in tqdm(
            range(int(iterations_FISTA)),
            desc="FISTA Accelerated TV Denoising",
            leave=not quiet,
        ):
            # update the tk factor
            tk_new = (1 + np.sqrt(1 + 4 * tk ** 2)) / 2
            tk_ratio = (tk - 1.0) / tk_new
            tk = tk_new

            # update accumulators
            if isotropic_R:
                b_norm[i] += iso_accumulator_update_4D_FISTA(
                    recon, acc1, acc2, d1, d2, tk_ratio, 0, 1, lambdaInv[0]
                )
            else:
                b_norm[i] += accumulator_update_4D_FISTA(
                    recon, acc1, d1, tk_ratio, 0, lambdaInv[0], BC_mode=BC_mode
                )
                b_norm[i] += accumulator_update_4D_FISTA(
                    recon, acc2, d2, tk_ratio, 1, lambdaInv[1], BC_mode=BC_mode
                )
            if isotropic_Q:
                b_norm[i] += iso_accumulator_update_4D_FISTA(
                    recon, acc3, acc4, d3, d4, tk_ratio, 2, 3, lambdaInv[2]
                )
            else:
                b_norm[i] += accumulator_update_4D_FISTA(
                    recon, acc3, d3, tk_ratio, 2, lambdaInv[2], BC_mode=BC_mode
                )
                b_norm[i] += accumulator_update_4D_FISTA(
                    recon, acc4, d4, tk_ratio, 3, lambdaInv[3], BC_mode=BC_mode
                )

            delta_recon[i] = datacube_update_4D(
                datacube, recon, acc1, acc2, acc3, acc4, lam_mu, BC_mode=BC_mode
            )

            if calculate_MSE:
                MSE[i + 1] = sum_square_error_4D(reference_data, recon)

            if (
                stopping_relative_change is not None
                and delta_recon[i] < stopping_relative_change
            ):
                # if we have converged, break out of the loop
                break
    if unaccelerated:
        for j in tqdm(
            range(int(iterations_unacc)),
            desc="Unaccelerated TV Denoising",
            leave=not quiet,
        ):
            i = j + iterations_FISTA
            # update accumulators

            if isotropic_R:
                b_norm[i] += iso_accumulator_update_4D(
                    recon, acc1, acc2, 0, 1, lambdaInv[0]
                )
            else:
                b_norm[i] += accumulator_update_4D(
                    recon, acc1, 0, lambdaInv[0], BC_mode=BC_mode
                )
                b_norm[i] += accumulator_update_4D(
                    recon, acc2, 1, lambdaInv[1], BC_mode=BC_mode
                )
            if isotropic_Q:
                b_norm[i] += iso_accumulator_update_4D(
                    recon, acc3, acc4, 2, 3, lambdaInv[2]
                )
            else:
                b_norm[i] += accumulator_update_4D(
                    recon, acc3, 2, lambdaInv[2], BC_mode=BC_mode
                )
                b_norm[i] += accumulator_update_4D(
                    recon, acc4, 3, lambdaInv[3], BC_mode=BC_mode
                )

            # update reconstruction
            delta_recon[i] = datacube_update_4D(
                datacube, recon, acc1, acc2, acc3, acc4, lam_mu, BC_mode=BC_mode
            )

            if calculate_MSE:
                MSE[i + 1] = sum_square_error_4D(reference_data, recon)

            if (
                stopping_relative_change is not None
                and delta_recon[i] < stopping_relative_change
            ):
                # if we have converged, break out of the loop
                if not quiet:
                    print(f"Stopping condition reached after {i} iterations, stopping.")
                break

    if calculate_MSE:
        return recon, b_norm, delta_recon, MSE
    else:
        return recon, b_norm, delta_recon


def denoise3D(
    datacube: np.ndarray,
    mu: np.ndarray,
    iterations: int = 7_500,
    stopping_relative_change: Optional[float] = None,
    BC_mode: int = 2,
    FISTA: bool = False,
    reference_data: Optional[np.ndarray] = None,
    lam: Optional[np.ndarray] = None,
    quiet: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Perform Proximal Anisotropic Total Variational denoising on a 3D datacube

    Arguments:
    datacube        (np.array) a C-contiguous 3D numpy array. Must be float32 or float64 dtype
    stopping_relative_change
                    (float) stopping criterion for relative change in reconstruction at each update step
                    if None, do iterations. if specified, stop when the relative change is below this value
                    A value of 0.05 seems to work best. 
    mu              (3-element np.array) TV weighting parameter for each dimension
    iterations      (int) number of iterations to perform TV update step
                    (list) first perform FISTA iterations, then unaccelerated
    BC_mode         (int) boundary conditions for evaluating difference operators:
                        0: Periodic
                        1: Mirror
                        2: Jia-Zhao Adv Comp Math 2010 33:231-241
    FISTA           (bool) whether to use FISA Acceleration. Converges much faster,
                    but involves much more memory use
    reference_data  (np.array) For testing convergence, pass an infinite signal dataset
                    and the mean square error will be calculated for each iteration
    lam             (np.array) TV weights for each dimension. Must be same dtype as datacube
    quiet           (bool) Suppress informational messages and clear the progress bar after running.

    The algorithm used is an extension of that shown in this paper:
    Jia, Rong-Qing, and Hanqing Zhao. "A fast algorithm for the total variation model of image denoising."
    Advances in Computational Mathematics 33.2 (2010): 231-241.
    """

    assert datacube.dtype in (
        np.float32,
        np.float64,
    ), "datacube must be floating point datatype."

    if lam is None:
        lam = mu / 16.0

    assert lam.dtype == datacube.dtype, "Lambda must have same dtype as datacube."

    assert datacube.flags[
        "C_CONTIGUOUS"
    ], "datacube must be C-contiguous. Try np.ascontiguousarray(datacube) on the array"

    lambdaInv = 1.0 / lam
    lam_mu = (lam / mu).astype(datacube.dtype)

    assert np.all(lam_mu <= (1.0 / 16.0)) & np.all(
        lam_mu > 0
    ), "Parameters must satisfy 0 < λ/μ <= 1/8"
    if not quiet:
        try:
            print(f"λ/μ ≈ [1/{mu/lam[0]:.0f}, 1/{mu/lam[1]:.0f}, 1/{mu/lam[2]:.0f}]")
        except Exception:
            print(
                "I tried to print with pretty characters but your system doesn't like Unicode..."
            )
        # warn about memory requirements
        print(
            f"Available RAM: {size(psutil.virtual_memory().available,system=alternative)}",
            flush=True,
        )

    unaccelerated = not FISTA
    if type(iterations) in (list, tuple):
        FISTA = True
        unaccelerated = True

        iterations_FISTA = iterations[0]
        iterations_unacc = iterations[1]
    else:
        iterations_FISTA = iterations * FISTA
        iterations_unacc = iterations * (not FISTA)

    if not quiet:
        if FISTA:
            print(
                f"FISTA Accelerated TV denoising will require {size(datacube.nbytes*7,system=alternative)} of RAM...",
                flush=True,
            )
        else:
            print(
                f"Unaccelerated TV denoising will require {size(datacube.nbytes*4,system=alternative)} of RAM...",
                flush=True,
            )

    calculate_error = reference_data is not None
    if calculate_error:
        MSE = np.zeros((iterations_FISTA + iterations_unacc + 1,), dtype=datacube.dtype)
        MSE[0] = sum_square_error_3D(datacube, reference_data)

    b_norm = np.zeros((iterations_FISTA + iterations_unacc), dtype=datacube.dtype)
    delta_recon = np.zeros_like(b_norm)

    # allocate memory for the accumulators and the output datacube
    acc1 = np.zeros_like(datacube)
    acc2 = np.zeros_like(datacube)
    acc3 = np.zeros_like(datacube)

    # if using FISTA, allocate the extra auxiliary arrays
    if FISTA:
        d1 = np.zeros_like(datacube)
        d2 = np.zeros_like(datacube)
        d3 = np.zeros_like(datacube)

        tk = 1.0

    recon = datacube.copy()

    if FISTA:
        for i in tqdm(
            range(int(iterations_FISTA)), desc="FISTA Accelerated TV Denoising", leave=not quiet,
        ):
            # update the tk factor
            tk_new = (1 + np.sqrt(1 + 4 * tk ** 2)) / 2
            tk_ratio = (tk - 1.0) / tk_new
            tk = tk_new

            # update accumulators
            b_norm[i] += accumulator_update_3D_FISTA(
                recon, acc1, d1, tk_ratio, 0, lambdaInv[0], BC_mode=BC_mode
            )
            b_norm[i] += accumulator_update_3D_FISTA(
                recon, acc2, d2, tk_ratio, 1, lambdaInv[1], BC_mode=BC_mode
            )
            b_norm[i] += accumulator_update_3D_FISTA(
                recon, acc3, d3, tk_ratio, 2, lambdaInv[2], BC_mode=BC_mode
            )

            delta_recon[i] = datacube_update_3D(
                datacube, recon, acc1, acc2, acc3, lam_mu, BC_mode=BC_mode
            )

            if calculate_error:
                MSE[i + 1] = sum_square_error_3D(reference_data, recon)

            if (
                stopping_relative_change is not None
                and delta_recon[i] < stopping_relative_change
            ):
                # if we have converged, break out of the loop
                break
    if unaccelerated:
        for j in tqdm(range(int(iterations_unacc)), desc="Unaccelerated TV Denoising", leave=not quiet):
            i = j + iterations_FISTA
            # update accumulators
            b_norm[i] += accumulator_update_3D(
                recon, acc1, 0, lambdaInv[0], BC_mode=BC_mode
            )
            b_norm[i] += accumulator_update_3D(
                recon, acc2, 1, lambdaInv[1], BC_mode=BC_mode
            )
            b_norm[i] += accumulator_update_3D(
                recon, acc3, 2, lambdaInv[2], BC_mode=BC_mode
            )

            # update reconstruction
            delta_recon[i] = datacube_update_3D(
                datacube, recon, acc1, acc2, acc3, lam_mu, BC_mode=BC_mode
            )

            if calculate_error:
                MSE[i + 1] = sum_square_error_3D(reference_data, recon)

            if (
                stopping_relative_change is not None
                and delta_recon[i] < stopping_relative_change
            ):
                # if we have converged, break out of the loop
                if not quiet:
                    print(f"Stopping condition reached after {i} iterations, stopping.")
                break

    if calculate_error:
        return recon, b_norm, delta_recon, MSE
    else:
        return recon, b_norm, delta_recon


def check_memory(datacube):
    """
    Determine if there is sufficient RAM to perform different TV denoising algorithms
    """
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
            ["(Half-)Isotropic Unaccelerated", fmt(dcsize * 5), checkmark(dcsize * 5)],
        ]
    elif len(datacube.shape) == 3:
        algos = [
            ["Anisotropic Unaccelerated", fmt(dcsize * 4), checkmark(dcsize * 4)],
            ["Anisotropic FISTA", fmt(dcsize * 11), checkmark(dcsize * 11)],
            ["(Half-)Isotropic Unaccelerated", fmt(dcsize * 5), checkmark(dcsize * 5)],
        ]

    print(f"Datacube size is {fmt(dcsize)} with dtype {datacube.dtype}")
    print(tabulate(algos, headers))
