"""Built in custom functions for pixelflow."""

import numpy as np
import numpy.typing as npt

from skimage import measure

from pixelflow.core import pixelflow_custom


@pixelflow_custom(requires_package="pyefd")
def efd(
    mask: npt.NDArray,
    image: npt.NDArray,
    *,
    threshold: float = 0.8,
    n_terms: int = 10,
    normalize: bool = True,
) -> npt.NDArray:
    """Custom function to calculate elliptic fourier descriptors for each object.

    Parameters
    ----------
    mask : array
    image : array
    threshold : float
        Contour threshold.
    n_terms : int
        The order of Fourier coefficients to calculate.
    normalize : bool
        If the coefficients should be normalized.

    Notes
    -----
    See documentation for ``pyefd`` for more details.
    """

    _nan_features = np.full((n_terms * 4,), np.nan)

    # if the size of the mask is too small return array of NaN
    if all(mask.shape[dim] < 3 for dim in range(mask.ndim)):
        return _nan_features

    contour = measure.find_contours(mask, threshold)

    # if we can't find a contour, return default features
    if not contour:
        return _nan_features

    import pyefd

    coeffs = pyefd.elliptic_fourier_descriptors(
        contour[0], order=n_terms, normalize=normalize
    )

    return coeffs.flatten()
