import dataclasses
import importlib
import functools
import os
import warnings

from typing import Callable, Optional, Union
from skimage.measure import label, regionprops_table

import numpy as np
import numpy.typing as npt
import pandas as pd
import porespy as ps


Numeric = Union[int, float]


class PixelflowImportWarning(UserWarning):
    pass


def pixelflow_custom(
    func: Optional[Callable] = None,
    *,
    channel_mask: tuple[int, ...] = (...),
    scale: tuple[float, ...] = (...),
    requires_package: Optional[str] = None,
    **kwargs,
) -> Callable:
    """Pixelflow custom analysis function decorator.

    Parameters
    ----------
    func : callable
        The analysis function to be wrapped
    channel_mask : tuple
        The channels to be masked in the image.

    """

    if func is None:
        return functools.partial(
            pixelflow_custom,
            channel_mask=channel_mask,
            scale=scale,
            requires_package=requires_package,
            **kwargs,
        )

    if requires_package is not None:
        module_spec = importlib.util.find_spec(requires_package)
        if module_spec is None:
            warnings.warn(
                f"Package {requires_package} is not installed.", PixelflowImportWarning
            )

    @functools.wraps(func)
    def wrapper(mask: npt.NDArray, image: npt.NDArray, **kwargs) -> Callable:
        # some internal logic, here just checks that we have at least a 2D image
        assert mask.ndim >= 2
        # assert image.ndim == mask.ndim

        return func(mask, image, **kwargs)

    return wrapper


class PixelflowLambda(functools.partial):
    """Custom partial lambda function to wrap analysis modules.

    Usage
    -----
    >>> pf_lambda = PixelflowLambda(analysis_func, some_kwarg=10)
    """

    @property
    def __name__(self) -> str:
        return self.func.__name__


@dataclasses.dataclass
class PixelflowResult:
    """Result container with additional `reduce` functionality."""

    features: pd.DataFrame
    image_intensity: Optional[npt.NDArray] = None

    def count(self) -> int:
        """Return the number of objects in the image."""
        return len(self.features)

    def sum(self, feature_name: str) -> Numeric:
        """Return the sum of the named feature across all objects."""
        return np.sum(self.features[feature_name])

    def __repr__(self) -> str:
        return repr(self.features)

    def _repr_html_(self) -> str:
        return self.features.to_html()

    def to_csv(self, path: os.PathLike, **kwargs) -> None:
        """Output the features dataframe as a .csv file."""
        self.features.to_csv(path, **kwargs)


def pixelflow(
    mask: npt.NDArray,
    image: npt.NDArray,
    *,
    features: Optional[tuple[str]] = None,
    custom: Optional[Callable] = None,
    dim_labels: Optional[str] = None,
) -> PixelflowResult:
    """Simple wrapper around `regionprops` to be extended or replaced.

    Parameters
    ----------
    mask : array
    image : array
    features : tuple, optional
    custom : tuple, optional
    dim_labels : str, optional.
        Dimension labels for the mask. Currently accepts YX or ZYX. Will be
        guessed if not supplied.

    Returns
    -------
    result : PixelflowResult
        A wrapped dataframe with additional functionality.

    Notes
    -----
    Things to consider:
    * ND arrays, perhaps we want to process successive images using the
        same functions (e.g. TYX) where T represents time but the functions
        operate on YX images
    * Inputs other than a segmentation mask, e.g. bounding boxes, regions
    * Distributing the task across cores/gpus for performance
    * Anisotropic scales in the image data
    * Using other inputs such as xarray, dask or pytorch tensors
    * Interfaces with scivision models
    * A progress bar for user feedback
    * Utility functions such as:
        * Image padding to take care of edges
        * Supplying different image channels to different functions
    """

    if dim_labels is None:
        if mask.ndim == 2:
            dim_labels = "YX"
            print("YX image detected")
        elif mask.ndim == 3:
            dim_labels = "ZYX"
            print("ZYX image detected")
        else:
            raise ValueError("Image must be YX or ZYX, check mask.ndim")

    # check dim_labels matches mask.ndim
    if len(dim_labels) != mask.ndim:
        raise ValueError("dim_labels doesn't match mask dimensions")

    # If image is YX then use regionprops_table
    if dim_labels == "YX":
        features_dat = regionprops_table(
            label(mask),
            image,
            properties=features,
            extra_properties=custom,
        )
        features_df = pd.DataFrame(features_dat)

    # If image is ZYX then use regionprops_3D
    elif dim_labels == "ZYX":
        features_2d = ("label", "bbox", "centroid")
        # if image_intensity is requested calculate it through regionprops_table
        if features is not None:
            features_2d += tuple(set(features).intersection({"image_intensity"}))
            features_3d = tuple(set(features).difference({"image_intensity"}))

        # calculate the regionprops features
        features_dat = regionprops_table(
            mask,
            image,
            properties=features_2d,
            extra_properties=custom,
        )
        features_df = pd.DataFrame(features_dat)

        # calculate the 3D features
        features_dat3d = ps.metrics.regionprops_3D(mask)
        features_df3d = ps.metrics.props_to_DataFrame(features_dat3d)

        # if only certain features are requested, then filter the dataframe
        if features is not None:
            features_df3d = features_df3d[list(features_3d)]
        # combine the regionprops and 3D features
        features_df = pd.merge(features_df, features_df3d)

    else:
        raise ValueError("Image type unsupported, expected 'YX' or 'ZYX'")

    pf_result = PixelflowResult(features=features_df)

    # If image_intensity is requested, extract it
    if features is not None and "image_intensity" in features:
        features_img = pf_result.features.pop("image_intensity")
        pf_result.image_intensity = features_img

    return pf_result
