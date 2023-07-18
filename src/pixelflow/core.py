import dataclasses
import functools
import numpy as np
import numpy.typing as npt
import pandas as pd

from skimage.measure import label, regionprops_table
from typing import Callable, Optional, Union


Numeric = Union[int, float]

def pixelflow_custom(
    func: Optional[Callable] = None,
    *, 
    channel_mask: tuple[int, ...] = (...),
    scale: tuple[float, ...] = (...),
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
        return functools.partial(pixelflow_custom, channel_mask=channel_mask)
    
    @functools.wraps(func)
    def wrapper(x: npt.NDArray, y: npt.NDArray) -> Callable:

        # some internal logic, here just checks that we have at least a 2D image
        assert x.ndim >= 2
        assert y.ndim == x.ndim

        return func(x, y)
    return wrapper


@dataclasses.dataclass
class PixelflowResult():
    """Result container with additional `reduce` functionality."""
    features: pd.DataFrame 

    def count(self) -> int:
        """Return the number of objects in the image."""
        return len(self.features)
    
    def sum(self, feature_name: str) -> Numeric:
        return np.sum(self.features[feature_name])

    def __repr__(self) -> str:
        return repr(self.features)
    
    def _repr_html_(self) -> str:
        return self.features.to_html()
    
    def to_csv(self, path: str, **kwargs) -> None:
        self.features.to_csv(path, **kwargs)
    
        

def pixelflow(
    mask: npt.NDArray, 
    image: npt.NDArray, 
    *, 
    features: Optional[tuple[str]] = None,
    custom: Optional[Callable] = None,
) -> PixelflowResult:
    """Simple wrapper around `regionprops` to be extended or replaced.
    
    Parameters
    ----------
    mask : array 
    image : array
    features : tuple, optional 
    custom : tuple, optional

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
    features = regionprops_table(
        label(mask),
        image,
        properties=features,
        extra_properties=custom,
    )
    return PixelflowResult(features=pd.DataFrame(features))
