"""Core functions for the pixelflow package."""

import dataclasses
import importlib
import functools
import os
import warnings

from typing import Callable, Optional, Union
from skimage.measure import label, regionprops_table
from porespy.metrics import regionprops_3D, props_to_DataFrame

import numpy as np
import numpy.typing as npt
import pandas as pd

import rasterio
from rasterio.plot import reshape_as_image
import geopandas as gpd
from rasterio.features import rasterize
import re
import matplotlib.pyplot as plt 
import glob

Numeric = Union[int, float]


class PixelflowImportWarning(UserWarning):
    """Warning for missing packages."""


class PixelflowMaskWarning(UserWarning):
    """Warning for empty masks."""


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
    """Result container with additional `reduce` functionality
    and the ability to output the features as a .csv file."""

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
    masks: npt.NDArray|str,
    images: Optional[npt.NDArray|str] = None,
    *,
    features: Optional[tuple[str]] = None,
    custom: Optional[Callable] = None,
    dim_labels: Optional[str] = None,
    labelled: bool = False,
    img_coords: Optional[tuple[float]] = None,
    spacing: Optional[tuple[float]] = None,
) -> PixelflowResult:
    """Simple wrapper around `regionprops` to be extended or replaced.

    Parameters
    ----------
    mask : array, filepath or directory
        The segmentation mask to be analysed either as an array (Required for 3D),
        an image file (JPEG/TIFF/GEOTIFF), an ESRI shapefile or a directory 
        containing any of the above. 
        The objects must be distinguished
        from the background, but do not need to be labelled. If there are no
        objects in the mask, the function will return None.

    image : array, filepath or directory, optional
        An image (array, JPEG/TIFF/GEOTIFF), or directory containing an 
        equal number of images the same size and dimensions as the mask. If present, features
        such as maximum image intensity can be calculated, and the objects
        can be segmented from the image.
    features : tuple, optional
        Currently accepts all the features in regionprops
        (https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops)
        for 2D / 3D images and the features in regionprops_3D
        (https://porespy.org/modules/generated/generated/porespy.metrics.regionprops_3D.html)
        for 3D images
    custom : tuple, optional
        A custom function to be applied to the image. It accepts the mask / image
        arguments
    dim_labels : str, optional
        Dimension labels for the mask. Currently accepts YX or ZYX. Will be
        guessed based on mask dimensions if not supplied.
    labelled : bool
        Whether the individual objects in the mask are labelled. If not,
        they will be labelled using `skimage.measure.label`. Defaults to False.
    img_coords : tuple, optional
        The coordinates of the image in chosen units in the format
        (top, left, bottom, right), which will be used to calculate the
        pixel size
    spacing : tuple, optional
        The pixel size in the chosen units

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
    #set up output list
    results=[]
    # load the data (file_list and crs_list not currently used but 
    # may be useful in future)
    if not images == None: 
        mask_list, image_list, file_list, crs_list = load_data(masks, images, spacing)
    else:
        mask_list, file_list, crs_list = load_data(masks, images, spacing)

    #iterate over masks
    for it, mask in enumerate(mask_list): 

        #Get corresponding image if provided
        if 'image_list' in locals():
            image=image_list[it]
        else:
            image=None

        # check if mask contains any objects
        if mask.max() - mask.min() == 0:
            warnings.warn("The mask doesn't contain any objects.", PixelflowMaskWarning)
            return None

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

        # check if image is labelled
        mask = mask if labelled else label(mask)

        # if image coordinates are supplied calculate spacing and origin
        if spacing is None:
            if img_coords is not None:
                spacing = calc_spacing(img_coords, mask)
            else:
                spacing = (1,) * mask.ndim

        if any(val == 0 for val in spacing):
            raise ValueError(f"Spacing cannot be zero: {spacing}")

        # add warning for spacing and custom functions
        if any(val != 1 for val in spacing) and custom is not None:
            warnings.warn("Spacing may not work as expected for custom functions.")

        # If image is YX then use regionprops_table
        if dim_labels == "YX":
            if features is None:
                features = (
                    "label",
                    "bbox",
                    "centroid",
                    "area",
                    "major_axis_length",
                    "eccentricity",
                    "orientation",
                    "solidity",
                )
            features_dat = regionprops_table(
                mask,
                image,
                properties=features,
                extra_properties=custom,
                spacing=spacing,
            )
            features_df = pd.DataFrame(features_dat)

        # If image is ZYX then use regionprops_3D
        elif dim_labels == "ZYX":
            features_2d = ("label",)
            features_3d = (
                "label",
                "bbox_volume",
                "convex_volume",
                "sphericity",
                "surface_area",
                "volume",
                "border",
                "inscribed_sphere",
                "skeleton",
                "slices",
                "surface_mesh_simplices",
                "surface_mesh_vertices",
            )
            # if image_intensity is requested calculate it through regionprops_table
            if features is not None:
                features_3d = tuple(set(features).intersection(features_3d))
                features_2d += tuple(set(features).difference(features_3d))
            # calculate the regionprops features
            features_dat = regionprops_table(
                mask,
                image,
                properties=features_2d,
                extra_properties=custom,
                spacing=spacing,
            )
            features_df = pd.DataFrame(features_dat)

            # calculate the 3D features
            features_dat3d = regionprops_3D(mask)
            features_df3d = props_to_DataFrame(features_dat3d)

            # filter the dataframe to only include the requested features
            features_df3d = features_df3d[features_df3d.columns.intersection(features_3d)]

            # convert volume columns to correct spacing
            px_vol = np.prod(spacing)
            if px_vol != 1:
                # multiply the volume columns by the pixel volumne
                vol_col = features_df3d.columns.str.contains("volume")
                if vol_col.any():
                    features_df3d.loc[:, vol_col] *= px_vol

                # multiply the surface area columns by pixel area
                vol_sa = features_df3d.columns.str.contains("surface_area")
                if vol_sa.any():
                    if len(np.unique(spacing)) == 1:
                        features_df3d.loc[:, vol_sa] *= spacing[0] ** 2
                    else:
                        raise NotImplementedError(
                            "surface_area supports isotropic spacings only"
                        )

                # give error if trying to calculate sphericity for anisotropic images
                vol_sp = features_df3d.columns.str.contains("sphericity")
                if vol_sp.any():
                    if len(np.unique(spacing)) != 1:
                        raise NotImplementedError(
                            "sphericity supports isotropic spacings only"
                        )

            # combine the regionprops and 3D features
            features_df = pd.merge(features_df, features_df3d)

        else:
            raise ValueError("Image type unsupported, expected 'YX' or 'ZYX'")

        pf_result = PixelflowResult(features=features_df)

        # If image_intensity is requested, extract it
        if features is not None and "image_intensity" in features:
            features_img = pf_result.features.pop("image_intensity")
            pf_result.image_intensity = features_img

        results.append(pf_result)


    if len(results)>1:
        return results
    else:
        return pf_result


def calc_spacing(
    coords: tuple[float],
    mask: npt.NDArray,
) -> tuple[float]:
    """Calculate the pixel size for the image in the chosen units.

    Parameters
    ----------
    coords : tuple
        The coordinates of the image in chosen units in the format
        (top, left, bottom, right) for a 2D image
    mask : array

    Returns
    -------
    spacing : tuple
        The pixel size in the chosen units
    """

    # calculate the spacing based on corner coords and number of pixels for each dim
    spacing = tuple(
        round(abs((coords[i] - coords[i + mask.ndim]) / mask.shape[i]), 10)
        for i in range(mask.ndim)
    )

    # check if coords are small enough for issues when rounding to 10 decimal places
    if any(val < 1e-8 for val in spacing):
        warnings.warn(
            "Small pixel size may cause rounding errors, consider using finer units."
        )

    # return a tuple of the spacing
    return spacing


def calc_coords(
    in_coords: pd.DataFrame,
    coord_bound: tuple[float],
    spacing: tuple[float],
) -> pd.DataFrame:
    """Convert the coordinates from the pixel inputs to the desired units.

    Parameters
    ----------
    in_coords : tuple
        The coordinates to convert
    coord_bound : tuple
        The corner coordinates of the object in chosen units
        in the format of a bbox (top, left, bottom, right) for a 2D image
    spacing : tuple
        The pixel size in the chosen units

    Returns
    -------
    out_coords : tuple
    """

    # calculate the number of dimensions
    ndim = len(spacing)
    out_coords = [
        None,
    ] * ndim

    # for each dimension
    for i in range(ndim):
        # check whether the coordinate system increases or decreases for that dimension
        if coord_bound[i] < coord_bound[i + ndim]:
            # calculate the rescaled coordinates
            out_coords[i] = coord_bound[i] + in_coords.iloc[:, i] * spacing[i]
        else:
            out_coords[i] = coord_bound[i] - in_coords.iloc[:, i] * spacing[i]

    return tuple(out_coords)

#%
def load_data(
    mask, 
    image: Optional [tuple]= None,
    spacing: Optional[tuple[float]] = None
):
    """load mask(s) and/or image(s) from various sources
        Handles (lists of) numpy arrays, JPEG, TIFF, GeoTIFF, ESRI Shapefiles
        Shapefiles are converted by rasterization
        If passing shapefile required img to be GeoTIFF with matching CRS
        If passed a directory, loads all files in that directory - NB this may not be tractable for large datasets. 
        Only works for 2D data

    Parameters
    ----------
    image : array, list of arrays or str, optional
        image(s) or path to image file(s)
        The coordinates to convert
    mask : array, list of arrays or str, optional
        mask(s) or path to mask file(s)
    spacing : tuple, optional
        Only needed when shapefile mask passed without img
        The pixel size for rasterization in the crs units

    Returns
    -------
    mask_array : list of npt.ndarray
        mask(s) in array format
    img_array : list of npt.ndarray 
        image(s) in array format
    file_list : List 
        mask file names to be associated with output df
    crs : list 
        EPSG codes defining CRS of source data
    
    dev ambitions: 
    *Add 3D functionality for raster imports
    *crop to ROI (either bbox or vector mask)
    *Reproject mismatched data if both have valid EPSG
    *Inherit labels from shapefile (/vector) data if present 
        as 'ID' column

    """

    
    # list of supported file types (to be added to as implemented)
    f_types=['.jpg', '.jpeg', '.tif', '.tiff', '.shp']

    # mask
    if isinstance(mask, np.ndarray): 
        mask_array=[mask]
        mask_crs_list=[None]
        f_list=[None]
    elif os.path.isfile(mask):
        mask_array, mask_crs = load_and_convert_to_array(mask, spacing)
        mask_array=[mask_array]
        mask_crs_list=[mask_crs]
        f_list=[mask]
    elif os.path.isdir(mask):
        mask_array=[]
        mask_crs_list=[]
        f_list=[]
        for f in f_types:
            f_list = f_list+glob.glob(mask+'/*'+f)
        f_list = sort_nicely(f_list)
        
        for file in f_list:
            mask_arr, mask_crs = load_and_convert_to_array(file, spacing)
            mask_array.append(mask_arr)
            mask_crs_list.append(mask_crs)

    #strip paths from file names in file list
    file_list=[]
    for f in f_list: 
        file_list.append(os.path.split(f)[1])

    # img 
    if not image == None:
        if isinstance(image, np.ndarray): 
            img_array=[image]
            img_crs_list=[None]
        elif os.path.isfile(image):
            img_array, img_crs = load_and_convert_to_array(image, spacing)
            img_array=[img_array]
            img_crs_list=[img_crs]
        elif os.path.isdir(image):
            img_array=[]
            img_crs_list=[]
            f_list_i=[]
            for f in f_types:
                f_list_i=f_list_i+glob.glob(image+'/*'+f)
            f_list_i = sort_nicely(f_list_i)
            assert len(f_list_i)==len(f_list), 'Images supplied but number does not match number of masks'
            
            for file in f_list_i:
                img_arr, img_crs = load_and_convert_to_array(file, spacing)
                img_array.append(img_arr)  
                img_crs_list.append(img_crs)

        
        # Check image and mask shapes agree in XY dimensions
        for i, m in enumerate(mask_array):
            assert(m.shape[0:2]==img_array[i].shape[0:2]), 'mask and image shapes index '+str(i)+' do not agree'

        # Check CRS of images and masks agree where mask has a crs
        for i,c in enumerate(mask_crs_list):
            assert c==img_crs_list[i], 'CRS definitions for mask and image at index '+str(i)+' do not match'

        # TODO - reproject-warp image to match mask if mismatch found

        return mask_array, img_array, file_list, mask_crs_list
    else:
        return mask_array, file_list, mask_crs_list



def load_and_convert_to_array(filepath, spacing):
    _, file_extension = os.path.splitext(filepath.lower())
    if file_extension == '.jpeg' or file_extension == '.jpg':
        img = rasterio.open(filepath)
        img_array = img.read().transpose(1, 2, 0)
        img.close()
        mask_crs=None
        return np.squeeze(img_array), mask_crs
    
    elif file_extension == '.tif' or file_extension == '.tiff':
        with rasterio.open(filepath) as src:
            img_array = src.read().transpose(1, 2, 0)
            #try to get crs info
            try:
                mask_crs = src.crs.to_epsg()
            except:
                mask_crs=None

            img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        return np.squeeze(img_array), mask_crs
    
    elif file_extension == '.shp':
        # First check that a valid spacing has been provided
        assert not spacing==None, 'Conversion from vector requires spacing to be defined, no spacing supplied'
        # Read shapefile
        gdf = gpd.read_file(filepath)
        #try to get crs
        try: 
            mask_crs=gdf.crs.to_epsg()
        except:
            mask_crs=None
        # Create a blank raster to rasterize shapefile onto 
        # TODO - currently assumes CRS units are metres
        height=int(np.ceil((gdf.total_bounds[2]-gdf.total_bounds[0])/spacing[0]))
        width=int(np.ceil((gdf.total_bounds[3]-gdf.total_bounds[1])/spacing[1]))
        
        raster = np.zeros((height, width))
        # Rasterize the shapefile onto the blank raster
        #TODO inherit labels from shapefile if present
        rasterized = rasterize(
            shapes=gdf.geometry,
            out_shape=(height, width),
            transform=rasterio.transform.from_bounds(*gdf.total_bounds, width, height),
            fill=0,
            dtype=np.uint8
        )
        return np.squeeze(rasterized), mask_crs
    else:
        raise ValueError("Unsupported file format")


def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)
    return l
# %%
#mask='C:/Users/benevans/OneDrive/OneDrive - NERC/Documents/Repos/Pixelflow/pixelflow/data/masks'
#image='C:/Users/benevans/OneDrive/OneDrive - NERC/Documents/Repos/Pixelflow/pixelflow/data/images'
# %%
#mask='C:/Users/benevans/OneDrive/OneDrive - NERC/Documents/Repos/Pixelflow/pixelflow/data/masks/Iceberg_Example_5F90.tif'
#image='C:/Users/benevans/OneDrive/OneDrive - NERC/Documents/Repos/Pixelflow/pixelflow/data/images'

