"""Tests for core functionality of pixelflow."""

import pytest
from skimage.measure import label, regionprops_table
import numpy as np
import pandas as pd

import pixelflow
from ._utils import simulate_image


def test_core_count(simulated_dataset):
    """Test counting using a simulated 2D or 3D image."""
    mask, img, coords, bbox = simulated_dataset

    num_blobs = len(bbox)
    assert img.ndim == coords.shape[-1]

    result = pixelflow.pixelflow(
        mask,
        img,
        features=("label",),
    )
    assert isinstance(result, pixelflow.core.PixelflowResult)
    assert result.count() == num_blobs


@pytest.mark.parametrize("dim", (2, 3))
def test_core_dim_labels(dim):
    """Test dimension guessing flags incorrect number"""
    size = (128,) * dim
    img, coords = simulate_image(size=size, num_blobs=5)
    mask = label(img > 2.3)
    with pytest.raises(ValueError, match=r"dim_labels doesn't match mask dimensions"):
        pixelflow.pixelflow(mask, features=("bbox",), dim_labels="X" * (dim + 1))


@pytest.mark.parametrize("dim", (2, 3))
def test_core_no_objects(dim):
    """Test mask with no objects gets flagged correctly"""
    size = (128,) * dim
    mask = np.zeros(size)
    with pytest.warns(UserWarning, match=r"The mask doesn't contain any objects"):
        result = pixelflow.pixelflow(mask)
    assert result is None


# tests to check that the spacing parameter works as expected
def test_core_no_spacing(simulated_dataset):
    """Test whether default spacing works as expected, e.g. (1,1)."""
    mask, img, coords, bbox = simulated_dataset
    result1 = pixelflow.pixelflow(
        mask,
        img,
        features=("label", "bbox", "centroid", "area", "major_axis_length", "extent"),
    )
    result2 = pixelflow.pixelflow(
        mask,
        img,
        features=("label", "bbox", "centroid", "area", "major_axis_length", "extent"),
        spacing=(1,) * mask.ndim,
    )
    pd.testing.assert_frame_equal(result1.features, result2.features)


@pytest.mark.parametrize("pixel", ((0.4, 0.4), (0.2, 0.3)))
def test_core_area_spacing(simulated_dataset, pixel):
    """Test whether spacing works as expected for area / volume."""
    mask, img, coords, bbox = simulated_dataset
    if mask.ndim == 2:
        features = (
            "label",
            "area",
        )
    else:
        features = (
            "label",
            "volume",
        )
        pixel = pixel + (0.4,)

    # pixelflow calculation with spacing
    result1 = pixelflow.pixelflow(
        mask,
        img,
        features=features,
        spacing=pixel,
    )
    # spacing calculated separately
    result2 = pixelflow.pixelflow(
        mask,
        img,
        features=features,
    )
    px_vol = np.prod(pixel)
    result2.features[features[1]] *= px_vol
    pd.testing.assert_frame_equal(result1.features, result2.features)


def test_core_sa_iso_spacing(simulated_dataset):
    """Test whether isotropic spacing works as expected for surface area / perimeter."""
    mask, img, coords, bbox = simulated_dataset
    if mask.ndim == 2:
        features = (
            "label",
            "perimeter",
        )
    else:
        features = (
            "label",
            "surface_area",
        )

    pixel = (0.4,) * mask.ndim

    # pixelflow calculation with spacing
    result1 = pixelflow.pixelflow(
        mask,
        img,
        features=features,
        spacing=pixel,
    )
    # spacing calculated separately
    result2 = pixelflow.pixelflow(
        mask,
        img,
        features=features,
    )
    px_vol = np.prod(pixel[:-1])
    result2.features[features[1]] *= px_vol
    pd.testing.assert_frame_equal(result1.features, result2.features)


def test_core_sa_aniso_spacing(simulated_dataset):
    """Test if anisotropic spacing gives expected error for surface area/perimeter"""
    mask, img, coords, bbox = simulated_dataset
    if mask.ndim == 2:
        features = (
            "label",
            "perimeter",
        )
        pixel = (0.3, 0.4)

    else:
        features = (
            "label",
            "surface_area",
        )
        pixel = (0.2, 0.3, 0.4)

    with pytest.raises(NotImplementedError, match=r"supports isotropic spacings only"):
        pixelflow.pixelflow(
            mask,
            img,
            features=features,
            spacing=pixel,
        )


@pytest.mark.parametrize("pixel", ((0.4, 0.4), (0.2, 0.3)))
def test_core_length_spacing(simulated_dataset, pixel):
    """Test whether spacing works as expected for length."""
    mask, img, coords, bbox = simulated_dataset

    features = (
        "label",
        "major_axis_length",
    )

    if mask.ndim == 3:
        pixel = pixel + (0.4,)

    # pixelflow calculation with spacing
    result1 = pixelflow.pixelflow(
        mask,
        img,
        features=features,
        spacing=pixel,
    )
    # spacing calculated separately
    result2 = regionprops_table(
        mask,
        img,
        properties=features,
        spacing=pixel,
    )
    pd.testing.assert_frame_equal(result1.features, pd.DataFrame(result2))


def test_core_ratio_spacing(simulated_dataset):
    """Test whether spacing works as expected for sphericity / eccentricity."""
    mask, img, coords, bbox = simulated_dataset
    if mask.ndim == 2:
        features = (
            "label",
            "eccentricity",
        )
        pixel = (0.4, 0.4)
    else:
        features = (
            "label",
            "sphericity",
        )
        pixel = (0.4, 0.4, 0.4)

    # pixelflow calculation with spacing
    result1 = pixelflow.pixelflow(
        mask,
        img,
        features=features,
        spacing=pixel,
    )
    # spacing not calculated
    result2 = pixelflow.pixelflow(
        mask,
        img,
        features=features,
    )

    pd.testing.assert_frame_equal(result1.features, result2.features)


def test_core_labels(simulated_dataset):
    """Test whether the labelled tag works as expected."""
    mask, img, coords, bbox = simulated_dataset

    # rotate mask to change label order
    mask = np.ndarray.transpose(mask)

    features = ("label", "area", "equivalent_diameter")

    # run pixelflow with label tag
    result1 = pixelflow.pixelflow(
        mask,
        features=features,
        labelled=True,
    )

    # run regionprops
    result2 = regionprops_table(
        mask,
        properties=features,
    )

    pd.testing.assert_frame_equal(
        result1.features[list(features)], pd.DataFrame(result2)[list(features)]
    )


def test_core_no_labels(simulated_dataset):
    """Test whether labelling works as expected."""
    mask, img, coords, bbox = simulated_dataset

    features = ("label", "area", "equivalent_diameter")

    result1 = pixelflow.pixelflow(
        mask,
        features=features,
    )

    # remove labels and run again
    mask[mask > 1] = 1

    result2 = pixelflow.pixelflow(
        mask,
        features=features,
    )

    pd.testing.assert_frame_equal(result1.features, result2.features)


def test_core_features(simulated_dataset):
    """Test whether not specifying features works as expected."""
    mask, img, coords, bbox = simulated_dataset

    if mask.ndim == 2:
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
    else:
        features = (
            "label",
            "volume",
            "bbox_volume",
            "sphericity",
            "surface_area",
            "convex_volume",
        )

    # run pixelflow with label tag
    result1 = pixelflow.pixelflow(
        mask,
        features=features,
    )

    # run regionprops
    result2 = pixelflow.pixelflow(
        mask,
    )

    pd.testing.assert_frame_equal(result1.features, result2.features)


def test_core_calc_spacing_2d():
    """Test whether the spacing is calculated as expected for 2D images."""
    mask = np.zeros((200, 200))

    assert pixelflow.calc_spacing((30, 10, 10, 30), mask) == (0.1, 0.1)
    assert pixelflow.calc_spacing((10, 10, 30, 30), mask) == (0.1, 0.1)
    assert pixelflow.calc_spacing((10, 10, -10, -10), mask) == (0.1, 0.1)
    assert pixelflow.calc_spacing((10, 10, 30, 20), mask) == (0.1, 0.05)
    assert pixelflow.calc_spacing((10, 10, 20, 30), mask) == (0.05, 0.1)
    with pytest.warns(UserWarning, match=r"Small pixel size may cause rounding errors"):
        result = pixelflow.calc_spacing((1e-8, 1e-8, 3e-8, 3e-8), mask)
    assert result == (1e-10, 1e-10)
    with pytest.warns(UserWarning, match=r"Small pixel size may cause rounding errors"):
        result1 = pixelflow.calc_spacing((10, 30, 10, 30), mask)
    assert result1 == (0.0, 0.0)


def test_core_calc_spacing_3d():
    """Test whether the spacing is calculated as expected for 3D images."""
    mask = np.zeros((200, 200, 200))

    assert pixelflow.calc_spacing((30, 10, 10, 10, 30, 30), mask) == (0.1, 0.1, 0.1)
    assert pixelflow.calc_spacing((10, 10, 10, 30, 30, 30), mask) == (0.1, 0.1, 0.1)
    assert pixelflow.calc_spacing((10, 10, 10, 30, 30, 20), mask) == (0.1, 0.1, 0.05)
    assert pixelflow.calc_spacing((10, 10, 10, 30, 20, 30), mask) == (0.1, 0.05, 0.1)
    assert pixelflow.calc_spacing((10, 10, 10, 20, 30, 30), mask) == (0.05, 0.1, 0.1)
    with pytest.warns(UserWarning, match=r"Small pixel size may cause rounding errors"):
        result = pixelflow.calc_spacing((1e-8, 1e-8, 1e-8, 3e-8, 3e-8, 3e-8), mask)
    assert result == (1e-10, 1e-10, 1e-10)
    with pytest.warns(UserWarning, match=r"Small pixel size may cause rounding errors"):
        result1 = pixelflow.calc_spacing((10, 20, 30, 10, 20, 30), mask)
    assert result1 == (0.0, 0.0, 0.0)


def test_core_zero_spacing(simulated_dataset):
    """Test 0 spacing is flagged correctly."""
    mask, img, coords, bbox = simulated_dataset

    spacing = (0,) * mask.ndim

    with pytest.raises(ValueError, match=r"Spacing cannot be zero"):
        pixelflow.pixelflow(mask, spacing=spacing)
