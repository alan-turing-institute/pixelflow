"""Tests for core functionality of pixelflow."""

import pytest
from skimage.measure import label
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
    pd.testing.assert_frame_equal(
        result1.features[list(features)], result2.features[list(features)]
    )


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
    pd.testing.assert_frame_equal(
        result1.features[list(features)], result2.features[list(features)]
    )


def test_core_sa_aniso_spacing(simulated_dataset):
    """Test whether anisotropic spacing produces expected error for surface area / perimeter."""
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
