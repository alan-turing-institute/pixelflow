"""Tests for core functionality of pixelflow."""

import pytest
from skimage.measure import label
import numpy as np
import warnings

import pixelflow
from ._utils import simulate_image


def test_core_count(simulated_dataset):
    """Test counting using a simulated 2D or 3D image."""
    mask, img, coords, bbox = simulated_dataset

    num_blobs = coords.shape[0]
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
    