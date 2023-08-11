import pytest
import pixelflow


@pytest.mark.parametrize("simulated_dataset", (2, 3), indirect=True)
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
