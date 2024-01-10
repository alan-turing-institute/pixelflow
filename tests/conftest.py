import pytest
import numpy as np

from skimage.measure import label, regionprops

from ._utils import simulate_image


@pytest.fixture(params=[2, 3])
def simulated_dataset(request):
    """Make a simulated dataset."""
    size = (128,) * request.param
    num_blobs = 5
    img, coords = simulate_image(size=size, num_blobs=num_blobs)
    mask = label(img > 2.3)
    props = regionprops(mask)
    bbox = [prop.bbox for prop in props]
    return mask, img, coords, bbox
