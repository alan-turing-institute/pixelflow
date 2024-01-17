import pytest
import numpy as np

from skimage.measure import label, regionprops

from ._utils import simulate_image


@pytest.fixture(params=[2, 3])
def simulated_dataset(request):
    """Make a simulated dataset."""
    size = (256,) * request.param
    num_blobs = 6
    img, coords = simulate_image(size=size, num_blobs=num_blobs)
    mask = label(img > 2.2)
    # remove small objects
    tmp = np.unique(mask, return_counts=True)
    if any(tmp[1] < 5):
        for val in tmp[0][tmp[1] < 5]:
            mask[mask == val] = 0
        mask = label(mask)
    props = regionprops(mask)
    bbox = [prop.bbox for prop in props]
    return mask, img, coords, bbox
