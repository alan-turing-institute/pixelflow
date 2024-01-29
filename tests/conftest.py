import pytest

from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects

from ._utils import simulate_image


@pytest.fixture(params=[2, 3])
def simulated_dataset(request):
    """Make a simulated dataset."""
    size = (256,) * request.param
    num_blobs = 6
    img, coords = simulate_image(size=size, num_blobs=num_blobs)
    mask = label(img > 2.2)
    # remove small objects
    if mask.max() > 1:
        mask = remove_small_objects(mask, min_size=5)
        mask = label(mask)
    props = regionprops(mask)
    bbox = [prop.bbox for prop in props]
    return mask, img, coords, bbox
