import numpy as np
import numpy.typing as npt

RNG = np.random.default_rng(seed=12345)


def simulate_image(
    *, size: tuple[int] = (256, 256), num_blobs: int = 5
) -> tuple[npt.NDArray, npt.NDArray]:
    """Create a simulated image with a number of blobs."""

    ndim = len(size)

    def dist_mat(centroid: tuple[float]) -> npt.NDArray:
        dist = np.zeros(size)

        coords = [
            np.linspace(-0.1, 1.1, size[ax]) - centroid[ax] for ax in range(dist.ndim)
        ]

        xyz = np.meshgrid(*coords)
        dist = 1.0 - np.log(np.sqrt(sum(ax * ax for ax in xyz)))
        return dist

    centroids = np.stack(
        [RNG.uniform(0, 1, size=(num_blobs,)) for sz in size],
        axis=-1,
    )

    img = np.mean(np.stack([dist_mat(c) for c in centroids], axis=0), axis=0)

    assert img.ndim == ndim

    return img, centroids
