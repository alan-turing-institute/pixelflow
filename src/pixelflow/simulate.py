import numpy as np
import numpy.typing as npt

RNG = np.random.default_rng()


def simulated_image(
    *, 
    size: tuple[int, int] = (256, 256), 
    num_blobs: int = 5
) -> npt.NDArray:
    """Create a simulated image with a number of blobs."""
    def dist_mat(centroid: tuple[float, float]) -> npt.NDArray:
        dist = np.zeros(size)
        x, y = np.meshgrid(
            np.linspace(-0.1, 1.1, size[0]) - centroid[0],
            np.linspace(-0.1, 1.1, size[1]) - centroid[1],
        )
        dist = 1.0 - np.log(np.sqrt(x*x + y*y))
        return dist
    
    centroids = np.stack(
        [RNG.uniform(0, 1, size=(num_blobs,)) for sz in size],
        axis=-1,
    )

    return np.mean(np.stack([dist_mat(c) for c in centroids], axis=0), axis=0)