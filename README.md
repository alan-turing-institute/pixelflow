<h1 align="center">
  <img src="https://github.com/alan-turing-institute/pixelflow/assets/8217795/c1ccd706-3c6f-4a0a-840e-dff8fd00c282" width=400 />
</h1>

[![Continuous integration status badge](https://github.com/alan-turing-institute/pixelflow/actions/workflows/tests.yml/badge.svg)](https://github.com/alan-turing-institute/pixelflow/actions/workflows/tests.yml)
[![Licence badge (BSD 3 Clause)](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://github.com/alan-turing-institute/pixelflow/blob/main/LICENSE)

Pixelflow is a tool for extracting information about the characteristics of objects in images. Segmentation models can be used to identify objects, but researchers typically then want to obtain more details about these objects, e.g. their number, size, shape etc. Pixelflow aims to simplify this process, working with a mask from 2D or 3D images and outputting a dataframe of the desired characteristics.

It is part of the [scivision](https://sci.vision) library of tools for working with computer vision models and data.

## Installation
Pixelflow is not currently available through PyPI, but it can be installed directly from this github repo

```bash
pip install git+https://github.com/alan-turing-institute/pixelflow.git
```

## Usage

``` python
from pixelflow import pixelflow, pixelflow_custom

@pixelflow_custom
def custom_func(x: npt.NDArray, y: npt.NDArray) -> float:
    return -np.sum(x * y)

pixelflow(
    mask,
    img,
    features=('label', 'bbox', 'centroid', 'area', 'major_axis_length', 'orientation', 'image_intensity'),
    custom=(custom_func,)
    dim_labels="YX",
    labelled=True
)
```

A more detailed example usecase of pixelflow, extracting information on seed size and shape, can be found in the [scivision-gallery](https://github.com/scivision-gallery/pixelflow_seed_demo).

## Contributing
Contributions of any kind welcome! Please feel free to open a pull request or raise an issue.
