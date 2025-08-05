<img height="200" src="https://raw.githubusercontent.com/cytomining/cytodataframe/main/logo/with-text-for-light-bg.png?raw=true">

# CytoDataFrame

[![PyPI - Version](https://img.shields.io/pypi/v/cytodataframe)](https://pypi.org/project/CytoDataFrame/)
[![Build Status](https://github.com/cytomining/CytoDataFrame/actions/workflows/run-tests.yml/badge.svg?branch=main)](https://github.com/cytomining/CytoDataFrame/actions/workflows/run-tests.yml?query=branch%3Amain)
![Coverage Status](https://raw.githubusercontent.com/cytomining/CytoDataFrame/main/media/coverage-badge.svg)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Poetry](https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json)](https://python-poetry.org/)
[![Software DOI badge](https://zenodo.org/badge/DOI/10.5281/zenodo.14797074.svg)](https://doi.org/10.5281/zenodo.14797074)

![](https://raw.githubusercontent.com/cytomining/coSMicQC/refs/heads/main/docs/presentations/2024-09-18-SBI2-Conference/images/cosmicqc-example-cytodataframe.png)
_CytoDataFrame extends Pandas functionality to help display single-cell profile data alongside related images._

CytoDataFrame is an advanced in-memory data analysis format designed for single-cell profiling, integrating not only the data profiles but also their corresponding microscopy images and segmentation masks.
Traditional single-cell profiling often excludes the associated images from analysis, limiting the scope of research.
CytoDataFrame bridges this gap, offering a purpose-built solution for comprehensive analysis that incorporates both the data and images, empowering more detailed and visual insights in single-cell research.

CytoDataFrame is best suited for work within Jupyter notebooks.
With CytoDataFrame you can:

- View image objects alongside their feature data using a Pandas DataFrame-like interface.
- Highlight image objects using mask or outline files to understand their segmentation.
- Adjust image displays on-the-fly using interactive slider widgets.

📓 ___Want to see CytoDataFrame in action?___ Check out our [example notebook](docs/src/examples/cytodataframe_at_a_glance.ipynb) for a quick tour of its key features.

> ✨ CytoDataFrame development began within **[coSMicQC](https://github.com/cytomining/coSMicQC)** - a single-cell profile quality control package.
> Please check out our work there as well!

## Installation

Install CytoDataFrame from source using the following:

```shell
# install from pypi
pip install cytodataframe

# or install directly from source
pip install git+https://github.com/cytomining/CytoDataFrame.git
```

## Contributing, Development, and Testing

Please see our [contributing](https://cytomining.github.io/CytoDataFrame/main/contributing) documentation for more details on contributions, development, and testing.

## References

- [coSMicQC](https://github.com/cytomining/coSMicQC)
- [pycytominer](https://github.com/cytomining/pycytominer)
- [CellProfiler](https://github.com/CellProfiler/CellProfiler)
- [CytoTable](https://github.com/cytomining/CytoTable)
