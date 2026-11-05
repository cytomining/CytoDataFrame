# Examples

Featured example:
[OME Arrow and CytoDataFrame](examples/omearrow_and_cytodataframe)

## marimo

[CytoDataFrame in marimo](examples/marimo_and_cytodataframe) follows all 26
display examples in [CytoDataFrame at a Glance](examples/cytodataframe_at_a_glance):
JUMP, NF1, nuclear speckles, pediatric cancer atlas, OME Parquet round trips,
and 3D nuclei with and without segmentation masks. It also includes reactive
brightness, row-count, texture-filter, and 3D mask-opacity controls. The docs
show an executed, static snapshot; run the Python notebook for interactivity.

Download the {download}`marimo notebook <examples/marimo_and_cytodataframe.py>`
and run it from a checkout of the `0.3.x` branch:

```sh
uv run --with marimo==0.25.2 marimo edit docs/src/examples/marimo_and_cytodataframe.py
```

Regenerate the docs snapshot after editing the Python notebook (do not use
Jupytext to synchronize this pair: the Python source is a native marimo app):

```sh
uv run --with marimo==0.25.2 marimo export ipynb \
  docs/src/examples/marimo_and_cytodataframe.py --include-outputs \
  -o docs/src/examples/marimo_and_cytodataframe.ipynb -f -- -static-3d true
uv run --with ruff==0.16.10 ruff format docs/src/examples/marimo_and_cytodataframe.ipynb
```

This example uses an explicit HTML bridge, not CytoDataFrame's Jupyter widgets.
Live 3D uses CytoDataFrame's vtk.js renderer inside iframes; it requires CDN
access and uses grayscale-on-black, percentile-normalized uint8 intensities and
a binary green mask overlay. Each masked image has an independent **Show mask**
checkbox, which preserves its camera and leaves the other images unchanged.
It supports rotation and zoom, but does not reproduce PyVista's
per-object label colors, center marker, or full toolbar. The saved notebook
uses PyVista PNGs for both 3D examples so GitHub and the docs do not need
JavaScript or a running Python process. Those PNGs are generated only by the
standalone export: invoking native VTK rendering in marimo's worker thread can
abort Python on macOS. The `-static-3d true` export argument
is required: marimo's execution mode alone does not identify ipynb exports.

Run the parity tests, which compare all examples against the original notebook:

```sh
uv run pytest tests/test_marimo_example.py
```

```{toctree}
---
caption: "List of examples"
maxdepth: 2
glob:
---
examples/*
```
