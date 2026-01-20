import logging
import pathlib

import numpy as np
import pandas as pd

from cytodataframe.frame import CytoDataFrame
from cytodataframe.volume import (
    build_3d_image_html_view,
    extract_volume_from_ome_arrow,
)


def _fake_ome_arrow_volume() -> dict:
    return {
        "type": "ome.arrow",
        "pixels_meta": {
            "size_x": 2,
            "size_y": 2,
            "size_z": 2,
            "size_c": 1,
            "size_t": 1,
        },
        "planes": [
            {"z": 0, "c": 0, "t": 0, "pixels": [0, 1, 2, 3]},
            {"z": 1, "c": 0, "t": 0, "pixels": [4, 5, 6, 7]},
        ],
    }


def test_extract_volume_from_ome_arrow_builds_volume():
    logger = logging.getLogger(__name__)
    cdf = CytoDataFrame(pd.DataFrame({"A": [1]}))
    volume_data = extract_volume_from_ome_arrow(
        _fake_ome_arrow_volume(),
        cdf._ensure_uint8,
        cdf._is_ome_arrow_value,
        logger,
    )

    assert volume_data is not None
    volume, dims = volume_data
    assert dims == (2, 2, 2)
    assert volume.shape == (2, 2, 2)
    assert volume[0, 0, 0] == 0
    assert volume[1, 1, 1] == 7


def test_build_3d_image_html_view_contains_vtk_script():
    volume = np.zeros((2, 2, 2), dtype=np.uint8)
    html = build_3d_image_html_view(
        volume=volume,
        dims=(2, 2, 2),
        data_value="volume.tiff",
        candidate_path=pathlib.Path("volume.tiff"),
        display_options={"width": "120px", "height": "120px"},
    )

    assert 'class="cyto-3d-image"' in html
    assert "data-volume=" in html
    assert "vtk.js" in html


def test_process_ome_arrow_volume_returns_vtk_html():
    cdf = CytoDataFrame(pd.DataFrame({"A": [1]}))
    html = cdf.process_ome_arrow_data_as_html_display(_fake_ome_arrow_volume())

    assert 'class="cyto-3d-image"' in html
    assert "vtk.js" in html
