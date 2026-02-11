import logging
import pathlib
import types
from importlib.machinery import ModuleSpec

import numpy as np
import pandas as pd
import pytest

from cytodataframe.frame import CytoDataFrame
from cytodataframe.volume import (
    build_3d_html_from_path,
    build_3d_image_html_stub,
    build_3d_image_html_view,
    build_3d_vtk_js_initializer,
    build_3d_vtk_js_script,
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


def test_build_3d_image_html_view_uses_stub_when_inline_volume_too_large():
    volume = np.zeros((8, 8, 8), dtype=np.uint8)
    html = build_3d_image_html_view(
        volume=volume,
        dims=(8, 8, 8),
        data_value="volume.tiff",
        candidate_path=pathlib.Path("volume.tiff"),
        display_options={"max_inline_volume_bytes": 16},
    )

    assert 'class="cyto-3d-image"' in html
    assert "data-volume=" not in html
    assert "too large for inline rendering" in html


def test_build_3d_image_html_view_defaults_when_limit_value_invalid():
    volume = np.zeros((2, 2, 2), dtype=np.uint8)
    html = build_3d_image_html_view(
        volume=volume,
        dims=(2, 2, 2),
        data_value="volume.tiff",
        candidate_path=pathlib.Path("volume.tiff"),
        display_options={"max_inline_volume_bytes": "not-an-int"},
    )

    assert "data-volume=" in html


def test_process_ome_arrow_volume_returns_vtk_html():
    cdf = CytoDataFrame(pd.DataFrame({"A": [1]}))
    html = cdf.process_ome_arrow_data_as_html_display(_fake_ome_arrow_volume())

    assert 'class="cyto-3d-image"' in html
    assert "vtk.js" in html


def test_build_3d_image_html_stub_respects_none_height():
    html = build_3d_image_html_stub(
        data_value="vol.tiff",
        candidate_path=pathlib.Path("vol.tiff"),
        display_options={"width": "111px", "height": None},
    )
    assert "width:111px" in html
    assert "height:" not in html
    assert "3D image" in html


def test_build_3d_image_html_stub_includes_default_height():
    html = build_3d_image_html_stub(
        data_value="vol.tiff",
        candidate_path=pathlib.Path("vol.tiff"),
        display_options={"width": "90px"},
    )
    assert "height:90px" in html


def test_vtk_js_helpers_include_expected_hooks():
    script = build_3d_vtk_js_script("abc")
    initializer = build_3d_vtk_js_initializer()
    assert "https://unpkg.com/@kitware/vtk.js@34.9.1/dist/vtk.js" in script
    assert "document.getElementById('abc')" in script
    assert "querySelectorAll('.cyto-3d-image[data-volume][data-dims]')" in initializer
    for token in (
        "ctfun.addRGBPoint(1,1,1,1);",
        "ctfun.addRGBPoint(255,1,1,1);",
        "ofun.addPoint(1,0.15);",
        "ofun.addPoint(255,0.2);",
    ):
        assert token in script
        assert token in initializer


def test_vtk_js_helpers_allow_custom_url_override():
    custom_url = "https://example.com/vtk-local.js"
    script = build_3d_vtk_js_script("abc", vtk_js_url=custom_url)
    initializer = build_3d_vtk_js_initializer(
        display_options={"vtk_js_url": custom_url}
    )
    assert custom_url in script
    assert custom_url in initializer


def test_build_3d_image_html_view_uses_env_vtk_js_url(
    monkeypatch: pytest.MonkeyPatch,
):
    custom_url = "https://example.com/vtk-env.js"
    monkeypatch.setenv("CYTODATAFRAME_VTK_JS_URL", custom_url)
    volume = np.zeros((2, 2, 2), dtype=np.uint8)
    html = build_3d_image_html_view(
        volume=volume,
        dims=(2, 2, 2),
        data_value="volume.tiff",
        candidate_path=pathlib.Path("volume.tiff"),
        display_options={},
    )
    assert custom_url in html


def test_extract_volume_from_ome_arrow_returns_none_for_invalid_inputs():
    cdf = CytoDataFrame(pd.DataFrame({"A": [1]}))
    logger = logging.getLogger(__name__)
    assert (
        extract_volume_from_ome_arrow(
            {"not": "ome"},
            cdf._ensure_uint8,
            cdf._is_ome_arrow_value,
            logger,
        )
        is None
    )
    assert (
        extract_volume_from_ome_arrow(
            {
                "type": "ome.arrow",
                "pixels_meta": {"size_x": 2, "size_y": 2, "size_z": 1},
                "planes": [],
            },
            cdf._ensure_uint8,
            cdf._is_ome_arrow_value,
            logger,
        )
        is None
    )


def test_extract_volume_from_ome_arrow_filters_invalid_planes():
    cdf = CytoDataFrame(pd.DataFrame({"A": [1]}))
    logger = logging.getLogger(__name__)
    data_value = {
        "type": "ome.arrow",
        "pixels_meta": {"size_x": 2, "size_y": 2, "size_z": 3},
        "planes": np.array(
            [
                {"z": 0, "c": 1, "t": 0, "pixels": [0, 1, 2, 3]},
                {"z": 1, "c": 0, "t": 0, "pixels": [4, 5, 6, 7]},
                {"z": 2, "c": 0, "t": 0, "pixels": [8, 9]},
            ],
            dtype=object,
        ),
    }
    volume_data = extract_volume_from_ome_arrow(
        data_value,
        cdf._ensure_uint8,
        cdf._is_ome_arrow_value,
        logger,
    )
    assert volume_data is not None
    volume, dims = volume_data
    assert dims == (2, 2, 3)
    assert int(volume[1, 1, 1]) == 7


def test_extract_volume_from_ome_arrow_logs_debug_on_exception():
    cdf = CytoDataFrame(pd.DataFrame({"A": [1]}))
    messages = []

    class FakeLogger:
        def debug(self, msg: str, *args: object) -> None:
            messages.append(msg % args if args else msg)

    volume_data = extract_volume_from_ome_arrow(
        {
            "type": "ome.arrow",
            "pixels_meta": {"size_x": "bad", "size_y": 2, "size_z": 2},
            "planes": [],
        },
        cdf._ensure_uint8,
        cdf._is_ome_arrow_value,
        FakeLogger(),
    )
    assert volume_data is None
    assert any("Unable to decode 3D OME-Arrow struct" in msg for msg in messages)


def test_build_3d_html_from_path_without_ome_arrow_returns_none():
    cdf = CytoDataFrame(pd.DataFrame({"A": [1]}))
    messages = []

    class FakeLogger:
        def debug(self, msg: str, *args: object) -> None:
            messages.append(msg % args if args else msg)

    original_import = __import__

    def fake_import(name: str, *args: object, **kwargs: object):  # noqa: ANN202
        if name == "ome_arrow":
            raise ImportError("missing ome_arrow")
        return original_import(name, *args, **kwargs)

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr("builtins.__import__", fake_import)
        html = build_3d_html_from_path(
            data_value="x",
            candidate_path=pathlib.Path("nope"),
            display_options={},
            ensure_uint8=cdf._ensure_uint8,
            is_ome_arrow_value=cdf._is_ome_arrow_value,
            logger=FakeLogger(),
        )
    assert html is None
    assert any("ome-arrow not available" in msg for msg in messages)


def test_extract_volume_from_ome_arrow_returns_none_when_no_valid_planes():
    cdf = CytoDataFrame(pd.DataFrame({"A": [1]}))
    logger = logging.getLogger(__name__)
    data_value = {
        "type": "ome.arrow",
        "pixels_meta": {"size_x": 2, "size_y": 2, "size_z": 2},
        "planes": [None, {"z": 3, "c": 0, "t": 0, "pixels": [1, 2, 3, 4]}, {"z": 1}],
    }
    assert (
        extract_volume_from_ome_arrow(
            data_value,
            cdf._ensure_uint8,
            cdf._is_ome_arrow_value,
            logger,
        )
        is None
    )


def test_build_3d_image_html_view_fallback_handles_percentile_and_write_errors(
    monkeypatch: pytest.MonkeyPatch,
):
    volume = np.ones((2, 2, 2), dtype=np.uint8)

    def raise_percentile_error(*_args: object) -> tuple[float, float]:
        raise ValueError("bad")

    def raise_write_error(*_args: object, **_kwargs: object) -> None:
        raise ValueError("write fail")

    monkeypatch.setattr("cytodataframe.volume.np.percentile", raise_percentile_error)
    monkeypatch.setattr("cytodataframe.volume.imageio.imwrite", raise_write_error)
    html = build_3d_image_html_view(
        volume=volume,
        dims=(2, 2, 2),
        data_value="volume.tiff",
        candidate_path=pathlib.Path("volume.tiff"),
        display_options={"width": "120px", "height": "120px"},
    )
    assert 'class="cyto-3d-image"' in html
    assert '<img class="cyto-3d-fallback"' not in html


def test_build_3d_html_from_path_handles_ome_arrow_load_failure(
    monkeypatch: pytest.MonkeyPatch,
):
    cdf = CytoDataFrame(pd.DataFrame({"A": [1]}))
    messages = []

    class FakeLogger:
        def debug(self, msg: str, *args: object) -> None:
            messages.append(msg % args if args else msg)

    class FailingOMEArrow:
        def __init__(self, data: str) -> None:
            raise RuntimeError("boom")

    fake_module = types.SimpleNamespace(
        OMEArrow=FailingOMEArrow,
        __spec__=ModuleSpec("ome_arrow", loader=None),
    )
    monkeypatch.setitem(__import__("sys").modules, "ome_arrow", fake_module)
    html = build_3d_html_from_path(
        data_value="x",
        candidate_path=pathlib.Path("x"),
        display_options={},
        ensure_uint8=cdf._ensure_uint8,
        is_ome_arrow_value=cdf._is_ome_arrow_value,
        logger=FakeLogger(),
    )
    assert html is None
    assert any("Failed to load OME-Arrow for 3D rendering" in msg for msg in messages)


def test_build_3d_html_from_path_with_fake_ome_arrow(monkeypatch: pytest.MonkeyPatch):
    cdf = CytoDataFrame(pd.DataFrame({"A": [1]}))

    class FakeOMEArrow:
        def __init__(self, data: str) -> None:
            self.data = _fake_ome_arrow_volume()

    fake_module = types.SimpleNamespace(
        OMEArrow=FakeOMEArrow,
        __spec__=ModuleSpec("ome_arrow", loader=None),
    )
    monkeypatch.setitem(__import__("sys").modules, "ome_arrow", fake_module)
    html = build_3d_html_from_path(
        data_value="volume.tiff",
        candidate_path=pathlib.Path("volume.tiff"),
        display_options={"width": "120px", "height": "120px"},
        ensure_uint8=cdf._ensure_uint8,
        is_ome_arrow_value=cdf._is_ome_arrow_value,
        logger=logging.getLogger(__name__),
    )
    assert html is not None
    assert 'class="cyto-3d-image"' in html


def test_build_3d_html_from_path_returns_none_when_volume_decode_fails(
    monkeypatch: pytest.MonkeyPatch,
):
    cdf = CytoDataFrame(pd.DataFrame({"A": [1]}))

    class BadOMEArrow:
        def __init__(self, data: str) -> None:
            self.data = {"type": "ome.arrow", "pixels_meta": {"size_x": 2}}

    fake_module = types.SimpleNamespace(
        OMEArrow=BadOMEArrow,
        __spec__=ModuleSpec("ome_arrow", loader=None),
    )
    monkeypatch.setitem(__import__("sys").modules, "ome_arrow", fake_module)
    html = build_3d_html_from_path(
        data_value="x",
        candidate_path=pathlib.Path("x"),
        display_options={},
        ensure_uint8=cdf._ensure_uint8,
        is_ome_arrow_value=cdf._is_ome_arrow_value,
        logger=logging.getLogger(__name__),
    )
    assert html is None
