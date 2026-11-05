"""Exercise the real marimo example cells with and without reactive overrides.

Run with: uv run --with marimo==0.25.2 pytest tests/test_marimo_example.py
"""

import ast
import base64
import io
import json
import pathlib
import re
import runpy
from html import unescape
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from cytodataframe.frame import CytoDataFrame

EXAMPLE = (
    Path(__file__).resolve().parents[1]
    / "docs/src/examples/marimo_and_cytodataframe.py"
)


@pytest.fixture(scope="module")
def notebook():
    return runpy.run_path(str(EXAMPLE))


@pytest.mark.parametrize(
    ("rows", "brightness", "outlines", "composite"),
    [(1, 10, False, False), (3, 50, False, False), (5, 90, True, True)],
)
def test_marimo_image_table(
    notebook: dict, rows: int, brightness: int, outlines: bool, composite: bool
):
    _, definitions = notebook["image_table"].run(
        row_count=SimpleNamespace(value=rows),
        brightness=SimpleNamespace(value=brightness),
        outlines=SimpleNamespace(value=outlines),
        composite=SimpleNamespace(value=composite),
    )
    html = definitions["table_html"]
    assert len(definitions["frame"]) == rows
    assert html.count("<img") == rows * (3 if composite else 2)
    assert "Image_Composite" in html if composite else "Image_Composite" not in html
    _assert_embedded_images(html)


def _assert_embedded_images(html: str) -> None:
    images = re.findall(r"data:image/[^;]+;base64,([^\"']+)", html)
    assert len(images) == html.count("<img")
    for encoded in images:
        with Image.open(io.BytesIO(base64.b64decode(encoded))) as image:
            image.verify()


def test_marimo_controls_change_pixels(notebook: dict):
    def render(brightness: int, outlines: bool) -> list[str]:
        _, definitions = notebook["image_table"].run(
            row_count=SimpleNamespace(value=1),
            brightness=SimpleNamespace(value=brightness),
            outlines=SimpleNamespace(value=outlines),
            composite=SimpleNamespace(value=False),
        )
        return re.findall(
            r"data:image/[^;]+;base64,([^\"']+)", definitions["table_html"]
        )

    baseline = render(50, False)
    assert baseline != render(90, False)
    assert baseline != render(50, True)


@pytest.fixture(scope="module")
def reference_frames(tmp_path_factory: pytest.TempPathFactory):
    """Execute the actual at-a-glance cells as the independent parity oracle."""
    data = EXAMPLE.parents[3] / "tests/data"
    snapshot = json.loads(
        EXAMPLE.with_name("cytodataframe_at_a_glance.ipynb").read_text()
    )
    namespace = {
        "CytoDataFrame": CytoDataFrame,
        "pathlib": pathlib,
        "pd": pd,
        "jump_data_path": str(data / "cytotable/JUMP_plate_BR00117006"),
        "nf1_cellpainting_path": str(data / "cytotable/NF1_cellpainting_data_shrunken"),
        "nuclear_speckles_path": str(data / "cytotable/nuclear_speckles"),
        "pediatric_cancer_atlas_path": str(
            data / "cytotable/pediatric_cancer_atlas_profiling"
        ),
        "cp_3d_path": str(data / "CP_tutorial_3D_noise_nuclei_segmentation"),
    }
    temporary_file = tmp_path_factory.mktemp("marimo-parity") / "example.ome.parquet"
    frames = {}
    for index, cell in enumerate(snapshot["cells"][2:], start=2):
        source = "".join(cell["source"]).replace("%%time\n", "")
        source = source.replace('"example.ome.parquet"', repr(str(temporary_file)))
        tree = ast.parse(source)
        if index == 26:
            # Keep the absolute fixture path in place of the notebook-relative one.
            tree.body = tree.body[1:]
        expression = tree.body[-1]
        assert isinstance(expression, ast.Expr)
        tree.body[-1] = ast.Assign(
            targets=[ast.Name(id="result", ctx=ast.Store())], value=expression.value
        )
        exec(
            compile(ast.fix_missing_locations(tree), "reference-notebook", "exec"),
            namespace,
        )
        frames[index] = namespace["result"]
    return frames


@pytest.mark.parametrize("cell_index", range(2, 28))
def test_at_a_glance_parity(notebook: dict, reference_frames: dict, cell_index: int):
    """Compare every example to the original notebook, not hand-made data."""
    overrides = {"static_3d": True} if cell_index >= 26 else {}
    _, definitions = notebook[f"example_{cell_index:02}"].run(**overrides)
    actual = definitions[f"frame_{cell_index:02}"]
    expected = reference_frames[cell_index]

    # OME exports stamp acquisition_datetime at export time. Compare all other
    # fields, including nested pixel arrays, without requiring identical clocks.
    def normalize(frame: CytoDataFrame) -> pd.DataFrame:
        return pd.DataFrame(frame).map(
            lambda value: (
                {**value, "acquisition_datetime": None}
                if isinstance(value, dict) and "acquisition_datetime" in value
                else value
            )
        )

    pd.testing.assert_frame_equal(normalize(actual), normalize(expected))
    assert (
        actual._custom_attrs["display_options"]
        == expected._custom_attrs["display_options"]
    )
    html = definitions[f"html_{cell_index:02}"]
    assert "<img" in html
    assert "3D snapshot unavailable" not in html
    _assert_embedded_images(html)
    if cell_index < 26:
        expected_html = expected._repr_html_(debug=True)
        # Compare actual encoded pixels, not only HTML tags or image counts.
        pattern = r"data:image/[^;]+;base64,([^\"']+)"
        assert re.findall(pattern, html) == re.findall(pattern, expected_html)
    else:
        for row in actual.index:
            volume, _ = actual._get_3d_volume_from_cell(row, "FileName_Nuclei")
            original, _ = expected._get_3d_volume_from_cell(row, "FileName_Nuclei")
            np.testing.assert_array_equal(volume, original)
            labels = actual._get_3d_label_overlay_from_cell(
                row, "FileName_Nuclei", expected_shape=volume.shape
            )
            if cell_index == 26:
                assert labels is None
            else:
                assert labels.shape == volume.shape
                assert np.count_nonzero(labels) > 0


def test_marimo_texture_filter(notebook: dict):
    _, definitions = notebook["example_22"].run()
    original = definitions["frame_22"]
    values = original["Nuclei_Texture_Variance_DAPI_3_03_256"]
    lower, upper = float(values.median()), float(values.max())
    _, filtered = notebook["filtered_speckles"].run(
        frame_22=original, texture_range=SimpleNamespace(value=(lower, upper))
    )
    frame = filtered["filtered_frame"]
    assert 0 < len(frame) < len(original)
    assert frame["Nuclei_Texture_Variance_DAPI_3_03_256"].between(lower, upper).all()
    _assert_embedded_images(filtered["filtered_html"])


@pytest.mark.parametrize("cell_index", [26, 27])
def test_marimo_interactive_3d(notebook: dict, cell_index: int):
    _, definitions = notebook[f"example_{cell_index}"].run(static_3d=True)
    _, renderer = notebook["interactive_rendering"].run()
    html = renderer["make_3d_table"](definitions[f"frame_{cell_index}"])
    assert html.count("<iframe") == 3
    assert "data-volume" in html
    assert "vtk.js@34.9.1" in html
    assert "data-volume-cmap=&quot;gray&quot;" in html
    assert "data-volume-background=&quot;0.000000,0.000000,0.000000&quot;" in html
    documents = unescape(html)
    payloads = re.findall(r'data-volume="([^"]+)"', documents)
    overlays = re.findall(r'data-label-volume="([^"]*)"', documents)
    assert len(payloads) == 3
    assert len(overlays) == 3
    toggles = re.findall(r'<input[^>]+class="cyto-3d-mask-toggle"', documents)
    assert len(toggles) == (3 if cell_index == 27 else 0)
    for row, payload, overlay in zip(
        definitions[f"frame_{cell_index}"].index, payloads, overlays, strict=True
    ):
        frame = definitions[f"frame_{cell_index}"]
        volume, _ = frame._get_3d_volume_from_cell(row, "FileName_Nuclei")
        pixels = np.frombuffer(base64.b64decode(payload), dtype=np.uint8)
        assert pixels.size == volume.size
        assert pixels.min() == 0
        assert pixels.max() == 255
        if cell_index == 26:
            assert not overlay
        else:
            labels = frame._get_3d_label_overlay_from_cell(
                row, "FileName_Nuclei", expected_shape=volume.shape
            )
            np.testing.assert_array_equal(
                np.frombuffer(base64.b64decode(overlay), dtype=np.uint8),
                np.where(labels > 0, 255, 0).astype(np.uint8).ravel(),
            )


@pytest.mark.parametrize("cell_index", [26, 27])
def test_live_3d_never_invokes_native_snapshot(notebook: dict, cell_index: int):
    def forbidden_snapshot(frame: CytoDataFrame) -> str:
        pytest.fail("Live marimo must not create a native VTK render window")

    output, definitions = notebook[f"example_{cell_index}"].run(
        static_3d=False, render_frame=forbidden_snapshot
    )
    assert definitions[f"html_{cell_index}"] == ""
    assert output.text.count("<iframe") == 3


def test_marimo_docs_snapshot():
    snapshot = json.loads(EXAMPLE.with_suffix(".ipynb").read_text())
    outputs = [
        output for cell in snapshot["cells"] for output in cell.get("outputs", [])
    ]
    assert not any(output["output_type"] == "error" for output in outputs)
    html = "".join(
        "".join(output.get("data", {}).get("text/html", [])) for output in outputs
    )
    assert "<iframe" not in html
    _assert_embedded_images(html)
    examples = {}
    for cell in snapshot["cells"]:
        match = re.search(r"html_(\d+) = render_frame", "".join(cell["source"]))
        if match:
            examples[int(match[1])] = "".join(
                "".join(output.get("data", {}).get("text/html", []))
                for output in cell.get("outputs", [])
            )
    assert set(examples) == set(range(2, 28))
    assert all("<img" in output for output in examples.values())
    assert examples[22].count("<img") == 25  # Match the full reference table.
    assert examples[26].count("<img") == 3
    assert examples[27].count("<img") == 3
    assert examples[26] != examples[27]
    assert "/Users/" not in EXAMPLE.with_suffix(".ipynb").read_text()
