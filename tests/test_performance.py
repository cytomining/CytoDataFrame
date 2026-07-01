"""
Performance regression tests for CytoDataFrame.

These guard the optimizations that keep CytoDataFrame responsive on wide
single-cell feature tables (which routinely have thousands of columns). They are
written to be stable across machines and CI runners by preferring:

* relative comparisons (optimized path vs. a naive baseline measured in the same
  process), which scale together regardless of hardware, and
* structural assertions (how many times an expensive operation runs), which are
  deterministic,

over brittle absolute wall-clock thresholds. A single, deliberately generous
wall-clock budget is included as a coarse guard against catastrophic
(e.g. accidentally quadratic) regressions.
"""

import pathlib
import re
import time
from typing import Any, Callable, List

import imageio.v2 as imageio
import numpy as np
import pandas as pd
import pytest

import cytodataframe.frame as cdf_frame
from cytodataframe import CytoDataFrame


def _wide_numeric_frame(n_numeric: int = 2000, n_rows: int = 50) -> pd.DataFrame:
    """Build a wide, mostly-numeric frame resembling a single-cell profile.

    It has many numeric feature columns plus a couple of string image-name
    columns, mirroring real CellProfiler/cytomining outputs where the numeric
    features vastly outnumber the image columns.
    """
    rng = np.random.default_rng(0)
    data = {
        f"Feature_{i}": rng.random(n_rows).astype(np.float64) for i in range(n_numeric)
    }
    data["Metadata_ImageNumber"] = np.arange(n_rows)
    data["Image_FileName_DNA"] = [f"img_{i}.tiff" for i in range(n_rows)]
    data["Image_FileName_RNA"] = [f"img_{i}_rna.tif" for i in range(n_rows)]
    return pd.DataFrame(data)


def _best_time(func: Callable[[], Any], repeats: int = 3, warmup: int = 1) -> float:
    """Return the fastest wall-clock time over a few runs (least noisy)."""
    for _ in range(warmup):
        func()
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        func()
        best = min(best, time.perf_counter() - start)
    return best


def _render_via_notebook_path(
    frame: CytoDataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Render a frame through the real notebook path (``_repr_html_`` with
    ``debug=False``), stubbing out notebook detection and display side effects.

    This exercises the full path a user hits in Jupyter -- including
    ``_try_render_trame_widget_table``/``_find_3d_columns_for_display`` and the
    static-snapshot + interactive renders -- rather than the lighter
    ``debug=True`` shortcut. That path drives ``find_image_columns`` and
    ``process_image_data_as_html_display`` more times per display, so it is the
    one worth guarding.
    """
    real_get_option = cdf_frame.get_option
    # Make the code believe it is running in a notebook, but leave every other
    # pandas display option untouched.
    monkeypatch.setattr(
        cdf_frame,
        "get_option",
        lambda name: (
            True if name == "display.notebook_repr_html" else real_get_option(name)
        ),
    )
    # Swallow IPython display side effects (widgets, HTML, javascript).
    monkeypatch.setattr(cdf_frame, "display", lambda *args, **kwargs: None)

    frame._repr_html_(debug=False)


def test_find_image_columns_skips_numeric_columns_and_is_faster():
    """Detecting image columns must not scan every numeric feature column.

    ``find_image_columns`` skips columns whose dtype cannot hold a filename
    string. We assert it is meaningfully faster than a naive scan of every
    column's values (the pre-optimization behavior) while returning the same
    result. The comparison is relative, so it holds regardless of the machine.
    """
    frame = CytoDataFrame(_wide_numeric_frame())
    pattern = re.compile(r".*\.(tif|tiff)$", flags=re.IGNORECASE)

    def naive_scan() -> list:
        # Mirrors the original implementation: scan the values of *every*
        # column, including numeric ones.
        return [
            column
            for column in frame.columns
            if frame[column]
            .apply(
                lambda value: (
                    isinstance(value, str) and pattern.match(str(value)) is not None
                )
            )
            .any()
        ]

    # Correctness: the optimized detector finds exactly the image columns.
    assert frame.find_image_columns() == ["Image_FileName_DNA", "Image_FileName_RNA"]
    # ...and matches what a full scan would find.
    assert frame.find_image_columns() == naive_scan()

    optimized = _best_time(frame.find_image_columns)
    naive = _best_time(naive_scan)

    # Skipping thousands of numeric columns should be at least 2x faster. In
    # practice it is far more, so this leaves generous headroom against noise.
    assert optimized < naive * 0.5, (
        f"find_image_columns not fast enough: optimized={optimized * 1e3:.1f}ms "
        f"naive={naive * 1e3:.1f}ms"
    )


def test_render_does_not_repeatedly_rescan_image_columns(
    monkeypatch: pytest.MonkeyPatch,
):
    """A notebook render must not re-scan for image columns per helper.

    Earlier, the HTML render reconstructed a fresh ``CytoDataFrame`` for each
    helper it needed, re-running image-column detection several times. Calling
    the helpers on the existing frame instead roughly halved the scans through
    the real notebook path (12 -> 6 for this frame). This structural check fails
    if the per-helper-reconstruction pattern is reintroduced, independent of
    hardware speed.

    The frame includes an ``Image_PathName_*`` column so it mirrors real
    CellProfiler/cytomining output (where image-path metadata is captured once
    at construction and reused).
    """
    frame = CytoDataFrame(
        pd.DataFrame(
            {
                "Metadata_ImageNumber": np.arange(5),
                "Image_FileName_DNA": [f"img_{i}.tiff" for i in range(5)],
                "Image_PathName_DNA": ["/data/images"] * 5,
            }
        )
    )

    calls = {"count": 0}
    original = CytoDataFrame.find_image_columns

    def counting_find_image_columns(self: CytoDataFrame) -> List[str]:
        calls["count"] += 1
        return original(self)

    monkeypatch.setattr(
        CytoDataFrame, "find_image_columns", counting_find_image_columns
    )

    _render_via_notebook_path(frame, monkeypatch)

    # The optimized path scans 6x for this frame; the pre-optimization behavior
    # was 12x. The bound catches a return to per-helper reconstruction while
    # leaving headroom, and does not scale with dataframe width.
    assert calls["count"] <= 8, (
        f"render scanned for image columns {calls['count']} times (expected <= 8)"
    )


def test_render_does_not_process_phantom_image_columns(
    monkeypatch: pytest.MonkeyPatch,
):
    """Rendering must only process the image columns actually on display.

    A frame can carry ``Image_URL_*`` columns (detected as image columns) plus
    matching ``Image_PathName_*`` metadata. A prior bug pulled the URL columns
    into the path metadata, so rendering re-joined and processed them once per
    row before dropping them -- doubling the per-image decode/equalize work.
    This asserts the render only processes the displayed image column.

    Exercised through the real notebook path, which builds the table twice (the
    static snapshot plus the interactive render), so the expected count is
    2 builds x 2 rows x 1 displayed image column == 4. With the phantom URL
    column it was 8.
    """
    frame = CytoDataFrame(
        pd.DataFrame(
            {
                "Image_FileName_DNA": ["a.tiff", "b.tiff"],
                "Image_PathName_DNA": ["/imgs", "/imgs"],
                "Image_URL_DNA": ["file:/imgs/a.tiff", "file:/imgs/b.tiff"],
                "Nuclei_AreaShape_BoundingBoxMinimum_X": [0, 0],
                "Nuclei_AreaShape_BoundingBoxMinimum_Y": [0, 0],
                "Nuclei_AreaShape_BoundingBoxMaximum_X": [5, 5],
                "Nuclei_AreaShape_BoundingBoxMaximum_Y": [5, 5],
            }
        )
    )[["Image_FileName_DNA"]]

    calls = {"count": 0}
    original = CytoDataFrame.process_image_data_as_html_display

    def counting_process(
        self: CytoDataFrame, *args: object, **kwargs: object
    ) -> object:
        calls["count"] += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(
        CytoDataFrame, "process_image_data_as_html_display", counting_process
    )

    _render_via_notebook_path(frame, monkeypatch)

    assert calls["count"] == 4, (
        f"rendering processed images {calls['count']} times (expected 4)"
    )


def _frame_sharing_one_image(tmp_path: pathlib.Path) -> CytoDataFrame:
    """Two displayed rows that reference the same field-of-view image on disk."""
    image = np.linspace(0, 255, 40 * 40, dtype=np.uint8).reshape(40, 40)
    imageio.imwrite(tmp_path / "img.tiff", image)
    return CytoDataFrame(
        pd.DataFrame(
            {
                "Image_FileName_DNA": ["img.tiff", "img.tiff"],
                "Nuclei_AreaShape_BoundingBoxMinimum_X": [0, 0],
                "Nuclei_AreaShape_BoundingBoxMinimum_Y": [0, 0],
                "Nuclei_AreaShape_BoundingBoxMaximum_X": [20, 20],
                "Nuclei_AreaShape_BoundingBoxMaximum_Y": [20, 20],
            }
        ),
        data_context_dir=str(tmp_path),
    )[["Image_FileName_DNA"]]


def _count_decodes(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Patch image decoding to count calls; returns a mutable ``{"n": int}``."""
    calls = {"n": 0}
    real_imread = cdf_frame.imageio.imread

    def counting_imread(path: object, *args: object, **kwargs: object) -> object:
        calls["n"] += 1
        return real_imread(path, *args, **kwargs)

    monkeypatch.setattr(cdf_frame.imageio, "imread", counting_imread)
    return calls


def test_repeat_renders_reuse_cached_images(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    """The decode + equalize step is cached across cells and re-renders.

    Reading and contrast-enhancing an image is the dominant render cost, and the
    same field-of-view image is frequently shared by multiple displayed objects
    and re-processed on each re-render (e.g. brightness/filter changes). The
    cache must decode each unique image once and reuse it thereafter.
    """
    frame = _frame_sharing_one_image(tmp_path)
    calls = _count_decodes(monkeypatch)

    html_first = frame._repr_html_(debug=True)
    first_render_decodes = calls["n"]

    calls["n"] = 0
    html_second = frame._repr_html_(debug=True)
    second_render_decodes = calls["n"]

    # Two rows share one image -> a single decode on the first render...
    assert first_render_decodes == 1, (
        f"expected 1 decode across shared cells, got {first_render_decodes}"
    )
    # ...and re-rendering the same frame decodes nothing (fully cached).
    assert second_render_decodes == 0, (
        f"expected 0 decodes on re-render, got {second_render_decodes}"
    )
    # The cache must not change what is rendered.
    assert html_first == html_second


def test_image_cache_can_be_disabled(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    """The ``image_disable_cache`` display option turns caching off."""
    frame = _frame_sharing_one_image(tmp_path)
    frame._custom_attrs["display_options"] = {"image_disable_cache": True}
    calls = _count_decodes(monkeypatch)

    frame._repr_html_(debug=True)
    frame._repr_html_(debug=True)

    # With caching disabled, both cells in both renders decode: 2 rows x 2 renders.
    assert calls["n"] == 4, f"expected 4 decodes with cache disabled, got {calls['n']}"


def test_wide_frame_construction_stays_within_budget():
    """Coarse guard against catastrophic (e.g. quadratic) construction cost.

    Constructing a CytoDataFrame runs metadata detection over every column.
    This uses a deliberately generous budget so it is not flaky on shared CI
    runners; it exists to catch order-of-magnitude regressions, not to police
    small fluctuations.
    """
    wide = _wide_numeric_frame(n_numeric=3000)

    elapsed = _best_time(lambda: CytoDataFrame(wide), repeats=2, warmup=1)

    assert elapsed < 2.5, f"wide-frame construction too slow: {elapsed:.2f}s"
