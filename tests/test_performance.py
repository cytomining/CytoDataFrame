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

import re
import time

import numpy as np
import pandas as pd

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


def _best_time(func, repeats: int = 3, warmup: int = 1) -> float:
    """Return the fastest wall-clock time over a few runs (least noisy)."""
    for _ in range(warmup):
        func()
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        func()
        best = min(best, time.perf_counter() - start)
    return best


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


def test_render_scans_for_image_columns_at_most_once(monkeypatch):
    """A single render must not repeatedly re-scan for image columns.

    Earlier, the HTML render reconstructed a fresh ``CytoDataFrame`` for each
    helper it needed, re-running image-column detection several times per
    render (measured at 4x). Calling the helpers on the existing frame instead
    brings this to a single scan. This structural check fails if the
    per-helper-reconstruction pattern is reintroduced, independent of hardware
    speed.

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

    def counting_find_image_columns(self):
        calls["count"] += 1
        return original(self)

    monkeypatch.setattr(
        CytoDataFrame, "find_image_columns", counting_find_image_columns
    )

    frame._repr_html_(debug=True)

    # One scan per render is expected; allow a small margin for future changes
    # while still catching a return to per-helper reconstruction (which was 4x).
    assert calls["count"] <= 2, (
        f"render scanned for image columns {calls['count']} times (expected <= 2)"
    )


def test_render_does_not_process_phantom_image_columns(monkeypatch):
    """Rendering must only process the image columns actually on display.

    A frame can carry ``Image_URL_*`` columns (detected as image columns) plus
    matching ``Image_PathName_*`` metadata. A prior bug pulled the URL columns
    into the path metadata, so rendering re-joined and processed them once per
    row before dropping them -- doubling the per-image decode/equalize work.
    This asserts each displayed row triggers exactly one image-processing call
    per displayed image column.
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

    def counting_process(self, *args, **kwargs):
        calls["count"] += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(
        CytoDataFrame, "process_image_data_as_html_display", counting_process
    )

    frame._repr_html_(debug=True)

    # 2 rows x 1 displayed image column == 2 calls (was 4 with the phantom
    # URL column being processed and then discarded).
    assert calls["count"] == 2, (
        f"rendering processed images {calls['count']} times (expected 2)"
    )


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
