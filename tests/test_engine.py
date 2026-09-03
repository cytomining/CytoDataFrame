"""
Tests for the CytoDataFrame backend abstraction layer (engine.py).

These tests check that converting data out of a CytoDataFrame into another
format and back ("round-tripping") never silently changes it. Concretely,
converting a CytoDataFrame to Arrow/Parquet/pandas/Polars and back to a
CytoDataFrame ("interchange") must preserve row counts, null values, column
dtypes, and column ordering:

    cdf -> Arrow   -> cdf
    cdf -> Parquet -> cdf
    cdf -> pandas  -> cdf
    cdf -> Polars  -> cdf
"""

import pathlib

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

from cytodataframe import CytoDataFrame, engine


@pytest.fixture(name="profiling_frame")
def fixture_profiling_frame() -> pd.DataFrame:
    """A small profiling-like frame with mixed dtypes and nulls."""
    return pd.DataFrame(
        {
            "Metadata_Well": ["A01", "A01", "B02", None],
            "Metadata_ObjectNumber": [1, 2, 1, 2],
            "Cells_AreaShape_Area": [10.0, np.nan, 30.0, 40.0],
            "Nuclei_Location_Center_X": [5.0, 6.0, 7.0, 8.0],
            "Nuclei_Intensity_MeanIntensity_DNA": [0.1, 0.2, 0.3, 0.4],
        }
    )


def _assert_tabular_equivalent(left: pd.DataFrame, right: pd.DataFrame) -> None:
    """Assert two frames share row count, columns, null mask, and values."""
    left = left.reset_index(drop=True)
    right = right.reset_index(drop=True)
    assert len(left) == len(right)
    assert list(left.columns) == list(right.columns)
    for col in left.columns:
        lnull = left[col].isna().to_numpy()
        rnull = right[col].isna().to_numpy()
        assert np.array_equal(lnull, rnull), f"null mask differs for {col}"
        lvals = left[col][~left[col].isna()].tolist()
        rvals = right[col][~right[col].isna()].tolist()
        assert lvals == rvals, f"values differ for {col}"


# --------------------------------------------------------------------------- #
# Conversions from every supported input type
# --------------------------------------------------------------------------- #
def test_engine_to_arrow_from_all_inputs(profiling_frame: pd.DataFrame):
    pdf = profiling_frame
    expected = pa.Table.from_pandas(pdf, preserve_index=False)
    for source in (
        pdf,
        pl.from_pandas(pdf),
        pl.from_pandas(pdf).lazy(),
        expected,
    ):
        table = engine.to_arrow(source)
        assert isinstance(table, pa.Table)
        assert table.num_rows == len(pdf)
        assert table.schema.names == list(pdf.columns)


def test_engine_to_polars_from_all_inputs(profiling_frame: pd.DataFrame):
    pdf = profiling_frame
    for source in (
        pdf,
        pl.from_pandas(pdf),
        pl.from_pandas(pdf).lazy(),
        pa.Table.from_pandas(pdf, preserve_index=False),
    ):
        out = engine.to_polars(source)
        assert isinstance(out, pl.DataFrame)
        assert out.height == len(pdf)
        assert out.columns == list(pdf.columns)


def test_engine_to_lazyframe_passthrough_and_convert(profiling_frame: pd.DataFrame):
    lf = pl.from_pandas(profiling_frame).lazy()
    # passthrough
    assert engine.to_lazyframe(lf) is lf
    # convert from pandas
    converted = engine.to_lazyframe(profiling_frame)
    assert isinstance(converted, pl.LazyFrame)
    assert converted.collect().height == len(profiling_frame)


def test_engine_to_pandas_returns_pandas_identity(profiling_frame: pd.DataFrame):
    # pandas inputs are returned untouched (object columns are never disturbed)
    assert engine.to_pandas(profiling_frame) is profiling_frame
    converted = engine.to_pandas(pl.from_pandas(profiling_frame))
    assert isinstance(converted, pd.DataFrame)
    _assert_tabular_equivalent(profiling_frame, converted)


def test_engine_rejects_unsupported_type():
    with pytest.raises(TypeError):
        engine.to_arrow(object())
    with pytest.raises(TypeError):
        engine.to_polars(42)


# --------------------------------------------------------------------------- #
# Round-trip interchange guarantees
# --------------------------------------------------------------------------- #
def test_roundtrip_arrow(profiling_frame: pd.DataFrame):
    cdf = CytoDataFrame(profiling_frame)
    table = cdf.to_arrow()
    restored = CytoDataFrame(table)
    assert isinstance(restored, CytoDataFrame)
    _assert_tabular_equivalent(profiling_frame, pd.DataFrame(restored))


def test_roundtrip_polars(profiling_frame: pd.DataFrame):
    cdf = CytoDataFrame(profiling_frame)
    restored = CytoDataFrame(cdf.to_polars())
    _assert_tabular_equivalent(profiling_frame, pd.DataFrame(restored))


def test_roundtrip_pandas(profiling_frame: pd.DataFrame):
    cdf = CytoDataFrame(profiling_frame)
    restored = CytoDataFrame(cdf.to_pandas())
    _assert_tabular_equivalent(profiling_frame, pd.DataFrame(restored))


def test_roundtrip_parquet_from_file(
    profiling_frame: pd.DataFrame, tmp_path: pathlib.Path
):
    cdf = CytoDataFrame(profiling_frame)
    out = tmp_path / "profiles.parquet"
    cdf.export(str(out))
    restored = CytoDataFrame(str(out))
    _assert_tabular_equivalent(profiling_frame, pd.DataFrame(restored))


def test_roundtrip_preserves_schema(profiling_frame: pd.DataFrame):
    cdf = CytoDataFrame(profiling_frame)
    # Arrow schema names + the inferred CytoSchema survive a polars round-trip.
    before = cdf.cyto_schema.to_dict()
    after = CytoDataFrame(cdf.to_polars()).cyto_schema.to_dict()
    assert before == after


def test_scan_parquet_helper(profiling_frame: pd.DataFrame, tmp_path: pathlib.Path):
    out = tmp_path / "profiles.parquet"
    profiling_frame.to_parquet(out)
    lf = engine.scan_parquet(str(out))
    assert isinstance(lf, pl.LazyFrame)
    assert lf.collect().height == len(profiling_frame)
