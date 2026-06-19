"""
Tests for the CytoLazyFrame lazy query builder (lazy.py).

Covers the lazy-execution surface from the evolution plan and differential
validation that lazy Polars execution matches the equivalent pandas result.
"""

import pathlib

import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

from cytodataframe import CytoDataFrame, CytoLazyFrame


@pytest.fixture(name="profiles")
def fixture_profiles() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Metadata_Well": ["A01", "A01", "B02", "B02", "C03"],
            "Metadata_ObjectNumber": [1, 2, 1, 2, 1],
            "Cells_AreaShape_Area": [10.0, 20.0, 30.0, 40.0, 50.0],
            "Nuclei_Location_Center_X": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )


def test_to_lazy_returns_cytolazyframe(profiles: pd.DataFrame):
    lazy = CytoDataFrame(profiles).to_lazy()
    assert isinstance(lazy, CytoLazyFrame)
    assert lazy.columns == list(profiles.columns)


def test_lazy_filter_matches_pandas(profiles: pd.DataFrame):
    cdf = CytoDataFrame(profiles)
    lazy_result = cdf.to_lazy().filter(pl.col("Cells_AreaShape_Area") >= 30.0).collect()
    pandas_result = profiles[profiles["Cells_AreaShape_Area"] >= 30.0]

    assert isinstance(lazy_result, CytoDataFrame)
    assert len(lazy_result) == len(pandas_result)
    assert (
        lazy_result["Cells_AreaShape_Area"].tolist()
        == pandas_result["Cells_AreaShape_Area"].tolist()
    )


def test_lazy_eager_equivalence(profiles: pd.DataFrame):
    """Lazy and eager polars execution produce identical results."""
    cdf = CytoDataFrame(profiles)
    lazy_df = cdf.to_lazy().filter(pl.col("Metadata_Well") == "B02").to_polars()
    eager_df = cdf.to_polars().filter(pl.col("Metadata_Well") == "B02")
    assert lazy_df.equals(eager_df)


def test_lazy_select_features(profiles: pd.DataFrame):
    cdf = CytoDataFrame(profiles)
    result = cdf.to_lazy().select_features().collect()
    # geometry column dropped; metadata + feature retained
    assert "Nuclei_Location_Center_X" not in result.columns
    assert "Cells_AreaShape_Area" in result.columns
    assert "Metadata_Well" in result.columns


def test_lazy_select_features_explicit_no_metadata(profiles: pd.DataFrame):
    cdf = CytoDataFrame(profiles)
    result = (
        cdf.to_lazy()
        .select_features(["Cells_AreaShape_Area"], keep_metadata=False)
        .collect()
    )
    assert list(result.columns) == ["Cells_AreaShape_Area"]


def test_lazy_group_by_agg(profiles: pd.DataFrame):
    cdf = CytoDataFrame(profiles)
    result = (
        cdf.to_lazy()
        .group_by("Metadata_Well")
        .agg(pl.col("Cells_AreaShape_Area").sum().alias("total"))
        .collect()
    )
    totals = dict(
        zip(
            result["Metadata_Well"].tolist(),
            result["total"].tolist(),
            strict=False,
        )
    )
    expected = profiles.groupby("Metadata_Well")["Cells_AreaShape_Area"].sum()
    assert totals["A01"] == expected["A01"]
    assert totals["B02"] == expected["B02"]


def test_lazy_join(profiles: pd.DataFrame):
    cdf = CytoDataFrame(profiles)
    annotations = pl.DataFrame(
        {"Metadata_Well": ["A01", "B02"], "treatment": ["drug", "ctrl"]}
    )
    result = cdf.to_lazy().join(annotations, on="Metadata_Well", how="inner").collect()
    assert "treatment" in result.columns
    # only A01 (2 rows) + B02 (2 rows) survive the inner join
    assert len(result) == 4


def test_lazy_rename_and_drop(profiles: pd.DataFrame):
    cdf = CytoDataFrame(profiles)
    result = (
        cdf.to_lazy()
        .rename({"Cells_AreaShape_Area": "area"})
        .drop("Nuclei_Location_Center_X")
        .collect()
    )
    assert "area" in result.columns
    assert "Nuclei_Location_Center_X" not in result.columns


def test_lazy_to_arrow_and_polars(profiles: pd.DataFrame):
    lazy = CytoDataFrame(profiles).to_lazy()
    assert isinstance(lazy.to_arrow(), pa.Table)
    assert isinstance(lazy.to_polars(), pl.DataFrame)
    assert isinstance(lazy.to_pandas(), pd.DataFrame)


def test_lazy_context_carry_through(profiles: pd.DataFrame, tmp_path: pathlib.Path):
    """Image/display context survives a lazy pipeline into the collected frame."""
    ctx_dir = str(tmp_path)
    cdf = CytoDataFrame(
        profiles,
        data_context_dir=ctx_dir,
        display_options={"width": 123},
    )
    collected = cdf.to_lazy().filter(pl.col("Metadata_Well") == "A01").collect()
    assert collected._custom_attrs["data_context_dir"] == ctx_dir
    assert collected._custom_attrs["display_options"] == {"width": 123}


def test_scan_parquet_pipeline(profiles: pd.DataFrame, tmp_path: pathlib.Path):
    out = tmp_path / "profiles.parquet"
    profiles.to_parquet(out)
    result = (
        CytoDataFrame.scan_parquet(str(out), data_context_dir=str(tmp_path))
        .filter(pl.col("Metadata_Well") == "A01")
        .select_features()
        .collect()
    )
    assert isinstance(result, CytoDataFrame)
    assert len(result) == 2
    assert result._custom_attrs["data_context_dir"] == str(tmp_path)


def test_scan_parquet_returns_lazyframe(profiles: pd.DataFrame, tmp_path: pathlib.Path):
    out = tmp_path / "profiles.parquet"
    profiles.to_parquet(out)
    scanned = CytoDataFrame.scan_parquet(str(out))
    assert isinstance(scanned, CytoLazyFrame)
    assert "CytoLazyFrame" in repr(scanned)
