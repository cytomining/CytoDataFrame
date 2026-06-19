"""
Tests for the CytoDataFrame formal schema system (schema.py).

Covers deterministic schema inference (differential against the hand-written
classification rules), property-based invariants via Hypothesis, and the
Arrow-native bounding-box / centroid struct helpers.
"""

import pandas as pd
import polars as pl
import pyarrow as pa
import pytest
from hypothesis import given
from hypothesis import strategies as st

from cytodataframe import CytoDataFrame, CytoSchema
from cytodataframe.schema import add_bbox_struct, add_centroid_struct


@pytest.fixture(name="cellprofiler_frame")
def fixture_cellprofiler_frame() -> pd.DataFrame:
    """A frame mirroring CellProfiler-style single-cell output."""
    return pd.DataFrame(
        {
            "Metadata_Well": ["A01", "B02"],
            "Metadata_Site": [1, 2],
            "ImageNumber": [1, 1],
            "ObjectNumber": [1, 2],
            "Image_FileName_DNA": ["a.tif", "b.tif"],
            "Image_PathName_DNA": ["/imgs", "/imgs"],
            "Cells_AreaShape_Area": [100.0, 200.0],
            "Cells_Intensity_MeanIntensity_DNA": [0.5, 0.6],
            "Nuclei_Location_Center_X": [5.0, 6.0],
            "Nuclei_Location_Center_Y": [7.0, 8.0],
            "Cells_AreaShape_BoundingBoxMinimum_X": [0, 1],
        }
    )


# --------------------------------------------------------------------------- #
# Deterministic / differential classification
# --------------------------------------------------------------------------- #
def test_schema_classification_buckets(cellprofiler_frame: pd.DataFrame):
    schema = CytoSchema.from_pandas(cellprofiler_frame)

    assert schema.image_key == "Image_FileName_DNA"
    assert schema.object_key == "ObjectNumber"

    # Features are numeric measurement columns only.
    assert set(schema.feature_columns) == {
        "Cells_AreaShape_Area",
        "Cells_Intensity_MeanIntensity_DNA",
    }
    # Geometry columns are spatial coordinates.
    assert set(schema.geometry_columns) == {
        "Nuclei_Location_Center_X",
        "Nuclei_Location_Center_Y",
        "Cells_AreaShape_BoundingBoxMinimum_X",
    }
    # Metadata holds identifiers + image references.
    assert "Metadata_Well" in schema.metadata_columns
    assert "Image_FileName_DNA" in schema.metadata_columns
    assert "ObjectNumber" in schema.metadata_columns


def test_schema_inference_matches_across_backends(cellprofiler_frame: pd.DataFrame):
    """pandas, polars, and Arrow inference agree."""
    from_pandas = CytoSchema.from_pandas(cellprofiler_frame).to_dict()
    from_polars = CytoSchema.from_polars(pl.from_pandas(cellprofiler_frame)).to_dict()
    from_arrow = CytoSchema.from_arrow(
        pa.Table.from_pandas(cellprofiler_frame, preserve_index=False).schema
    ).to_dict()
    assert from_pandas == from_polars == from_arrow


def test_schema_infer_dispatch(cellprofiler_frame: pd.DataFrame):
    table = pa.Table.from_pandas(cellprofiler_frame, preserve_index=False)
    assert (
        CytoSchema.infer(table).to_dict()
        == CytoSchema.infer(cellprofiler_frame).to_dict()
    )
    assert CytoSchema.infer(pl.from_pandas(cellprofiler_frame).lazy()).to_dict() == (
        CytoSchema.infer(cellprofiler_frame).to_dict()
    )


def test_schema_validate_and_require(cellprofiler_frame: pd.DataFrame):
    schema = CytoSchema.from_pandas(cellprofiler_frame)
    assert schema.validate() == []
    assert schema.require("image_key", "object_key") is schema

    bare = CytoSchema.from_columns(["Cells_AreaShape_Area"])
    with pytest.raises(ValueError, match="missing required key"):
        bare.require("image_key")


def test_schema_validate_detects_overlap():
    bad = CytoSchema(
        feature_columns=["x"],
        metadata_columns=["x"],
    )
    issues = bad.validate()
    assert any("feature and metadata" in issue for issue in issues)
    with pytest.raises(ValueError):
        bad.validate(strict=True)


def test_cytodataframe_cyto_schema_property(cellprofiler_frame: pd.DataFrame):
    cdf = CytoDataFrame(cellprofiler_frame)
    assert cdf.cyto_schema.image_key == "Image_FileName_DNA"


# --------------------------------------------------------------------------- #
# Property-based invariants
# --------------------------------------------------------------------------- #
_NAME_VOCAB = [
    "Metadata_Well",
    "Metadata_Plate",
    "ImageNumber",
    "ObjectNumber",
    "Image_FileName_DNA",
    "Image_PathName_DNA",
    "Cells_AreaShape_Area",
    "Nuclei_Intensity_MeanIntensity",
    "Cells_AreaShape_BoundingBox_Minimum_X",
    "Nuclei_Location_Center_X",
    "RandomFeature_1",
    "AnnotationLabel",
]


@given(
    columns=st.lists(
        st.sampled_from(_NAME_VOCAB), min_size=1, max_size=12, unique=True
    ),
    numeric_seed=st.lists(st.booleans(), min_size=12, max_size=12),
)
def test_schema_partition_invariants(columns: list, numeric_seed: list):
    numeric = {
        name: numeric_seed[idx % len(numeric_seed)] for idx, name in enumerate(columns)
    }
    schema = CytoSchema.from_columns(columns, numeric=numeric)

    meta = set(schema.metadata_columns)
    feat = set(schema.feature_columns)
    geom = set(schema.geometry_columns)

    # Every column is classified into exactly one of the three buckets.
    assert meta | feat | geom == set(columns)
    assert meta.isdisjoint(feat)
    assert feat.isdisjoint(geom)
    assert meta.isdisjoint(geom)

    # A non-numeric column is never treated as a feature.
    for name in columns:
        if not numeric[name]:
            assert name not in feat


# --------------------------------------------------------------------------- #
# Arrow-native struct helpers (Phase 3)
# --------------------------------------------------------------------------- #
def test_add_bbox_struct_keeps_flattened():
    df = pl.DataFrame(
        {
            "Cells_AreaShape_BoundingBoxMinimum_X": [0, 1],
            "Cells_AreaShape_BoundingBoxMinimum_Y": [0, 1],
            "Cells_AreaShape_BoundingBoxMaximum_X": [10, 11],
            "Cells_AreaShape_BoundingBoxMaximum_Y": [10, 11],
        }
    )
    out = add_bbox_struct(df)
    assert "bbox" in out.columns
    # flattened compatibility columns remain available
    assert "Cells_AreaShape_BoundingBoxMinimum_X" in out.columns
    struct = out["bbox"][0]
    assert struct["min_x"] == 0
    assert struct["max_y"] == 10


def test_add_centroid_struct_xy():
    df = pl.DataFrame(
        {
            "Nuclei_Location_Center_X": [5.0, 6.0],
            "Nuclei_Location_Center_Y": [7.0, 8.0],
        }
    )
    out = add_centroid_struct(df)
    assert "centroid" in out.columns
    assert out["centroid"][0]["x"] == 5.0
    assert out["centroid"][0]["y"] == 7.0


def test_struct_helpers_noop_without_columns():
    df = pl.DataFrame({"a": [1, 2]})
    assert add_bbox_struct(df).columns == ["a"]
    assert add_centroid_struct(df).columns == ["a"]
