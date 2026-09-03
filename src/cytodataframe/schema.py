"""
Formal schema system for CytoDataFrame.

This module classifies a profiling table's columns by role (metadata,
feature, geometry, image) and folds flattened CellProfiler-style geometry
columns into nested Arrow structs. It provides:

    * :class:`CytoSchema` - an explicit, inspectable classification of a
      profiling table's columns into image / object keys, metadata, feature, and
      geometry roles. The classification reduces reliance on ad-hoc naming
      conventions scattered through the codebase and gives downstream operations
      a single source of truth.
    * Arrow-native struct helpers that fold the flattened CellProfiler-style
      bounding-box / centroid columns into nested Arrow structs while keeping the
      flattened compatibility columns available for existing consumers.

Schema inference is deterministic and works from a pandas DataFrame, a polars
DataFrame/LazyFrame, or a :class:`pyarrow.Schema`/:class:`pyarrow.Table`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List, Mapping, Optional, Sequence

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd
    import polars as pl
    import pyarrow as pa


# --------------------------------------------------------------------------- #
# Column-role detection patterns
#
# These are kept as module-level constants (rather than a separate config
# file) because they're small, static, and specific to this one classifier;
# CellProfiler's naming conventions for these roles don't vary per pipeline.
# If per-project overrides are ever needed (e.g. non-CellProfiler naming
# schemes), that would be a reason to move them out into configuration.
# --------------------------------------------------------------------------- #

# "Geometry" here means per-object *location* coordinates (where an object
# sits within its parent image), not shape/size measurements. In CellProfiler
# output, ``Location_Center_*``, ``AreaShape_Center_*``/``Center_Mass``, and
# ``BoundingBox`` columns are all pixel coordinates in the whole-image
# coordinate system for that object (not relative to the object itself), so
# they behave the same way for our purposes: they should never be treated as
# a normalizable morphology feature, but they also aren't identifier/
# annotation metadata, hence the separate "geometry" role. Shape/size
# descriptors like Area, Perimeter, or Eccentricity are unaffected by this
# pattern and remain classified as features.
_GEOMETRY_PATTERN = re.compile(
    r"(boundingbox"
    r"|location_center"
    r"|areashape_center"
    r"|center_mass"
    r"|_center_[xyz]\b"
    r"|_center_[xyz]$)",
    flags=re.IGNORECASE,
)

# Image filename / path columns reference images rather than measurements.
_IMAGE_FILENAME_PATTERN = re.compile(r"filename", flags=re.IGNORECASE)
_IMAGE_PATHNAME_PATTERN = re.compile(r"pathname", flags=re.IGNORECASE)

# Known single-cell object identifier columns, in preference order.
_OBJECT_KEY_PRIORITY = (
    "metadata_objectnumber",
    "metadata_object_number",
    "objectnumber",
    "object_number",
)
_OBJECT_KEY_SUFFIX = "number_object_number"

# Known identifier-style metadata columns (casefolded exact names).
_KNOWN_ID_COLUMNS = frozenset(
    {
        "imagenumber",
        "objectnumber",
        "object_number",
        "tablenumber",
        "table_number",
        "plate",
        "well",
        "site",
    }
)


def _is_image_column(name: str) -> bool:
    """Return True when a column name references an image filename or path."""
    return bool(
        _IMAGE_FILENAME_PATTERN.search(name) or _IMAGE_PATHNAME_PATTERN.search(name)
    )


def _is_geometry_column(name: str) -> bool:
    """Return True when a column name encodes spatial geometry."""
    return bool(_GEOMETRY_PATTERN.search(name))


def _is_identifier_metadata(name: str) -> bool:
    """Return True when a column name looks like an identifier/metadata column."""
    lowered = name.casefold()
    if lowered.startswith("metadata"):
        return True
    if lowered in _KNOWN_ID_COLUMNS:
        return True
    return lowered.endswith(_OBJECT_KEY_SUFFIX)


@dataclass
class CytoSchema:
    """
    An explicit classification of a profiling table's columns.

    Attributes:
        image_key:
            The primary image filename column, if present.
        object_key:
            The single-cell object identifier column, if present.
        metadata_columns:
            Identifier / annotation / image-reference / non-numeric columns.
        feature_columns:
            Numeric measurement columns (the modeling features).
        geometry_columns:
            Spatial coordinate columns (bounding boxes, centroids).
        image_columns:
            All image filename/path columns (``image_key`` is the first).
    """

    image_key: Optional[str] = None
    object_key: Optional[str] = None
    metadata_columns: List[str] = field(default_factory=list)
    feature_columns: List[str] = field(default_factory=list)
    geometry_columns: List[str] = field(default_factory=list)
    image_columns: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    # Construction / inference
    # ------------------------------------------------------------------ #
    @classmethod
    def from_columns(
        cls,
        columns: Sequence[str],
        numeric: Optional[Mapping[str, bool]] = None,
    ) -> "CytoSchema":
        """
        Classify ``columns`` into schema roles.

        Args:
            columns:
                Ordered column names.
            numeric:
                Optional mapping of column name -> whether the column holds a
                numeric dtype. When a column is absent from the mapping (or the
                mapping is ``None``) the column is treated as numeric for the
                purpose of feature detection, so name-based rules still apply.
        """
        numeric = dict(numeric) if numeric is not None else None

        metadata: List[str] = []
        features: List[str] = []
        geometry: List[str] = []
        image_columns: List[str] = []

        for name in columns:
            col = str(name)
            is_numeric = True if numeric is None else bool(numeric.get(col, True))

            if _is_image_column(col):
                image_columns.append(col)
                metadata.append(col)
                continue
            if _is_geometry_column(col):
                geometry.append(col)
                continue
            if _is_identifier_metadata(col) or not is_numeric:
                metadata.append(col)
                continue
            features.append(col)

        image_key = image_columns[0] if image_columns else None
        object_key = cls._detect_object_key(columns)

        return cls(
            image_key=image_key,
            object_key=object_key,
            metadata_columns=metadata,
            feature_columns=features,
            geometry_columns=geometry,
            image_columns=image_columns,
        )

    @staticmethod
    def _detect_object_key(columns: Sequence[str]) -> Optional[str]:
        """Return the best single-cell object identifier column, if any."""
        lowered = {str(c).casefold(): str(c) for c in columns}
        for candidate in _OBJECT_KEY_PRIORITY:
            if candidate in lowered:
                return lowered[candidate]
        for col in columns:
            if str(col).casefold().endswith(_OBJECT_KEY_SUFFIX):
                return str(col)
        return None

    @classmethod
    def from_pandas(cls, data: "pd.DataFrame") -> "CytoSchema":
        """Infer a schema from a :class:`pandas.DataFrame`."""
        import pandas as pd

        numeric = {
            str(col): (
                pd.api.types.is_numeric_dtype(dtype)
                and not pd.api.types.is_bool_dtype(dtype)
            )
            for col, dtype in data.dtypes.items()
        }
        return cls.from_columns(list(data.columns), numeric=numeric)

    @classmethod
    def from_arrow(cls, schema: "pa.Schema") -> "CytoSchema":
        """Infer a schema from a :class:`pyarrow.Schema`."""
        import pyarrow as pa

        def _numeric(dtype: "pa.DataType") -> bool:
            return (
                pa.types.is_integer(dtype)
                or pa.types.is_floating(dtype)
                or pa.types.is_decimal(dtype)
            )

        numeric = {field.name: _numeric(field.type) for field in schema}
        return cls.from_columns(list(schema.names), numeric=numeric)

    @classmethod
    def from_polars(cls, data: "pl.DataFrame | pl.LazyFrame") -> "CytoSchema":
        """Infer a schema from a polars DataFrame or LazyFrame."""
        schema = data.collect_schema() if hasattr(data, "collect_schema") else None
        if schema is None:
            schema = data.schema
        numeric = {name: dtype.is_numeric() for name, dtype in schema.items()}
        return cls.from_columns(list(schema.keys()), numeric=numeric)

    @classmethod
    def infer(cls, data: Any) -> "CytoSchema":
        """
        Infer a schema from any supported tabular input.

        Dispatches on the runtime type so callers can pass pandas, polars, or
        Arrow data without converting first.
        """
        import pandas as pd

        # pyarrow first: a Table exposes ``.schema``.
        try:
            import pyarrow as pa

            if isinstance(data, pa.Table):
                return cls.from_arrow(data.schema)
            if isinstance(data, pa.Schema):
                return cls.from_arrow(data)
        except ImportError:  # pragma: no cover - pyarrow is a hard dependency
            pass

        try:
            import polars as pl

            if isinstance(data, (pl.DataFrame, pl.LazyFrame)):
                return cls.from_polars(data)
        except ImportError:  # pragma: no cover - polars is a hard dependency
            pass

        if isinstance(data, pd.DataFrame):
            return cls.from_pandas(data)

        raise TypeError(
            f"Cannot infer a CytoSchema from object of type {type(data)!r}."
        )

    # ------------------------------------------------------------------ #
    # Introspection / validation
    # ------------------------------------------------------------------ #
    @property
    def columns(self) -> List[str]:
        """
        All classified columns in metadata/geometry/feature order.

        ``image_columns`` is not iterated separately here because
        ``from_columns`` always also appends every image column to
        ``metadata_columns``, so image columns are already covered by that
        bucket; nothing is silently dropped.
        """
        ordered: List[str] = []
        seen: set[str] = set()
        for bucket in (
            self.metadata_columns,
            self.geometry_columns,
            self.feature_columns,
        ):
            for col in bucket:
                if col not in seen:
                    seen.add(col)
                    ordered.append(col)
        return ordered

    def validate(self, strict: bool = False) -> List[str]:
        """
        Check schema self-consistency.

        For a schema produced by ``from_columns``/``from_pandas``/``from_arrow``/
        ``from_polars``, the checks below never fire: those classifiers assign
        each column to exactly one bucket. They exist to catch inconsistency in
        a ``CytoSchema`` built or edited directly (e.g. constructed by hand, or
        with its column lists mutated after inference).

        Returns a list of human-readable issues. When ``strict`` is True and any
        issue is found, a :class:`ValueError` is raised instead.
        """
        issues: List[str] = []

        feature_set = set(self.feature_columns)
        metadata_set = set(self.metadata_columns)
        geometry_set = set(self.geometry_columns)

        overlap_fm = feature_set & metadata_set
        overlap_fg = feature_set & geometry_set
        if overlap_fm:
            issues.append(
                f"Columns classified as both feature and metadata: {sorted(overlap_fm)}"
            )
        if overlap_fg:
            issues.append(
                f"Columns classified as both feature and geometry: {sorted(overlap_fg)}"
            )
        if self.image_key is not None and self.image_key not in metadata_set:
            issues.append(
                f"image_key {self.image_key!r} is not present in metadata columns."
            )

        if strict and issues:
            raise ValueError("Invalid CytoSchema: " + "; ".join(issues))
        return issues

    def require(self, *keys: str) -> "CytoSchema":
        """
        Assert that the named required keys are present.

        Args:
            keys:
                Any of ``"image_key"`` / ``"object_key"``. Raises
                :class:`ValueError` when a required key is ``None``.
        """
        missing = [key for key in keys if getattr(self, key, None) is None]
        if missing:
            raise ValueError(f"CytoSchema is missing required key(s): {missing}")
        return self

    def to_dict(self) -> dict:
        """Return a plain-dict view of the schema (handy for tests/serialization)."""
        return {
            "image_key": self.image_key,
            "object_key": self.object_key,
            "metadata_columns": list(self.metadata_columns),
            "feature_columns": list(self.feature_columns),
            "geometry_columns": list(self.geometry_columns),
            "image_columns": list(self.image_columns),
        }


# --------------------------------------------------------------------------- #
# Arrow-native struct helpers (Phase 3)
# --------------------------------------------------------------------------- #

# Bounding-box column groups keyed by compartment, mirroring the flattened
# CellProfiler naming convention. Order is (min_x, min_y, max_x, max_y).
_BBOX_GROUPS = {
    "cytoplasm": "Cytoplasm_AreaShape_BoundingBox",
    "nuclei": "Nuclei_AreaShape_BoundingBox",
    "cells": "Cells_AreaShape_BoundingBox",
    "generic": "AreaShape_BoundingBox",
}

# Centroid column groups keyed by compartment, mirroring the flattened naming.
_CENTROID_GROUPS = {
    "nuclei": "Nuclei_Location_Center",
    "nuclei_meta": "Metadata_Nuclei_Location_Center",
    "cells": "Cells_Location_Center",
    "cells_meta": "Metadata_Cells_Location_Center",
    "cytoplasm": "Cytoplasm_Location_Center",
    "cytoplasm_meta": "Metadata_Cytoplasm_Location_Center",
}


def _bbox_field_columns(prefix: str) -> dict:
    """Return the flattened bounding-box column names for a prefix."""
    return {
        "min_x": f"{prefix}Minimum_X",
        "min_y": f"{prefix}Minimum_Y",
        "max_x": f"{prefix}Maximum_X",
        "max_y": f"{prefix}Maximum_Y",
        "min_z": f"{prefix}Minimum_Z",
        "max_z": f"{prefix}Maximum_Z",
    }


def add_bbox_struct(
    data: "pl.DataFrame",
    struct_name: str = "bbox",
    keep_flattened: bool = True,
) -> "pl.DataFrame":
    """
    Fold flattened bounding-box columns into a nested Arrow struct.

    The first matching compartment group (cytoplasm -> nuclei -> cells ->
    generic) is used. The flattened compatibility columns are retained by
    default so existing consumers keep working.

    Returns the input unchanged when no bounding-box columns are present.
    """
    import polars as pl

    required_keys = ("min_x", "min_y", "max_x", "max_y")
    available = set(data.columns)
    for prefix in _BBOX_GROUPS.values():
        cols = _bbox_field_columns(prefix)
        required = {k: v for k, v in cols.items() if k in required_keys}
        if not all(col in available for col in required.values()):
            continue
        fields = [
            pl.col(cols["min_x"]).alias("min_x"),
            pl.col(cols["min_y"]).alias("min_y"),
            pl.col(cols["max_x"]).alias("max_x"),
            pl.col(cols["max_y"]).alias("max_y"),
        ]
        if cols["min_z"] in available and cols["max_z"] in available:
            fields.append(pl.col(cols["min_z"]).alias("min_z"))
            fields.append(pl.col(cols["max_z"]).alias("max_z"))
        result = data.with_columns(pl.struct(fields).alias(struct_name))
        if not keep_flattened:
            drop = [c for c in cols.values() if c in available]
            result = result.drop(drop)
        return result
    return data


def add_centroid_struct(
    data: "pl.DataFrame",
    struct_name: str = "centroid",
    keep_flattened: bool = True,
) -> "pl.DataFrame":
    """
    Fold flattened centroid columns into a nested Arrow struct ``{x, y[, z]}``.

    The first matching compartment group is used. Flattened compatibility
    columns are retained by default. Returns the input unchanged when no
    centroid columns are present.
    """
    import polars as pl

    available = set(data.columns)
    for prefix in _CENTROID_GROUPS.values():
        x_col = f"{prefix}_X"
        y_col = f"{prefix}_Y"
        z_col = f"{prefix}_Z"
        if x_col not in available or y_col not in available:
            continue
        fields = [
            pl.col(x_col).alias("x"),
            pl.col(y_col).alias("y"),
        ]
        if z_col in available:
            fields.append(pl.col(z_col).alias("z"))
        result = data.with_columns(pl.struct(fields).alias(struct_name))
        if not keep_flattened:
            drop = [c for c in (x_col, y_col, z_col) if c in available]
            result = result.drop(drop)
        return result
    return data
