"""
Backend abstraction layer for CytoDataFrame.

This module is the execution/interchange boundary described in the CytoDataFrame
evolution plan. It treats Apache Arrow as the canonical schema and memory
contract, Polars as the execution engine, and pandas as a compatibility layer.

The functions here normalize the supported tabular inputs

    * :class:`pandas.DataFrame` / :class:`pandas.Series`
    * :class:`polars.DataFrame`
    * :class:`polars.LazyFrame`
    * :class:`pyarrow.Table`
    * :class:`cytodataframe.frame.CytoDataFrame` (a ``pandas.DataFrame`` subclass)

into the representation requested by the caller while preserving row counts,
null semantics, column ordering, and schema.

Design notes:
    * Arrow is used as the bridge whenever a schema/serialization contract is
      requested (``to_arrow``).
    * Conversions intentionally avoid forcing existing *pandas* object columns
      (which may hold numpy image arrays or OME-Arrow structs) through Arrow,
      because Arrow cannot always round-trip arbitrary Python objects. Such
      columns are only converted when the caller explicitly asks for an Arrow or
      Polars representation.
"""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, Any, Union

import pandas as pd

if TYPE_CHECKING:  # pragma: no cover - typing only
    import polars as pl
    import pyarrow as pa

# Public alias describing every tabular input CytoDataFrame's engine understands.
TabularData = Union[
    "pd.DataFrame",
    "pd.Series",
    "pl.DataFrame",
    "pl.LazyFrame",
    "pa.Table",
]


def _polars() -> Any:
    """Import polars lazily so importing this module stays cheap."""
    import polars as pl

    return pl


def _pyarrow() -> Any:
    """Import pyarrow lazily so importing this module stays cheap."""
    import pyarrow as pa

    return pa


def is_polars_dataframe(data: Any) -> bool:
    """Return True when ``data`` is a :class:`polars.DataFrame`."""
    try:
        pl = _polars()
    except ImportError:
        return False
    return isinstance(data, pl.DataFrame)


def is_polars_lazyframe(data: Any) -> bool:
    """Return True when ``data`` is a :class:`polars.LazyFrame`."""
    try:
        pl = _polars()
    except ImportError:
        return False
    return isinstance(data, pl.LazyFrame)


def is_arrow_table(data: Any) -> bool:
    """Return True when ``data`` is a :class:`pyarrow.Table`."""
    try:
        pa = _pyarrow()
    except ImportError:
        return False
    return isinstance(data, pa.Table)


def is_supported(data: Any) -> bool:
    """Return True when ``data`` is one of the supported tabular inputs."""
    return (
        isinstance(data, (pd.DataFrame, pd.Series))
        or is_polars_dataframe(data)
        or is_polars_lazyframe(data)
        or is_arrow_table(data)
    )


def to_pandas(data: TabularData) -> pd.DataFrame:
    """
    Convert any supported tabular input to a :class:`pandas.DataFrame`.

    pandas inputs (including ``CytoDataFrame``) are returned as-is so that object
    columns holding images or OME-Arrow structs are never disturbed.
    """
    if isinstance(data, pd.DataFrame):
        return data
    if isinstance(data, pd.Series):
        return data.to_frame()
    if is_polars_lazyframe(data):
        return data.collect().to_pandas()
    if is_polars_dataframe(data):
        return data.to_pandas()
    if is_arrow_table(data):
        return data.to_pandas()
    raise TypeError(
        f"Unsupported type for CytoDataFrame engine conversion: {type(data)!r}"
    )


def to_polars(data: TabularData) -> "pl.DataFrame":
    """Convert any supported tabular input to an eager :class:`polars.DataFrame`."""
    pl = _polars()
    if isinstance(data, pl.DataFrame):
        return data
    if isinstance(data, pl.LazyFrame):
        return data.collect()
    if is_arrow_table(data):
        return pl.from_arrow(data)
    if isinstance(data, pd.Series):
        data = data.to_frame()
    if isinstance(data, pd.DataFrame):
        # Strip any pandas subclass (e.g. CytoDataFrame) and index before handing
        # the frame to polars, which has no index concept.
        try:
            return pl.from_pandas(pd.DataFrame(data))
        except Exception as exc:
            raise TypeError(
                "Could not convert pandas data to polars. Columns holding "
                "non-Arrow-compatible Python objects (e.g. numpy image arrays) "
                "cannot be represented in polars/Arrow."
            ) from exc
    raise TypeError(
        f"Unsupported type for CytoDataFrame engine conversion: {type(data)!r}"
    )


def to_lazyframe(data: TabularData) -> "pl.LazyFrame":
    """Convert any supported tabular input to a :class:`polars.LazyFrame`."""
    pl = _polars()
    if isinstance(data, pl.LazyFrame):
        return data
    return to_polars(data).lazy()


def to_arrow(data: TabularData, *, preserve_index: bool = False) -> "pa.Table":
    """
    Convert any supported tabular input to a :class:`pyarrow.Table`.

    Arrow is the canonical schema/serialization contract, so this is the
    conversion used whenever schema or interchange guarantees matter.
    """
    pa = _pyarrow()
    if is_arrow_table(data):
        return data
    if is_polars_lazyframe(data):
        return data.collect().to_arrow()
    if is_polars_dataframe(data):
        return data.to_arrow()
    if isinstance(data, pd.Series):
        data = data.to_frame()
    if isinstance(data, pd.DataFrame):
        try:
            return pa.Table.from_pandas(
                pd.DataFrame(data), preserve_index=preserve_index
            )
        except (pa.ArrowInvalid, pa.ArrowTypeError, TypeError) as exc:
            raise TypeError(
                "Could not convert pandas data to an Arrow table. Columns "
                "holding non-Arrow-compatible Python objects (e.g. numpy image "
                "arrays) cannot be represented in Arrow."
            ) from exc
    raise TypeError(
        f"Unsupported type for CytoDataFrame engine conversion: {type(data)!r}"
    )


def normalize_to_pandas(data: TabularData) -> pd.DataFrame:
    """
    Normalize a supported input to pandas for the compatibility facade.

    This is the ingestion entry point used by ``CytoDataFrame.__init__`` to wrap
    Polars/Arrow inputs while keeping pandas as the backing store.
    """
    return to_pandas(data)


def scan_parquet(
    source: Union[str, pathlib.Path], **kwargs: Any
) -> "pl.LazyFrame":
    """
    Lazily scan a Parquet file/dataset into a :class:`polars.LazyFrame`.

    This enables predicate/projection pushdown for large profiling datasets
    without materializing them eagerly.
    """
    pl = _polars()
    return pl.scan_parquet(source, **kwargs)


def read_parquet(source: Union[str, pathlib.Path], **kwargs: Any) -> "pl.DataFrame":
    """Eagerly read a Parquet file into a :class:`polars.DataFrame`."""
    pl = _polars()
    return pl.read_parquet(source, **kwargs)
