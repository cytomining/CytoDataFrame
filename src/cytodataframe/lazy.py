"""
Lazy Polars query builder for CytoDataFrame.

``CytoLazyFrame`` wraps a :class:`polars.LazyFrame` and carries the
CytoDataFrame "context" (image directories, display options, ...) so that a
lazy pipeline can be materialized back into a fully-configured
:class:`~cytodataframe.frame.CytoDataFrame`.

This is the surface that powers the lazy-execution example from the evolution
plan::

    (
        CytoDataFrame.scan_parquet("profiles.parquet")
        .filter(pl.col("Metadata_Well") == "A01")
        .select_features()
        .collect()
    )

It is intentionally a *separate* type from ``CytoDataFrame`` so that its
polars-native ``filter``/``select`` semantics never collide with pandas' own
``DataFrame.filter`` (which CytoDataFrame inherits and relies on internally).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence

from . import engine
from .schema import CytoSchema

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd
    import polars as pl
    import pyarrow as pa

    from .frame import CytoDataFrame


# Constructor kwargs that carry image/display context and should survive a lazy
# pipeline so ``collect()`` rebuilds an equivalently-configured CytoDataFrame.
_CONTEXT_KEYS = (
    "data_context_dir",
    "data_mask_context_dir",
    "data_outline_context_dir",
    "segmentation_file_regex",
    "image_adjustment",
    "display_options",
)

# Number of column names shown in a CytoLazyFrame repr before truncating.
_REPR_PREVIEW_COLS = 8


class CytoLazyGroupBy:
    """Thin wrapper around a polars lazy group-by that returns a CytoLazyFrame."""

    def __init__(self, group_by: Any, context: Dict[str, Any]) -> None:
        self._group_by = group_by
        self._context = context

    def agg(self, *aggs: Any, **named_aggs: Any) -> "CytoLazyFrame":
        """Aggregate grouped data, returning a :class:`CytoLazyFrame`."""
        return CytoLazyFrame(
            self._group_by.agg(*aggs, **named_aggs), context=self._context
        )


class CytoLazyFrame:
    """
    A lazy, Polars-backed view over CytoDataFrame data.

    The wrapped :class:`polars.LazyFrame` is the canonical execution engine;
    operations build up a query plan and only execute on :meth:`collect`.
    """

    def __init__(
        self,
        data: Any,
        *,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._lf: "pl.LazyFrame" = engine.to_lazyframe(data)
        self._context: Dict[str, Any] = dict(context or {})

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    @property
    def lazyframe(self) -> "pl.LazyFrame":
        """The underlying :class:`polars.LazyFrame`."""
        return self._lf

    @property
    def context(self) -> Dict[str, Any]:
        """The CytoDataFrame context carried through the pipeline."""
        return dict(self._context)

    @property
    def columns(self) -> List[str]:
        """Column names of the (lazily) resolved schema."""
        return list(self._lf.collect_schema().names())

    @property
    def cyto_schema(self) -> CytoSchema:
        """Infer a :class:`CytoSchema` from the lazy schema (no data scan)."""
        return CytoSchema.from_polars(self._lf)

    def _wrap(self, lazyframe: "pl.LazyFrame") -> "CytoLazyFrame":
        """Wrap a derived LazyFrame, preserving context."""
        return CytoLazyFrame(lazyframe, context=self._context)

    # ------------------------------------------------------------------ #
    # Table operations (delegated to polars, return CytoLazyFrame)
    # ------------------------------------------------------------------ #
    def filter(self, *predicates: Any, **constraints: Any) -> "CytoLazyFrame":
        """Filter rows. Mirrors :meth:`polars.LazyFrame.filter`."""
        return self._wrap(self._lf.filter(*predicates, **constraints))

    def select(self, *exprs: Any, **named_exprs: Any) -> "CytoLazyFrame":
        """Select/transform columns. Mirrors :meth:`polars.LazyFrame.select`."""
        return self._wrap(self._lf.select(*exprs, **named_exprs))

    def with_columns(self, *exprs: Any, **named_exprs: Any) -> "CytoLazyFrame":
        """Add/replace columns. Mirrors :meth:`polars.LazyFrame.with_columns`."""
        return self._wrap(self._lf.with_columns(*exprs, **named_exprs))

    def rename(self, mapping: Dict[str, str], **kwargs: Any) -> "CytoLazyFrame":
        """Rename columns. Mirrors :meth:`polars.LazyFrame.rename`."""
        return self._wrap(self._lf.rename(mapping, **kwargs))

    def drop(self, *columns: Any, **kwargs: Any) -> "CytoLazyFrame":
        """Drop columns. Mirrors :meth:`polars.LazyFrame.drop`."""
        return self._wrap(self._lf.drop(*columns, **kwargs))

    def sort(self, *by: Any, **kwargs: Any) -> "CytoLazyFrame":
        """Sort rows. Mirrors :meth:`polars.LazyFrame.sort`."""
        return self._wrap(self._lf.sort(*by, **kwargs))

    def unique(self, *args: Any, **kwargs: Any) -> "CytoLazyFrame":
        """Drop duplicate rows. Mirrors :meth:`polars.LazyFrame.unique`."""
        return self._wrap(self._lf.unique(*args, **kwargs))

    def head(self, n: int = 5) -> "CytoLazyFrame":
        """Return the first ``n`` rows lazily."""
        return self._wrap(self._lf.head(n))

    def tail(self, n: int = 5) -> "CytoLazyFrame":
        """Return the last ``n`` rows lazily."""
        return self._wrap(self._lf.tail(n))

    def limit(self, n: int = 5) -> "CytoLazyFrame":
        """Limit to ``n`` rows lazily."""
        return self._wrap(self._lf.limit(n))

    def join(
        self,
        other: "CytoLazyFrame | pl.LazyFrame | pl.DataFrame | pd.DataFrame",
        *args: Any,
        **kwargs: Any,
    ) -> "CytoLazyFrame":
        """
        Join against another frame. Mirrors :meth:`polars.LazyFrame.join`.

        ``other`` may be a CytoLazyFrame, polars LazyFrame/DataFrame, or pandas
        DataFrame; it is normalized to a LazyFrame first.
        """
        if isinstance(other, CytoLazyFrame):
            other_lf = other._lf
        else:
            other_lf = engine.to_lazyframe(other)
        return self._wrap(self._lf.join(other_lf, *args, **kwargs))

    def group_by(self, *by: Any, **kwargs: Any) -> CytoLazyGroupBy:
        """Group rows for aggregation. Mirrors :meth:`polars.LazyFrame.group_by`."""
        return CytoLazyGroupBy(self._lf.group_by(*by, **kwargs), self._context)

    def select_features(
        self,
        features: Optional[Iterable[str]] = None,
        *,
        keep_metadata: bool = True,
    ) -> "CytoLazyFrame":
        """
        Select feature columns (optionally keeping metadata identifiers).

        When ``features`` is omitted, the schema-inferred feature columns are
        used. When ``keep_metadata`` is True, metadata/identifier/image columns
        are retained alongside the selected features, preserving original column
        order.
        """
        import polars as pl

        schema = self.cyto_schema
        available = self.columns
        available_set = set(available)

        if features is None:
            feature_set = set(schema.feature_columns)
        else:
            feature_set = {str(f) for f in features}

        keep = set(feature_set)
        if keep_metadata:
            keep.update(schema.metadata_columns)

        selected = [col for col in available if col in keep and col in available_set]
        return self._wrap(self._lf.select([pl.col(c) for c in selected]))

    # ------------------------------------------------------------------ #
    # Materialization
    # ------------------------------------------------------------------ #
    def collect_polars(self, **kwargs: Any) -> "pl.DataFrame":
        """Execute the query plan, returning an eager :class:`polars.DataFrame`."""
        return self._lf.collect(**kwargs)

    def to_polars(self, **kwargs: Any) -> "pl.DataFrame":
        """Alias for :meth:`collect_polars`."""
        return self.collect_polars(**kwargs)

    def to_arrow(self, **kwargs: Any) -> "pa.Table":
        """Execute and return a :class:`pyarrow.Table`."""
        return self.collect_polars(**kwargs).to_arrow()

    def to_pandas(self, **kwargs: Any) -> "pd.DataFrame":
        """Execute and return a :class:`pandas.DataFrame`."""
        return self.collect_polars(**kwargs).to_pandas()

    def collect(self, **kwargs: Any) -> "CytoDataFrame":
        """
        Execute the query plan and return a configured ``CytoDataFrame``.

        The CytoDataFrame is rebuilt with the image/display context that was
        carried through the lazy pipeline.
        """
        # Imported lazily to avoid a circular import at module load time.
        from .frame import CytoDataFrame

        pandas_df = self.collect_polars(**kwargs).to_pandas()
        context = {k: v for k, v in self._context.items() if k in _CONTEXT_KEYS}
        return CytoDataFrame(pandas_df, **context)

    def __repr__(self) -> str:
        try:
            cols = self.columns
            head = cols[:_REPR_PREVIEW_COLS]
            preview = ", ".join(head) + (
                " ..." if len(cols) > _REPR_PREVIEW_COLS else ""
            )
        except Exception:  # repr must never raise
            preview = "<unresolved schema>"
        return f"CytoLazyFrame(columns=[{preview}])"


def build_context(custom_attrs: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the carry-through context from a CytoDataFrame ``_custom_attrs``."""
    return {key: custom_attrs.get(key) for key in _CONTEXT_KEYS}


def scan_parquet(
    source: Any,
    *,
    context: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> CytoLazyFrame:
    """Lazily scan a Parquet source into a :class:`CytoLazyFrame`."""
    return CytoLazyFrame(engine.scan_parquet(source, **kwargs), context=context)


def from_sequence_context(keys: Sequence[str], values: Sequence[Any]) -> Dict[str, Any]:
    """Build a context dict from parallel key/value sequences (helper for tests)."""
    return dict(zip(keys, values, strict=False))
