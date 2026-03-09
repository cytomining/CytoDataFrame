"""
Defines a CytoDataFrame class.
"""

import base64
import contextlib
import logging
import os
import pathlib
import re
import sys
import tempfile
import uuid
import warnings
from collections import OrderedDict
from io import BytesIO, StringIO
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

import imageio.v2 as imageio
import ipywidgets as widgets
import numpy as np
import pandas as pd
import skimage
from IPython import get_ipython
from IPython.display import HTML, Javascript, display
from pandas._config import (
    get_option,
)
from pandas.io.formats import (
    format as fmt,
)
from skimage.util import img_as_ubyte

from .image import (
    add_image_scale_bar,
    adjust_with_adaptive_histogram_equalization,
    draw_outline_on_image_from_mask,
    draw_outline_on_image_from_outline,
    get_pixel_bbox_from_offsets,
)
from .volume import (
    build_3d_html_from_path,
    build_3d_image_html_stub,
    build_3d_image_html_view,
    build_3d_vtk_js_initializer,
    extract_volume_from_ome_arrow,
)

logger = logging.getLogger(__name__)
MIN_VOLUME_NDIM = 3
RGB_LIKE_CHANNEL_COUNTS = (MIN_VOLUME_NDIM, 4)
MIN_RGB_SPATIAL_DIM = 8
MAX_RGB_ASPECT_RATIO = 4.0
MIN_POSITION_COMPONENTS = 2

# provide backwards compatibility for Self type in earlier Python versions.
# see: https://peps.python.org/pep-0484/#annotating-instance-and-class-methods
CytoDataFrame_type = TypeVar("CytoDataFrame_type", bound="CytoDataFrame")


class CytoDataFrame(pd.DataFrame):
    """
    A class designed to enhance single-cell data handling by wrapping
    pandas DataFrame capabilities, providing advanced methods for quality control,
    comprehensive analysis, and image-based data processing.

    This class can initialize with either a pandas DataFrame or a file path (CSV, TSV,
    TXT, or Parquet). When initialized with a file path, it reads the data into a
    pandas DataFrame. It also includes capabilities to export data.

    Attributes:
        _metadata (ClassVar[list[str]]):
            A class-level attribute that includes custom attributes.
        _custom_attrs (dict):
            A dictionary to store custom attributes, such as data source,
            context directory, and bounding box information.
    """

    _metadata: ClassVar = ["_custom_attrs"]
    _HTML_3D_STUB_KEY: ClassVar[str] = "_cyto_3d_html_stub"

    def __init__(  # noqa: PLR0913
        self: CytoDataFrame_type,
        data: Union[CytoDataFrame_type, pd.DataFrame, str, pathlib.Path],
        data_context_dir: Optional[str] = None,
        data_image_paths: Optional[pd.DataFrame] = None,
        data_bounding_box: Optional[pd.DataFrame] = None,
        compartment_center_xy: Optional[Union[pd.DataFrame, bool]] = None,
        data_mask_context_dir: Optional[str] = None,
        data_outline_context_dir: Optional[str] = None,
        segmentation_file_regex: Optional[Dict[str, str]] = None,
        image_adjustment: Optional[Callable] = None,
        display_options: Optional[Dict[str, Any]] = None,
        *args: Tuple[Any, ...],
        **kwargs: Dict[str, Any],
    ) -> None:
        """
        Initializes the CytoDataFrame with either a DataFrame or a file path.

        Args:
            data (Union[CytoDataFrame_type, pd.DataFrame, str, pathlib.Path]):
                The data source, either a pandas DataFrame or a file path.
            data_context_dir (Optional[str]):
                Directory context for the image data within the DataFrame.
            data_image_paths (Optional[pd.DataFrame]):
                Image path data for the image files.
            data_bounding_box (Optional[pd.DataFrame]):
                Bounding box data for the DataFrame images.
            compartment_center_xy: Optional[Union[pd.DataFrame, bool]]:
                Center coordinates for the compartments in the DataFrame.
                If the value is None the default behavior is to find columns
                related to the compartment center xy data and indicate red dots
                where those points are within the cropped image display through
                Jupyter notebooks. If the value is False then no compartment
                center xy data will be used for the DataFrame.
            data_mask_context_dir: Optional[str]:
                Directory context for the mask data for images.
            data_outline_context_dir: Optional[str]:
                Directory context for the outline data for images.
            segmentation_file_regex: Optional[Dict[str, str]]:
                A dictionary which includes regex strings for mapping segmentation
                images (masks or outlines) to unsegmented images.
            image_adjustment: Callable
                A callable function which will be used to make image adjustments
                when they are processed by CytoDataFrame. The function should
                include a single parameter which takes as input a np.ndarray and
                return the same after adjustments. Defaults to None,
                which will incur an adaptive histogram equalization on images.
                Reference histogram equalization for more information:
                https://scikit-image.org/docs/stable/auto_examples/color_exposure/
            display_options: Optional[Dict[str, Any]]:
                A dictionary of display options for the DataFrame images.
                This can include options like 'width', 'height', etc.
                which are used to specify the display size of images in HTML.
                Options:
                - 'outline_color': Color of the outline to be drawn on the image.
                e.g. {'outline_color': (255, 0, 0)} for red.
                - 'brightness': Sets dynamic brightness for the images and
                sets a default for the interactive widget slider.
                The value should be between 0 and 100.
                e.g. {'brightness': 20} to set the brightness to 20%.
                - 'width': Width of the displayed image in pixels. A value of
                None will default to use automatic / default adjustments.
                e.g. {'width': 300} for 300 pixels width.
                - 'height': Height of the displayed image in pixels. A value of
                None will default to use automatic / default adjustments.
                e.g. {'height': 300} for 300 pixels height.
                - 'center_dot': Whether to draw a red dot at the compartment center
                None will default to display a center dot.
                e.g. {'center_dot': True} to draw a red dot at the compartment center.
                - 'offset_bounding_box': declare a relative bounding box using
                the nuclei center xy coordinates to dynamically crop all images
                by offsets from the center of the bounding box.
                (overriding the bounding box data from the dataframe).
                e.g. {'bounding_box':
                {'x_min': -100, 'y_min': -100, 'x_max': 100, 'y_max': 100}
                }
                - 'scale_bar': Adds a physical scale bar to each displayed crop.
                  note: um / pixel details can often be found within the metadata
                  of the images themselves or within the experiment documentation.
                  e.g. {
                      'um_per_pixel': 0.325,        # required if not set globally
                      'pixel_per_um': 3.07692307692,# required if not set globally
                      'length_um': 10.0,            # default 10
                      'thickness_px': 4,            # default 4
                      'color': (255, 255, 255),     # RGB, default white
                      'location': 'lower right',    # 'lower/upper left/right'
                      'margin_px': 10,              # default 10
                      'font_size_px': 14,           # best-effort with PIL default font
                  }
                - Alternatively, set a global pixel size in 'display_options':
                  {'um_per_pixel': 0.325}  # used if not provided under 'scale_bar'
                - 'ignore_image_path_columns': When True and a data_context_dir is set,
                  ignore any PathName_* or other image path columns and resolve images
                  only via data_context_dir + filename.
                - 'view': Optional UI preference for 3D rendering. Use "trame" to
                  prefer the trame backend when backend is not explicitly set.
            **kwargs:
                Additional keyword arguments to pass to the pandas read functions.
        """

        initial_brightness = (
            # set to 50 if no display options are provided
            50
            if not (display_options and display_options.get("brightness"))
            # otherwise use the brightness value from display options
            else display_options.get("brightness")
        )

        self._custom_attrs = {
            "data_source": None,
            "data_context_dir": (
                data_context_dir if data_context_dir is not None else None
            ),
            "data_image_paths": None,
            "data_bounding_box": None,
            "compartment_center_xy": None,
            "data_mask_context_dir": (
                data_mask_context_dir if data_mask_context_dir is not None else None
            ),
            "data_outline_context_dir": (
                data_outline_context_dir
                if data_outline_context_dir is not None
                else None
            ),
            "segmentation_file_regex": (
                segmentation_file_regex if segmentation_file_regex is not None else None
            ),
            "image_adjustment": (
                image_adjustment if image_adjustment is not None else None
            ),
            "display_options": (
                display_options if display_options is not None else None
            ),
            "is_transposed": False,
            # add widget control meta
            "_widget_state": {
                "scale": initial_brightness,
                "shown": False,  # whether VBox has been displayed
                "observing": False,  # whether slider observer is attached
            },
            "_snapshot_cache": {},
            "_volume_cache": {},
            "_scale_slider": widgets.IntSlider(
                value=initial_brightness,
                min=0,
                max=100,
                step=1,
                description="Image adjustment:",
                continuous_update=False,
                style={"description_width": "auto"},
            ),
            "_output": widgets.Output(),
        }

        if self._custom_attrs["data_context_dir"] is not None:
            logger.debug(
                "CytoDataFrame data_context_dir set to: %s",
                self._custom_attrs["data_context_dir"],
            )

        if isinstance(data, CytoDataFrame):
            self._custom_attrs["data_source"] = data._custom_attrs["data_source"]
            self._custom_attrs["data_context_dir"] = data._custom_attrs[
                "data_context_dir"
            ]
            self._custom_attrs["data_mask_context_dir"] = data._custom_attrs[
                "data_mask_context_dir"
            ]
            self._custom_attrs["data_outline_context_dir"] = data._custom_attrs[
                "data_outline_context_dir"
            ]
            super().__init__(data)
        elif isinstance(data, (pd.DataFrame, pd.Series)):
            self._custom_attrs["data_source"] = (
                "pandas.DataFrame"
                if isinstance(data, pd.DataFrame)
                else "pandas.Series"
            )
            super().__init__(data)
        elif isinstance(data, (str, pathlib.Path)):
            data_path = pathlib.Path(data)
            self._custom_attrs["data_source"] = str(data_path)

            if data_context_dir is None:
                self._custom_attrs["data_context_dir"] = str(data_path.parent)
            else:
                self._custom_attrs["data_context_dir"] = data_context_dir

            if data_path.suffix in {".csv", ".tsv", ".txt"} or data_path.suffixes == [
                ".csv",
                ".gz",
            ]:
                data = pd.read_csv(data_path, **kwargs)
            elif data_path.suffix == ".parquet":
                data = pd.read_parquet(data_path, **kwargs)
            else:
                raise ValueError("Unsupported file format for CytoDataFrame.")

            super().__init__(data)

        else:
            super().__init__(data)

        self._custom_attrs["data_bounding_box"] = (
            self.get_bounding_box_from_data()
            if data_bounding_box is None
            else data_bounding_box
        )

        self._custom_attrs["compartment_center_xy"] = (
            self.get_compartment_center_xy_from_data()
            if compartment_center_xy is None or compartment_center_xy is True
            else compartment_center_xy
            if compartment_center_xy is not False
            else None
        )

        self._custom_attrs["data_image_paths"] = (
            self.get_image_paths_from_data(image_cols=self.find_image_columns())
            if data_image_paths is None
            else data_image_paths
        )

        # Wrap methods so they return CytoDataFrames
        # instead of Pandas DataFrames.
        self._wrap_methods()

    def __getitem__(self: CytoDataFrame_type, key: Union[int, str]) -> Any:
        """
        Returns an element or a slice of the underlying pandas DataFrame.

        Args:
            key:
                The key or slice to access the data.

        Returns:
            pd.DataFrame or any:
                The selected element or slice of data.
        """

        result = super().__getitem__(key)

        if isinstance(result, pd.Series):
            return result

        elif isinstance(result, pd.DataFrame):
            cdf = CytoDataFrame(
                super().__getitem__(key),
                data_context_dir=self._custom_attrs["data_context_dir"],
                data_image_paths=self._custom_attrs["data_image_paths"],
                data_bounding_box=self._custom_attrs["data_bounding_box"],
                compartment_center_xy=self._custom_attrs["compartment_center_xy"],
                data_mask_context_dir=self._custom_attrs["data_mask_context_dir"],
                data_outline_context_dir=self._custom_attrs["data_outline_context_dir"],
                segmentation_file_regex=self._custom_attrs["segmentation_file_regex"],
                image_adjustment=self._custom_attrs["image_adjustment"],
                display_options=self._custom_attrs["display_options"],
            )

            # add widget control meta
            cdf._custom_attrs["_widget_state"] = self._custom_attrs["_widget_state"]
            cdf._custom_attrs["_scale_slider"] = self._custom_attrs["_scale_slider"]
            cdf._custom_attrs["_output"] = self._custom_attrs["_output"]

            return cdf

    def _return_cytodataframe(
        self: CytoDataFrame_type,
        method: Callable,
        method_name: str,
        *args: Tuple[Any, ...],
        **kwargs: Dict[str, Any],
    ) -> Any:
        """
        Wraps a given method to ensure that the returned result
        is an CytoDataFrame if applicable.

        Args:
            method (Callable):
                The method to be called and wrapped.
            method_name (str):
                The name of the method to be wrapped.
            *args (Tuple[Any, ...]):
                Positional arguments to be passed to the method.
            **kwargs (Dict[str, Any]):
                Keyword arguments to be passed to the method.

        Returns:
            Any:
                The result of the method call. If the result is a pandas DataFrame,
                it is wrapped in an CytoDataFrame instance with additional context
                information (data context directory and data bounding box).

        """

        result = method(*args, **kwargs)

        if isinstance(result, pd.DataFrame):
            cdf = CytoDataFrame(
                data=result,
                data_context_dir=self._custom_attrs["data_context_dir"],
                data_image_paths=self._custom_attrs["data_image_paths"],
                data_bounding_box=self._custom_attrs["data_bounding_box"],
                compartment_center_xy=self._custom_attrs["compartment_center_xy"],
                data_mask_context_dir=self._custom_attrs["data_mask_context_dir"],
                data_outline_context_dir=self._custom_attrs["data_outline_context_dir"],
                segmentation_file_regex=self._custom_attrs["segmentation_file_regex"],
                image_adjustment=self._custom_attrs["image_adjustment"],
                display_options=self._custom_attrs["display_options"],
            )
            # If the method name is transpose we know that
            # the dataframe has been transposed.
            if method_name == "transpose" and not self._custom_attrs["is_transposed"]:
                cdf._custom_attrs["is_transposed"] = True

            # add widget control meta
            cdf._custom_attrs["_widget_state"] = self._custom_attrs["_widget_state"]
            cdf._custom_attrs["_scale_slider"] = self._custom_attrs["_scale_slider"]
            cdf._custom_attrs["_output"] = self._custom_attrs["_output"]

        return cdf

    def _wrap_method(self: CytoDataFrame_type, method_name: str) -> Callable:
        """
        Creates a wrapper for the specified method
        to ensure it returns a CytoDataFrame.

        This method dynamically wraps a given
        method of the CytoDataFrame class to ensure
        that the returned result is a CytoDataFrame
        instance, preserving custom attributes.

        Args:
            method_name (str):
                The name of the method to wrap.

        Returns:
            Callable:
                The wrapped method that ensures
                the result is a CytoDataFrame.
        """

        def wrapper(*args: Tuple[Any, ...], **kwargs: Dict[str, Any]) -> Any:
            """
            Wraps the specified method to ensure
            it returns a CytoDataFrame.

            This function dynamically wraps a given
            method of the CytoDataFrame class
            to ensure that the returned result
            is a CytoDataFrame instance, preserving
            custom attributes.

            Args:
                *args (Tuple[Any, ...]):
                    Positional arguments to be passed to the method.
                **kwargs (Dict[str, Any]):
                    Keyword arguments to be passed to the method.

            Returns:
                Any:
                    The result of the method call.
                    If the result is a pandas DataFrame,
                    it is wrapped in a CytoDataFrame
                    instance with additional context
                    information (data context directory
                    and data bounding box).
            """

            method = getattr(super(CytoDataFrame, self), method_name)

            return self._return_cytodataframe(
                # important: we pass method and method_name
                # as positional args to avoid collisions
                # with the method signatures and chained
                # calls which might be made.
                method,
                method_name,
                *args,
                **kwargs,
            )

        return wrapper

    def _wrap_methods(self) -> None:
        """
        Method to wrap extended Pandas DataFrame methods
        so they return a CytoDataFrame instead of a
        Pandas DataFrame.
        """

        # list of methods by name from Pandas DataFrame class
        methods_to_wrap = ["head", "tail", "sort_values", "sample", "transpose"]

        # set the wrapped method for the class instance
        for method_name in methods_to_wrap:
            setattr(self, method_name, self._wrap_method(method_name=method_name))

    def _on_slider_change(self: CytoDataFrame_type, change: Dict[str, Any]) -> None:
        """
        Callback triggered when the image brightness/contrast
        slider is adjusted.

        This method updates the internal `_widget_state` to reflect
        the new slider value, clears the current output display, and
        triggers a re-render of the CytoDataFrame's HTML representation
        (including image thumbnails) based on the new scale setting.

        Args:
            change (dict):
                A dictionary provided by the
                ipywidgets observer mechanism.
                Expected to contain a `'new'`
                key representing the updated slider value.
        """

        self._custom_attrs["_widget_state"]["scale"] = change["new"]
        self._custom_attrs["_output"].clear_output(wait=True)

        # redraw output after adjustments to scale state
        self._render_output()

    def get_bounding_box_from_data(
        self: CytoDataFrame_type,
    ) -> Optional[CytoDataFrame_type]:
        """
        Retrieves bounding box data from the DataFrame based
        on predefined column groups.

        This method identifies specific groups of columns representing bounding box
        coordinates for different cellular components (cytoplasm, nuclei, cells). If
        those are not present, it falls back to a generic AreaShape bounding box. If
        all required columns are present, it filters and returns a new CytoDataFrame
        instance containing only these columns.

        Returns:
            Optional[CytoDataFrame_type]:
                A new instance of CytoDataFrame containing the bounding box columns if
                they exist in the DataFrame. Returns None if the required columns
                are not found.

        """
        # Define column groups and their corresponding conditions
        column_groups = {
            "cyto": [
                "Cytoplasm_AreaShape_BoundingBoxMaximum_X",
                "Cytoplasm_AreaShape_BoundingBoxMaximum_Y",
                "Cytoplasm_AreaShape_BoundingBoxMinimum_X",
                "Cytoplasm_AreaShape_BoundingBoxMinimum_Y",
            ],
            "nuclei": [
                "Nuclei_AreaShape_BoundingBoxMaximum_X",
                "Nuclei_AreaShape_BoundingBoxMaximum_Y",
                "Nuclei_AreaShape_BoundingBoxMinimum_X",
                "Nuclei_AreaShape_BoundingBoxMinimum_Y",
            ],
            "cells": [
                "Cells_AreaShape_BoundingBoxMaximum_X",
                "Cells_AreaShape_BoundingBoxMaximum_Y",
                "Cells_AreaShape_BoundingBoxMinimum_X",
                "Cells_AreaShape_BoundingBoxMinimum_Y",
            ],
            "generic": [
                "AreaShape_BoundingBoxMaximum_X",
                "AreaShape_BoundingBoxMaximum_Y",
                "AreaShape_BoundingBoxMinimum_X",
                "AreaShape_BoundingBoxMinimum_Y",
            ],
        }
        column_groups_z = {
            "cyto": [
                "Cytoplasm_AreaShape_BoundingBoxMaximum_Z",
                "Cytoplasm_AreaShape_BoundingBoxMinimum_Z",
            ],
            "nuclei": [
                "Nuclei_AreaShape_BoundingBoxMaximum_Z",
                "Nuclei_AreaShape_BoundingBoxMinimum_Z",
            ],
            "cells": [
                "Cells_AreaShape_BoundingBoxMaximum_Z",
                "Cells_AreaShape_BoundingBoxMinimum_Z",
            ],
            "generic": [
                "AreaShape_BoundingBoxMaximum_Z",
                "AreaShape_BoundingBoxMinimum_Z",
            ],
        }

        # Determine which group of columns to select based on availability in self.data
        selected_group = None
        ordered_groups = ("cyto", "nuclei", "cells", "generic")
        for group in ordered_groups:
            cols = column_groups[group]
            if all(col in self.columns.tolist() for col in cols):
                selected_group = group
                break

        # Assign the selected columns to self.bounding_box_df
        if selected_group:
            z_cols = column_groups_z.get(selected_group, [])
            if z_cols and all(col in self.columns.tolist() for col in z_cols):
                column_groups[selected_group] = column_groups[selected_group] + z_cols
            logger.debug(
                "Bounding box columns found: %s",
                column_groups[selected_group],
            )
            return self.filter(items=column_groups[selected_group])

        logger.debug(
            "Found no bounding box columns.",
        )

        return None

    def get_compartment_center_xy_from_data(
        self: CytoDataFrame_type,
    ) -> Optional[CytoDataFrame_type]:
        """
        Retrieves compartment center xy data from the
        DataFrame based on predefined column groups.

        This method identifies specific groups of columns representing center xy
        coordinates for different cellular components (cytoplasm, nuclei, cells) and
        checks for their presence in the DataFrame. If all required columns are present,
        it filters and returns a new CytoDataFrame instance containing only these
        columns.

        Returns:
            Optional[CytoDataFrame_type]:
                A new instance of CytoDataFrame containing the bounding box columns if
                they exist in the DataFrame. Returns None if the required columns
                are not found.

        """
        # Define column groups and their corresponding conditions
        column_groups = {
            "nuclei": [
                "Nuclei_Location_Center_X",
                "Nuclei_Location_Center_Y",
            ],
            "nuclei_w_meta": [
                "Metadata_Nuclei_Location_Center_X",
                "Metadata_Nuclei_Location_Center_Y",
            ],
            "cells": [
                "Cells_Location_Center_X",
                "Cells_Location_Center_Y",
            ],
            "cells_w_meta": [
                "Metadata_Cells_Location_Center_X",
                "Metadata_Cells_Location_Center_Y",
            ],
            "cyto": [
                "Cytoplasm_Location_Center_X",
                "Cytoplasm_Location_Center_Y",
            ],
            "cyto_w_meta": [
                "Metadata_Cytoplasm_Location_Center_X",
                "Metadata_Cytoplasm_Location_Center_Y",
            ],
        }

        # Determine which group of columns to select based on availability in self.data
        selected_group = None
        for group, cols in column_groups.items():
            if all(col in self.columns.tolist() for col in cols):
                selected_group = group
                break

        # Assign the selected columns to self.compartment_center_xy
        if selected_group:
            logger.debug(
                "Compartment center xy columns found: %s",
                column_groups[selected_group],
            )
            return self.filter(items=column_groups[selected_group])

        logger.debug(
            "Found no compartment center xy columns.",
        )

        return None

    def export(
        self: CytoDataFrame_type, file_path: str, **kwargs: Dict[str, Any]
    ) -> None:
        """
        Exports the underlying pandas DataFrame to a file.

        Args:
            file_path (str):
                The path where the DataFrame should be saved.
            **kwargs:
                Additional keyword arguments to pass to the pandas to_* methods.
        """

        data_path = pathlib.Path(file_path)

        # export to csv
        if ".csv" in data_path.suffixes:
            self.to_csv(file_path, **kwargs)
        # export to tsv
        elif any(elem in data_path.suffixes for elem in (".tsv", ".txt")):
            self.to_csv(file_path, sep="\t", **kwargs)
        # export to parquet
        elif data_path.suffix == ".parquet":
            self.to_parquet(file_path, **kwargs)
        else:
            raise ValueError("Unsupported file format for export.")

    def to_ome_parquet(  # noqa: PLR0915, PLR0912, C901
        self: CytoDataFrame_type,
        file_path: Union[str, pathlib.Path],
        arrow_column_suffix: str = "_OMEArrow",
        include_original: bool = True,
        include_mask_outline: bool = True,
        include_composite: bool = True,
        **kwargs: Dict[str, Any],
    ) -> None:
        """Export the dataframe with cropped images encoded as OMEArrow structs."""

        try:
            from ome_arrow import OMEArrow  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "CytoDataFrame.to_ome_parquet requires the optional 'ome-arrow' "
                "dependency. Install it via `pip install ome-arrow`."
            ) from exc
        try:
            from ome_arrow import from_numpy as ome_from_numpy  # type: ignore
        except ImportError:
            ome_from_numpy = None

        try:
            import importlib.metadata as importlib_metadata
        except ImportError:  # pragma: no cover
            import importlib_metadata  # type: ignore

        try:
            ome_arrow_version = importlib_metadata.version("ome-arrow")
        except importlib_metadata.PackageNotFoundError:
            module = sys.modules.get("ome_arrow")
            ome_arrow_version = getattr(module, "__version__", None)

        if not any((include_original, include_mask_outline, include_composite)):
            raise ValueError(
                "At least one of include_original, include_mask_outline, or "
                "include_composite must be True."
            )

        image_cols = self.find_image_columns() or []
        if not image_cols:
            logger.debug(
                "No image filename columns detected. Falling back to to_parquet()."
            )
            self.to_parquet(file_path, **kwargs)
            return

        bounding_box_df = self._custom_attrs.get("data_bounding_box")
        if bounding_box_df is None:
            raise ValueError(
                "to_ome_parquet requires bounding box metadata to crop images."
            )

        bounding_box_cols = bounding_box_df.columns.tolist()
        bbox_column_map = {
            "x_min": next(
                (col for col in bounding_box_cols if "Minimum_X" in str(col)), None
            ),
            "y_min": next(
                (col for col in bounding_box_cols if "Minimum_Y" in str(col)), None
            ),
            "x_max": next(
                (col for col in bounding_box_cols if "Maximum_X" in str(col)), None
            ),
            "y_max": next(
                (col for col in bounding_box_cols if "Maximum_Y" in str(col)), None
            ),
        }

        if any(value is None for value in bbox_column_map.values()):
            raise ValueError(
                "Unable to identify all bounding box coordinate columns for export."
            )

        working_df = self.copy()

        missing_bbox_cols = [
            col for col in bounding_box_cols if col not in working_df.columns
        ]
        if missing_bbox_cols:
            working_df = working_df.join(bounding_box_df[missing_bbox_cols])

        comp_center_df = self._custom_attrs.get("compartment_center_xy")
        comp_center_cols: List[str] = []
        missing_comp_cols: List[str] = []
        if comp_center_df is not None:
            comp_center_cols = comp_center_df.columns.tolist()
            missing_comp_cols = [
                col for col in comp_center_cols if col not in working_df.columns
            ]
            if missing_comp_cols:
                working_df = working_df.join(comp_center_df[missing_comp_cols])

        image_path_df = self._custom_attrs.get("data_image_paths")
        missing_path_cols: List[str] = []
        if image_path_df is not None:
            image_path_cols_all = image_path_df.columns.tolist()
            missing_path_cols = [
                col for col in image_path_cols_all if col not in working_df.columns
            ]
            if missing_path_cols:
                working_df = working_df.join(image_path_df[missing_path_cols])

        all_cols_str, all_cols_back = self._normalize_labels(working_df.columns)
        image_cols_str = [str(col) for col in image_cols]
        image_path_cols_str = self.find_image_path_columns(
            image_cols=image_cols_str, all_cols=all_cols_str
        )
        display_options = self._custom_attrs.get("display_options", {}) or {}
        if self._custom_attrs.get("data_context_dir") and display_options.get(
            "ignore_image_path_columns"
        ):
            logger.debug("Ignoring image path columns due to display option.")
            image_path_cols_str = {}
        image_path_cols = {}
        for image_col in image_cols:
            key = str(image_col)
            if key in image_path_cols_str:
                mapped_col = image_path_cols_str[key]
                image_path_cols[image_col] = all_cols_back.get(
                    str(mapped_col), mapped_col
                )

        comp_center_x = next((col for col in comp_center_cols if "X" in str(col)), None)
        comp_center_y = next((col for col in comp_center_cols if "Y" in str(col)), None)

        kwargs.setdefault("engine", "pyarrow")

        from cytodataframe import __version__ as cytodataframe_version

        metadata = {
            "cytodataframe:data-producer": "https://github.com/cytomining/CytoDataFrame",
            "cytodataframe:data-producer-version": cytodataframe_version,
        }
        if ome_arrow_version is not None:
            metadata["cytodataframe:ome-arrow-version"] = ome_arrow_version

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = pathlib.Path(tmpdir)
            for image_col in image_cols:
                image_path_col = image_path_cols.get(image_col)

                layer_configs: List[Tuple[str, str]] = []
                if include_original:
                    layer_configs.append(
                        ("original", f"{image_col}{arrow_column_suffix}_ORIG")
                    )
                if include_mask_outline:
                    layer_configs.append(
                        ("mask", f"{image_col}{arrow_column_suffix}_LABL")
                    )
                if include_composite:
                    layer_configs.append(
                        ("composite", f"{image_col}{arrow_column_suffix}_COMP")
                    )

                column_values = {col_name: [] for _, col_name in layer_configs}

                for _, row in working_df.iterrows():
                    image_value = row.get(image_col)
                    if image_value is None or pd.isna(image_value):
                        for _, col_name in layer_configs:
                            column_values[col_name].append(None)
                        continue

                    try:
                        bbox_values = (
                            row[bbox_column_map["x_min"]],
                            row[bbox_column_map["y_min"]],
                            row[bbox_column_map["x_max"]],
                            row[bbox_column_map["y_max"]],
                        )
                    except KeyError:
                        for _, col_name in layer_configs:
                            column_values[col_name].append(None)
                        continue

                    if any(pd.isna(value) for value in bbox_values):
                        for _, col_name in layer_configs:
                            column_values[col_name].append(None)
                        continue

                    bounding_box = tuple(int(value) for value in bbox_values)

                    compartment_center = None
                    if comp_center_x and comp_center_y:
                        center_vals = (row.get(comp_center_x), row.get(comp_center_y))
                        if not any(val is None or pd.isna(val) for val in center_vals):
                            compartment_center = tuple(int(v) for v in center_vals)

                    image_path_value = (
                        row.get(image_path_col) if image_path_col is not None else None
                    )

                    layers = self._prepare_cropped_image_layers(
                        data_value=image_value,
                        bounding_box=bounding_box,
                        compartment_center_xy=compartment_center,
                        image_path=image_path_value,
                        include_original=include_original,
                        include_mask_outline=include_mask_outline,
                        include_composite=include_composite,
                    )

                    sanitized_col = re.sub(r"[^A-Za-z0-9_.-]", "_", str(image_col))

                    for layer_key, col_name in layer_configs:
                        layer_array = layers.get(layer_key)
                        if layer_array is None:
                            column_values[col_name].append(None)
                            continue

                        try:
                            # Prefer direct in-memory conversion when available.
                            # This avoids TIFF round-trips and keeps channel
                            # layout explicit.
                            if (
                                ome_from_numpy is not None
                                and layer_array.ndim < MIN_VOLUME_NDIM
                            ):
                                ome_struct = ome_from_numpy(
                                    np.asarray(layer_array),
                                    dim_order="YX",
                                )
                            elif (
                                ome_from_numpy is not None
                                and layer_array.ndim == MIN_VOLUME_NDIM
                                and layer_array.shape[-1] in RGB_LIKE_CHANNEL_COUNTS
                            ):
                                # OME-Arrow expects channels-first for
                                # 2D multi-channel arrays.
                                channel_first = np.moveaxis(
                                    np.asarray(layer_array), -1, 0
                                )
                                ome_struct = ome_from_numpy(
                                    channel_first,
                                    dim_order="CYX",
                                )
                            elif (
                                layer_array.ndim == MIN_VOLUME_NDIM
                                and layer_array.shape[-1] in RGB_LIKE_CHANNEL_COUNTS
                            ):
                                # Compatibility fallback for environments where
                                # `ome_arrow.from_numpy` is not available.
                                temp_path = tmpdir_path / (
                                    f"{sanitized_col}_{layer_key}_"
                                    f"{uuid.uuid4().hex}.tiff"
                                )
                                with warnings.catch_warnings():
                                    warnings.simplefilter("ignore", UserWarning)
                                    imageio.imwrite(
                                        temp_path,
                                        layer_array,
                                        format="tiff",
                                    )
                                ome_struct = OMEArrow(data=str(temp_path)).data
                            else:
                                # Generic fallback for all other array shapes.
                                temp_path = tmpdir_path / (
                                    f"{sanitized_col}_{layer_key}_"
                                    f"{uuid.uuid4().hex}.tiff"
                                )
                                with warnings.catch_warnings():
                                    warnings.simplefilter("ignore", UserWarning)
                                    imageio.imwrite(
                                        temp_path,
                                        layer_array,
                                        format="tiff",
                                    )
                                ome_struct = OMEArrow(data=str(temp_path)).data
                            if hasattr(ome_struct, "as_py"):
                                ome_struct = ome_struct.as_py()
                        except Exception as exc:
                            logger.error(
                                "Failed to create OMEArrow struct for %s: %s",
                                layer_key,
                                exc,
                            )
                            column_values[col_name].append(None)
                            continue
                        column_values[col_name].append(ome_struct)

                for _, col_name in layer_configs:
                    working_df[col_name] = column_values[col_name]

        if missing_bbox_cols:
            working_df = working_df.drop(columns=missing_bbox_cols)

        if missing_comp_cols:
            working_df = working_df.drop(columns=missing_comp_cols)

        if missing_path_cols:
            working_df = working_df.drop(columns=missing_path_cols)

        final_kwargs = kwargs.copy()
        engine = final_kwargs.pop("engine", None)
        existing_metadata = final_kwargs.pop("metadata", {}) or {}
        merged_metadata = {**metadata, **existing_metadata}

        index_arg = final_kwargs.pop("index", None)
        if merged_metadata:
            import pyarrow as pa
            import pyarrow.parquet as pq

            table = pa.Table.from_pandas(
                working_df,
                preserve_index=True if index_arg is None else index_arg,
            )
            existing = table.schema.metadata or {}
            new_metadata = {
                **existing,
                **{
                    str(k).encode(): str(v).encode()
                    for k, v in merged_metadata.items()
                    if v is not None
                },
            }
            table = table.replace_schema_metadata(new_metadata)
            pq.write_table(table, file_path, **final_kwargs)
        else:
            if index_arg is not None:
                final_kwargs["index"] = index_arg
            if engine is not None:
                final_kwargs["engine"] = engine
            working_df.to_parquet(file_path, **final_kwargs)

    @staticmethod
    def is_notebook_or_lab() -> bool:
        """
        Determines if the code is being executed in a Jupyter notebook (.ipynb)
        returning false if it is not.

        This method attempts to detect the interactive shell environment
        using IPython's `get_ipython` function. It checks the class name of the current
        IPython shell to distinguish between different execution environments.

        Returns:
            bool:
                - `True`
                    if the code is being executed in a Jupyter notebook (.ipynb).
                - `False`
                    otherwise (e.g., standard Python shell, terminal IPython shell,
                    or scripts).
        """
        try:
            # check for type of session via ipython
            shell = get_ipython().__class__.__name__
            if "ZMQInteractiveShell" in shell:
                return True
            elif "TerminalInteractiveShell" in shell:
                return False
            else:
                return False
        except NameError:
            return False

    def find_image_columns(self: CytoDataFrame_type) -> List[str]:
        """
        Find columns containing image file names.

        This method searches for columns in the DataFrame
        that contain image file names with extensions .tif
        or .tiff (case insensitive).

        Returns:
            List[str]:
                A list of column names that contain
                image file names.

        """
        # build a pattern to match image file names
        pattern = r".*\.(tif|tiff)$"

        # search for columns containing image file names
        # based on pattern above.
        image_cols = [
            column
            for column in self.columns
            if self[column]
            .apply(
                lambda value: (
                    isinstance(value, (str, os.PathLike))
                    and re.match(pattern, str(value), flags=re.IGNORECASE)
                )
            )
            .any()
        ]

        logger.debug("Found image columns: %s", image_cols)

        return image_cols

    @staticmethod
    def _is_ome_arrow_value(value: Any) -> bool:
        """Check whether a value looks like an OME-Arrow struct."""

        return (
            isinstance(value, dict)
            and value.get("type") == "ome.arrow"
            and value.get("planes") is not None
            and value.get("pixels_meta") is not None
        )

    def find_ome_arrow_columns(
        self: CytoDataFrame_type, data: pd.DataFrame
    ) -> List[str]:
        """Identify columns that contain OME-Arrow structs."""

        ome_cols: List[str] = []
        for column in data.columns:
            series = data[column]
            if series.apply(self._is_ome_arrow_value).any():
                ome_cols.append(column)

        if ome_cols:
            logger.debug("Found OME-Arrow columns: %s", ome_cols)

        return ome_cols

    def get_image_paths_from_data(
        self: CytoDataFrame_type, image_cols: List[str]
    ) -> Dict[str, str]:
        """
        Gather data containing image path names
        (the directory storing the images but not the file
        names). We do this by seeking the pattern:
        Image_FileName_X --> Image_PathName_X.

        Args:
            image_cols: List[str]:
                A list of column names that contain
                image file names.

        Returns:
            Dict[str, str]:
                A list of column names that contain
                image file names.

        """

        image_path_columns = [
            col.replace("FileName", "PathName")
            for col in image_cols
            if col.replace("FileName", "PathName") in self.columns
        ]

        logger.debug("Found image path columns: %s", image_path_columns)

        return self.filter(items=image_path_columns) if image_path_columns else None

    def find_image_path_columns(
        self: CytoDataFrame_type, image_cols: List[str], all_cols: List[str]
    ) -> Dict[str, str]:
        """
        Find columns containing image path names
        (the directory storing the images but not the file
        names). We do this by seeking the pattern:
        Image_FileName_X --> Image_PathName_X.

        Args:
            image_cols: List[str]:
                A list of column names that contain
                image file names.
            all_cols: List[str]:
                A list of all column names.

        Returns:
            Dict[str, str]:
                A list of column names that contain
                image file names.

        """

        return {
            str(col): str(col).replace("FileName", "PathName")
            for col in image_cols
            if str(col).replace("FileName", "PathName") in all_cols
        }

    def search_for_mask_or_outline(  # noqa: PLR0913, PLR0911, C901
        self: CytoDataFrame_type,
        data_value: str,
        pattern_map: dict,
        file_dir: str,
        candidate_path: pathlib.Path,
        orig_image: np.ndarray,
        mask: bool = True,
    ) -> Tuple[Optional[np.ndarray], Optional[pathlib.Path]]:
        """
        Search for a mask or outline image file based on the
        provided patterns and apply it to the target image.

        Args:
            data_value (str):
                The value used to match patterns for locating
                mask or outline files.
            pattern_map (dict):
                A dictionary of file patterns and their corresponding
                original patterns for matching.
            file_dir (str):
                The directory where image files are stored.
            candidate_path (pathlib.Path):
                The path to the candidate image file to apply
                the mask or outline to.
            orig_image (np.ndarray):
                The image which will have a mask or outline applied.
            mask (bool, optional):
                Whether to search for a mask (True) or an outline (False).
                Default is True.

        Returns:
            np.ndarray:
                The target image with the applied mask or outline,
                or None if no relevant file is found.
        """
        logger.debug(
            "Searching for %s in %s", "mask" if mask else "outline", data_value
        )

        if file_dir is None:
            logger.debug("No mask or outline directory specified.")
            return None, None

        if pattern_map is None:
            matching_mask_file = list(
                pathlib.Path(file_dir).rglob(f"{pathlib.Path(candidate_path).stem}*")
            )
            if matching_mask_file:
                logger.debug(
                    "Found matching mask or outline: %s", matching_mask_file[0]
                )
                # gather display options if specified
                display_options = self._custom_attrs.get("display_options", {})
                if display_options is None:
                    display_options = {}
                # gather the outline color if specified
                outline_color = display_options.get("outline_color", (0, 255, 0))

                if mask:
                    return (
                        draw_outline_on_image_from_mask(
                            orig_image=orig_image,
                            mask_image_path=matching_mask_file[0],
                            outline_color=outline_color,
                        ),
                        matching_mask_file[0],
                    )
                else:
                    return (
                        draw_outline_on_image_from_outline(
                            orig_image=orig_image,
                            outline_image_path=matching_mask_file[0],
                            outline_color=outline_color,
                        ),
                        matching_mask_file[0],
                    )
            return None, None

        for file_pattern, original_pattern in pattern_map.items():
            if re.search(original_pattern, data_value):
                matching_files = [
                    file
                    for file in pathlib.Path(file_dir).rglob("*")
                    if re.search(file_pattern, file.name)
                ]
                if matching_files:
                    logger.debug(
                        "Found matching mask or outline using regex pattern %s : %s",
                        file_pattern,
                        matching_files[0],
                    )
                    # gather display options if specified
                    display_options = self._custom_attrs.get("display_options", {})
                    if display_options is None:
                        display_options = {}
                    # gather the outline color if specified
                    outline_color = display_options.get("outline_color", (0, 255, 0))
                    if mask:
                        return (
                            draw_outline_on_image_from_mask(
                                orig_image=orig_image,
                                mask_image_path=matching_files[0],
                                outline_color=outline_color,
                            ),
                            matching_files[0],
                        )
                    else:
                        return (
                            draw_outline_on_image_from_outline(
                                orig_image=orig_image,
                                outline_image_path=matching_files[0],
                                outline_color=outline_color,
                            ),
                            matching_files[0],
                        )

        logger.debug("No mask or outline found for: %s", data_value)

        return None, None

    @staticmethod
    def _find_matching_segmentation_path(
        data_value: str,
        pattern_map: Optional[dict],
        file_dir: Optional[str],
        candidate_path: pathlib.Path,
    ) -> Optional[pathlib.Path]:
        """Resolve a mask/outline file path for an image value.

        Args:
            data_value: Raw image value from the table row.
            pattern_map: Optional regex mapping from segmentation filename
                patterns to source image patterns.
            file_dir: Root directory containing mask/outline files.
            candidate_path: Resolved image candidate path used for stem matching.

        Returns:
            The first matching segmentation file path, or ``None`` when no match is
            found.
        """
        if file_dir is None:
            return None

        root = pathlib.Path(file_dir)
        if not root.exists():
            return None

        if pattern_map is None:
            matching_files = sorted(root.rglob(f"{pathlib.Path(candidate_path).stem}*"))
            return matching_files[0] if matching_files else None

        for file_pattern, original_pattern in pattern_map.items():
            if not re.search(original_pattern, data_value):
                continue
            for file in sorted(root.rglob("*")):
                if re.search(file_pattern, file.name):
                    return file
        return None

    def _prepare_3d_label_overlay(
        self: CytoDataFrame_type,
        segmentation_path: pathlib.Path,
        expected_shape: Tuple[int, ...],
        row: Optional[Any] = None,
    ) -> Optional[np.ndarray]:
        """Load and normalize a 3D segmentation image for volume overlays.

        Args:
            segmentation_path: Path to the mask/outline image file.
            expected_shape: Expected ``(z, y, x)`` array shape.
            row: Optional row label/index used to apply 3D bounding-box cropping.

        Returns:
            A uint8 binary array (0/255) matching ``expected_shape``, or ``None``
            when loading or shape validation fails.
        """
        try:
            mask_array = np.asarray(imageio.imread(segmentation_path))
        except (FileNotFoundError, ValueError):
            return None

        if mask_array.ndim > MIN_VOLUME_NDIM and mask_array.shape[-1] in (1, 3, 4):
            mask_array = mask_array[..., 0]

        if row is not None:
            bounds = self._get_3d_bbox_crop_bounds(
                row=row,
                volume_shape=tuple(int(v) for v in mask_array.shape),
            )
            if bounds is not None:
                x_min, x_max, y_min, y_max, z_min, z_max = bounds
                mask_array = mask_array[z_min:z_max, y_min:y_max, x_min:x_max]

        if mask_array.shape != expected_shape:
            return None

        return np.where(mask_array > 0, 255, 0).astype(np.uint8, copy=False)

    def _extract_array_from_ome_arrow(  # noqa: C901, PLR0911, PLR0912
        self: CytoDataFrame_type,
        data_value: Any,
    ) -> Optional[np.ndarray]:
        """Convert an OME-Arrow struct (dict) into an ndarray."""

        if not self._is_ome_arrow_value(data_value):
            return None

        try:
            pixels_meta = data_value.get("pixels_meta", {})
            size_x = int(pixels_meta.get("size_x"))
            size_y = int(pixels_meta.get("size_y"))
            size_z = int(pixels_meta.get("size_z") or 1)
            size_c = int(pixels_meta.get("size_c") or 1)
            planes = data_value.get("planes")

            if size_x <= 0 or size_y <= 0 or planes is None:
                return None
            if size_z > 1:
                return None

            if isinstance(planes, np.ndarray):
                plane_entries = planes.tolist()
            else:
                plane_entries = list(planes)

            if not plane_entries:
                return None

            base = size_x * size_y
            if base <= 0:
                return None

            if size_c > 1:
                planes_by_c = {}
                for plane in plane_entries:
                    if int(plane.get("t") or 0) != 0 or int(plane.get("z") or 0) != 0:
                        continue
                    channel_index = int(plane.get("c") or 0)
                    pixels = plane.get("pixels")
                    if pixels is None:
                        continue
                    np_pixels = np.asarray(pixels)
                    if np_pixels.size != base:
                        continue
                    planes_by_c[channel_index] = np_pixels.reshape((size_y, size_x))

                if len(planes_by_c) != size_c:
                    return None

                array = np.stack([planes_by_c[c] for c in range(size_c)], axis=-1)
            else:
                plane = plane_entries[0]
                pixels = plane.get("pixels")
                if pixels is None:
                    return None

                np_pixels = np.asarray(pixels)
                if np_pixels.size == 0 or np_pixels.size % base != 0:
                    return None

                channel_count = np_pixels.size // base
                if channel_count == 1:
                    array = np_pixels.reshape((size_y, size_x))
                else:
                    array = np_pixels.reshape((size_y, size_x, channel_count))

            return self._ensure_uint8(array)
        except Exception as exc:
            logger.debug("Unable to decode OME-Arrow struct: %s", exc)
            return None

    @staticmethod
    def _ensure_uint8(array: np.ndarray) -> np.ndarray:
        """Convert the provided array to uint8 without unnecessary warnings."""

        arr = np.asarray(array)
        if np.issubdtype(arr.dtype, np.integer):
            min_val = arr.min(initial=0)
            max_val = arr.max(initial=0)
            if 0 <= min_val <= 255 and 0 <= max_val <= 255:  # noqa: PLR2004
                return arr.astype(np.uint8, copy=False)
        return img_as_ubyte(arr)

    @staticmethod
    def _is_3d_image_array(array: np.ndarray) -> bool:
        if array.ndim < MIN_VOLUME_NDIM:
            return False
        if array.ndim != MIN_VOLUME_NDIM:
            return True
        if array.shape[-1] not in RGB_LIKE_CHANNEL_COUNTS:
            return True

        height, width = int(array.shape[0]), int(array.shape[1])
        short_side = min(height, width)
        long_side = max(height, width)
        if short_side < MIN_RGB_SPATIAL_DIM:
            return True

        aspect_ratio = long_side / max(short_side, 1)
        looks_like_rgb_2d = aspect_ratio <= MAX_RGB_ASPECT_RATIO
        return not looks_like_rgb_2d

    def _prepare_cropped_image_layers(  # noqa: C901, PLR0915, PLR0912, PLR0913
        self: CytoDataFrame_type,
        data_value: Any,
        bounding_box: Tuple[int, int, int, int],
        compartment_center_xy: Optional[Tuple[int, int]] = None,
        image_path: Optional[str] = None,
        include_original: bool = False,
        include_mask_outline: bool = False,
        include_composite: bool = True,
    ) -> Dict[str, Optional[np.ndarray]]:
        """Return requested cropped image layers for downstream consumers."""

        logger.debug(
            (
                "Preparing cropped layers. Data value: %s, Bounding box: %s, "
                "Compartment center xy: %s, Image path: %s"
            ),
            data_value,
            bounding_box,
            compartment_center_xy,
            image_path,
        )

        layers: Dict[str, Optional[np.ndarray]] = {}

        if array := self._extract_array_from_ome_arrow(data_value):
            if include_original:
                layers["original"] = array
            if include_mask_outline:
                layers["mask"] = array
            if include_composite:
                layers["composite"] = array
            return layers

        data_value = str(data_value)

        display_options = self._custom_attrs.get("display_options", {}) or {}
        if display_options.get("ignore_image_path_columns"):
            image_path = None

        if self._custom_attrs.get("data_context_dir"):
            normalized = data_value
            if normalized.startswith("file:"):
                normalized = normalized[len("file:") :]
            if "/" in normalized or "\\" in normalized:
                normalized = pathlib.Path(normalized).name
            data_value = normalized
        candidate_path = None

        if image_path is not None and pd.isna(image_path):
            image_path = None

        pattern_map = self._custom_attrs.get("segmentation_file_regex")

        provided_path = pathlib.Path(data_value)
        if provided_path.is_file():
            candidate_path = provided_path
        elif (
            self._custom_attrs["data_context_dir"] is None
            and image_path is not None
            and (
                existing_image_from_path := pathlib.Path(image_path)
                / pathlib.Path(data_value)
            ).is_file()
        ):
            logger.debug("Found existing image from path: %s", existing_image_from_path)
            candidate_path = existing_image_from_path
        elif self._custom_attrs["data_context_dir"] is not None and (
            candidate_paths := list(
                pathlib.Path(self._custom_attrs["data_context_dir"]).rglob(data_value)
            )
        ):
            logger.debug(
                "Found candidate paths (and attempting to use the first): %s",
                candidate_paths,
            )
            candidate_path = candidate_paths[0]
        else:
            if self._custom_attrs.get("data_context_dir") is not None:
                logger.debug(
                    "Checked data_context_dir %s for %s but found no matches.",
                    self._custom_attrs["data_context_dir"],
                    data_value,
                )
            logger.debug("No candidate file found for: %s", data_value)
            return layers

        try:
            orig_image_array = imageio.imread(candidate_path)
        except (FileNotFoundError, ValueError) as exc:
            logger.error(exc)
            return layers

        if self._is_3d_image_array(orig_image_array):
            logger.debug(
                "Detected 3D image at %s; returning HTML view.", candidate_path
            )
            html_view = None
            volume_array = np.asarray(orig_image_array)
            if (
                volume_array.ndim > MIN_VOLUME_NDIM
                and volume_array.shape[-1] in RGB_LIKE_CHANNEL_COUNTS
            ):
                volume_array = volume_array[..., 0]

            if volume_array.ndim == MIN_VOLUME_NDIM:
                label_overlay = None
                segmentation_path = self._find_matching_segmentation_path(
                    data_value=data_value,
                    pattern_map=pattern_map,
                    file_dir=self._custom_attrs.get("data_mask_context_dir"),
                    candidate_path=candidate_path,
                )
                if segmentation_path is None:
                    segmentation_path = self._find_matching_segmentation_path(
                        data_value=data_value,
                        pattern_map=pattern_map,
                        file_dir=self._custom_attrs.get("data_outline_context_dir"),
                        candidate_path=candidate_path,
                    )
                if segmentation_path is not None:
                    label_overlay = self._prepare_3d_label_overlay(
                        segmentation_path=segmentation_path,
                        expected_shape=volume_array.shape,
                    )
                with contextlib.suppress(Exception):
                    volume = self._ensure_uint8(volume_array)
                    dims = (volume.shape[2], volume.shape[1], volume.shape[0])
                    html_view = build_3d_image_html_view(
                        volume=volume,
                        dims=dims,
                        data_value=data_value,
                        candidate_path=candidate_path,
                        display_options=self._custom_attrs.get("display_options"),
                        label_volume=label_overlay,
                    )

            if html_view is None:
                html_view = build_3d_html_from_path(
                    data_value=data_value,
                    candidate_path=candidate_path,
                    display_options=self._custom_attrs.get("display_options"),
                    ensure_uint8=self._ensure_uint8,
                    is_ome_arrow_value=self._is_ome_arrow_value,
                    logger=logger,
                )
            layers[self._HTML_3D_STUB_KEY] = (
                html_view
                if html_view is not None
                else build_3d_image_html_stub(
                    data_value=data_value,
                    candidate_path=candidate_path,
                    display_options=self._custom_attrs.get("display_options"),
                )
            )
            return layers

        if self._custom_attrs["image_adjustment"] is not None:
            logger.debug("Adjusting image with custom image adjustment function.")
            orig_image_array = self._custom_attrs["image_adjustment"](
                orig_image_array, self._custom_attrs["_widget_state"]["scale"]
            )
        else:
            logger.debug("Adjusting image with adaptive histogram equalization.")
            orig_image_array = adjust_with_adaptive_histogram_equalization(
                image=orig_image_array,
                brightness=self._custom_attrs["_widget_state"]["scale"],
            )

        orig_image_array = self._ensure_uint8(orig_image_array)

        original_image_copy = orig_image_array.copy() if include_original else None

        prepared_image, mask_source_path = self.search_for_mask_or_outline(
            data_value=data_value,
            pattern_map=pattern_map,
            file_dir=self._custom_attrs["data_mask_context_dir"],
            candidate_path=candidate_path,
            orig_image=orig_image_array,
            mask=True,
        )

        if prepared_image is None:
            prepared_image, mask_source_path = self.search_for_mask_or_outline(
                data_value=data_value,
                pattern_map=pattern_map,
                file_dir=self._custom_attrs["data_outline_context_dir"],
                candidate_path=candidate_path,
                orig_image=orig_image_array,
                mask=False,
            )

        if prepared_image is None:
            prepared_image = orig_image_array

        mask_source_array = None
        if include_mask_outline and mask_source_path is not None:
            try:
                loaded_mask = imageio.imread(mask_source_path)
                if loaded_mask.ndim == 3:  # noqa: PLR2004
                    mask_gray = np.max(loaded_mask[..., :3], axis=2)
                else:
                    mask_gray = loaded_mask
                mask_binary = mask_gray > 0
                mask_uint8 = np.zeros(mask_binary.shape, dtype=np.uint8)
                mask_uint8[mask_binary] = 255
                mask_source_array = mask_uint8
            except (FileNotFoundError, ValueError) as exc:
                logger.error(
                    "Unable to read mask/outline image %s: %s", mask_source_path, exc
                )
                mask_source_array = None

        if (
            compartment_center_xy is not None
            and self._custom_attrs.get("display_options", None) is None
        ) or (
            self._custom_attrs.get("display_options", None) is not None
            and self._custom_attrs["display_options"].get("center_dot", True)
        ):
            center_x, center_y = map(int, compartment_center_xy)

            if len(prepared_image.shape) == 2:  # noqa: PLR2004
                prepared_image = skimage.color.gray2rgb(prepared_image)

            if (
                0 <= center_y < prepared_image.shape[0]
                and 0 <= center_x < prepared_image.shape[1]
            ):
                x_min, y_min, x_max, y_max = map(int, bounding_box)
                box_width = x_max - x_min
                box_height = y_max - y_min
                radius = max(1, int(min(box_width, box_height) * 0.03))

                rr, cc = skimage.draw.disk(
                    (center_y, center_x), radius=radius, shape=prepared_image.shape[:2]
                )
                prepared_image[rr, cc] = [255, 0, 0]

        try:
            x_min, y_min, x_max, y_max = map(int, bounding_box)

            if self._custom_attrs.get("display_options", None) and self._custom_attrs[
                "display_options"
            ].get("offset_bounding_box", None):
                center_x, center_y = map(int, compartment_center_xy)

                offset_bounding_box = self._custom_attrs["display_options"].get(
                    "offset_bounding_box"
                )
                x_min, y_min, x_max, y_max = get_pixel_bbox_from_offsets(
                    center_x=center_x,
                    center_y=center_y,
                    rel_bbox=(
                        offset_bounding_box["x_min"],
                        offset_bounding_box["y_min"],
                        offset_bounding_box["x_max"],
                        offset_bounding_box["y_max"],
                    ),
                )

            cropped_img_array = prepared_image[y_min:y_max, x_min:x_max]

            cropped_original = (
                original_image_copy[y_min:y_max, x_min:x_max]
                if include_original and original_image_copy is not None
                else None
            )
            if include_mask_outline and mask_source_array is not None:
                try:
                    cropped_mask = mask_source_array[y_min:y_max, x_min:x_max]
                except Exception as exc:
                    logger.debug(
                        "Failed to crop mask/outline array for %s: %s",
                        mask_source_path,
                        exc,
                    )
                    cropped_mask = None
            else:
                cropped_mask = None

            try:
                display_options = self._custom_attrs.get("display_options", {}) or {}
                scale_cfg = display_options.get("scale_bar", None)

                if scale_cfg:
                    um_per_pixel = None
                    if isinstance(scale_cfg, dict):
                        um_per_pixel = scale_cfg.get("um_per_pixel") or scale_cfg.get(
                            "pixel_size_um"
                        )
                    if um_per_pixel is None:
                        um_per_pixel = display_options.get(
                            "um_per_pixel"
                        ) or display_options.get("pixel_size_um")

                    if um_per_pixel is None:
                        ppu = None
                        if isinstance(scale_cfg, dict):
                            ppu = scale_cfg.get("pixels_per_um") or scale_cfg.get(
                                "pixel_per_um"
                            )
                        if ppu is None:
                            ppu = display_options.get(
                                "pixels_per_um"
                            ) or display_options.get("pixel_per_um")
                        if ppu:
                            try:
                                ppu = float(ppu)
                                if ppu > 0:
                                    um_per_pixel = 1.0 / ppu
                            except (TypeError, ValueError):
                                pass

                    if um_per_pixel:
                        params = {
                            "length_um": 10.0,
                            "thickness_px": 4,
                            "color": (255, 255, 255),
                            "location": "lower right",
                            "margin_px": 10,
                            "font_size_px": 14,
                        }
                        if isinstance(scale_cfg, dict):
                            params.update(
                                {
                                    k: v
                                    for k, v in scale_cfg.items()
                                    if k in params
                                    or k
                                    in (
                                        "um_per_pixel",
                                        "pixel_size_um",
                                        "pixels_per_um",
                                        "pixel_per_um",
                                    )
                                }
                            )

                        cropped_img_array = add_image_scale_bar(
                            cropped_img_array,
                            um_per_pixel=float(um_per_pixel),
                            **{
                                k: v
                                for k, v in params.items()
                                if k
                                not in (
                                    "um_per_pixel",
                                    "pixel_size_um",
                                    "pixels_per_um",
                                    "pixel_per_um",
                                )
                            },
                        )
            except Exception as exc:
                logger.debug("Skipping scale bar due to error: %s", exc)

        except ValueError as exc:
            raise ValueError(
                f"Bounding box contains invalid values: {bounding_box}"
            ) from exc
        except IndexError as exc:
            raise IndexError(
                f"Bounding box {bounding_box} is out of bounds for image dimensions "
                f"{prepared_image.shape}"
            ) from exc

        logger.debug("Cropped image array shape: %s", cropped_img_array.shape)

        if include_composite:
            layers["composite"] = cropped_img_array
        if include_original:
            layers["original"] = cropped_original
        if include_mask_outline:
            layers["mask"] = cropped_mask

        return layers

    def _prepare_cropped_image_array(
        self: CytoDataFrame_type,
        data_value: Any,
        bounding_box: Tuple[int, int, int, int],
        compartment_center_xy: Optional[Tuple[int, int]] = None,
        image_path: Optional[str] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[str]]:
        layers = self._prepare_cropped_image_layers(
            data_value=data_value,
            bounding_box=bounding_box,
            compartment_center_xy=compartment_center_xy,
            image_path=image_path,
            include_composite=True,
        )
        return layers.get("composite"), layers.get(self._HTML_3D_STUB_KEY)

    def _image_array_to_html(self: CytoDataFrame_type, image_array: np.ndarray) -> str:
        """Encode an image array as an HTML <img> tag."""

        try:
            png_bytes_io = BytesIO()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                imageio.imwrite(png_bytes_io, image_array, format="png")
            png_bytes = png_bytes_io.getvalue()
        except (FileNotFoundError, ValueError) as exc:
            logger.error(exc)
            raise

        display_options = self._custom_attrs.get("display_options", {}) or {}
        width = display_options.get("width", "300px")
        height = display_options.get("height")

        html_style = [f"width:{width}"]
        if height is not None:
            html_style.append(f"height:{height}")

        html_style_joined = ";".join(html_style)
        base64_image_bytes = base64.b64encode(png_bytes).decode("utf-8")

        return (
            '<img src="data:image/png;base64,'
            f'{base64_image_bytes}" style="{html_style_joined}"/>'
        )

    def process_ome_arrow_data_as_html_display(
        self: CytoDataFrame_type,
        data_value: Any,
    ) -> str:
        """Render an OME-Arrow struct as an HTML <img> element."""

        array = self._extract_array_from_ome_arrow(data_value)
        if array is None:
            volume_data = extract_volume_from_ome_arrow(
                data_value,
                self._ensure_uint8,
                self._is_ome_arrow_value,
                logger,
            )
            if volume_data is None:
                return str(data_value)
            volume, dims = volume_data
            return build_3d_image_html_view(
                volume=volume,
                dims=dims,
                data_value="ome-arrow",
                candidate_path=pathlib.Path("ome-arrow"),
                display_options=self._custom_attrs.get("display_options"),
            )

        try:
            return self._image_array_to_html(array)
        except Exception:
            return str(data_value)

    def process_image_data_as_html_display(
        self: CytoDataFrame_type,
        data_value: Any,
        bounding_box: Tuple[int, int, int, int],
        compartment_center_xy: Optional[Tuple[int, int]] = None,
        image_path: Optional[str] = None,
    ) -> str:
        """
        Process the image data based on the provided data value
        and bounding box, applying masks or outlines where
        applicable, and return an HTML representation of the
        cropped image for display.

        Args:
            data_value (Any):
                The value to search for in the file system or as the image data.
            bounding_box (Tuple[int, int, int, int]):
                The bounding box to crop the image.
            compartment_center_xy (Optional[Tuple[int, int]]):
                The center coordinates of the compartment.
            image_path (Optional[str]):
                The path to the image file.

        Returns:
            str:
                The HTML image display string, or the unmodified data
                value if the image cannot be processed.
        """

        logger.debug(
            (
                "Processing image data as HTML for display."
                " Data value: %s , Bounding box: %s , "
                "Compartment center xy: %s, Image path: %s"
            ),
            data_value,
            bounding_box,
            compartment_center_xy,
            image_path,
        )

        data_value = str(data_value)
        cropped_img_array, html_stub = self._prepare_cropped_image_array(
            data_value=data_value,
            bounding_box=bounding_box,
            compartment_center_xy=compartment_center_xy,
            image_path=image_path,
        )

        if html_stub is not None:
            return html_stub

        if cropped_img_array is None:
            return data_value

        logger.debug("Image processed successfully and being sent to HTML for display.")

        try:
            return self._image_array_to_html(cropped_img_array)
        except Exception:
            return data_value

    def _get_3d_volume_from_cell(  # noqa: C901, PLR0912, PLR0915
        self: CytoDataFrame_type,
        row: Any,
        column: Any,
    ) -> Tuple[np.ndarray, Tuple[int, int, int]]:
        display_options = self._custom_attrs.get("display_options", {}) or {}
        cache_disabled = bool(display_options.get("volume_disable_cache"))
        cache_max_entries_raw = display_options.get("volume_cache_max_entries", 32)
        try:
            cache_max_entries = max(1, int(cache_max_entries_raw))
        except (TypeError, ValueError):
            cache_max_entries = 32

        cache: "OrderedDict[str, Tuple[np.ndarray, Tuple[int, int, int]]]" = (
            OrderedDict()
        )
        if not cache_disabled:
            raw_cache = self._custom_attrs.get("_volume_cache", {})
            if isinstance(raw_cache, OrderedDict):
                cache = raw_cache
            else:
                cache = OrderedDict(raw_cache or {})
                self._custom_attrs["_volume_cache"] = cache
        cache_key = f"{row}::{column}"
        if not cache_disabled and cache_key in cache:
            cached = cache.pop(cache_key)
            cache[cache_key] = cached
            return cached

        try:
            value = self.loc[row, column]
        except Exception:
            value = self.iloc[row][column]

        volume = None
        dims = None

        if isinstance(value, np.ndarray) and self._is_3d_image_array(value):
            volume = np.asarray(value)
            dims = (volume.shape[2], volume.shape[1], volume.shape[0])
        elif self._is_ome_arrow_value(value):
            volume_data = extract_volume_from_ome_arrow(
                value,
                self._ensure_uint8,
                self._is_ome_arrow_value,
                logger,
            )
            if volume_data is not None:
                volume, dims = volume_data
        elif isinstance(value, (str, pathlib.Path)):
            volume_ndim = 3
            color_channel_counts = (1, volume_ndim, 4)
            data_value = str(value)
            context_dir = self._custom_attrs.get("data_context_dir")
            if context_dir:
                normalized = data_value
                if normalized.startswith("file:"):
                    normalized = normalized[len("file:") :]
                if "/" in normalized or "\\" in normalized:
                    normalized = pathlib.Path(normalized).name
                data_value = normalized

            data_path = pathlib.Path(data_value)
            candidate_paths: List[pathlib.Path] = []
            seen_candidates: set[str] = set()

            def _add_candidate(path_value: pathlib.Path) -> None:
                if not path_value.is_file():
                    return
                key = str(path_value.resolve())
                if key not in seen_candidates:
                    seen_candidates.add(key)
                    candidate_paths.append(path_value)

            _add_candidate(data_path)
            if context_dir:
                _add_candidate(pathlib.Path(context_dir) / data_path)

            candidate_filenames = {pathlib.Path(data_value).name}
            needs_path_column_lookup = context_dir is None or not candidate_paths
            if needs_path_column_lookup:
                image_cols = self.find_image_columns() or []
                all_cols = self.columns.tolist()
                path_df = self._custom_attrs.get("data_image_paths")
                if path_df is not None:
                    all_cols = list(
                        dict.fromkeys([*all_cols, *path_df.columns.tolist()])
                    )
                image_path_cols = self.find_image_path_columns(image_cols, all_cols)

                def _row_value(col_name: str) -> Any:
                    if col_name in self.columns:
                        try:
                            return self.loc[row, col_name]
                        except Exception:
                            return self.iloc[row][col_name]
                    if (
                        path_df is not None
                        and col_name in path_df.columns
                        and row in path_df.index
                    ):
                        return path_df.loc[row, col_name]
                    return None

                for image_col, path_col in image_path_cols.items():
                    row_filename = _row_value(image_col)
                    row_path = _row_value(path_col)
                    if row_filename is not None and not pd.isna(row_filename):
                        candidate_filenames.add(pathlib.Path(str(row_filename)).name)
                    if (
                        row_filename is not None
                        and not pd.isna(row_filename)
                        and row_path is not None
                        and not pd.isna(row_path)
                    ):
                        _add_candidate(pathlib.Path(str(row_path)) / str(row_filename))
                    if (
                        image_col == str(column)
                        and row_path is not None
                        and not pd.isna(row_path)
                    ):
                        _add_candidate(
                            pathlib.Path(str(row_path)) / pathlib.Path(data_value).name
                        )

            if context_dir:
                context_root = pathlib.Path(context_dir)
                for filename in sorted(candidate_filenames):
                    _add_candidate(context_root / filename)
                    if not candidate_paths:
                        for found in context_root.rglob(filename):
                            _add_candidate(found)

            if candidate_paths:
                data_path = candidate_paths[0]

            # First attempt direct image loading for TIFF/Zarr-backed 3D arrays.
            for file_candidate in candidate_paths:
                with contextlib.suppress(Exception):
                    image_volume = np.asarray(imageio.imread(file_candidate))
                    if self._is_3d_image_array(image_volume):
                        if (
                            image_volume.ndim > volume_ndim
                            and image_volume.shape[-1] in color_channel_counts
                        ):
                            image_volume = image_volume[..., 0]
                        if image_volume.ndim == volume_ndim:
                            volume = image_volume
                            dims = (volume.shape[2], volume.shape[1], volume.shape[0])
                            data_path = file_candidate
                            break

            # Fallback to OME-Arrow path decoding for string/path cells.
            try:
                from ome_arrow import OMEArrow  # type: ignore

                if volume is None:
                    decode_candidates = candidate_paths or [data_path]
                    for decode_path in decode_candidates:
                        ome_struct = OMEArrow(data=str(decode_path)).data
                        if hasattr(ome_struct, "as_py"):
                            ome_struct = ome_struct.as_py()
                        volume_data = extract_volume_from_ome_arrow(
                            ome_struct,
                            self._ensure_uint8,
                            self._is_ome_arrow_value,
                            logger,
                        )
                        if volume_data is not None:
                            volume, dims = volume_data
                            data_path = decode_path
                            break
            except Exception as exc:
                logger.debug(
                    (
                        "OME-Arrow fallback decode failed for row=%s, "
                        "column=%s, path=%s: %s"
                    ),
                    row,
                    column,
                    data_path,
                    exc,
                )

        if volume is None or dims is None:
            raise ValueError("Selected cell does not contain a 3D volume.")

        # Apply per-row bounding box cropping when available (XYZ).
        try:
            bounds = self._get_3d_bbox_crop_bounds(
                row=row,
                volume_shape=tuple(int(v) for v in volume.shape),
            )
            if bounds is not None:
                x_min, x_max, y_min, y_max, z_min, z_max = bounds
                volume = volume[z_min:z_max, y_min:y_max, x_min:x_max]
                dims = (volume.shape[2], volume.shape[1], volume.shape[0])
                logger.debug(
                    "Applied 3D bounding box crop: x(%s,%s) y(%s,%s) z(%s,%s)",
                    x_min,
                    x_max,
                    y_min,
                    y_max,
                    z_min,
                    z_max,
                )
        except Exception as exc:
            logger.debug("Skipping 3D bounding box crop due to error: %s", exc)

        if not cache_disabled:
            cache[cache_key] = (volume, dims)
            while len(cache) > cache_max_entries:
                cache.popitem(last=False)
        return volume, dims

    def _get_3d_label_overlay_from_cell(  # noqa: C901
        self: CytoDataFrame_type,
        row: Any,
        column: Any,
        expected_shape: Tuple[int, ...],
    ) -> Optional[np.ndarray]:
        """Build a 3D label overlay for a specific table cell.

        Args:
            row: Row label or index containing the 3D image value.
            column: Column label containing the 3D image value.
            expected_shape: Target ``(z, y, x)`` shape for the overlay.

        Returns:
            A uint8 binary label volume aligned to ``expected_shape`` when a
            compatible mask/outline can be found; otherwise ``None``.
        """
        try:
            value = self.loc[row, column]
        except Exception:
            value = self.iloc[row][column]

        if not isinstance(value, (str, pathlib.Path)):
            return None

        data_value = str(value)
        if self._custom_attrs.get("data_context_dir"):
            normalized = data_value
            if normalized.startswith("file:"):
                normalized = normalized[len("file:") :]
            if "/" in normalized or "\\" in normalized:
                normalized = pathlib.Path(normalized).name
            data_value = normalized

        candidate_path = pathlib.Path(data_value)
        context_dir = self._custom_attrs.get("data_context_dir")
        if not candidate_path.is_file() and context_dir:
            matches = sorted(
                pathlib.Path(context_dir).rglob(pathlib.Path(data_value).name)
            )
            if matches:
                candidate_path = matches[0]

        pattern_map = self._custom_attrs.get("segmentation_file_regex")
        segmentation_path = self._find_matching_segmentation_path(
            data_value=data_value,
            pattern_map=pattern_map,
            file_dir=self._custom_attrs.get("data_mask_context_dir"),
            candidate_path=candidate_path,
        )
        if segmentation_path is None:
            segmentation_path = self._find_matching_segmentation_path(
                data_value=data_value,
                pattern_map=pattern_map,
                file_dir=self._custom_attrs.get("data_outline_context_dir"),
                candidate_path=candidate_path,
            )
        if segmentation_path is None:
            logger.debug("No 3D mask/outline found for image %s", data_value)
            return None

        logger.debug(
            "Found 3D mask/outline for image %s at %s",
            data_value,
            segmentation_path,
        )
        overlay = self._prepare_3d_label_overlay(
            segmentation_path=segmentation_path,
            expected_shape=expected_shape,
            row=row,
        )
        if overlay is not None:
            logger.debug(
                "Prepared 3D mask/outline overlay for image %s with shape %s",
                data_value,
                overlay.shape,
            )
        else:
            logger.warning(
                (
                    "Found 3D mask/outline for image %s at %s but could not align "
                    "it with expected volume shape %s"
                ),
                data_value,
                segmentation_path,
                expected_shape,
            )
        return overlay

    def _get_3d_bbox_crop_bounds(
        self: CytoDataFrame_type,
        row: Any,
        volume_shape: Tuple[int, ...],
    ) -> Optional[Tuple[int, int, int, int, int, int]]:
        """Return clamped 3D bbox crop bounds.

        Args:
            row: Row label or index used to read bounding-box metadata.
            volume_shape: ``(z, y, x)`` shape of the source volume.

        Returns:
            A tuple of ``(x_min, x_max, y_min, y_max, z_min, z_max)`` bounds, or
            ``None`` when bounding-box columns are unavailable or cropping is
            disabled.
        """
        display_options = self._custom_attrs.get("display_options", {}) or {}
        if display_options.get("volume_disable_bbox_crop"):
            return None

        if len(volume_shape) < MIN_VOLUME_NDIM:
            return None

        bbox_source = self._custom_attrs.get("data_bounding_box")
        bbox_cols = (
            bbox_source.columns.tolist()
            if bbox_source is not None
            else self.columns.tolist()
        )

        def _find_col(tag: str) -> Optional[str]:
            return next((col for col in bbox_cols if tag in str(col)), None)

        x_min_col = _find_col("Minimum_X")
        x_max_col = _find_col("Maximum_X")
        y_min_col = _find_col("Minimum_Y")
        y_max_col = _find_col("Maximum_Y")
        z_min_col = _find_col("Minimum_Z")
        z_max_col = _find_col("Maximum_Z")

        if not all(
            col is not None for col in (x_min_col, x_max_col, y_min_col, y_max_col)
        ):
            return None

        try:
            row_data = (
                bbox_source.loc[row]
                if bbox_source is not None and row in bbox_source.index
                else self.loc[row]
            )
        except Exception:
            row_data = self.iloc[row]

        x_min = int(row_data[x_min_col])
        x_max = int(row_data[x_max_col])
        y_min = int(row_data[y_min_col])
        y_max = int(row_data[y_max_col])

        z_min = 0
        z_max = volume_shape[0]
        if z_min_col is not None and z_max_col is not None:
            z_min = int(row_data[z_min_col])
            z_max = int(row_data[z_max_col])

        z_min = max(0, min(z_min, volume_shape[0]))
        z_max = max(z_min + 1, min(z_max, volume_shape[0]))
        y_min = max(0, min(y_min, volume_shape[1]))
        y_max = max(y_min + 1, min(y_max, volume_shape[1]))
        x_min = max(0, min(x_min, volume_shape[2]))
        x_max = max(x_min + 1, min(x_max, volume_shape[2]))
        return x_min, x_max, y_min, y_max, z_min, z_max

    def _find_3d_columns_for_display(
        self: CytoDataFrame_type,
        max_rows: int = 5,
    ) -> List[Any]:
        """Find columns that contain at least one renderable 3D value."""

        image_cols = self.find_image_columns() or []
        ome_cols = self.find_ome_arrow_columns(self)
        candidate_columns = list(dict.fromkeys([*image_cols, *ome_cols]))
        if not candidate_columns:
            return []

        sample_rows = [
            row for row in self.get_displayed_rows() if str(row) != "\u2026"
        ][:max_rows]
        if not sample_rows:
            sample_rows = self.index.tolist()[:max_rows]

        columns_3d: List[Any] = []
        for column in candidate_columns:
            for row in sample_rows:
                try:
                    self._get_3d_volume_from_cell(row=row, column=column)
                    columns_3d.append(column)
                    break
                except Exception:
                    continue

        return columns_3d

    def _add_label_overlay_to_plotter(  # noqa: PLR0913
        self: CytoDataFrame_type,
        plotter: Any,
        volume: np.ndarray,
        label_volume: Optional[np.ndarray],
        spacing: Tuple[float, float, float],
        base_sample: float,
        display_options: dict[str, Any],
    ) -> List[Any]:
        """Add a 3D label overlay to an existing PyVista plotter.

        Args:
            plotter: Target PyVista plotter receiving overlay actors.
            volume: Source image volume in ``(z, y, x)`` order.
            label_volume: Optional label/mask volume aligned to ``volume``.
            spacing: Voxel spacing tuple used when building label image data.
            base_sample: Base sampling distance used for volume overlays.
            display_options: Display options controlling overlay mode/style.
        Returns:
            A list of added overlay actor objects.
        """
        overlay_actors: List[Any] = []
        if label_volume is None:
            return overlay_actors

        try:
            import pyvista as pv  # type: ignore
        except Exception:
            return overlay_actors

        try:
            label_arr = np.asarray(label_volume)
            if label_arr.shape != volume.shape:
                logger.warning(
                    (
                        "Skipping 3D label overlay due to shape mismatch: "
                        "label=%s volume=%s"
                    ),
                    label_arr.shape,
                    volume.shape,
                )
                return overlay_actors

            label_xyz = np.transpose((label_arr > 0).astype(np.uint8), (2, 1, 0))
            label_grid = pv.ImageData()
            label_grid.dimensions = tuple(int(v) for v in label_xyz.shape)
            label_grid.spacing = spacing
            label_grid.origin = (0.0, 0.0, 0.0)
            label_grid.point_data.clear()
            label_grid.point_data["label_scalars"] = np.asfortranarray(label_xyz).ravel(
                order="F"
            )

            overlay_mode = str(display_options.get("label_overlay_mode", "surface"))
            overlay_mode = overlay_mode.lower()
            overlay_color = display_options.get("label_overlay_color", (0, 255, 0))
            if (
                isinstance(overlay_color, (tuple, list))
                and len(overlay_color) >= MIN_VOLUME_NDIM
            ):
                overlay_color = tuple(
                    (float(v) / 255.0 if float(v) > 1.0 else float(v))
                    for v in overlay_color[:3]
                )
            overlay_opacity = float(display_options.get("label_overlay_opacity", 0.95))

            if overlay_mode == "surface":
                contour = label_grid.contour(isosurfaces=[0.5], scalars="label_scalars")
                edge_opacity = min(1.0, overlay_opacity + 0.15)
                overlay_actors.append(
                    plotter.add_mesh(
                        contour,
                        color=overlay_color,
                        opacity=overlay_opacity,
                        smooth_shading=False,
                        ambient=1.0,
                        diffuse=0.0,
                        specular=0.0,
                    )
                )
                overlay_actors.append(
                    plotter.add_mesh(
                        contour,
                        color=overlay_color,
                        style="wireframe",
                        opacity=edge_opacity,
                        line_width=2.5,
                    )
                )
            else:
                label_xyz_u8 = np.where(label_xyz > 0, 255, 0).astype(
                    np.uint8, copy=False
                )
                label_grid.point_data["label_scalars"] = np.asfortranarray(
                    label_xyz_u8
                ).ravel(order="F")
                filled_opacity = np.zeros(256, dtype=np.float32)
                filled_opacity[255] = overlay_opacity
                label_actor = plotter.add_volume(
                    label_grid,
                    scalars="label_scalars",
                    clim=(0, 255),
                    cmap=[(0.0, 0.0, 0.0), overlay_color],
                    opacity=filled_opacity,
                    shade=False,
                    show_scalar_bar=False,
                    opacity_unit_distance=base_sample,
                    blending="maximum",
                )
                overlay_actors.append(label_actor)
                with contextlib.suppress(Exception):
                    label_prop = (
                        getattr(label_actor, "prop", None) or label_actor.GetProperty()
                    )
                    label_prop.SetInterpolationTypeToNearest()
                with contextlib.suppress(Exception):
                    contour = label_grid.contour(
                        isosurfaces=[0.5], scalars="label_scalars"
                    )
                    overlay_actors.append(
                        plotter.add_mesh(
                            contour,
                            color=overlay_color,
                            style="wireframe",
                            opacity=min(1.0, overlay_opacity + 0.05),
                            line_width=2.0,
                        )
                    )

            logger.info(
                "Added 3D label overlay (shape=%s, opacity=%s, mode=%s)",
                label_arr.shape,
                overlay_opacity,
                overlay_mode,
            )
        except Exception as exc:
            logger.debug("Unable to add 3D label overlay: %s", exc)
            return overlay_actors
        return overlay_actors

    @staticmethod
    def _set_overlay_actor_visibility(actor: Any, visible: bool) -> None:
        """Set visibility for supported actor object variants."""
        visible_flag = 1 if visible else 0
        if hasattr(actor, "SetVisibility"):
            actor.SetVisibility(visible_flag)
            return
        if hasattr(actor, "visibility"):
            actor.visibility = bool(visible)
            return
        prop = getattr(actor, "prop", None)
        if hasattr(prop, "SetOpacity"):
            prop.SetOpacity(float(visible_flag))
            return
        getter = getattr(actor, "GetProperty", None)
        if callable(getter):
            with contextlib.suppress(Exception):
                getter().SetOpacity(float(visible_flag))

    @staticmethod
    def _resolve_overlay_toggle_position(
        plotter: Any,
        display_options: dict[str, Any],
        size: int,
    ) -> Tuple[int, int]:
        """Resolve checkbox position in display pixels (default lower-right)."""
        configured = display_options.get("label_overlay_toggle_position")
        if (
            isinstance(configured, (tuple, list))
            and len(configured) >= MIN_POSITION_COMPONENTS
            and all(
                isinstance(v, (int, float))
                for v in configured[:MIN_POSITION_COMPONENTS]
            )
        ):
            return int(configured[0]), int(configured[1])

        width_px = 300
        window_size = getattr(plotter, "window_size", None)
        if (
            isinstance(window_size, (tuple, list))
            and len(window_size) >= MIN_POSITION_COMPONENTS
            and isinstance(window_size[0], (int, float))
        ):
            width_px = int(window_size[0])
        else:
            configured_width = display_options.get("width", "300px")
            width_digits = re.search(r"\d+", str(configured_width))
            if width_digits:
                width_px = int(width_digits.group(0))

        margin = 10
        x_pos = max(margin, width_px - int(size) - margin)
        y_pos = int(display_options.get("label_overlay_toggle_vertical_offset", 10))
        return x_pos, y_pos

    @staticmethod
    def _resolve_plotter_window_height(
        plotter: Any,
        display_options: dict[str, Any],
    ) -> int:
        """Resolve plotter pixel height for viewport text placement."""
        window_size = getattr(plotter, "window_size", None)
        if (
            isinstance(window_size, (tuple, list))
            and len(window_size) >= MIN_POSITION_COMPONENTS
            and isinstance(window_size[1], (int, float))
        ):
            return int(window_size[1])
        configured_height = display_options.get("height", "300px")
        height_digits = re.search(r"\d+", str(configured_height))
        if height_digits:
            return int(height_digits.group(0))
        return 300

    def _add_label_overlay_toggle_control(
        self: CytoDataFrame_type,
        plotter: Any,
        overlay_actors: List[Any],
        display_options: dict[str, Any],
    ) -> None:
        """Add a PyVista checkbox widget to toggle label overlay visibility."""
        if not overlay_actors:
            return
        if not bool(display_options.get("label_overlay_toggle", True)):
            return
        if not hasattr(plotter, "add_checkbox_button_widget"):
            return

        def _toggle_overlay(state: Any) -> None:
            visible = bool(state)
            for actor in overlay_actors:
                with contextlib.suppress(Exception):
                    self._set_overlay_actor_visibility(actor=actor, visible=visible)
            with contextlib.suppress(Exception):
                plotter.render()

        size = int(display_options.get("label_overlay_toggle_size", 24))
        position = self._resolve_overlay_toggle_position(
            plotter=plotter,
            display_options=display_options,
            size=size,
        )
        with contextlib.suppress(Exception):
            plotter.add_checkbox_button_widget(
                callback=_toggle_overlay,
                value=True,
                size=size,
                position=position,
            )
            logger.debug("Added 3D label overlay toggle checkbox to plotter view.")

    def _toggle_overlay_actors_visibility(
        self: CytoDataFrame_type,
        plotter: Any,
        overlay_actors: List[Any],
        visible: bool,
    ) -> None:
        """Toggle all overlay actors and trigger a render."""
        for actor in overlay_actors:
            with contextlib.suppress(Exception):
                self._set_overlay_actor_visibility(actor=actor, visible=visible)
        with contextlib.suppress(Exception):
            plotter.render()

    def _build_pyvista_viewer(  # noqa: C901, PLR0912, PLR0913, PLR0915
        self: CytoDataFrame_type,
        volume: np.ndarray,
        backend: str,
        widget_height: str,
        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        opacity: Any = "sigmoid",
        shade: bool = False,
        label_volume: Optional[np.ndarray] = None,
        include_plotter_overlay_toggle: bool = True,
        **kwargs: Any,
    ) -> Any:
        try:
            import pyvista as pv  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "PyVista is required for trame-based 3D rendering."
            ) from exc

        display_options = self._custom_attrs.get("display_options", {}) or {}
        cmap = kwargs.pop("cmap", display_options.get("volume_cmap", "gray"))
        background = display_options.get("volume_background", "black")
        percentile_clim = display_options.get("volume_percentile_clim", (1.0, 99.9))
        interpolation = display_options.get("volume_interpolation", "nearest")
        sampling_scale = display_options.get("volume_sampling_scale", 0.5)
        show_axes = display_options.get("volume_show_axes", True)

        vol_xyz = np.transpose(volume, (2, 1, 0))
        if vol_xyz.dtype != np.float32:
            vol_xyz = vol_xyz.astype(np.float32, copy=False)
        grid = pv.ImageData()
        grid.dimensions = tuple(int(v) for v in vol_xyz.shape)
        grid.spacing = spacing
        grid.origin = (0.0, 0.0, 0.0)
        grid.point_data.clear()
        grid.point_data["scalars"] = np.asfortranarray(vol_xyz).ravel(order="F")
        try:
            grid.point_data.set_active_scalars("scalars")
        except AttributeError:
            try:
                grid.point_data.active_scalars_name = "scalars"
            except Exception:
                grid.set_active_scalars("scalars")

        plotter = pv.Plotter(notebook=True)
        plotter.set_background(background)

        if vol_xyz.size:
            try:
                nz = vol_xyz[vol_xyz > 0]
                data_for_clim = nz if nz.size else vol_xyz
                vmin, vmax = np.percentile(data_for_clim, percentile_clim)
                vmin = float(vmin)
                vmax = float(vmax if vmax > vmin else vmin + 1.0)
            except Exception:
                vmin = float(np.min(vol_xyz))
                vmax = float(np.max(vol_xyz) if np.max(vol_xyz) > vmin else vmin + 1.0)
        else:
            vmin, vmax = 0.0, 1.0

        if opacity == "sigmoid" and vmax <= 1.0:
            opacity = "linear"

        base_sample = max(min(spacing), 1e-6)
        vol_actor = plotter.add_volume(
            grid,
            scalars="scalars",
            opacity=opacity,
            shade=shade,
            cmap=cmap,
            clim=(vmin, vmax),
            show_scalar_bar=False,
            opacity_unit_distance=base_sample,
        )
        try:
            prop = getattr(vol_actor, "prop", None) or vol_actor.GetProperty()
            if interpolation.lower().startswith("near"):
                prop.SetInterpolationTypeToNearest()
            else:
                prop.SetInterpolationTypeToLinear()
            if hasattr(prop, "SetInterpolateScalarsBeforeMapping"):
                prop.SetInterpolateScalarsBeforeMapping(False)
            if hasattr(prop, "SetScalarOpacityUnitDistance"):
                prop.SetScalarOpacityUnitDistance(base_sample)
        except Exception as exc:
            logger.debug("Unable to configure volume property interpolation: %s", exc)
        try:
            mapper = getattr(vol_actor, "mapper", None) or vol_actor.GetMapper()
            if hasattr(mapper, "SetAutoAdjustSampleDistances"):
                mapper.SetAutoAdjustSampleDistances(False)
            if hasattr(mapper, "SetUseJittering"):
                mapper.SetUseJittering(False)
            if hasattr(mapper, "SetSampleDistance"):
                mapper.SetSampleDistance(float(base_sample * sampling_scale))
        except Exception as exc:
            logger.debug("Unable to configure volume mapper sampling: %s", exc)

        overlay_actors = self._add_label_overlay_to_plotter(
            plotter=plotter,
            volume=volume,
            label_volume=label_volume,
            spacing=spacing,
            base_sample=base_sample,
            display_options=display_options,
        )
        if include_plotter_overlay_toggle and overlay_actors:
            self._add_label_overlay_toggle_control(
                plotter=plotter,
                overlay_actors=overlay_actors,
                display_options=display_options,
            )

        if show_axes:
            with contextlib.suppress(Exception):
                plotter.add_axes()
        jupyter_kwargs = kwargs.pop("jupyter_kwargs", {})
        jupyter_kwargs.setdefault("width", "100%")
        jupyter_kwargs.setdefault("height", "100%")
        jupyter_kwargs.setdefault("add_menu", False)
        jupyter_kwargs.setdefault("collapse_menu", True)
        viewer = plotter.show(
            jupyter_backend=backend,
            return_viewer=True,
            jupyter_kwargs=jupyter_kwargs,
            **kwargs,
        )
        with contextlib.suppress(Exception):
            setattr(viewer, "_cdf_plotter", plotter)
        with contextlib.suppress(Exception):
            setattr(viewer, "_cdf_overlay_actors", overlay_actors)
        if hasattr(viewer, "layout"):
            try:
                import ipywidgets as widgets  # type: ignore

                viewer.layout = widgets.Layout(
                    width="100%",
                    height="100%",
                    min_height=widget_height,
                    margin="0",
                    padding="0",
                )
            except Exception as exc:
                logger.debug("Unable to assign viewer widget layout: %s", exc)
        if hasattr(viewer, "value") and isinstance(viewer.value, str):
            updated = viewer.value.replace("border: 1px solid", "border: 0 solid")
            if "width:" in updated or "height:" in updated:
                updated = re.sub(r"width:\\s*[^;]+;", "width: 100%;", updated)
                updated = re.sub(r"height:\\s*[^;]+;", "height: 100%;", updated)
                updated = re.sub(
                    r"class=\"pyvista\"",
                    'class="pyvista" style="width: 100%; height: 100%; '
                    'display: block; position: absolute; top: 0; left: 0;"',
                    updated,
                )
            viewer.value = updated
        return viewer

    def show_trame(  # noqa: PLR0915
        self: CytoDataFrame_type,
        row: Any,
        column: Any,
        backend: Optional[str] = "trame",
        **kwargs: Any,
    ) -> Any:
        """Render the dataframe HTML with a trame-backed 3D view.

        Args:
            row: Row label or index of the 3D cell.
            column: Column label containing the 3D data.
            backend: PyVista Jupyter backend name.
            **kwargs: Extra options forwarded to the viewer builder.

        Returns:
            A trame layout when available, otherwise an ipywidgets container.
        """

        volume, _dims = self._get_3d_volume_from_cell(row=row, column=column)
        label_overlay = self._get_3d_label_overlay_from_cell(
            row=row,
            column=column,
            expected_shape=volume.shape,
        )
        html_content = self._generate_jupyter_dataframe_html()

        if backend is None:
            display_options = self._custom_attrs.get("display_options", {}) or {}
            if display_options.get("view") == "trame":
                backend = "trame"

        try:
            import pyvista as pv  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "PyVista is required for trame-based 3D rendering."
            ) from exc

        if hasattr(pv, "set_jupyter_backend"):
            with contextlib.suppress(Exception):
                pv.set_jupyter_backend(backend)

        spacing = kwargs.pop("spacing", (1.0, 1.0, 1.0))
        opacity = kwargs.pop("opacity", "sigmoid")
        shade = kwargs.pop("shade", False)
        widget_height = kwargs.pop("widget_height", "700px")
        table_width = kwargs.pop("table_width", "60%")
        view_width = kwargs.pop("view_width", "40%")
        table_max_height = kwargs.pop(
            "table_max_height",
            (self._custom_attrs.get("display_options") or {}).get(
                "table_max_height", "700px"
            ),
        )

        viewer = self._build_pyvista_viewer(
            volume=volume,
            backend=backend,
            widget_height=widget_height,
            spacing=spacing,
            opacity=opacity,
            shade=shade,
            label_volume=label_overlay,
        )

        try:
            from trame.app import get_server  # type: ignore
            from trame.widgets import html as trame_html  # type: ignore

            server = (
                getattr(viewer, "server", None)
                or getattr(viewer, "_server", None)
                or get_server()
            )
            client_type = getattr(server, "client_type", "vue2")
            if client_type == "vue3":
                from trame.ui.vuetify3 import SinglePageLayout  # type: ignore
                from trame.widgets import vuetify3 as vuetify  # type: ignore
            else:
                from trame.ui.vuetify import SinglePageLayout  # type: ignore
                from trame.widgets import vuetify  # type: ignore

            with SinglePageLayout(server) as layout:
                layout.content.children = []
                with layout.content:  # noqa: SIM117
                    with vuetify.VContainer(
                        fluid=True,
                        style="padding:0;height:100%;overflow:auto;",
                    ):
                        with vuetify.VRow(style="margin:0;height:100%;"):
                            with vuetify.VCol(cols=12, md=7, style="padding: 0;"):
                                trame_html.Div(
                                    v_html=html_content,
                                    style=(
                                        f"width:{table_width};max-width:100%;"
                                        f"max-height:{table_max_height};"
                                        "overflow:auto;border:1px solid #e0e0e0;"
                                    ),
                                )
                            with vuetify.VCol(cols=12, md=5, style="padding: 0;"):
                                trame_html.Div(
                                    children=[viewer],
                                    style=f"width:{view_width};",
                                )
            try:
                url = getattr(server, "url", None)
                if callable(url):
                    logger.debug("Trame server URL: %s", url())
            except Exception as exc:
                logger.debug("Unable to fetch trame server URL: %s", exc)
            display(layout)
            return layout
        except Exception as exc:
            logger.debug("Falling back to ipywidgets layout: %s", exc)
            try:
                import ipywidgets as widgets  # type: ignore
            except Exception as widget_exc:
                raise RuntimeError(
                    "ipywidgets is required for notebook layout."
                ) from widget_exc

            html_widget = widgets.HTML(
                value=html_content,
                layout=widgets.Layout(
                    width=table_width,
                    height=table_max_height,
                    max_height=table_max_height,
                    overflow="auto",
                    border="1px solid #e0e0e0",
                ),
            )
            view_box = widgets.Box(
                [viewer],
                layout=widgets.Layout(
                    width=view_width,
                    height=table_max_height,
                    max_height=table_max_height,
                    overflow="auto",
                ),
            )
            container = widgets.HBox(
                [html_widget, view_box],
                layout=widgets.Layout(
                    width="100%",
                    height=table_max_height,
                    max_height=table_max_height,
                    overflow="auto",
                ),
            )
            return container

    def show_widget_table(  # noqa: C901, PLR0912, PLR0915
        self: CytoDataFrame_type,
        column: Any,
        rows: Optional[List[Any]] = None,
        backend: Optional[str] = "trame",
        **kwargs: Any,
    ) -> Any:
        """Render a widget-based table with 3D views embedded in columns."""

        if backend is None:
            display_options = self._custom_attrs.get("display_options", {}) or {}
            if display_options.get("view") == "trame":
                backend = "trame"

        try:
            import ipywidgets as widgets  # type: ignore
        except Exception as exc:
            raise RuntimeError("ipywidgets is required for widget tables.") from exc

        import html as html_lib

        columns_3d = kwargs.pop("columns_3d", None)
        if columns_3d is None:
            columns_3d = [column]
        if not columns_3d:
            raise ValueError("columns_3d must include at least one column.")
        target_columns = set(columns_3d)

        def _coerce_scalar(value: Any) -> Any:
            if isinstance(value, (np.integer, np.floating)):
                return value.item()
            return value

        display_rows = rows
        if display_rows is None:
            display_rows = self.get_displayed_rows()
            display_rows = [_coerce_scalar(v) for v in display_rows]
            max_rows_setting = pd.get_option("display.max_rows")
            if len(self) > max_rows_setting and display_rows:
                ellipsis_marker = "\u2026"
                if display_rows[-1] != ellipsis_marker:
                    display_rows.insert(len(display_rows) // 2, ellipsis_marker)
        max_rows = kwargs.pop("max_rows", None)
        if max_rows is not None:
            display_rows = display_rows[:max_rows]

        columns = kwargs.pop("columns", list(self.columns))
        max_cols = kwargs.pop("max_columns", None)
        if max_cols is not None and len(columns) > max_cols:
            head = max_cols // 2
            tail = max_cols - head
            columns = columns[:head] + columns[-tail:]

        display_options = self._custom_attrs.get("display_options") or {}
        default_height = display_options.get("height") or display_options.get("width")
        default_width = display_options.get("width") or "300px"

        def _css_size(value: Any, default: str) -> str:
            if value is None:
                return default
            if isinstance(value, (np.integer, np.floating)):
                value = value.item()
            if isinstance(value, (int, float)):
                return f"{int(value)}px"
            return str(value)

        widget_height = _css_size(kwargs.pop("widget_height", "100%"), "100%")
        cell_width = kwargs.pop("cell_width", None)
        index_width = _css_size(kwargs.pop("index_width", "140px"), "140px")
        row_height = _css_size(
            kwargs.pop("row_height", default_height or "300px"), "300px"
        )
        debug = kwargs.pop("debug", False)

        grid = widgets.GridspecLayout(
            len(display_rows) + 1,
            len(columns) + 1,
            layout=widgets.Layout(
                width="auto",
                max_width="100%",
                max_height=_css_size(kwargs.pop("table_max_height", "700px"), "700px"),
                overflow="auto",
            ),
        )
        column_width = _css_size(
            kwargs.pop("column_width", cell_width or default_width), "300px"
        )
        grid.layout.grid_template_columns = f"{index_width} " + " ".join(
            [column_width] * len(columns)
        )
        grid.layout.grid_auto_rows = row_height
        grid.layout.grid_column_gap = _css_size(
            kwargs.pop("column_gap", "12px"), "12px"
        )
        grid.layout.grid_row_gap = _css_size(kwargs.pop("row_gap", "8px"), "8px")
        grid.layout.align_items = "stretch"
        grid.layout.justify_items = "flex-start"

        header_row_height = kwargs.pop("header_row_height", "28px")

        def _safe_text(value: Any) -> str:
            if isinstance(value, (np.integer, np.floating)):
                return str(value.item())
            return str(value)

        grid[0, 0] = widgets.HTML(
            value=(
                "<div style='width:100%;height:100%;background:#EBEBEB;"
                "padding:5px 4px 5px 4px;line-height:1;'></div>"
            ),
            layout=widgets.Layout(height=header_row_height, width="100%"),
        )
        for col_idx, col in enumerate(columns, start=1):
            grid[0, col_idx] = widgets.HTML(
                value=(
                    "<div style='width:100%;height:100%;background:#EBEBEB;"
                    "padding:5px 4px 5px 4px;line-height:1;'>"
                    f"<b>{html_lib.escape(_safe_text(col))}</b></div>"
                ),
                layout=widgets.Layout(
                    height=header_row_height,
                    width="100%",
                ),
            )

        for row_idx, row_label in enumerate(display_rows, start=1):
            row_bg = "#FFFFFF" if row_idx % 2 == 1 else "#F5F5F5"
            row_label_html = (
                f"<div style='width:100%;height:100%;background:{row_bg};"
                "padding:4px;'>"
                f"<b>{html_lib.escape(_safe_text(row_label))}</b></div>"
            )
            grid[row_idx, 0] = widgets.HTML(
                value=row_label_html,
                layout=widgets.Layout(
                    height="100%",
                    width="100%",
                ),
            )
            for col_idx, col in enumerate(columns, start=1):
                if str(row_label) == "\u2026":
                    grid[row_idx, col_idx] = widgets.HTML(
                        value=(
                            f"<div style='width:100%;height:100%;background:{row_bg};"
                            "padding:4px;text-align:center;'>\u2026</div>"
                        ),
                        layout=widgets.Layout(width="100%", height="100%"),
                    )
                    continue
                if col in target_columns:
                    try:
                        volume, _dims = self._get_3d_volume_from_cell(
                            row=row_label, column=col
                        )
                        label_overlay = self._get_3d_label_overlay_from_cell(
                            row=row_label,
                            column=col,
                            expected_shape=volume.shape,
                        )
                        effective_height = (
                            row_height if widget_height == "100%" else widget_height
                        )
                        viewer = self._build_pyvista_viewer(
                            volume=volume,
                            backend=backend,
                            widget_height=effective_height,
                            label_volume=label_overlay,
                        )
                        grid[row_idx, col_idx] = widgets.Box(
                            [viewer],
                            layout=widgets.Layout(
                                width="100%",
                                height=row_height,
                                position="relative",
                                display="flex",
                                align_items="stretch",
                                justify_content="flex-start",
                                overflow="hidden",
                                margin="0",
                                padding="0",
                            ),
                        )
                        continue
                    except Exception as exc:
                        if debug:
                            raise
                        logger.debug("3D widget render failed: %s", exc)
                        grid[row_idx, col_idx] = widgets.HTML(
                            value="3D render failed",
                            layout=widgets.Layout(width="100%", height=row_height),
                        )
                        continue

                value = self.loc[row_label, col]
                text_value = html_lib.escape(_safe_text(value))
                grid[row_idx, col_idx] = widgets.HTML(
                    value=(
                        f"<div style='width:100%;height:100%;background:{row_bg};"
                        f"padding:4px;'>{text_value}</div>"
                    ),
                    layout=widgets.Layout(
                        width="100%",
                        height="100%",
                    ),
                )

        return grid

    def get_displayed_rows(self: CytoDataFrame_type) -> List[int]:
        """
        Get the indices of the rows that are currently
        displayed based on the pandas display settings.

        Returns:
            List[int]:
                A list of indices of the rows that
                are currently displayed.
        """

        # Get the current display settings
        max_rows = pd.get_option("display.max_rows")
        min_rows = pd.get_option("display.min_rows")

        if len(self) <= max_rows:
            # If the DataFrame has fewer rows than max_rows, all rows will be displayed
            return self.index.tolist()
        else:
            # Calculate how many rows will be displayed at the beginning and end
            half_min_rows = min_rows // 2
            start_display = self.index[:half_min_rows].tolist()
            end_display = self.index[-half_min_rows:].tolist()
            logger.debug("Detected display rows: %s", start_display + end_display)
            return start_display + end_display

    @staticmethod
    def _normalize_labels(labels: pd.Index) -> Tuple[pd.Index, Dict[str, Any]]:
        """
        Return (labels_as_str: pd.Index, backmap: dict[str, Any])
        """
        labels_as_str = pd.Index(map(str, labels))
        backmap = dict(zip(labels_as_str, labels))
        return labels_as_str, backmap

    def _generate_jupyter_dataframe_html(  # noqa: C901, PLR0912, PLR0915
        self: CytoDataFrame_type,
    ) -> str:
        """
        Returns HTML representation of the underlying pandas DataFrame
        for use within Juypyter notebook environments and similar.

        Referenced with modifications from:
        https://github.com/pandas-dev/pandas/blob/v2.2.2/pandas/core/frame.py#L1216

        Modifications added to help achieve image-based output for single-cell data
        within the context of CytoDataFrame and coSMicQC.

        Mainly for Jupyter notebooks.

        Returns:
            str: The data in a pandas DataFrame.
        """

        # handles DataFrame.info representations
        if self._info_repr():
            buf = StringIO()
            self.info(buf=buf)
            # need to escape the <class>, should be the first line.
            val = buf.getvalue().replace("<", r"&lt;", 1)
            val = val.replace(">", r"&gt;", 1)
            return f"<pre>{val}</pre>"

        # if we're in a notebook process as though in a jupyter environment
        if get_option("display.notebook_repr_html"):
            max_rows = get_option("display.max_rows")
            min_rows = get_option("display.min_rows")
            max_cols = get_option("display.max_columns")
            show_dimensions = get_option("display.show_dimensions")

            if self._custom_attrs["is_transposed"]:
                # if the data are transposed,
                # we transpose them back to keep
                # logic the same here.
                data = self.transpose()

            # Re-add bounding box columns if they are no longer available
            bounding_box_externally_joined = False
            if self._custom_attrs["data_bounding_box"] is not None and not all(
                col in self.columns.tolist()
                for col in self._custom_attrs["data_bounding_box"].columns.tolist()
            ):
                logger.debug("Re-adding bounding box columns.")
                data = (
                    self.join(other=self._custom_attrs["data_bounding_box"])
                    if not self._custom_attrs["is_transposed"]
                    else data.join(other=self._custom_attrs["data_bounding_box"])
                )
                bounding_box_externally_joined = True
            else:
                data = self.copy() if not bounding_box_externally_joined else data

            # Re-add compartment center xy columns if they are no longer available
            compartment_center_externally_joined = False
            if self._custom_attrs["compartment_center_xy"] is not None and not all(
                col
                in (data if bounding_box_externally_joined else self).columns.tolist()
                for col in self._custom_attrs["compartment_center_xy"].columns.tolist()
            ):
                logger.debug("Re-adding compartment center xy columns.")
                data = (
                    data.join(other=self._custom_attrs["compartment_center_xy"])
                    if bounding_box_externally_joined
                    else self.join(other=self._custom_attrs["compartment_center_xy"])
                )
                compartment_center_externally_joined = True
            else:
                data = (
                    data
                    if bounding_box_externally_joined
                    or compartment_center_externally_joined
                    else self.copy()
                )

            # Re-add image path columns if they are no longer available
            image_paths_externally_joined = False
            if self._custom_attrs["data_image_paths"] is not None and not all(
                col
                in (
                    data if compartment_center_externally_joined else self
                ).columns.tolist()
                for col in self._custom_attrs["data_image_paths"].columns.tolist()
            ):
                logger.debug("Re-adding image path columns.")
                logger.debug(
                    "bounding_box: %s",
                    compartment_center_externally_joined
                    or bounding_box_externally_joined,
                )
                data = (
                    data.join(other=self._custom_attrs["data_image_paths"])
                    if compartment_center_externally_joined
                    or bounding_box_externally_joined
                    else self.join(other=self._custom_attrs["data_image_paths"])
                )
                image_paths_externally_joined = True
            else:
                data = (
                    data
                    if image_paths_externally_joined or bounding_box_externally_joined
                    else self.copy()
                )

            # determine if we have image_cols to display
            image_cols = CytoDataFrame(data).find_image_columns() or []
            # normalize both the set of image cols and the pool of all cols to strings
            all_cols_str, all_cols_back = self._normalize_labels(data.columns)
            image_cols_str = [str(c) for c in image_cols]

            # If your helper expects strings, pass strings; then map the result back
            image_path_cols_str = (
                CytoDataFrame(data).find_image_path_columns(
                    image_cols=image_cols_str, all_cols=all_cols_str
                )
                or {}
            )
            display_options = self._custom_attrs.get("display_options", {}) or {}
            if self._custom_attrs.get("data_context_dir") and display_options.get(
                "ignore_image_path_columns"
            ):
                logger.debug("Ignoring image path columns due to display option.")
                image_path_cols_str = {}

            # Remap any returned path-column names back to the
            # original (possibly non-string) labels
            image_path_cols = {}
            for img_col in image_cols:
                key = str(img_col)
                if key in image_path_cols_str:
                    path_col_str = image_path_cols_str[key]
                    # path_col_str should be one of all_cols_str; map back to original
                    image_path_cols[img_col] = all_cols_back.get(
                        str(path_col_str), path_col_str
                    )

            logger.debug("Image columns found: %s", image_cols)

            # gather indices which will be displayed based on pandas configuration
            display_indices = CytoDataFrame(data).get_displayed_rows()

            # gather bounding box columns for use below
            if self._custom_attrs["data_bounding_box"] is not None:
                bounding_box_cols = self._custom_attrs[
                    "data_bounding_box"
                ].columns.tolist()

                # gather compartment_xy columns for use below
                if self._custom_attrs["compartment_center_xy"] is not None:
                    compartment_center_xy_cols = self._custom_attrs[
                        "compartment_center_xy"
                    ].columns.tolist()

                for image_col in image_cols:
                    data.loc[display_indices, image_col] = data.loc[
                        display_indices
                    ].apply(
                        lambda row: self.process_image_data_as_html_display(
                            data_value=row[image_col],
                            bounding_box=(
                                # rows below are specified using the column name to
                                # determine which part of the bounding box the columns
                                # relate to (the list of column names could be in
                                # various order).
                                row[
                                    next(
                                        col
                                        for col in bounding_box_cols
                                        if "Minimum_X" in col
                                    )
                                ],
                                row[
                                    next(
                                        col
                                        for col in bounding_box_cols
                                        if "Minimum_Y" in col
                                    )
                                ],
                                row[
                                    next(
                                        col
                                        for col in bounding_box_cols
                                        if "Maximum_X" in col
                                    )
                                ],
                                row[
                                    next(
                                        col
                                        for col in bounding_box_cols
                                        if "Maximum_Y" in col
                                    )
                                ],
                            ),
                            compartment_center_xy=(
                                (
                                    # rows below are specified using the column name to
                                    # determine which part of the bounding box the
                                    # columns relate to (the list of column names
                                    # could be in various order).
                                    row[
                                        next(
                                            col
                                            for col in compartment_center_xy_cols
                                            if "X" in col
                                        )
                                    ],
                                    row[
                                        next(
                                            col
                                            for col in compartment_center_xy_cols
                                            if "Y" in col
                                        )
                                    ],
                                )
                                if self._custom_attrs["compartment_center_xy"]
                                is not None
                                else None
                            ),
                            # set the image path based on the image_path cols.
                            image_path=(
                                row[image_path_cols[image_col]]
                                if image_path_cols is not None and image_path_cols != {}
                                else None
                            ),
                        ),
                        axis=1,
                    )

            if bounding_box_externally_joined:
                data = data.drop(
                    self._custom_attrs["data_bounding_box"].columns.tolist(), axis=1
                )

            if compartment_center_externally_joined:
                data = data.drop(
                    self._custom_attrs["compartment_center_xy"].columns.tolist(), axis=1
                )

            if image_paths_externally_joined:
                data = data.drop(
                    self._custom_attrs["data_image_paths"].columns.tolist(), axis=1
                )

            ome_arrow_cols = self.find_ome_arrow_columns(data)
            if ome_arrow_cols:
                for ome_col in ome_arrow_cols:
                    data.loc[display_indices, ome_col] = data.loc[
                        display_indices, ome_col
                    ].apply(self.process_ome_arrow_data_as_html_display)

            if self._custom_attrs["is_transposed"]:
                # retranspose to return the
                # data in the shape expected
                # by the user.
                data = data.transpose()

            formatter = fmt.DataFrameFormatter(
                data,
                columns=None,
                col_space=None,
                na_rep="NaN",
                formatters=None,
                float_format=None,
                sparsify=None,
                justify=None,
                index_names=True,
                header=True,
                index=True,
                bold_rows=True,
                # note: we avoid escapes to allow HTML rendering for images
                escape=False,
                max_rows=max_rows,
                min_rows=min_rows,
                max_cols=max_cols,
                show_dimensions=show_dimensions,
                decimal=".",
            )

            table_html = fmt.DataFrameRenderer(formatter).to_html()
            style = (
                "<style>"
                "table.dataframe thead tr {background:#EBEBEB;}"
                "table.dataframe thead th {background:#EBEBEB;}"
                "table.dataframe tbody tr {background:#FFFFFF;}"
                "table.dataframe tbody tr:nth-child(even) {background:#F5F5F5;}"
                "</style>"
            )
            return style + table_html

        else:
            return None

    def _render_output(self: CytoDataFrame_type) -> None:
        # Return a hidden div that nbconvert will keep but Jupyter will ignore
        html_content = self._generate_jupyter_dataframe_html()

        with self._custom_attrs["_output"]:
            display(HTML(html_content))
            if "cyto-3d-image" in html_content and "data-volume" in html_content:
                display(
                    Javascript(
                        build_3d_vtk_js_initializer(
                            display_options=self._custom_attrs.get("display_options")
                        )
                    )
                )

        # Only emit static HTML outside notebooks to avoid duplicate
        # tables inside ipywidgets output.
        if not get_option("display.notebook_repr_html"):
            display(
                HTML(
                    f"""
                    <style>
                        /* Show only when printing */
                        @media print {{
                            .print-view {{
                                display: block !important;
                                margin-top: 1em;
                            }}
                        }}

                    </style>
                    <div class="print-view" style="display:none;">
                            {html_content}
                    </div>
                    """
                )
            )

    def _pyvista_volume_snapshot_html(  # noqa: C901, PLR0912, PLR0915
        self: CytoDataFrame_type,
        volume: np.ndarray,
        dims: Tuple[int, int, int],
        label_volume: Optional[np.ndarray] = None,
    ) -> Optional[str]:
        """Render a static PyVista snapshot for a 3D volume.

        Args:
            volume: Source volume in ``(z, y, x)`` order.
            dims: Declared vtk dimensions for the volume.
            label_volume: Optional binary label volume aligned to ``volume``.

        Returns:
            A PNG-backed ``<img>`` HTML string, or ``None`` if snapshot rendering
            cannot be completed.
        """
        try:
            import pyvista as pv  # type: ignore
        except Exception:
            return None

        display_options = self._custom_attrs.get("display_options", {}) or {}
        width = display_options.get("width", "300px")
        height = display_options.get("height", width)
        cmap = display_options.get("volume_cmap", "gray")
        background = display_options.get("volume_background", "black")
        percentile_clim = display_options.get("volume_percentile_clim", (1.0, 99.9))
        interpolation = display_options.get("volume_interpolation", "nearest")
        sampling_scale = display_options.get("volume_sampling_scale", 0.5)

        vol_xyz = np.transpose(volume, (2, 1, 0))
        expected_dims = (volume.shape[2], volume.shape[1], volume.shape[0])
        if dims != expected_dims:
            logger.debug(
                "Snapshot dims %s do not match volume-derived dims %s.",
                dims,
                expected_dims,
            )
        if vol_xyz.dtype != np.float32:
            vol_xyz = vol_xyz.astype(np.float32, copy=False)

        if vol_xyz.size:
            try:
                nz = vol_xyz[vol_xyz > 0]
                data_for_clim = nz if nz.size else vol_xyz
                vmin, vmax = np.percentile(data_for_clim, percentile_clim)
                vmin = float(vmin)
                vmax = float(vmax if vmax > vmin else vmin + 1.0)
            except Exception:
                vmin = float(np.min(vol_xyz))
                vmax = float(np.max(vol_xyz) if np.max(vol_xyz) > vmin else vmin + 1.0)
        else:
            vmin, vmax = 0.0, 1.0

        spacing = (1.0, 1.0, 1.0)
        base_sample = max(min(spacing), 1e-6)
        grid = pv.ImageData()
        grid.dimensions = tuple(int(v) for v in vol_xyz.shape)
        grid.spacing = spacing
        grid.origin = (0.0, 0.0, 0.0)
        grid.point_data.clear()
        grid.point_data["scalars"] = np.asfortranarray(vol_xyz).ravel(order="F")
        try:
            grid.point_data.set_active_scalars("scalars")
        except AttributeError:
            try:
                grid.point_data.active_scalars_name = "scalars"
            except Exception:
                grid.set_active_scalars("scalars")

        plotter = pv.Plotter(off_screen=True)
        plotter.set_background(background)
        vol_actor = plotter.add_volume(
            grid,
            scalars="scalars",
            opacity="sigmoid",
            shade=False,
            cmap=cmap,
            clim=(vmin, vmax),
            show_scalar_bar=False,
            opacity_unit_distance=base_sample,
        )
        try:
            prop = getattr(vol_actor, "prop", None) or vol_actor.GetProperty()
            if interpolation.lower().startswith("near"):
                prop.SetInterpolationTypeToNearest()
            else:
                prop.SetInterpolationTypeToLinear()
            if hasattr(prop, "SetInterpolateScalarsBeforeMapping"):
                prop.SetInterpolateScalarsBeforeMapping(False)
            if hasattr(prop, "SetScalarOpacityUnitDistance"):
                prop.SetScalarOpacityUnitDistance(base_sample)
        except Exception as exc:
            logger.debug("Unable to configure snapshot volume property: %s", exc)
        try:
            mapper = getattr(vol_actor, "mapper", None) or vol_actor.GetMapper()
            if hasattr(mapper, "SetAutoAdjustSampleDistances"):
                mapper.SetAutoAdjustSampleDistances(False)
            if hasattr(mapper, "SetUseJittering"):
                mapper.SetUseJittering(False)
            if hasattr(mapper, "SetSampleDistance"):
                mapper.SetSampleDistance(float(base_sample * sampling_scale))
        except Exception as exc:
            logger.debug("Unable to configure snapshot mapper sampling: %s", exc)

        self._add_label_overlay_to_plotter(
            plotter=plotter,
            volume=volume,
            label_volume=label_volume,
            spacing=spacing,
            base_sample=base_sample,
            display_options=display_options,
        )

        try:
            img = plotter.screenshot(return_img=True)
            if img is None:
                return None
            from PIL import Image as PILImage  # type: ignore

            buf = BytesIO()
            PILImage.fromarray(img).save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            html_style = ";".join([f"width:{width}", f"height:{height}"])
            return f'<img src="data:image/png;base64,{b64}" style="{html_style}"/>'
        except Exception as exc:
            logger.debug("Failed to render PyVista snapshot: %s", exc)
            return None

    def _snapshot_cache_key(self: CytoDataFrame_type, row: Any, column: Any) -> str:
        return f"{row}::{column}"

    def _enqueue_snapshot_tasks(
        self: CytoDataFrame_type,
        rows: List[Any],
        columns: List[Any],
    ) -> None:
        """Placeholder hook for async snapshot pre-rendering.

        TODO: Add optional background workers to precompute `_snapshot_cache`
        entries for the provided rows/columns when async rendering is enabled.
        """
        logger.debug(
            "Snapshot task queueing not implemented yet (rows=%d, columns=%d).",
            len(rows),
            len(columns),
        )

    def _generate_trame_snapshot_html(self: CytoDataFrame_type) -> str:  # noqa: C901
        """Generate a static HTML table with PyVista 3D snapshots."""
        html_content = self._generate_jupyter_dataframe_html()
        try:
            if self._custom_attrs.get("data_bounding_box") is None:
                return html_content

            data = self.copy()
            image_cols = self.find_image_columns() or []
            if not image_cols:
                return html_content

            display_indices = self.get_displayed_rows()
            cache = self._custom_attrs.get("_snapshot_cache", {})
            cache_lock = self._custom_attrs.get("_snapshot_cache_lock")
            for image_col in image_cols:

                def _render_cell(
                    row: pd.Series,
                    bound_image_col: Any = image_col,
                ) -> str:
                    try:
                        key = self._snapshot_cache_key(row.name, bound_image_col)
                        if cache_lock is not None:
                            with cache_lock:
                                snapshot = cache.get(key)
                        else:
                            snapshot = cache.get(key)
                        if snapshot:
                            return snapshot
                        volume, dims = self._get_3d_volume_from_cell(
                            row=row.name, column=bound_image_col
                        )
                        snapshot = self._pyvista_volume_snapshot_html(
                            volume,
                            dims,
                            label_volume=self._get_3d_label_overlay_from_cell(
                                row=row.name,
                                column=bound_image_col,
                                expected_shape=volume.shape,
                            ),
                        )
                        if cache_lock is not None:
                            with cache_lock:
                                cache[key] = snapshot
                        else:
                            cache[key] = snapshot
                        if snapshot:
                            return snapshot
                    except Exception as exc:
                        logger.debug(
                            "Snapshot rendering failed for row=%s column=%s: %s",
                            row.name,
                            bound_image_col,
                            exc,
                        )
                    return (
                        "<div style='padding:4px;color:#000;'>"
                        "Snapshot unavailable</div>"
                    )

                data.loc[display_indices, image_col] = data.loc[display_indices].apply(
                    _render_cell, axis=1
                )

            formatter = fmt.DataFrameFormatter(
                data,
                columns=None,
                col_space=None,
                na_rep="NaN",
                formatters=None,
                float_format=None,
                sparsify=None,
                justify=None,
                index_names=True,
                header=True,
                index=True,
                bold_rows=True,
                escape=False,
                max_rows=get_option("display.max_rows"),
                min_rows=get_option("display.min_rows"),
                max_cols=get_option("display.max_columns"),
                show_dimensions=get_option("display.show_dimensions"),
                decimal=".",
            )
            table_html = fmt.DataFrameRenderer(formatter).to_html()
            style = (
                "<style>table.dataframe th, table.dataframe td {color:#000;}</style>"
            )
            return style + table_html
        except Exception as exc:
            logger.debug("Failed to build trame snapshot HTML: %s", exc)
            return html_content

    def _repr_html_(self: CytoDataFrame_type, debug: bool = False) -> str:
        """
        Returns HTML representation of the underlying pandas DataFrame
        for use within Juypyter notebook environments and similar.

        We modify this to be a delivery mechanism for ipywidgets
        in order to dynamically adjust the dataframe display
        within Jupyter environments.

        Mainly for Jupyter notebooks.

        Returns:
            str: The data in a pandas DataFrame.
        """

        display_options = self._custom_attrs.get("display_options", {}) or {}
        force_trame = display_options.get("view") == "trame"
        auto_trame_for_3d = display_options.get("auto_trame_for_3d", True)
        columns_3d = self._find_3d_columns_for_display() if auto_trame_for_3d else []
        if (force_trame or columns_3d) and not debug:
            if force_trame and not columns_3d:
                columns_3d = list(
                    dict.fromkeys(
                        [
                            *(self.find_image_columns() or []),
                            *self.find_ome_arrow_columns(self),
                        ]
                    )
                )
            if columns_3d:
                try:
                    widget_table = self.show_widget_table(
                        column=columns_3d[0],
                        columns_3d=columns_3d,
                        backend=None,
                    )
                    display(widget_table)
                    html_content = self._generate_trame_snapshot_html()
                    details_html = (
                        '<details class="cyto-static-snapshot">'
                        "<summary>Static snapshot (for non-interactive view)</summary>"
                        f"{html_content}</details>"
                    )
                    display(HTML(details_html))
                    return None
                except Exception as exc:
                    logger.debug(
                        "Trame widget table render failed, falling back to HTML: %s",
                        exc,
                    )

        # if we're in a notebook process as though in a jupyter environment
        if get_option("display.notebook_repr_html") and not debug:
            if not self._custom_attrs["_widget_state"]["shown"]:
                display(
                    widgets.VBox(
                        [
                            self._custom_attrs["_scale_slider"],
                            self._custom_attrs["_output"],
                        ]
                    )
                )
                self._custom_attrs["_widget_state"]["shown"] = True

            # Attach the slider observer exactly once
            if not self._custom_attrs["_widget_state"]["observing"]:
                self._custom_attrs["_scale_slider"].observe(
                    self._on_slider_change, names="value"
                )
                self._custom_attrs["_widget_state"]["observing"] = True

            # render fresh HTML for this cell
            self._render_output()

        # allow for debug mode to be set which returns the HTML
        # without widgets.

        elif debug:
            return self._generate_jupyter_dataframe_html()

        else:
            return None

    def __repr__(self: CytoDataFrame_type, debug: bool = False) -> str:
        """
        Return the string representation of the CytoDataFrame.

        In notebook environments, this method suppresses the default string
        representation to prevent interference with the interactive `_repr_html_`
        output (e.g., ipywidgets-based GUI). When `debug` is set to True, the
        standard string representation is returned even in notebook contexts.

        Args:
            debug (bool, optional):
                If True, always return the standard representation regardless
                of notebook environment. Defaults to False.

        Returns:
            str:
                The string representation of the DataFrame (or an empty string
                in notebook view mode when debug is False).
        """

        if get_option("display.notebook_repr_html") and not debug:
            return ""
        else:
            return super().__repr__()

    def _enbable_debug_mode(self: CytoDataFrame_type) -> None:
        """
        Enable debug mode for the CytoDataFrame instance.
        This method sets the logger level to DEBUG and
        enables debug mode for the instance.
        """
        logger.setLevel(logging.DEBUG)

        # Only add a handler if none exist (to avoid duplicates)
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setLevel(logging.DEBUG)  # This is critical
            formatter = logging.Formatter("%(levelname)s: %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
