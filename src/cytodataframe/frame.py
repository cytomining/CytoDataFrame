"""
Defines a CytoDataFrame class.
"""

import base64
import logging
import pathlib
import re
import sys
import warnings
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

import imageio
import ipywidgets as widgets
import numpy as np
import pandas as pd
import skimage
import skimage.measure
from IPython import get_ipython
from IPython.display import HTML, display
from pandas._config import (
    get_option,
)
from pandas.io.formats import (
    format as fmt,
)
from skimage.util import img_as_ubyte

from .image import (
    adjust_with_adaptive_histogram_equalization,
    draw_outline_on_image_from_mask,
    draw_outline_on_image_from_outline,
    get_pixel_bbox_from_offsets,
)

logger = logging.getLogger(__name__)

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
                "shown": False,
            },
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

    def __getitem__(self: CytoDataFrame_type, key: Union[int, str]) -> Any:  # noqa: ANN401
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
    ) -> Any:  # noqa: ANN401
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

        def wrapper(*args: Tuple[Any, ...], **kwargs: Dict[str, Any]) -> Any:  # noqa: ANN401
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
        }

        # Determine which group of columns to select based on availability in self.data
        selected_group = None
        for group, cols in column_groups.items():
            if all(col in self.columns.tolist() for col in cols):
                selected_group = group
                break

        # Assign the selected columns to self.bounding_box_df
        if selected_group:
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
                lambda value: isinstance(value, str)
                and re.match(pattern, value, flags=re.IGNORECASE)
            )
            .any()
        ]

        logger.debug("Found image columns: %s", image_cols)

        return image_cols

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
    ) -> np.ndarray:
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
            return None

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
                    return draw_outline_on_image_from_mask(
                        orig_image=orig_image,
                        mask_image_path=matching_mask_file[0],
                        outline_color=outline_color,
                    )
                else:
                    return draw_outline_on_image_from_outline(
                        orig_image=orig_image,
                        outline_image_path=matching_mask_file[0],
                        outline_color=outline_color,
                    )
            return None

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
                        return draw_outline_on_image_from_mask(
                            orig_image=orig_image,
                            mask_image_path=matching_files[0],
                            outline_color=outline_color,
                        )
                    else:
                        return draw_outline_on_image_from_outline(
                            orig_image=orig_image,
                            outline_image_path=matching_files[0],
                            outline_color=outline_color,
                        )

        logger.debug("No mask or outline found for: %s", data_value)

        return None

    def process_image_data_as_html_display(  # noqa: PLR0912, C901, PLR0915
        self: CytoDataFrame_type,
        data_value: Any,  # noqa: ANN401
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

        # stringify the data value in case it isn't a string
        data_value = str(data_value)

        candidate_path = None
        # Get the pattern map for segmentation file regex
        pattern_map = self._custom_attrs.get("segmentation_file_regex")

        # Step 1: Find the candidate file if the data value is not already a file
        if not pathlib.Path(data_value).is_file():
            # determine if we have a file from the path (dir) + filename
            if (
                self._custom_attrs["data_context_dir"] is None
                and image_path is not None
                and (
                    existing_image_from_path := pathlib.Path(
                        f"{image_path}/{data_value}"
                    )
                ).is_file()
            ):
                logger.debug(
                    "Found existing image from path: %s", existing_image_from_path
                )
                candidate_path = existing_image_from_path

            # Search for the data value in the data context directory
            elif self._custom_attrs["data_context_dir"] is not None and (
                candidate_paths := list(
                    pathlib.Path(self._custom_attrs["data_context_dir"]).rglob(
                        data_value
                    )
                )
            ):
                logger.debug(
                    "Found candidate paths (and attempting to use the first): %s",
                    candidate_paths,
                )
                # If a candidate file is found, use the first one
                candidate_path = candidate_paths[0]

            else:
                logger.debug("No candidate file found for: %s", data_value)
                # If no candidate file is found, return the original data value
                return data_value

        # read the image as an array
        orig_image_array = imageio.imread(candidate_path)

        # Adjust the image with image adjustment callable
        # or adaptive histogram equalization
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

        # Normalize to 0-255 for image saving
        orig_image_array = img_as_ubyte(orig_image_array)

        prepared_image = None
        # Step 2: Search for a mask
        prepared_image = self.search_for_mask_or_outline(
            data_value=data_value,
            pattern_map=pattern_map,
            file_dir=self._custom_attrs["data_mask_context_dir"],
            candidate_path=candidate_path,
            orig_image=orig_image_array,
            mask=True,
        )

        # If no mask is found, proceed to search for an outline
        if prepared_image is None:
            # Step 3: Search for an outline if no mask was found
            prepared_image = self.search_for_mask_or_outline(
                data_value=data_value,
                pattern_map=pattern_map,
                file_dir=self._custom_attrs["data_outline_context_dir"],
                candidate_path=candidate_path,
                orig_image=orig_image_array,
                mask=False,
            )

        # Step 4: If neither mask nor outline is found, use the original image array
        if prepared_image is None:
            prepared_image = orig_image_array

        # Step 5: Add a red dot for the compartment center before cropping
        if (
            compartment_center_xy is not None
            and self._custom_attrs.get("display_options", None) is None
        ) or (
            self._custom_attrs.get("display_options", None) is not None
            and self._custom_attrs["display_options"].get("center_dot", True)
        ):
            center_x, center_y = map(int, compartment_center_xy)  # Ensure integers

            # Convert grayscale image to RGB if necessary
            # Check if the image is grayscale
            if len(prepared_image.shape) == 2:  # noqa: PLR2004
                prepared_image = skimage.color.gray2rgb(prepared_image)

            if (
                0 <= center_y < prepared_image.shape[0]
                and 0 <= center_x < prepared_image.shape[1]
            ):
                # Calculate the radius as a fraction of the bounding box size
                x_min, y_min, x_max, y_max = map(int, bounding_box)
                box_width = x_max - x_min
                box_height = y_max - y_min
                radius = max(
                    1, int(min(box_width, box_height) * 0.03)
                )  # 3% of the smaller dimension

                rr, cc = skimage.draw.disk(
                    (center_y, center_x), radius=radius, shape=prepared_image.shape[:2]
                )
                prepared_image[rr, cc] = [255, 0, 0]  # Red color in RGB

        # Step 6: Crop the image based on the bounding box and encode it to PNG format
        try:
            # set a default bounding box
            x_min, y_min, x_max, y_max = map(int, bounding_box)

            # if we have custom offset bounding box information, use it
            if self._custom_attrs.get("display_options", None) and self._custom_attrs[
                "display_options"
            ].get("offset_bounding_box", None):
                try:
                    # note: this will default to the nuclei centers based
                    # on earlier input for this parameter.
                    center_x, center_y = map(int, compartment_center_xy)

                    offset_bounding_box = self._custom_attrs["display_options"].get(
                        "offset_bounding_box"
                    )
                    # generate offset bounding box positions
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
                except IndexError:
                    logger.debug(
                        (
                            "Bounding box %s is out of bounds for image %s ."
                            " Defaulting to use bounding box from data."
                        ),
                        (x_min, y_min, x_max, y_max),
                        image_path,
                    )

            cropped_img_array = prepared_image[
                y_min:y_max, x_min:x_max
            ]  # Perform slicing
        except ValueError as e:
            raise ValueError(
                f"Bounding box contains invalid values: {bounding_box}"
            ) from e
        except IndexError as e:
            raise IndexError(
                f"Bounding box {bounding_box} is out of bounds for image dimensions "
                f"{prepared_image.shape}"
            ) from e

        logger.debug("Cropped image array shape: %s", cropped_img_array.shape)

        # Step 7:
        try:
            # Save cropped image to buffer
            png_bytes_io = BytesIO()

            # catch warnings about low contrast images and avoid displaying them
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                skimage.io.imsave(
                    png_bytes_io, cropped_img_array, plugin="imageio", extension=".png"
                )
            png_bytes = png_bytes_io.getvalue()

        except (FileNotFoundError, ValueError) as exc:
            # Handle errors if image processing fails
            logger.error(exc)
            return data_value

        logger.debug("Image processed successfully and being sent to HTML for display.")

        # Step 8: Return HTML image display as a base64-encoded PNG
        # we dynamically style the image so that it will be displayed based
        # on automatic or user-based settings from the display_options custom
        # attribute.
        display_options = self._custom_attrs.get("display_options", {})
        if display_options is None:
            display_options = {}
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
            # Show widget only once per instance per notebook session
            if not self._custom_attrs["_widget_state"]["shown"]:
                display(self._custom_attrs["_scale_slider"])
                self._custom_attrs["_widget_state"]["shown"] = True

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
            if image_cols := CytoDataFrame(data).find_image_columns():
                # attempt to find the image path columns
                image_path_cols = CytoDataFrame(data).find_image_path_columns(
                    image_cols=image_cols, all_cols=data.columns
                )
            logger.debug("Image columns found: %s", image_cols)

            # gather indices which will be displayed based on pandas configuration
            display_indices = CytoDataFrame(data).get_displayed_rows()

            # gather bounding box columns for use below
            bounding_box_cols = self._custom_attrs["data_bounding_box"].columns.tolist()

            # gather compartment_xy columns for use below
            if self._custom_attrs["compartment_center_xy"] is not None:
                compartment_center_xy_cols = self._custom_attrs[
                    "compartment_center_xy"
                ].columns.tolist()

            for image_col in image_cols:
                data.loc[display_indices, image_col] = data.loc[display_indices].apply(
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
                                # determine which part of the bounding box the columns
                                # relate to (the list of column names could be in
                                # various order).
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
                            if self._custom_attrs["compartment_center_xy"] is not None
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

            return fmt.DataFrameRenderer(formatter).to_html()

        else:
            return None

    def _render_output(self: CytoDataFrame_type) -> str:
        # Return a hidden div that nbconvert will keep but Jupyter will ignore
        html_content = self._generate_jupyter_dataframe_html()

        with self._custom_attrs["_output"]:
            display(HTML(html_content))

        # We duplicate the display so that the jupyter notebook
        # retains printable output (which appears in static exports
        # such as PDFs or GitHub webpages). Ipywidget output
        # rendering is not retained in these formats, so we must
        # add this in order to retain visibility of the data.
        display(
            HTML(
                f"""
                <style>
                    /* Hide by default on screen */
                    .print-view {{
                        display: none;
                        margin-top: 1em;
                    }}

                    /* Show only when printing */
                    @media print {{
                        .print-view {{
                            display: block;
                            margin-top: 1em;
                        }}
                    }}

                </style>
                <div class="print-view">
                        {html_content}
                </div>
                """
            )
        )

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

        # if we're in a notebook process as though in a jupyter environment
        if get_option("display.notebook_repr_html") and not debug:
            # always clear old output and show fresh slider + output
            self._custom_attrs["_output"].clear_output(wait=True)
            display(
                widgets.VBox(
                    [
                        self._custom_attrs["_scale_slider"],
                        self._custom_attrs["_output"],
                    ]
                )
            )
            # render fresh HTML for this cell
            self._render_output()
            # ensure slider continues to control the output
            self._custom_attrs["_scale_slider"].observe(
                self._on_slider_change, names="value"
            )

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
