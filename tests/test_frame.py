"""
Tests cosmicqc CytoDataFrame module
"""

import base64
import logging
import pathlib
import re
import sys
import types
import warnings
from collections import OrderedDict
from contextlib import nullcontext
from importlib.machinery import ModuleSpec
from io import BytesIO

import imageio.v2 as imageio
import ipywidgets as widgets
import numpy as np
import pandas as pd
import pytest
import tifffile
from _pytest.monkeypatch import MonkeyPatch
from PIL import Image
from pyarrow import parquet

from cytodataframe.frame import (
    FILTER_SLIDER_LABEL_WIDTH_PX,
    FILTER_SLIDER_READOUT_WIDTH_PX,
    FILTER_SLIDER_TOTAL_WIDTH_PX,
    MAX_FILTER_SLIDER_STOPS,
    CytoDataFrame,
)
from tests.utils import (
    cytodataframe_image_display_contains_pixels,
)


def test_to_ome_parquet_adds_arrow_column(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    image_path = image_dir / "sample.tiff"
    imageio.imwrite(image_path, np.zeros((10, 10), dtype=np.uint8))

    data = pd.DataFrame(
        {
            "Image_FileName_DNA": [image_path.name],
            "Image_PathName_DNA": [str(image_dir)],
            "Cells_AreaShape_BoundingBoxMinimum_X": [0],
            "Cells_AreaShape_BoundingBoxMinimum_Y": [0],
            "Cells_AreaShape_BoundingBoxMaximum_X": [10],
            "Cells_AreaShape_BoundingBoxMaximum_Y": [10],
        }
    )

    cdf = CytoDataFrame(data=data)

    class TestOMEArrow:
        def __init__(self, data: str):  # noqa: ANN204
            self.data = data

    test_module = types.SimpleNamespace(
        OMEArrow=TestOMEArrow,
        __version__="test",
        __spec__=types.SimpleNamespace(loader=None),
    )
    monkeypatch.setitem(sys.modules, "ome_arrow", test_module)

    captured: dict = {}

    def fake_write_table(table, file_path, **kwargs):  # noqa: ANN001, ANN202, ANN003
        captured["df"] = table.to_pandas()
        captured["file_path"] = file_path
        captured["kwargs"] = kwargs
        captured["metadata"] = table.schema.metadata or {}

    monkeypatch.setattr("pyarrow.parquet.write_table", fake_write_table, raising=False)

    output_path = tmp_path / "out.parquet"
    cdf.to_ome_parquet(output_path)

    composite_col = "Image_FileName_DNA_OMEArrow_COMP"
    orig_col = "Image_FileName_DNA_OMEArrow_ORIG"
    mask_col = "Image_FileName_DNA_OMEArrow_LABL"
    for column in (composite_col, orig_col, mask_col):
        assert column in captured["df"].columns

    comp_value = captured["df"].loc[0, composite_col]
    orig_value = captured["df"].loc[0, orig_col]
    mask_value = captured["df"].loc[0, mask_col]

    assert isinstance(comp_value, str) and comp_value.endswith(".tiff")
    assert isinstance(orig_value, str) and orig_value.endswith(".tiff")
    assert mask_value is None
    assert captured["file_path"] == output_path
    metadata = captured["metadata"]
    assert metadata[b"cytodataframe:data-producer"]
    assert metadata[b"cytodataframe:data-producer-version"]


def test_to_ome_parquet_real_data(
    tmp_path: pathlib.Path, cytotable_NF1_data_parquet_shrunken: str
) -> None:
    pytest.importorskip(
        "ome_arrow", reason="to_ome_parquet real-data test requires ome-arrow"
    )

    parquet_path = pathlib.Path(cytotable_NF1_data_parquet_shrunken)
    image_dir = parquet_path.parent / "Plate_2_images"
    mask_dir = parquet_path.parent / "Plate_2_masks"

    cdf = CytoDataFrame(
        data=cytotable_NF1_data_parquet_shrunken,
        data_context_dir=str(image_dir),
        data_mask_context_dir=str(mask_dir),
    )

    output_path = tmp_path / "nf1.ome.parquet"
    image_cols = cdf.find_image_columns()

    cdf.to_ome_parquet(output_path)

    assert output_path.exists()
    table = parquet.read_table(output_path)
    expected_arrow_cols = []
    for col in image_cols:
        expected_arrow_cols.extend(
            [
                f"{col}_OMEArrow_COMP",
                f"{col}_OMEArrow_ORIG",
                f"{col}_OMEArrow_LABL",
            ]
        )
    for column in expected_arrow_cols:
        assert column in table.column_names

    mask_cols = [f"{col}_OMEArrow_LABL" for col in image_cols]
    mask_df = table.select(mask_cols).to_pandas()
    assert mask_df.notna().any().any()


def test_to_ome_parquet_layer_flags(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    image_path = image_dir / "sample.tiff"
    imageio.imwrite(image_path, np.zeros((10, 10), dtype=np.uint8))

    data = pd.DataFrame(
        {
            "Image_FileName_DNA": [image_path.name],
            "Image_PathName_DNA": [str(image_dir)],
            "Cells_AreaShape_BoundingBoxMinimum_X": [0],
            "Cells_AreaShape_BoundingBoxMinimum_Y": [0],
            "Cells_AreaShape_BoundingBoxMaximum_X": [10],
            "Cells_AreaShape_BoundingBoxMaximum_Y": [10],
        }
    )

    cdf = CytoDataFrame(data=data)

    class TestOMEArrow:
        def __init__(self, data: str):  # noqa: ANN204
            self.data = data

    test_module = types.SimpleNamespace(
        OMEArrow=TestOMEArrow,
        __version__="test",
        __spec__=types.SimpleNamespace(loader=None),
    )
    monkeypatch.setitem(sys.modules, "ome_arrow", test_module)

    captured: dict = {}

    def fake_write_table(table, file_path, **kwargs):  # noqa: ANN001, ANN202, ANN003
        captured["df"] = table.to_pandas()

    monkeypatch.setattr("pyarrow.parquet.write_table", fake_write_table, raising=False)

    cdf.to_ome_parquet(
        tmp_path / "out.parquet",
        include_original=False,
        include_mask_outline=False,
        include_composite=True,
    )

    columns = captured["df"].columns
    assert "Image_FileName_DNA_OMEArrow_COMP" in columns
    assert "Image_FileName_DNA_OMEArrow_ORIG" not in columns
    assert "Image_FileName_DNA_OMEArrow_LABL" not in columns


def test_ome_arrow_columns_render_html(
    tmp_path: pathlib.Path, cytotable_NF1_data_parquet_shrunken: str
) -> None:
    pytest.importorskip(
        "ome_arrow", reason="OME-Arrow rendering test requires ome-arrow"
    )

    parquet_path = pathlib.Path(cytotable_NF1_data_parquet_shrunken)
    image_dir = parquet_path.parent / "Plate_2_images"
    mask_dir = parquet_path.parent / "Plate_2_masks"

    raw_cdf = CytoDataFrame(
        data=cytotable_NF1_data_parquet_shrunken,
        data_context_dir=str(image_dir),
        data_mask_context_dir=str(mask_dir),
    )

    ome_path = tmp_path / "nf1.arrow.parquet"
    raw_cdf.to_ome_parquet(ome_path)

    arrow_cdf = CytoDataFrame(data=ome_path)
    arrow_cols = [col for col in arrow_cdf.columns if col.endswith("_OMEArrow_COMP")]
    assert arrow_cols

    html_output = arrow_cdf[arrow_cols]._repr_html_(debug=True)
    assert "data:image/png;base64" in html_output


def test_prepare_layers_mask_binary(tmp_path: pathlib.Path) -> None:
    image_array = np.zeros((6, 6), dtype=np.uint8)
    image_path = tmp_path / "cell.tiff"
    imageio.imwrite(image_path, image_array)

    mask_array = np.zeros((6, 6, 3), dtype=np.uint8)
    mask_array[1:4, 1:4] = (0, 255, 0)
    mask_path = tmp_path / "cell_mask.png"
    imageio.imwrite(mask_path, mask_array)

    data = pd.DataFrame(
        {
            "Image_FileName_DNA": ["cell.tiff"],
            "Image_PathName_DNA": [str(tmp_path)],
            "Cells_AreaShape_BoundingBoxMinimum_X": [0],
            "Cells_AreaShape_BoundingBoxMinimum_Y": [0],
            "Cells_AreaShape_BoundingBoxMaximum_X": [6],
            "Cells_AreaShape_BoundingBoxMaximum_Y": [6],
        }
    )

    cdf = CytoDataFrame(
        data=data,
        data_context_dir=str(tmp_path),
        data_mask_context_dir=str(tmp_path),
    )

    layers = cdf._prepare_cropped_image_layers(
        data_value="cell.tiff",
        bounding_box=(0, 0, 6, 6),
        include_mask_outline=True,
        include_original=False,
        include_composite=False,
    )

    mask_layer = layers["mask"]
    assert mask_layer is not None
    assert mask_layer.dtype == np.uint8
    assert set(np.unique(mask_layer).tolist()).issubset({0, 255})


def test_prepare_layers_3d_uses_loaded_volume_without_ome_arrow_fallback(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    volume = np.arange(4 * 5 * 6, dtype=np.uint8).reshape(4, 5, 6)
    image_path = tmp_path / "vol3d.tiff"
    tifffile.imwrite(image_path, volume)

    cdf = CytoDataFrame(
        data=pd.DataFrame({"Image_FileName_DNA": [image_path.name]}),
        data_context_dir=str(tmp_path),
    )

    def fail_ome_arrow_path(**_kwargs: object) -> str:
        raise AssertionError("OME-Arrow fallback should not be used for 3D TIFF")

    monkeypatch.setattr(
        "cytodataframe.frame.build_3d_html_from_path",
        fail_ome_arrow_path,
    )
    layers = cdf._prepare_cropped_image_layers(
        data_value=image_path.name,
        bounding_box=(0, 0, 6, 5),
        include_composite=False,
        include_original=False,
        include_mask_outline=False,
    )

    html_value = layers.get(CytoDataFrame._HTML_3D_STUB_KEY)
    assert isinstance(html_value, str)
    assert "data-volume=" in html_value


def test_prepare_layers_3d_includes_label_overlay_from_mask_dir(
    tmp_path: pathlib.Path,
) -> None:
    volume = np.arange(4 * 5 * 6, dtype=np.uint8).reshape(4, 5, 6)
    image_path = tmp_path / "vol3d.tiff"
    tifffile.imwrite(image_path, volume)

    mask_dir = tmp_path / "masks"
    mask_dir.mkdir()
    label = np.zeros((4, 5, 6), dtype=np.uint8)
    label[1:3, 2:4, 1:5] = 255
    tifffile.imwrite(mask_dir / "vol3d_mask.tiff", label)

    cdf = CytoDataFrame(
        data=pd.DataFrame({"Image_FileName_DNA": [image_path.name]}),
        data_context_dir=str(tmp_path),
        data_mask_context_dir=str(mask_dir),
    )

    layers = cdf._prepare_cropped_image_layers(
        data_value=image_path.name,
        bounding_box=(0, 0, 6, 5),
        include_composite=False,
        include_original=False,
        include_mask_outline=False,
    )

    html_value = layers.get(CytoDataFrame._HTML_3D_STUB_KEY)
    assert isinstance(html_value, str)
    assert "data-volume=" in html_value
    assert 'data-label-volume="' in html_value


def test_get_3d_label_overlay_from_cell_applies_bbox_crop(
    tmp_path: pathlib.Path,
) -> None:
    volume = np.arange(4 * 5 * 6, dtype=np.uint8).reshape(4, 5, 6)
    image_path = tmp_path / "vol3d.tiff"
    tifffile.imwrite(image_path, volume)

    mask_dir = tmp_path / "masks"
    mask_dir.mkdir()
    label = np.zeros((4, 5, 6), dtype=np.uint8)
    label[1:3, 1:4, 1:5] = 255
    tifffile.imwrite(mask_dir / "vol3d_mask.tiff", label)

    data = pd.DataFrame(
        {
            "Image_FileName_DNA": [image_path.name],
            "AreaShape_BoundingBoxMinimum_X": [1],
            "AreaShape_BoundingBoxMaximum_X": [5],
            "AreaShape_BoundingBoxMinimum_Y": [1],
            "AreaShape_BoundingBoxMaximum_Y": [4],
            "AreaShape_BoundingBoxMinimum_Z": [1],
            "AreaShape_BoundingBoxMaximum_Z": [3],
        }
    )
    cdf = CytoDataFrame(
        data=data,
        data_context_dir=str(tmp_path),
        data_mask_context_dir=str(mask_dir),
    )

    cropped_volume, _ = cdf._get_3d_volume_from_cell(row=0, column="Image_FileName_DNA")
    overlay = cdf._get_3d_label_overlay_from_cell(
        row=0,
        column="Image_FileName_DNA",
        expected_shape=cropped_volume.shape,
    )

    assert overlay is not None
    assert overlay.shape == cropped_volume.shape
    assert overlay.dtype == np.uint8
    assert overlay.max() == 255


def test_get_3d_bbox_crop_bounds_prefers_cellprofiler_columns() -> None:
    cdf = CytoDataFrame(
        data=pd.DataFrame(
            {
                "Other_Minimum_X": [0],
                "Other_Maximum_X": [10],
                "Other_Minimum_Y": [0],
                "Other_Maximum_Y": [10],
                "Cells_AreaShape_BoundingBoxMinimum_X": [2],
                "Cells_AreaShape_BoundingBoxMaximum_X": [6],
                "Cells_AreaShape_BoundingBoxMinimum_Y": [3],
                "Cells_AreaShape_BoundingBoxMaximum_Y": [7],
                "Cells_AreaShape_BoundingBoxMinimum_Z": [1],
                "Cells_AreaShape_BoundingBoxMaximum_Z": [4],
            }
        )
    )

    bounds = cdf._get_3d_bbox_crop_bounds(row=0, volume_shape=(8, 8, 8))

    assert bounds == (2, 6, 3, 7, 1, 4)


def test_get_3d_bbox_crop_bounds_accepts_custom_column_map() -> None:
    cdf = CytoDataFrame(
        data=pd.DataFrame(
            {
                "bbox_x0": [1],
                "bbox_x1": [5],
                "bbox_y0": [2],
                "bbox_y1": [6],
                "bbox_z0": [0],
                "bbox_z1": [3],
            }
        ),
        display_options={
            "volume_bbox_column_map": {
                "x_min": "bbox_x0",
                "x_max": "bbox_x1",
                "y_min": "bbox_y0",
                "y_max": "bbox_y1",
                "z_min": "bbox_z0",
                "z_max": "bbox_z1",
            }
        },
    )

    bounds = cdf._get_3d_bbox_crop_bounds(row=0, volume_shape=(8, 8, 8))

    assert bounds == (1, 5, 2, 6, 0, 3)


def test_find_matching_segmentation_path_filters_by_image_identifier(
    tmp_path: pathlib.Path,
) -> None:
    mask_dir = tmp_path / "masks"
    mask_dir.mkdir()
    (mask_dir / "img_a_mask.tiff").write_bytes(b"")
    (mask_dir / "img_b_mask.tiff").write_bytes(b"")

    matched = CytoDataFrame._find_matching_segmentation_path(
        data_value="img_a.tiff",
        pattern_map={r".*_mask\.tiff$": r".*"},
        file_dir=str(mask_dir),
        candidate_path=pathlib.Path("img_a.tiff"),
    )

    assert matched is not None
    assert matched.name == "img_a_mask.tiff"


def test_find_matching_segmentation_path_prefers_candidate_parent_tree(
    tmp_path: pathlib.Path,
) -> None:
    mask_dir = tmp_path / "masks"
    (mask_dir / "plate_a").mkdir(parents=True)
    (mask_dir / "plate_b").mkdir(parents=True)
    (mask_dir / "plate_a" / "nuclei1_mask.tiff").write_bytes(b"")
    (mask_dir / "plate_b" / "nuclei1_mask.tiff").write_bytes(b"")

    matched = CytoDataFrame._find_matching_segmentation_path(
        data_value="plate_a/nuclei1.tiff",
        pattern_map={r".*_mask\.tiff$": r".*"},
        file_dir=str(mask_dir),
        candidate_path=pathlib.Path("/tmp/plate_a/nuclei1.tiff"),
    )

    assert matched is not None
    assert matched.parent.name == "plate_a"


def test_cytodataframe_input(
    tmp_path: pathlib.Path,
    basic_outlier_dataframe: pd.DataFrame,
    basic_outlier_csv: str,
    basic_outlier_csv_gz: str,
    basic_outlier_tsv: str,
    basic_outlier_parquet: str,
):
    # Tests CytoDataFrame with pd.DataFrame input.
    sc_df = CytoDataFrame(data=basic_outlier_dataframe)

    # test that we ingested the data properly
    assert sc_df._custom_attrs["data_source"] == "pandas.DataFrame"
    assert sc_df.equals(basic_outlier_dataframe)

    # test export
    basic_outlier_dataframe.to_parquet(
        control_path := f"{tmp_path}/df_input_example.parquet"
    )
    sc_df.export(test_path := f"{tmp_path}/df_input_example1.parquet")

    assert parquet.read_table(control_path).equals(parquet.read_table(test_path))

    # Tests CytoDataFrame with pd.Series input.
    sc_df = CytoDataFrame(data=basic_outlier_dataframe.loc[0])

    # test that we ingested the data properly
    assert sc_df._custom_attrs["data_source"] == "pandas.Series"
    assert sc_df.equals(pd.DataFrame(basic_outlier_dataframe.loc[0]))

    # Tests CytoDataFrame with CSV input.
    sc_df = CytoDataFrame(data=basic_outlier_csv)
    expected_df = pd.read_csv(basic_outlier_csv)

    # test that we ingested the data properly
    assert sc_df._custom_attrs["data_source"] == str(basic_outlier_csv)
    assert sc_df.equals(expected_df)

    # test export
    sc_df.export(test_path := f"{tmp_path}/df_input_example.csv", index=False)

    pd.testing.assert_frame_equal(expected_df, pd.read_csv(test_path))

    # Tests CytoDataFrame with CSV input.
    sc_df = CytoDataFrame(data=basic_outlier_csv_gz)
    expected_df = pd.read_csv(basic_outlier_csv_gz)

    # test that we ingested the data properly
    assert sc_df._custom_attrs["data_source"] == str(basic_outlier_csv_gz)
    assert sc_df.equals(expected_df)

    # test export
    sc_df.export(test_path := f"{tmp_path}/df_input_example.csv.gz", index=False)

    pd.testing.assert_frame_equal(
        expected_df, pd.read_csv(test_path, compression="gzip")
    )

    # Tests CytoDataFrame with TSV input.
    sc_df = CytoDataFrame(data=basic_outlier_tsv)
    expected_df = pd.read_csv(basic_outlier_tsv, delimiter="\t")

    # test that we ingested the data properly
    assert sc_df._custom_attrs["data_source"] == str(basic_outlier_tsv)
    assert sc_df.equals(expected_df)

    # test export
    sc_df.export(test_path := f"{tmp_path}/df_input_example.tsv", index=False)

    pd.testing.assert_frame_equal(expected_df, pd.read_csv(test_path, sep="\t"))

    # Tests CytoDataFrame with parquet input.
    sc_df = CytoDataFrame(data=basic_outlier_parquet)
    expected_df = pd.read_parquet(basic_outlier_parquet)

    # test that we ingested the data properly
    assert sc_df._custom_attrs["data_source"] == str(basic_outlier_parquet)
    assert sc_df.equals(expected_df)

    # test export
    sc_df.export(test_path := f"{tmp_path}/df_input_example2.parquet")

    assert parquet.read_table(basic_outlier_parquet).equals(
        parquet.read_table(test_path)
    )

    # test CytoDataFrame with CytoDataFrame input
    copy_sc_df = CytoDataFrame(data=sc_df)

    pd.testing.assert_frame_equal(copy_sc_df, sc_df)


def test_repr_html_green_pixels(
    cytotable_NF1_data_parquet_shrunken: str,
    cytotable_nuclear_speckles_data_parquet: str,
    cytotable_pediatric_cancer_atlas_parquet: str,
):
    """
    Tests how images are rendered through customized repr_html in CytoDataFrame.
    """

    # Ensure there's at least one greenish pixel in the image
    # when context dirs are set for the NF1 dataset.
    assert cytodataframe_image_display_contains_pixels(
        frame=CytoDataFrame(
            data=cytotable_NF1_data_parquet_shrunken,
            data_context_dir=f"{pathlib.Path(cytotable_NF1_data_parquet_shrunken).parent}/Plate_2_images",
            data_mask_context_dir=f"{pathlib.Path(cytotable_NF1_data_parquet_shrunken).parent}/Plate_2_masks",
        ),
        image_cols=["Image_FileName_DAPI", "Image_FileName_GFP", "Image_FileName_RFP"],
        color_conditions={"green": 255, "red": None, "blue": None},
    ), "The NF1 images do not contain green outlines."

    # Ensure there's at least one greenish pixel in the image
    # when context dirs are NOT set for the NF1 dataset.
    nf1_dataset_with_modified_image_paths = pd.read_parquet(
        path=cytotable_NF1_data_parquet_shrunken
    )
    nf1_dataset_with_modified_image_paths.loc[
        :, ["Image_PathName_DAPI", "Image_PathName_GFP", "Image_PathName_RFP"]
    ] = f"{pathlib.Path(cytotable_NF1_data_parquet_shrunken).parent}/Plate_2_images"

    assert cytodataframe_image_display_contains_pixels(
        frame=CytoDataFrame(
            data=nf1_dataset_with_modified_image_paths,
            data_mask_context_dir=f"{pathlib.Path(cytotable_NF1_data_parquet_shrunken).parent}/Plate_2_masks",
        ),
        image_cols=["Image_FileName_DAPI", "Image_FileName_GFP", "Image_FileName_RFP"],
        color_conditions={"green": 255, "red": None, "blue": None},
    ), "The NF1 images do not contain green outlines."

    # Ensure there's at least one greenish pixel in the image
    # when context dirs are set for the nuclear speckles dataset.
    assert cytodataframe_image_display_contains_pixels(
        frame=CytoDataFrame(
            data=cytotable_nuclear_speckles_data_parquet,
            data_context_dir=f"{pathlib.Path(cytotable_nuclear_speckles_data_parquet).parent}/images",
            data_mask_context_dir=f"{pathlib.Path(cytotable_nuclear_speckles_data_parquet).parent}/masks",
        ),
        image_cols=[
            "Image_FileName_A647",
            "Image_FileName_DAPI",
            "Image_FileName_GOLD",
        ],
        color_conditions={"green": 255, "red": None, "blue": None},
    ), "The nuclear speckles images do not contain green outlines."

    # Ensure there's at least one greenish pixel in the image
    # when context dirs are set for the pediatric cancer dataset.
    assert cytodataframe_image_display_contains_pixels(
        frame=CytoDataFrame(
            data=cytotable_pediatric_cancer_atlas_parquet,
            data_context_dir=f"{pathlib.Path(cytotable_pediatric_cancer_atlas_parquet).parent}/images/orig",
            data_outline_context_dir=f"{pathlib.Path(cytotable_pediatric_cancer_atlas_parquet).parent}/images/outlines",
            segmentation_file_regex={
                r"CellsOutlines_BR(\d+)_C(\d{2})_\d+\.tiff": r".*ch3.*\.tiff",
                r"NucleiOutlines_BR(\d+)_C(\d{2})_\d+\.tiff": r".*ch5.*\.tiff",
            },
        ),
        image_cols=[
            "Image_FileName_OrigAGP",
            "Image_FileName_OrigDNA",
        ],
        color_conditions={"green": 255, "red": None, "blue": None},
    ), "The pediatric cancer atlas speckles images do not contain green outlines."

    # Ensure there's at least one greenish pixel in the image
    # when context dirs are NOT set for the pediatric cancer dataset.
    # (tests the regex associations with default image paths)
    pediatric_cancer_dataset_with_modified_image_paths = pd.read_parquet(
        path=cytotable_pediatric_cancer_atlas_parquet
    )
    # fmt: off
    pediatric_cancer_dataset_with_modified_image_paths = (
        pediatric_cancer_dataset_with_modified_image_paths.assign(
        Image_PathName_OrigAGP=(
            f"{pathlib.Path(cytotable_pediatric_cancer_atlas_parquet).parent}/images/orig"
        ),
        Image_PathName_OrigDNA=(
            f"{pathlib.Path(cytotable_pediatric_cancer_atlas_parquet).parent}/images/orig"
        ),
    )
    )
    # fmt: on

    assert cytodataframe_image_display_contains_pixels(
        frame=CytoDataFrame(
            data=pediatric_cancer_dataset_with_modified_image_paths,
            data_outline_context_dir=f"{pathlib.Path(cytotable_pediatric_cancer_atlas_parquet).parent}/images/outlines",
            segmentation_file_regex={
                r"CellsOutlines_BR(\d+)_C(\d{2})_\d+\.tiff": r".*ch3.*\.tiff",
                r"NucleiOutlines_BR(\d+)_C(\d{2})_\d+\.tiff": r".*ch5.*\.tiff",
            },
        ),
        image_cols=[
            "Image_FileName_OrigAGP",
            "Image_FileName_OrigDNA",
        ],
        color_conditions={"green": 255, "red": None, "blue": None},
    ), "The pediatric cancer atlas speckles images do not contain green outlines."


def test_repr_html_red_pixels(
    cytotable_NF1_data_parquet_shrunken: str,
    cytotable_nuclear_speckles_data_parquet: str,
    cytotable_pediatric_cancer_atlas_parquet: str,
):
    """
    Tests how images are rendered through customized repr_html in CytoDataFrame.
    """

    # Ensure there's at least one reddish pixel in the image
    # when context dirs are set for the NF1 dataset.
    assert cytodataframe_image_display_contains_pixels(
        frame=CytoDataFrame(
            data=cytotable_NF1_data_parquet_shrunken,
            data_context_dir=f"{pathlib.Path(cytotable_NF1_data_parquet_shrunken).parent}/Plate_2_images",
            data_mask_context_dir=f"{pathlib.Path(cytotable_NF1_data_parquet_shrunken).parent}/Plate_2_masks",
        ),
        image_cols=["Image_FileName_DAPI", "Image_FileName_GFP", "Image_FileName_RFP"],
        color_conditions={"green": None, "red": 255, "blue": None},
    ), "The NF1 images do not contain red dots."

    # Ensure there are no reddish pixels in the image
    # when context dirs are set for the NF1 dataset.
    assert not cytodataframe_image_display_contains_pixels(
        frame=CytoDataFrame(
            data=cytotable_NF1_data_parquet_shrunken,
            data_context_dir=f"{pathlib.Path(cytotable_NF1_data_parquet_shrunken).parent}/Plate_2_images",
            data_mask_context_dir=f"{pathlib.Path(cytotable_NF1_data_parquet_shrunken).parent}/Plate_2_masks",
            compartment_center_xy=False,
        ),
        image_cols=["Image_FileName_DAPI", "Image_FileName_GFP", "Image_FileName_RFP"],
        color_conditions={"green": None, "red": 255, "blue": None},
    ), "The NF1 images contain red pixels when it shouldn't."

    # Ensure there's at least one greenish pixel in the image
    # when context dirs are NOT set for the NF1 dataset.
    nf1_dataset_with_modified_image_paths = pd.read_parquet(
        path=cytotable_NF1_data_parquet_shrunken
    )
    nf1_dataset_with_modified_image_paths.loc[
        :, ["Image_PathName_DAPI", "Image_PathName_GFP", "Image_PathName_RFP"]
    ] = f"{pathlib.Path(cytotable_NF1_data_parquet_shrunken).parent}/Plate_2_images"

    assert cytodataframe_image_display_contains_pixels(
        frame=CytoDataFrame(
            data=nf1_dataset_with_modified_image_paths,
            data_mask_context_dir=f"{pathlib.Path(cytotable_NF1_data_parquet_shrunken).parent}/Plate_2_masks",
        ),
        image_cols=["Image_FileName_DAPI", "Image_FileName_GFP", "Image_FileName_RFP"],
        color_conditions={"green": None, "red": 255, "blue": None},
    ), "The NF1 images do not contain red dots."

    # Ensure there's at least one reddish pixel in the image
    # when context dirs are set for the nuclear speckles dataset.
    assert cytodataframe_image_display_contains_pixels(
        frame=CytoDataFrame(
            data=cytotable_nuclear_speckles_data_parquet,
            data_context_dir=f"{pathlib.Path(cytotable_nuclear_speckles_data_parquet).parent}/images",
            data_mask_context_dir=f"{pathlib.Path(cytotable_nuclear_speckles_data_parquet).parent}/masks",
        ),
        image_cols=[
            "Image_FileName_A647",
            "Image_FileName_DAPI",
            "Image_FileName_GOLD",
        ],
        color_conditions={"green": None, "red": 255, "blue": None},
    ), "The nuclear speckles images do not contain red dots."

    # Ensure there's at least one reddish pixel in the image
    # when context dirs are set for the pediatric cancer dataset.
    assert cytodataframe_image_display_contains_pixels(
        frame=CytoDataFrame(
            data=cytotable_pediatric_cancer_atlas_parquet,
            data_context_dir=f"{pathlib.Path(cytotable_pediatric_cancer_atlas_parquet).parent}/images/orig",
            data_outline_context_dir=f"{pathlib.Path(cytotable_pediatric_cancer_atlas_parquet).parent}/images/outlines",
            segmentation_file_regex={
                r"CellsOutlines_BR(\d+)_C(\d{2})_\d+\.tiff": r".*ch3.*\.tiff",
                r"NucleiOutlines_BR(\d+)_C(\d{2})_\d+\.tiff": r".*ch5.*\.tiff",
            },
        ),
        image_cols=[
            "Image_FileName_OrigAGP",
            "Image_FileName_OrigDNA",
        ],
        color_conditions={"green": None, "red": 255, "blue": None},
    ), "The pediatric cancer atlas speckles images do not contain red dots."

    # Ensure there's at least one reddish pixel in the image
    # when context dirs are NOT set for the pediatric cancer dataset.
    # (tests the regex associations with default image paths)
    pediatric_cancer_dataset_with_modified_image_paths = pd.read_parquet(
        path=cytotable_pediatric_cancer_atlas_parquet
    )
    # fmt: off
    pediatric_cancer_dataset_with_modified_image_paths = (
        pediatric_cancer_dataset_with_modified_image_paths.assign(
        Image_PathName_OrigAGP=(
            f"{pathlib.Path(cytotable_pediatric_cancer_atlas_parquet).parent}/images/orig"
        ),
        Image_PathName_OrigDNA=(
            f"{pathlib.Path(cytotable_pediatric_cancer_atlas_parquet).parent}/images/orig"
        ),
    )
    )
    # fmt: on

    assert cytodataframe_image_display_contains_pixels(
        frame=CytoDataFrame(
            data=pediatric_cancer_dataset_with_modified_image_paths,
            data_outline_context_dir=f"{pathlib.Path(cytotable_pediatric_cancer_atlas_parquet).parent}/images/outlines",
            segmentation_file_regex={
                r"CellsOutlines_BR(\d+)_C(\d{2})_\d+\.tiff": r".*ch3.*\.tiff",
                r"NucleiOutlines_BR(\d+)_C(\d{2})_\d+\.tiff": r".*ch5.*\.tiff",
            },
        ),
        image_cols=[
            "Image_FileName_OrigAGP",
            "Image_FileName_OrigDNA",
        ],
        color_conditions={"green": None, "red": 255, "blue": None},
    ), "The pediatric cancer atlas speckles images do not contain red dots."


def test_repr_html_offset_bounding_box_without_bounding_box_columns(
    cytotable_NF1_data_parquet_shrunken: str,
):
    """
    Tests that images are cropped via the ``offset_bounding_box`` display option
    when the input lacks AreaShape bounding box columns (e.g. older CellProfiler
    outputs such as LINCS), relying instead on compartment center coordinates.

    See https://github.com/cytomining/CytoDataFrame/issues/215.
    """

    image_dir = (
        f"{pathlib.Path(cytotable_NF1_data_parquet_shrunken).parent}/Plate_2_images"
    )
    image_cols = [
        "Image_FileName_DAPI",
        "Image_FileName_GFP",
        "Image_FileName_RFP",
    ]

    # Simulate old CellProfiler outputs by removing all bounding box columns
    # while keeping the Location_Center columns.
    nf1_data = pd.read_parquet(path=cytotable_NF1_data_parquet_shrunken)
    data_without_bounding_box = nf1_data.drop(
        columns=[col for col in nf1_data.columns if "BoundingBox" in col],
    )

    def displayed_image_sizes(frame: CytoDataFrame) -> list:
        """Decode every cropped image embedded in the HTML and return its size."""
        html_output = frame[image_cols]._repr_html_(debug=True)
        matches = re.findall(r"data:image/png;base64,([^\"]+)", html_output)
        return [Image.open(BytesIO(base64.b64decode(match))).size for match in matches]

    # Sanity check: no bounding box columns but compartment centers are present.
    no_offset_frame = CytoDataFrame(
        data=data_without_bounding_box,
        data_context_dir=image_dir,
    )
    assert no_offset_frame._custom_attrs["data_bounding_box"] is None
    assert no_offset_frame._custom_attrs["compartment_center_xy"] is not None

    # Without an offset_bounding_box and without bounding box columns there is
    # nothing to crop with, so no cropped images are produced.
    assert displayed_image_sizes(no_offset_frame) == []

    # With an offset_bounding_box, images are cropped using the compartment
    # center coordinates plus the provided offsets.
    offset = {"x_min": -20, "y_min": -20, "x_max": 20, "y_max": 20}
    offset_frame = CytoDataFrame(
        data=data_without_bounding_box,
        data_context_dir=image_dir,
        display_options={"offset_bounding_box": offset},
    )
    sizes = displayed_image_sizes(offset_frame)

    # One cropped image is produced per displayed row and image column.
    assert len(sizes) == len(offset_frame) * len(image_cols)
    # Each crop spans the full offset range (clamped at the image edges), so it
    # should be no larger than the requested width/height.
    expected_width = offset["x_max"] - offset["x_min"]
    expected_height = offset["y_max"] - offset["y_min"]
    for width, height in sizes:
        assert 0 < width <= expected_width
        assert 0 < height <= expected_height

    # The cropped images should still include the red compartment center dot.
    assert cytodataframe_image_display_contains_pixels(
        frame=offset_frame,
        image_cols=image_cols,
        color_conditions={"green": None, "red": 255, "blue": None},
    ), "The offset-cropped NF1 images do not contain red dots."


def test_repr_html_offset_bounding_box_without_center_columns_warns(
    cytotable_NF1_data_parquet_shrunken: str,
    caplog: pytest.LogCaptureFixture,
):
    """
    Tests that an offset_bounding_box with no bounding box and no compartment
    center columns logs a warning and produces no cropped images.

    See https://github.com/cytomining/CytoDataFrame/issues/215.
    """

    image_dir = (
        f"{pathlib.Path(cytotable_NF1_data_parquet_shrunken).parent}/Plate_2_images"
    )
    image_cols = ["Image_FileName_DAPI", "Image_FileName_GFP", "Image_FileName_RFP"]

    nf1_data = pd.read_parquet(path=cytotable_NF1_data_parquet_shrunken)
    data_without_bbox_or_center = nf1_data.drop(
        columns=[
            col
            for col in nf1_data.columns
            if "BoundingBox" in col or "Location_Center" in col
        ],
    )

    frame = CytoDataFrame(
        data=data_without_bbox_or_center,
        data_context_dir=image_dir,
        display_options={
            "offset_bounding_box": {
                "x_min": -20,
                "y_min": -20,
                "x_max": 20,
                "y_max": 20,
            }
        },
    )
    assert frame._custom_attrs["data_bounding_box"] is None
    assert frame._custom_attrs["compartment_center_xy"] is None

    with caplog.at_level(logging.WARNING):
        html_output = frame[image_cols]._repr_html_(debug=True)

    assert "no compartment center xy columns were found" in caplog.text
    assert re.findall(r"data:image/png;base64,([^\"]+)", html_output) == []


def test_repr_html_render_whole_image_without_bounding_box_or_center(
    cytotable_NF1_data_parquet_shrunken: str,
):
    """
    Tests that the ``render_whole_image`` display option renders full fields of
    view for image-level inputs that have neither bounding box nor compartment
    center columns (e.g. whole-FOV quality control metrics).

    See https://github.com/cytomining/CytoDataFrame/issues/202.
    """

    image_dir = (
        f"{pathlib.Path(cytotable_NF1_data_parquet_shrunken).parent}/Plate_2_images"
    )
    image_cols = ["Image_FileName_DAPI", "Image_FileName_GFP", "Image_FileName_RFP"]

    # Simulate whole-FOV inputs by removing both bounding box and center columns.
    nf1_data = pd.read_parquet(path=cytotable_NF1_data_parquet_shrunken)
    fov_data = nf1_data.drop(
        columns=[
            col
            for col in nf1_data.columns
            if "BoundingBox" in col or "Location_Center" in col
        ],
    )

    def displayed_image_sizes(frame: CytoDataFrame) -> list:
        """Decode every image embedded in the HTML and return its size."""
        html_output = frame[image_cols]._repr_html_(debug=True)
        matches = re.findall(r"data:image/png;base64,([^\"]+)", html_output)
        return [Image.open(BytesIO(base64.b64decode(match))).size for match in matches]

    # Without the flag, there is nothing to crop with so nothing renders.
    plain_frame = CytoDataFrame(data=fov_data, data_context_dir=image_dir)
    assert plain_frame._custom_attrs["data_bounding_box"] is None
    assert plain_frame._custom_attrs["compartment_center_xy"] is None
    assert displayed_image_sizes(plain_frame) == []

    # With render_whole_image, full fields of view render uncropped.
    whole_frame = CytoDataFrame(
        data=fov_data,
        data_context_dir=image_dir,
        display_options={"render_whole_image": True},
    )
    sizes = displayed_image_sizes(whole_frame)
    assert len(sizes) == len(whole_frame) * len(image_cols)

    # Each rendered image matches the dimensions of its source field of view.
    source_image = imageio.imread(
        next(pathlib.Path(image_dir).rglob(fov_data["Image_FileName_DAPI"].iloc[0]))
    )
    expected_height, expected_width = source_image.shape[:2]
    assert all(size == (expected_width, expected_height) for size in sizes)


def test_repr_html_offset_bounding_box_warns_when_centers_missing_with_bbox(
    cytotable_NF1_data_parquet_shrunken: str,
    caplog: pytest.LogCaptureFixture,
):
    """
    Tests that an offset_bounding_box with bounding box columns but no
    compartment center columns warns that the offset is ignored, while still
    cropping via the bounding box columns.

    See https://github.com/cytomining/CytoDataFrame/issues/215.
    """

    image_dir = (
        f"{pathlib.Path(cytotable_NF1_data_parquet_shrunken).parent}/Plate_2_images"
    )
    image_cols = ["Image_FileName_DAPI", "Image_FileName_GFP", "Image_FileName_RFP"]

    nf1_data = pd.read_parquet(path=cytotable_NF1_data_parquet_shrunken)
    data_without_center = nf1_data.drop(
        columns=[col for col in nf1_data.columns if "Location_Center" in col],
    )

    frame = CytoDataFrame(
        data=data_without_center,
        data_context_dir=image_dir,
        display_options={
            "offset_bounding_box": {
                "x_min": -20,
                "y_min": -20,
                "x_max": 20,
                "y_max": 20,
            }
        },
    )
    assert frame._custom_attrs["data_bounding_box"] is not None
    assert frame._custom_attrs["compartment_center_xy"] is None

    with caplog.at_level(logging.WARNING):
        html_output = frame[image_cols]._repr_html_(debug=True)

    assert "no compartment center xy columns were found" in caplog.text
    assert "offset_bounding_box will be ignored" in caplog.text
    # the bounding box columns are still used to crop the images.
    assert re.findall(r"data:image/png;base64,([^\"]+)", html_output) != []


def test_repr_html_offset_bounding_box_with_missing_key_raises_value_error(
    cytotable_NF1_data_parquet_shrunken: str,
):
    """
    Tests that a misspelled/missing offset_bounding_box key raises a clear
    ValueError instead of a raw KeyError.

    See https://github.com/cytomining/CytoDataFrame/issues/215.
    """

    image_dir = (
        f"{pathlib.Path(cytotable_NF1_data_parquet_shrunken).parent}/Plate_2_images"
    )
    image_cols = ["Image_FileName_DAPI", "Image_FileName_GFP", "Image_FileName_RFP"]

    frame = CytoDataFrame(
        data=cytotable_NF1_data_parquet_shrunken,
        data_context_dir=image_dir,
        display_options={
            # "ymin" is a typo for "y_min".
            "offset_bounding_box": {
                "x_min": -20,
                "ymin": -20,
                "x_max": 20,
                "y_max": 20,
            }
        },
    )

    with pytest.raises(
        ValueError, match=r"offset_bounding_box.*missing a required key"
    ):
        frame[image_cols]._repr_html_(debug=True)


def test_return_cytodataframe(cytotable_NF1_data_parquet_shrunken: str):
    """
    Tests to ensure we return a CytoDataFrame
    from extended Pandas methods.
    """

    cdf = CytoDataFrame(data=cytotable_NF1_data_parquet_shrunken)

    assert isinstance(cdf.head(), CytoDataFrame)
    assert isinstance(cdf.tail(), CytoDataFrame)
    assert isinstance(cdf.sort_values(by="Metadata_ImageNumber"), CytoDataFrame)
    assert isinstance(cdf.sample(n=5), CytoDataFrame)
    assert isinstance(cdf[0:2], CytoDataFrame)
    assert isinstance(cdf[1:1], CytoDataFrame)
    assert isinstance(cdf[0:5:2], CytoDataFrame)
    assert isinstance(cdf.iloc[0:2], CytoDataFrame)
    assert isinstance(cdf.iloc[1:1], CytoDataFrame)
    assert isinstance(cdf.iloc[0:5:2], CytoDataFrame)


def test_return_cytodataframe_passthroughs_non_dataframe_results() -> None:
    """Ensure helper methods return scalar-like results without wrapping."""

    cdf = CytoDataFrame(pd.DataFrame({"a": [1, 2, 3]}))

    result = cdf._return_cytodataframe(lambda: 3, "dummy_method")

    assert result == 3


def test_iloc_slice_preserves_cytodataframe_html_formatting():
    """Ensure ``iloc`` slices keep the CytoDataFrame notebook HTML renderer."""

    cdf = CytoDataFrame(pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}))

    bracket_sliced = cdf[0:3:2]
    bracket_empty_sliced = cdf[1:1]
    sliced = cdf.iloc[0:3:2]
    empty_sliced = cdf.iloc[1:1]

    assert isinstance(bracket_sliced, CytoDataFrame)
    assert isinstance(bracket_empty_sliced, CytoDataFrame)
    assert bracket_sliced._custom_attrs["_output"] is cdf._custom_attrs["_output"]
    assert (
        bracket_sliced._custom_attrs["_widget_state"]
        is cdf._custom_attrs["_widget_state"]
    )
    assert bracket_empty_sliced._custom_attrs["_output"] is cdf._custom_attrs["_output"]
    assert (
        bracket_empty_sliced._custom_attrs["_widget_state"]
        is cdf._custom_attrs["_widget_state"]
    )
    assert isinstance(sliced, CytoDataFrame)
    assert isinstance(empty_sliced, CytoDataFrame)
    assert sliced._custom_attrs["_output"] is cdf._custom_attrs["_output"]
    assert sliced._custom_attrs["_widget_state"] is cdf._custom_attrs["_widget_state"]
    assert empty_sliced._custom_attrs["_output"] is cdf._custom_attrs["_output"]
    assert (
        empty_sliced._custom_attrs["_widget_state"]
        is cdf._custom_attrs["_widget_state"]
    )
    assert "background:#EBEBEB" in bracket_sliced._repr_html_(debug=True)
    assert "background:#EBEBEB" in bracket_empty_sliced._repr_html_(debug=True)
    assert "background:#EBEBEB" in sliced._repr_html_(debug=True)
    assert "background:#EBEBEB" in empty_sliced._repr_html_(debug=True)


def test_transpose_toggles_transposed_state() -> None:
    """Ensure repeated transposes flip the transposed rendering state back."""

    cdf = CytoDataFrame(pd.DataFrame({"a": [1, 2], "b": [3, 4]}))

    transposed = cdf.T
    double_transposed = transposed.T

    assert transposed._custom_attrs["is_transposed"] is True
    assert double_transposed._custom_attrs["is_transposed"] is False


def test_cytodataframe_dynamic_width_and_height(
    cytotable_NF1_data_parquet_shrunken: str,
):
    """
    Tests to ensure we return a CytoDataFrame
    from extended Pandas methods.
    """

    cdf = CytoDataFrame(
        data=cytotable_NF1_data_parquet_shrunken,
        data_context_dir=f"{pathlib.Path(cytotable_NF1_data_parquet_shrunken).parent}/Plate_2_images",
        data_mask_context_dir=f"{pathlib.Path(cytotable_NF1_data_parquet_shrunken).parent}/Plate_2_masks",
        # set the width to 100px and height to auto for images
        display_options={"width": "100px", "height": "auto"},
    )

    # gather the html of the output for the dataframe
    cdf_image_html = cdf[
        ["Image_FileName_DAPI", "Image_FileName_GFP", "Image_FileName_RFP"]
    ][1:2]._repr_html_(debug=True)

    # test that the html string contains the customized width and height
    # constraints on the 3 images which display within the html output.
    assert cdf_image_html.count("width:100px") == 3
    assert cdf_image_html.count("height:auto") == 3

    # transpose and test for the same to ensure the images are
    # formatted despite being transposed (that we didn't lose them
    # in the process).
    cdf_image_html = cdf[
        ["Image_FileName_DAPI", "Image_FileName_GFP", "Image_FileName_RFP"]
    ][1:2].T._repr_html_(debug=True)

    assert cdf_image_html.count("width:100px") == 3
    assert cdf_image_html.count("height:auto") == 3


def test_slider_updates_state(monkeypatch: MonkeyPatch):
    """
    Test that the slider for image adjustments updates the internal
    widget state and triggers the render method.
    """

    # Minimal test dataframe
    df = pd.DataFrame({"Image_FileName_DNA": ["example.tif"]})
    cdf = CytoDataFrame(df)

    # Simulate the change dictionary sent by ipywidgets
    change = {"new": 75}

    # Track render calls using monkeypatch or a flag
    render_called = {}
    loading_called = {}

    def mock_render_output() -> None:
        render_called["called"] = True

    def mock_show_loading() -> None:
        loading_called["called"] = True

    monkeypatch.setattr(cdf, "_render_output", mock_render_output)
    monkeypatch.setattr(cdf, "_show_output_loading_indicator", mock_show_loading)

    # Call the method manually
    cdf._on_slider_change(change)

    # Check if internal widget state updated
    assert cdf._custom_attrs["_widget_state"]["scale"] == 75

    # Check if the render method was triggered
    assert render_called.get("called", False)
    assert loading_called.get("called", False)


def test_filter_slider_updates_state(monkeypatch: MonkeyPatch):
    """Test that the filter slider updates internal state and triggers render."""
    cdf = CytoDataFrame(
        pd.DataFrame({"Image_FileName_DNA": ["example.tif"], "AreaShape_Area": [2.0]}),
        display_options={"filter_column": "AreaShape_Area"},
    )
    cdf._custom_attrs["_widget_state"]["filter_column"] = "AreaShape_Area"
    render_called = {}
    loading_called = {}

    def mock_render_output() -> None:
        render_called["called"] = True

    def mock_show_loading() -> None:
        loading_called["called"] = True

    monkeypatch.setattr(cdf, "_render_output", mock_render_output)
    monkeypatch.setattr(cdf, "_show_output_loading_indicator", mock_show_loading)
    cdf._on_filter_slider_change({"new": (1.5, 2.5)})

    assert cdf._custom_attrs["_widget_state"]["filter_range"] == (1.5, 2.5)
    assert render_called.get("called", False)
    assert loading_called.get("called", False)


def test_filter_display_indices_by_widget_range() -> None:
    cdf = CytoDataFrame(pd.DataFrame({"FilterScore": [1.0, 2.0, 3.0]}))
    cdf._custom_attrs["_widget_state"]["filter_column"] = "FilterScore"
    cdf._custom_attrs["_widget_state"]["filter_range"] = (1.5, 2.5)

    filtered = cdf._filter_display_indices_by_widget_range(
        data=cdf, display_indices=[0, 1, 2]
    )

    assert filtered == [1]


def test_filter_display_indices_by_widget_range_multiple_columns() -> None:
    cdf = CytoDataFrame(
        pd.DataFrame(
            {
                "FilterScoreA": [1.0, 2.0, 3.0, 4.0],
                "FilterScoreB": [10.0, 20.0, 30.0, 40.0],
            }
        )
    )
    cdf._custom_attrs["_widget_state"]["filter_columns"] = [
        "FilterScoreA",
        "FilterScoreB",
    ]
    cdf._custom_attrs["_widget_state"]["filter_ranges"] = {
        "FilterScoreA": (1.5, 3.5),
        "FilterScoreB": (15.0, 35.0),
    }

    filtered = cdf._filter_display_indices_by_widget_range(
        data=cdf, display_indices=[0, 1, 2, 3]
    )

    assert filtered == [1, 2]


def test_filter_display_indices_by_widget_range_preserves_duplicate_labels() -> None:
    cdf = CytoDataFrame(pd.DataFrame({"FilterScore": [1.0, 2.0, 3.0]}, index=[0, 0, 1]))
    cdf._custom_attrs["_widget_state"]["filter_column"] = "FilterScore"
    cdf._custom_attrs["_widget_state"]["filter_range"] = (0.5, 2.5)

    filtered = cdf._filter_display_indices_by_widget_range(
        data=cdf, display_indices=[0, 0, 0, 1]
    )

    assert filtered == [0, 0]


def test_filter_slider_rounds_labels_but_preserves_values() -> None:
    cdf = CytoDataFrame(
        pd.DataFrame({"FilterScore": [0.0123, 0.456, 9.87]}),
        display_options={"filter_column": "FilterScore"},
    )

    slider = cdf._ensure_filter_range_slider()

    assert isinstance(slider, widgets.SelectionRangeSlider)
    assert "cdf-filter-range-slider" in getattr(slider, "_dom_classes", ())
    options = list(slider.options)
    assert options == [("0.01", 0.0123), ("0.46", 0.456), ("9.87", 9.87)]


def test_get_filter_slider_columns_falls_back_to_single_when_many_is_empty() -> None:
    cdf = CytoDataFrame(
        pd.DataFrame({"FilterScore": [1.0, 2.0]}),
        display_options={"filter_columns": [], "filter_column": "FilterScore"},
    )

    columns = cdf._get_filter_slider_columns()

    assert columns == ["FilterScore"]


def test_filter_slider_caps_option_count_for_near_unique_values() -> None:
    values = np.arange(MAX_FILTER_SLIDER_STOPS + 200, dtype=np.float64)
    cdf = CytoDataFrame(
        pd.DataFrame({"FilterScore": values}),
        display_options={"filter_column": "FilterScore"},
    )

    slider = cdf._ensure_filter_range_slider()

    assert isinstance(slider, widgets.SelectionRangeSlider)
    options = list(slider.options)
    assert len(options) == MAX_FILTER_SLIDER_STOPS
    assert options[0][1] == float(values.min())
    assert options[-1][1] == float(values.max())
    option_vals = np.array([float(option[1]) for option in options], dtype=np.float64)
    deltas = np.diff(option_vals)
    assert np.all(deltas > 0)
    assert np.allclose(deltas, deltas[0], rtol=1e-6, atol=1e-12)
    assert slider.value == (float(values.min()), float(values.max()))


def test_filter_slider_reuses_cached_widget_instance() -> None:
    cdf = CytoDataFrame(
        pd.DataFrame({"FilterScore": [1.0, 2.0, 3.0]}),
        display_options={"filter_column": "FilterScore"},
    )

    first_slider = cdf._ensure_filter_range_slider(filter_col="FilterScore")
    assert isinstance(first_slider, widgets.SelectionRangeSlider)
    first_options = list(first_slider.options)
    assert first_options[-1][1] == 3.0

    cdf.loc[:, "FilterScore"] = [1.0, 2.0, 4.0]
    second_slider = cdf._ensure_filter_range_slider(filter_col="FilterScore")

    assert isinstance(second_slider, widgets.SelectionRangeSlider)
    assert second_slider is first_slider
    second_options = list(second_slider.options)
    assert second_options[-1][1] == 4.0


def test_filter_distribution_constant_values_stays_centered() -> None:
    html = CytoDataFrame._build_filter_distribution_html(
        values=pd.Series([0.47, 0.47, 0.47, 0.47]),
        selected_range=(0.47, 0.47),
        size_px=(FILTER_SLIDER_TOTAL_WIDTH_PX, 52),
        track_padding_px=(
            FILTER_SLIDER_LABEL_WIDTH_PX,
            FILTER_SLIDER_READOUT_WIDTH_PX,
        ),
    )

    match = re.search(r"<polyline[^>]*points='([^']+)'", html)
    assert match is not None
    points = [
        (float(part.split(",")[0]), float(part.split(",")[1]))
        for part in match.group(1).split()
    ]
    peak_x = min(points, key=lambda point: point[1])[0]
    track_left = float(FILTER_SLIDER_LABEL_WIDTH_PX)
    track_right = float(FILTER_SLIDER_TOTAL_WIDTH_PX - FILTER_SLIDER_READOUT_WIDTH_PX)
    track_mid = (track_left + track_right) / 2.0
    assert abs(peak_x - track_mid) < 30.0


def test_filter_slider_control_renders_threshold_line() -> None:
    cdf = CytoDataFrame(
        pd.DataFrame({"FilterScore": [1.0, 2.0, 3.0]}),
        display_options={
            "filter_column": "FilterScore",
            "filter_plot_threshold": 2.0,
        },
    )
    cdf._custom_attrs["_widget_state"]["filter_column"] = "FilterScore"
    cdf._custom_attrs["_widget_state"]["filter_ranges"] = {"FilterScore": (1.0, 3.0)}

    _slider, control = cdf._build_filter_slider_control_for_column("FilterScore")

    assert isinstance(control, widgets.VBox)
    assert isinstance(control.children[0], widgets.HTML)
    assert "stroke='#dc2626'" in control.children[0].value
    assert "y1='6.00'" in control.children[0].value
    assert "y2='22.00'" in control.children[0].value


def test_filter_slider_control_warns_and_clamps_out_of_range_threshold(
    caplog: pytest.LogCaptureFixture,
) -> None:
    cdf = CytoDataFrame(
        pd.DataFrame({"FilterScore": [1.0, 2.0, 3.0]}),
        display_options={
            "filter_column": "FilterScore",
            "filter_plot_threshold": 9.0,
        },
    )
    cdf._custom_attrs["_widget_state"]["filter_column"] = "FilterScore"
    cdf._custom_attrs["_widget_state"]["filter_ranges"] = {"FilterScore": (1.0, 3.0)}

    with caplog.at_level(logging.WARNING):
        _slider, control = cdf._build_filter_slider_control_for_column("FilterScore")

    assert isinstance(control, widgets.VBox)
    assert isinstance(control.children[0], widgets.HTML)
    assert "stroke='#dc2626'" in control.children[0].value
    assert "outside data range" in caplog.text


def test_filter_slider_control_ignores_non_numeric_threshold(
    caplog: pytest.LogCaptureFixture,
) -> None:
    cdf = CytoDataFrame(
        pd.DataFrame({"FilterScore": [1.0, 2.0, 3.0]}),
        display_options={
            "filter_column": "FilterScore",
            "filter_plot_threshold": "not-a-number",
        },
    )
    cdf._custom_attrs["_widget_state"]["filter_column"] = "FilterScore"
    cdf._custom_attrs["_widget_state"]["filter_ranges"] = {"FilterScore": (1.0, 3.0)}

    with caplog.at_level(logging.WARNING):
        _slider, control = cdf._build_filter_slider_control_for_column("FilterScore")

    assert isinstance(control, widgets.VBox)
    assert isinstance(control.children[0], widgets.HTML)
    assert "stroke='#dc2626'" not in control.children[0].value
    assert "is not numeric" in caplog.text


def test_filter_slider_threshold_key_match_is_case_and_whitespace_tolerant() -> None:
    cdf = CytoDataFrame(
        pd.DataFrame({"FilterScore": [1.0, 2.0, 3.0]}),
        display_options={
            "filter_column": "FilterScore",
            "filter_plot_thresholds": {"  filterscore  ": 2.0},
        },
    )
    cdf._custom_attrs["_widget_state"]["filter_column"] = "FilterScore"
    cdf._custom_attrs["_widget_state"]["filter_ranges"] = {"FilterScore": (1.0, 3.0)}

    _slider, control = cdf._build_filter_slider_control_for_column("FilterScore")

    assert isinstance(control, widgets.VBox)
    assert isinstance(control.children[0], widgets.HTML)
    assert "stroke='#dc2626'" in control.children[0].value


def test_filter_slider_threshold_aligns_with_selection_slider_domain() -> None:
    cdf = CytoDataFrame(
        pd.DataFrame({"FilterScore": [0.0, 1.0, 100.0]}),
        display_options={
            "filter_column": "FilterScore",
            "filter_plot_threshold": 1.0,
        },
    )
    cdf._custom_attrs["_widget_state"]["filter_column"] = "FilterScore"
    cdf._custom_attrs["_widget_state"]["filter_ranges"] = {"FilterScore": (0.0, 100.0)}

    _slider, control = cdf._build_filter_slider_control_for_column("FilterScore")
    assert isinstance(control, widgets.VBox)
    assert isinstance(control.children[0], widgets.HTML)
    html = control.children[0].value
    x_match = re.search(r"x1='([0-9.]+)' y1='[0-9.]+'", html)
    assert x_match is not None
    x_val = float(x_match.group(1))

    track_left = float(FILTER_SLIDER_LABEL_WIDTH_PX)
    track_right = float(FILTER_SLIDER_TOTAL_WIDTH_PX - FILTER_SLIDER_READOUT_WIDTH_PX)
    track_mid = (track_left + track_right) / 2.0
    assert abs(x_val - track_mid) < 8.0


def test_filter_distribution_is_not_flat_for_clustered_values() -> None:
    html = CytoDataFrame._build_filter_distribution_html(
        values=pd.Series([0.0] * 60 + [0.1] * 30 + [2.0] * 10),
        selected_range=(0.0, 100.0),
        size_px=(FILTER_SLIDER_TOTAL_WIDTH_PX, 52),
        track_padding_px=(
            FILTER_SLIDER_LABEL_WIDTH_PX,
            FILTER_SLIDER_READOUT_WIDTH_PX,
        ),
    )
    match = re.search(r"<polyline[^>]*points='([^']+)'", html)
    assert match is not None
    ys = [float(part.split(",")[1]) for part in match.group(1).split()]
    assert max(ys) - min(ys) > 2.0


def test_filter_distribution_avoids_runtime_warnings_for_large_ranges() -> None:
    values = pd.Series(
        np.concatenate(
            [
                np.full(2000, 0.0),
                np.full(1500, 1.0),
                np.linspace(2.0, 5000.0, 2000),
            ]
        )
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        html = CytoDataFrame._build_filter_distribution_html(
            values=values,
            selected_range=(0.0, 5000.0),
            size_px=(FILTER_SLIDER_TOTAL_WIDTH_PX, 52),
            track_padding_px=(
                FILTER_SLIDER_LABEL_WIDTH_PX,
                FILTER_SLIDER_READOUT_WIDTH_PX,
            ),
        )

    runtime_warnings = [
        warning for warning in caught if issubclass(warning.category, RuntimeWarning)
    ]
    assert html
    assert not runtime_warnings


def test_generate_html_removes_rows_outside_filter_range(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cdf = CytoDataFrame(
        pd.DataFrame(
            {
                "Label": ["keep-row", "drop-row"],
                "FilterScore": [2.0, 9.0],
            }
        ),
        display_options={"filter_column": "FilterScore"},
    )
    cdf._custom_attrs["_widget_state"]["filter_column"] = "FilterScore"
    cdf._custom_attrs["_widget_state"]["filter_range"] = (1.5, 2.5)

    options = {
        "display.notebook_repr_html": True,
        "display.max_rows": 10,
        "display.min_rows": 10,
        "display.max_columns": 10,
        "display.show_dimensions": False,
    }
    monkeypatch.setattr("cytodataframe.frame.get_option", lambda name: options[name])

    html = cdf._generate_jupyter_dataframe_html()

    assert "keep-row" in html
    assert "drop-row" not in html


def test_generate_html_filters_full_frame_before_display_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels = [f"row-{idx}" for idx in range(20)]
    scores = [0.0] * 20
    labels[10] = "middle-keep"
    scores[10] = 5.0

    cdf = CytoDataFrame(
        pd.DataFrame({"Label": labels, "FilterScore": scores}),
        display_options={"filter_column": "FilterScore"},
    )
    cdf._custom_attrs["_widget_state"]["filter_column"] = "FilterScore"
    cdf._custom_attrs["_widget_state"]["filter_range"] = (4.9, 5.1)
    cdf._custom_attrs["_widget_state"]["filter_columns"] = ["FilterScore"]
    cdf._custom_attrs["_widget_state"]["filter_ranges"] = {"FilterScore": (4.9, 5.1)}

    options = {
        "display.notebook_repr_html": True,
        "display.max_rows": 8,
        "display.min_rows": 4,
        "display.max_columns": 10,
        "display.show_dimensions": False,
    }
    monkeypatch.setattr("cytodataframe.frame.get_option", lambda name: options[name])
    monkeypatch.setattr("pandas.get_option", lambda name: options[name])

    html = cdf._generate_jupyter_dataframe_html()

    assert "middle-keep" in html
    assert "row-0" not in html
    assert "row-19" not in html


def test_get_3d_volume_from_cell_loads_3d_tiff(tmp_path: pathlib.Path) -> None:
    volume = np.arange(4 * 5 * 6, dtype=np.uint8).reshape(4, 5, 6)
    image_path = tmp_path / "volume.tiff"
    tifffile.imwrite(image_path, volume)

    cdf = CytoDataFrame(
        data=pd.DataFrame({"Image_FileName_DNA": [image_path.name]}),
        data_context_dir=str(tmp_path),
    )

    loaded_volume, dims = cdf._get_3d_volume_from_cell(
        row=0, column="Image_FileName_DNA"
    )

    assert loaded_volume.shape == (4, 5, 6)
    assert dims == (6, 5, 4)


def test_get_3d_volume_from_cell_normalizes_file_uri_with_context_dir(
    tmp_path: pathlib.Path,
) -> None:
    volume = np.arange(3 * 4 * 5, dtype=np.uint8).reshape(3, 4, 5)
    image_path = tmp_path / "volume_uri.tiff"
    tifffile.imwrite(image_path, volume)

    cdf = CytoDataFrame(
        data=pd.DataFrame({"Image_FileName_DNA": [f"file:{image_path}"]}),
        data_context_dir=str(tmp_path),
    )

    loaded_volume, dims = cdf._get_3d_volume_from_cell(
        row=0, column="Image_FileName_DNA"
    )

    assert loaded_volume.shape == (3, 4, 5)
    assert dims == (5, 4, 3)


def test_find_image_columns_accepts_pathlike_values(tmp_path: pathlib.Path) -> None:
    cdf = CytoDataFrame(
        pd.DataFrame(
            {
                "PathLikeCol": [tmp_path / "img.tiff"],
                "NotImage": [tmp_path / "table.csv"],
            }
        )
    )
    assert "PathLikeCol" in cdf.find_image_columns()
    assert "NotImage" not in cdf.find_image_columns()


def test_get_3d_volume_from_cell_uses_image_pathname_column(
    tmp_path: pathlib.Path,
) -> None:
    volume = np.arange(3 * 4 * 5, dtype=np.uint8).reshape(3, 4, 5)
    image_path = tmp_path / "via_path_col.tiff"
    tifffile.imwrite(image_path, volume)

    cdf = CytoDataFrame(
        data=pd.DataFrame(
            {
                "Image_FileName_DNA": [image_path.name],
                "Image_PathName_DNA": [str(tmp_path)],
            }
        ),
    )

    loaded_volume, dims = cdf._get_3d_volume_from_cell(
        row=0, column="Image_FileName_DNA"
    )
    assert loaded_volume.shape == (3, 4, 5)
    assert dims == (5, 4, 3)


def test_get_3d_volume_from_cell_uses_data_image_paths_helper(
    tmp_path: pathlib.Path,
) -> None:
    volume = np.arange(2 * 4 * 6, dtype=np.uint8).reshape(2, 4, 6)
    image_path = tmp_path / "helper_path_col.tiff"
    tifffile.imwrite(image_path, volume)

    cdf = CytoDataFrame(
        data=pd.DataFrame({"Image_FileName_DNA": [image_path.name]}),
        data_image_paths=pd.DataFrame({"Image_PathName_DNA": [str(tmp_path)]}),
    )

    loaded_volume, dims = cdf._get_3d_volume_from_cell(
        row=0, column="Image_FileName_DNA"
    )
    assert loaded_volume.shape == (2, 4, 6)
    assert dims == (6, 4, 2)


def test_get_3d_volume_from_cell_rglob_in_context_dir(tmp_path: pathlib.Path) -> None:
    nested_dir = tmp_path / "nested" / "images"
    nested_dir.mkdir(parents=True)
    volume = np.arange(2 * 3 * 4, dtype=np.uint8).reshape(2, 3, 4)
    image_path = nested_dir / "rglob_volume.tiff"
    tifffile.imwrite(image_path, volume)

    cdf = CytoDataFrame(
        data=pd.DataFrame({"Image_FileName_DNA": [image_path.name]}),
        data_context_dir=str(tmp_path),
    )

    loaded_volume, dims = cdf._get_3d_volume_from_cell(
        row=0, column="Image_FileName_DNA"
    )
    assert loaded_volume.shape == (2, 3, 4)
    assert dims == (4, 3, 2)


def test_get_3d_volume_from_cell_uses_bounded_lru_cache() -> None:
    cdf = CytoDataFrame(
        pd.DataFrame(
            {
                "A": [
                    np.zeros((2, 2, 2), dtype=np.uint8),
                    np.ones((2, 2, 2), dtype=np.uint8),
                    np.full((2, 2, 2), 2, dtype=np.uint8),
                ]
            }
        ),
        display_options={"volume_cache_max_entries": 2},
    )

    cdf._get_3d_volume_from_cell(row=0, column="A")
    cdf._get_3d_volume_from_cell(row=1, column="A")
    cache = cdf._custom_attrs["_volume_cache"]
    assert isinstance(cache, OrderedDict)
    assert list(cache.keys()) == ["0::A", "1::A"]

    # Access row 0 again, making it the most-recent entry.
    cdf._get_3d_volume_from_cell(row=0, column="A")
    assert list(cache.keys()) == ["1::A", "0::A"]

    # Inserting a third entry evicts the least-recently used one (row 1).
    cdf._get_3d_volume_from_cell(row=2, column="A")
    assert list(cache.keys()) == ["0::A", "2::A"]
    assert len(cache) == 2


def test_get_3d_volume_from_cell_skips_cache_when_disabled() -> None:
    cdf = CytoDataFrame(
        pd.DataFrame({"A": [np.ones((2, 2, 2), dtype=np.uint8)]}),
        display_options={"volume_disable_cache": True},
    )
    sentinel_cache = {"0::A": (np.zeros((1, 1, 1), dtype=np.uint8), (1, 1, 1))}
    cdf._custom_attrs["_volume_cache"] = sentinel_cache

    volume, dims = cdf._get_3d_volume_from_cell(row=0, column="A")

    assert volume.shape == (2, 2, 2)
    assert dims == (2, 2, 2)
    assert cdf._custom_attrs["_volume_cache"] is sentinel_cache
    assert cdf._custom_attrs["_volume_cache"]["0::A"][0].shape == (1, 1, 1)


def test_repr_html_auto_trame_for_3d_inputs(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    volume = np.arange(3 * 4 * 5, dtype=np.uint8).reshape(3, 4, 5)
    image_path = tmp_path / "auto_trame_volume.tiff"
    tifffile.imwrite(image_path, volume)

    cdf = CytoDataFrame(
        data=pd.DataFrame({"Image_FileName_DNA": [image_path.name]}),
        data_context_dir=str(tmp_path),
    )

    captured: dict = {}

    def fake_show_widget_table(column: str, **kwargs: object) -> str:
        captured["column"] = column
        captured["columns_3d"] = kwargs.get("columns_3d")
        captured["backend"] = kwargs.get("backend")
        return "widget_table"

    displayed: list = []

    def fake_snapshot_html() -> str:
        return "<table/>"

    def capture_display(value: object) -> None:
        displayed.append(value)

    monkeypatch.setattr(cdf, "show_widget_table", fake_show_widget_table)
    monkeypatch.setattr(cdf, "_generate_trame_snapshot_html", fake_snapshot_html)
    monkeypatch.setattr("cytodataframe.frame.display", capture_display)

    assert cdf._repr_html_() is None
    assert captured["column"] == "Image_FileName_DNA"
    assert captured["columns_3d"] == ["Image_FileName_DNA"]
    assert captured["backend"] is None
    assert displayed


def test_find_3d_columns_for_display_skips_ellipsis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cdf = CytoDataFrame(pd.DataFrame({"Image_FileName_DNA": ["volume.tiff"]}))
    attempted_rows: list = []

    monkeypatch.setattr(cdf, "find_image_columns", lambda: ["Image_FileName_DNA"])
    monkeypatch.setattr(CytoDataFrame, "find_ome_arrow_columns", lambda _self, _df: [])
    monkeypatch.setattr(cdf, "get_displayed_rows", lambda: [0, "\u2026", 1])

    def fake_get_3d_volume(row: int, column: str):  # noqa: ANN202
        attempted_rows.append(row)
        if row == 1:
            return np.zeros((2, 2, 2), dtype=np.uint8), (2, 2, 2)
        raise ValueError("not 3d")

    monkeypatch.setattr(cdf, "_get_3d_volume_from_cell", fake_get_3d_volume)

    assert cdf._find_3d_columns_for_display() == ["Image_FileName_DNA"]
    assert attempted_rows == [0, 1]


def test_find_3d_columns_for_display_falls_back_to_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cdf = CytoDataFrame(
        pd.DataFrame({"Image_FileName_DNA": ["volume.tiff"]}, index=[7])
    )
    attempted_rows: list = []

    monkeypatch.setattr(cdf, "find_image_columns", lambda: ["Image_FileName_DNA"])
    monkeypatch.setattr(CytoDataFrame, "find_ome_arrow_columns", lambda _self, _df: [])
    monkeypatch.setattr(cdf, "get_displayed_rows", lambda: ["\u2026"])

    def fake_get_3d_volume(row: int, column: str):  # noqa: ANN202
        attempted_rows.append(row)
        return np.zeros((2, 2, 2), dtype=np.uint8), (2, 2, 2)

    monkeypatch.setattr(cdf, "_get_3d_volume_from_cell", fake_get_3d_volume)

    assert cdf._find_3d_columns_for_display() == ["Image_FileName_DNA"]
    assert attempted_rows == [7]


def test_repr_html_force_trame_falls_back_to_candidate_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cdf = CytoDataFrame(
        pd.DataFrame({"Image_FileName_DNA": ["volume.tiff"], "OMEArrowCol": [None]}),
        display_options={"view": "trame", "auto_trame_for_3d": False},
    )
    captured: dict = {}
    displayed: list = []

    def fake_image_columns() -> list:
        return ["Image_FileName_DNA"]

    def fake_ome_columns(_self: CytoDataFrame, _df: pd.DataFrame) -> list:
        return ["OMEArrowCol"]

    def fake_snapshot_html() -> str:
        return "<table/>"

    def capture_display(value: object) -> None:
        displayed.append(value)

    monkeypatch.setattr(cdf, "find_image_columns", fake_image_columns)
    monkeypatch.setattr(CytoDataFrame, "find_ome_arrow_columns", fake_ome_columns)
    monkeypatch.setattr(cdf, "_generate_trame_snapshot_html", fake_snapshot_html)
    monkeypatch.setattr("cytodataframe.frame.display", capture_display)

    def fake_show_widget_table(column: str, **kwargs: object) -> str:
        captured["column"] = column
        captured["columns_3d"] = kwargs.get("columns_3d")
        return "widget_table"

    monkeypatch.setattr(cdf, "show_widget_table", fake_show_widget_table)

    assert cdf._repr_html_() is None
    assert captured["column"] == "Image_FileName_DNA"
    assert captured["columns_3d"] == ["Image_FileName_DNA", "OMEArrowCol"]
    assert displayed


def test_repr_html_trame_with_filter_columns_uses_notebook_widget_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cdf = CytoDataFrame(
        pd.DataFrame(
            {
                "Image_FileName_DNA": ["volume.tiff"],
                "FilterScoreA": [1.0],
                "FilterScoreB": [2.0],
            }
        ),
        display_options={
            "view": "trame",
            "filter_columns": ["FilterScoreA", "FilterScoreB"],
        },
    )
    calls = {"show_widget_table": 0, "render_notebook": 0}

    monkeypatch.setattr(
        cdf, "_find_3d_columns_for_display", lambda: ["Image_FileName_DNA"]
    )

    def fake_show_widget_table(**_kwargs: object) -> str:
        calls["show_widget_table"] += 1
        return "widget_table"

    def fake_render_notebook_widget_output(**_kwargs: object) -> None:
        calls["render_notebook"] += 1

    monkeypatch.setattr(cdf, "show_widget_table", fake_show_widget_table)
    monkeypatch.setattr(
        cdf, "_render_notebook_widget_output", fake_render_notebook_widget_output
    )
    monkeypatch.setattr("cytodataframe.frame.get_option", lambda _name: True)

    assert cdf._repr_html_() is None
    assert calls["show_widget_table"] == 0
    assert calls["render_notebook"] == 1


def test_repr_html_2d_displays_static_snapshot_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cdf = CytoDataFrame(pd.DataFrame({"A": [1]}))
    displayed: list[object] = []

    monkeypatch.setattr(cdf, "_find_3d_columns_for_display", lambda: [])
    monkeypatch.setattr(cdf, "_render_output", lambda: None)
    monkeypatch.setattr(cdf, "_generate_jupyter_dataframe_html", lambda: "<table/>")
    monkeypatch.setattr("cytodataframe.frame.get_option", lambda _name: True)

    def capture_display(value: object) -> None:
        displayed.append(value)

    monkeypatch.setattr("cytodataframe.frame.display", capture_display)

    assert cdf._repr_html_() is None
    html_blocks = [
        str(getattr(widget, "data", ""))
        for widget in displayed
        if hasattr(widget, "data")
    ]
    assert any("cyto-static-snapshot" in block for block in html_blocks)


def test_repr_html_2d_places_filter_slider_next_to_image_adjustment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cdf = CytoDataFrame(
        pd.DataFrame({"FilterScore": [1.0, 2.0, 3.0]}),
        display_options={"filter_column": "FilterScore"},
    )
    displayed: list[object] = []

    monkeypatch.setattr(cdf, "_find_3d_columns_for_display", lambda: [])
    monkeypatch.setattr(cdf, "_render_output", lambda: None)
    monkeypatch.setattr(cdf, "_generate_jupyter_dataframe_html", lambda: "<table/>")
    monkeypatch.setattr("cytodataframe.frame.get_option", lambda _name: True)

    def capture_display(value: object) -> None:
        displayed.append(value)

    monkeypatch.setattr("cytodataframe.frame.display", capture_display)

    assert cdf._repr_html_() is None

    container = next(widget for widget in displayed if isinstance(widget, widgets.VBox))
    controls_row = container.children[0]
    assert isinstance(controls_row, widgets.HBox)
    assert len(controls_row.children) == 2
    filter_wrapper = controls_row.children[1]
    assert isinstance(filter_wrapper, widgets.VBox)
    assert len(filter_wrapper.children) == 1
    filter_control = filter_wrapper.children[0]
    assert isinstance(filter_control, widgets.VBox)
    assert isinstance(filter_control.children[0], widgets.HTML)
    assert "<svg " in filter_control.children[0].value
    assert isinstance(filter_control.children[1], widgets.SelectionRangeSlider)


def test_repr_html_2d_uses_accordion_for_multiple_filter_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cdf = CytoDataFrame(
        pd.DataFrame(
            {
                "FilterScoreA": [1.0, 2.0, 3.0],
                "FilterScoreB": [10.0, 20.0, 30.0],
            }
        ),
        display_options={"filter_columns": ["FilterScoreA", "FilterScoreB"]},
    )
    displayed: list[object] = []

    monkeypatch.setattr(cdf, "_find_3d_columns_for_display", lambda: [])
    monkeypatch.setattr(cdf, "_render_output", lambda: None)
    monkeypatch.setattr(cdf, "_generate_jupyter_dataframe_html", lambda: "<table/>")
    monkeypatch.setattr("cytodataframe.frame.get_option", lambda _name: True)

    def capture_display(value: object) -> None:
        displayed.append(value)

    monkeypatch.setattr("cytodataframe.frame.display", capture_display)

    assert cdf._repr_html_() is None

    container = next(widget for widget in displayed if isinstance(widget, widgets.VBox))
    controls_row = container.children[0]
    assert isinstance(controls_row, widgets.HBox)
    assert len(controls_row.children) == 2
    accordion = controls_row.children[1]
    assert isinstance(accordion, widgets.Accordion)
    assert len(accordion.children) == 1
    assert isinstance(accordion.children[0], widgets.VBox)
    assert len(accordion.children[0].children) == 2


def test_is_notebook_or_lab_detects_zmq_shell(monkeypatch: pytest.MonkeyPatch) -> None:
    zmq_shell = type("ZMQInteractiveShell", (), {})()
    monkeypatch.setattr("cytodataframe.frame.get_ipython", lambda: zmq_shell)
    assert CytoDataFrame.is_notebook_or_lab() is True


def test_is_notebook_or_lab_detects_terminal_shell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    term_shell = type("TerminalInteractiveShell", (), {})()
    monkeypatch.setattr("cytodataframe.frame.get_ipython", lambda: term_shell)
    assert CytoDataFrame.is_notebook_or_lab() is False


def test_is_notebook_or_lab_handles_unknown_shell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unknown_shell = type("CustomShell", (), {})()
    monkeypatch.setattr("cytodataframe.frame.get_ipython", lambda: unknown_shell)
    assert CytoDataFrame.is_notebook_or_lab() is False


def test_is_notebook_or_lab_handles_name_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_name_error():  # noqa: ANN202
        raise NameError("missing")

    monkeypatch.setattr("cytodataframe.frame.get_ipython", raise_name_error)
    assert CytoDataFrame.is_notebook_or_lab() is False


def test_show_widget_table_rejects_empty_columns_3d():
    cdf = CytoDataFrame(pd.DataFrame({"A": [1]}))
    with pytest.raises(ValueError, match="columns_3d must include at least one column"):
        cdf.show_widget_table(column="A", rows=[0], columns_3d=[])


def test_show_widget_table_renders_fallback_when_3d_fails():
    cdf = CytoDataFrame(pd.DataFrame({"A": [1], "B": [2]}))
    grid = cdf.show_widget_table(
        column="A",
        rows=[0, "\u2026"],
        columns=["A", "B"],
        columns_3d=["A"],
    )
    # Header + 2 rows, index + 2 columns
    assert grid.n_rows == 3
    assert grid.n_columns == 3
    assert grid.layout.width == "100%"
    assert grid.layout.height == "700px"
    assert grid.layout.overflow == "auto"
    assert "3D render failed" in grid[1, 1].value
    assert "\u2026" in grid[2, 1].value


def test_show_widget_table_raises_in_debug_mode_when_3d_fails():
    cdf = CytoDataFrame(pd.DataFrame({"A": [1]}))
    with pytest.raises(ValueError, match="does not contain a 3D volume"):
        cdf.show_widget_table(
            column="A",
            rows=[0],
            columns=["A"],
            columns_3d=["A"],
            debug=True,
        )


def test_show_widget_table_renders_3d_viewer_cells_successfully(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cdf = CytoDataFrame(
        pd.DataFrame(
            {
                "A": [1, 2, 3, 4, 5],
                "B": [10, 20, 30, 40, 50],
                "C": [100, 200, 300, 400, 500],
                "D": [1000, 2000, 3000, 4000, 5000],
            }
        ),
        display_options={"view": "trame", "height": 220, "width": 150},
    )
    monkeypatch.setattr(cdf, "get_displayed_rows", lambda: [np.int64(0), np.int64(1)])
    monkeypatch.setattr("cytodataframe.frame.pd.get_option", lambda _name: 3)
    monkeypatch.setattr(
        cdf,
        "_get_3d_volume_from_cell",
        lambda row, column: (np.ones((2, 2, 2), dtype=np.uint8), (2, 2, 2)),
    )
    monkeypatch.setattr(
        cdf,
        "_get_3d_label_overlay_from_cell",
        lambda row, column, expected_shape: np.ones(expected_shape, dtype=np.uint8),
    )

    captured: dict[str, object] = {}

    def fake_build_pyvista_viewer(**kwargs: object):  # noqa: ANN202
        captured.update(kwargs)

        return widgets.HTML(value="viewer")

    monkeypatch.setattr(cdf, "_build_pyvista_viewer", fake_build_pyvista_viewer)

    grid = cdf.show_widget_table(
        column="A",
        backend=None,
        columns=["A", "B", "C", "D"],
        max_columns=3,
        max_rows=2,
        columns_3d=["A"],
        widget_height=np.int64(140),
        index_width=np.int64(90),
    )

    assert grid.n_rows == 3
    assert grid.n_columns == 4
    assert "…" in grid[2, 1].value
    assert captured["backend"] == "trame"
    assert captured["widget_height"] == "140px"
    assert isinstance(captured.get("label_volume"), np.ndarray)
    assert isinstance(grid[1, 1], widgets.Box)
    assert len(grid[1, 1].children) == 1


def test_get_displayed_rows_when_under_limit(monkeypatch: pytest.MonkeyPatch):
    cdf = CytoDataFrame(pd.DataFrame({"A": [1, 2, 3]}, index=[10, 20, 30]))

    monkeypatch.setattr("cytodataframe.frame.pd.get_option", lambda name: 10)
    assert cdf.get_displayed_rows() == [10, 20, 30]


def test_get_displayed_rows_when_over_limit(monkeypatch: pytest.MonkeyPatch):
    cdf = CytoDataFrame(pd.DataFrame({"A": list(range(8))}, index=list(range(8))))

    def fake_get_option(name: str) -> int:
        return 6 if name == "display.max_rows" else 4

    monkeypatch.setattr("cytodataframe.frame.pd.get_option", fake_get_option)
    assert cdf.get_displayed_rows() == [0, 1, 6, 7]


def test_normalize_labels_returns_string_index_and_backmap():
    labels = pd.Index([1, "x", 2.5])
    labels_as_str, backmap = CytoDataFrame._normalize_labels(labels)
    assert list(labels_as_str) == ["1", "x", "2.5"]
    assert backmap["1"] == 1
    assert backmap["x"] == "x"
    assert backmap["2.5"] == 2.5


def test_is_3d_image_array_detects_rgb_like_images_as_not_3d() -> None:
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    rgba = np.zeros((64, 64, 4), dtype=np.uint8)
    assert CytoDataFrame._is_3d_image_array(rgb) is False
    assert CytoDataFrame._is_3d_image_array(rgba) is False


def test_is_3d_image_array_accepts_thin_small_volume_shapes() -> None:
    thin_x = np.zeros((5, 20, 3), dtype=np.uint8)
    singleton_x = np.zeros((5, 20, 1), dtype=np.uint8)
    assert CytoDataFrame._is_3d_image_array(thin_x) is True
    assert CytoDataFrame._is_3d_image_array(singleton_x) is True


def test_extract_array_from_ome_arrow_rebuilds_multichannel_planes() -> None:
    cdf = CytoDataFrame(pd.DataFrame({"A": [1]}))
    data_value = {
        "type": "ome.arrow",
        "pixels_meta": {"size_x": 2, "size_y": 2, "size_z": 1, "size_c": 3},
        "planes": [
            {"t": 0, "c": 0, "z": 0, "pixels": [10, 20, 30, 40]},
            {"t": 0, "c": 1, "z": 0, "pixels": [50, 60, 70, 80]},
            {"t": 0, "c": 2, "z": 0, "pixels": [90, 100, 110, 120]},
        ],
    }

    image = cdf._extract_array_from_ome_arrow(data_value)

    assert image is not None
    assert image.shape == (2, 2, 3)
    assert np.array_equal(image[0, 0], np.array([10, 50, 90], dtype=np.uint8))


def _install_fake_pyvista(  # noqa: C901
    monkeypatch: pytest.MonkeyPatch,
    screenshot_image: np.ndarray | None = None,
) -> None:
    class FakePointData:
        def __init__(self) -> None:
            self.data = {}
            self.active_scalars_name = None

        def clear(self) -> None:
            self.data = {}

        def __setitem__(self, key: str, value: object) -> None:
            self.data[key] = value

        def set_active_scalars(self, _name: str) -> None:
            raise AttributeError

    class FakeImageData:
        def __init__(self) -> None:
            self.dimensions = None
            self.spacing = None
            self.origin = None
            self.point_data = FakePointData()

        def set_active_scalars(self, _name: str) -> None:
            return None

        def contour(self, *args: object, **kwargs: object) -> object:
            return object()

    class FakeProp:
        def SetInterpolationTypeToNearest(self) -> None:
            return None

        def SetInterpolationTypeToLinear(self) -> None:
            return None

        def SetInterpolateScalarsBeforeMapping(self, _value: bool) -> None:
            return None

        def SetScalarOpacityUnitDistance(self, _value: float) -> None:
            return None

    class FakeMapper:
        def SetAutoAdjustSampleDistances(self, _value: bool) -> None:
            return None

        def SetUseJittering(self, _value: bool) -> None:
            return None

        def SetSampleDistance(self, _value: float) -> None:
            return None

    class FakeActor:
        def __init__(self) -> None:
            self.prop = FakeProp()
            self.mapper = FakeMapper()

        def GetProperty(self) -> FakeProp:
            return self.prop

        def GetMapper(self) -> FakeMapper:
            return self.mapper

    class FakeViewer:
        def __init__(self) -> None:
            self.layout = None
            self.value = (
                'class="pyvista" style="border: 1px solid; width: 200px; '
                'height: 200px;"'
            )

    class FakePlotter:
        def __init__(self, notebook: bool = False, off_screen: bool = False) -> None:
            self.notebook = notebook
            self.off_screen = off_screen
            self.background = None

        def set_background(self, value: str) -> None:
            self.background = value

        def add_volume(self, *args: object, **kwargs: object) -> FakeActor:
            return FakeActor()

        def add_axes(self) -> None:
            return None

        def add_mesh(self, *args: object, **kwargs: object) -> object:
            return object()

        def show(self, **_kwargs: object) -> FakeViewer:
            return FakeViewer()

        def screenshot(self, return_img: bool = True) -> np.ndarray | None:
            if not return_img:
                return None
            return screenshot_image

    fake_module = types.SimpleNamespace(
        ImageData=FakeImageData,
        Plotter=FakePlotter,
        set_jupyter_backend=lambda _backend: None,
        __spec__=ModuleSpec("pyvista", loader=None),
    )
    monkeypatch.setitem(sys.modules, "pyvista", fake_module)


def test_build_pyvista_viewer_with_fake_module(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_pyvista(
        monkeypatch,
        screenshot_image=np.zeros((2, 2, 3), dtype=np.uint8),
    )
    cdf = CytoDataFrame(pd.DataFrame({"A": [1]}))

    viewer = cdf._build_pyvista_viewer(
        volume=np.ones((2, 2, 2), dtype=np.uint8),
        backend="trame",
        widget_height="120px",
    )
    assert hasattr(viewer, "_cdf_plotter")
    assert "width: 100%;" in viewer.value
    assert "height: 100%;" in viewer.value


def test_build_pyvista_viewer_with_filled_label_overlay_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_pyvista(
        monkeypatch,
        screenshot_image=np.zeros((2, 2, 3), dtype=np.uint8),
    )
    cdf = CytoDataFrame(
        pd.DataFrame({"A": [1]}),
        display_options={"label_overlay_mode": "filled"},
    )

    viewer = cdf._build_pyvista_viewer(
        volume=np.ones((2, 2, 2), dtype=np.uint8),
        backend="trame",
        widget_height="120px",
        label_volume=np.ones((2, 2, 2), dtype=np.uint8),
    )
    assert hasattr(viewer, "_cdf_plotter")


def test_build_pyvista_viewer_with_surface_label_overlay_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_pyvista(
        monkeypatch,
        screenshot_image=np.zeros((2, 2, 3), dtype=np.uint8),
    )
    cdf = CytoDataFrame(
        pd.DataFrame({"A": [1]}),
        display_options={"label_overlay_mode": "surface"},
    )

    viewer = cdf._build_pyvista_viewer(
        volume=np.ones((2, 2, 2), dtype=np.uint8),
        backend="trame",
        widget_height="120px",
        label_volume=np.ones((2, 2, 2), dtype=np.uint8),
    )
    assert hasattr(viewer, "_cdf_plotter")


def test_add_label_overlay_toggle_control_toggles_overlay_actor_visibility() -> None:
    cdf = CytoDataFrame(pd.DataFrame({"A": [1]}))
    toggles: list[int] = []
    renders: list[bool] = []
    checkbox_kwargs: dict[str, object] = {}
    label_kwargs: dict[str, object] = {}

    class FakeActor:
        def SetVisibility(self, value: int) -> None:
            toggles.append(value)

    class FakePlotter:
        window_size = (300, 300)

        def render(self) -> None:
            renders.append(True)

        def add_checkbox_button_widget(
            self,
            callback: object,
            value: bool,
            size: int,
            position: tuple[int, int],
        ) -> None:
            checkbox_kwargs["callback"] = callback
            checkbox_kwargs["value"] = value
            checkbox_kwargs["size"] = size
            checkbox_kwargs["position"] = position

        def add_text(self, *_args: object, **kwargs: object) -> None:
            label_kwargs.update(kwargs)

    actor = FakeActor()
    plotter = FakePlotter()
    cdf._add_label_overlay_toggle_control(
        plotter=plotter,
        overlay_actors=[actor],
        display_options={},
    )

    assert checkbox_kwargs["value"] is True
    assert checkbox_kwargs["size"] == 24
    assert checkbox_kwargs["position"] == (266, 10)
    label_position = label_kwargs["position"]
    assert isinstance(label_position, tuple)
    assert label_position == pytest.approx((0.01, 20 / 300))
    assert label_kwargs["viewport"] is True
    assert label_kwargs["color"] == "white"
    assert label_kwargs["font_size"] == 9
    callback = checkbox_kwargs["callback"]
    assert callable(callback)

    callback(False)
    callback(True)
    assert toggles == [0, 1]
    assert len(renders) == 2


def test_add_label_overlay_toggle_control_respects_disable_option() -> None:
    cdf = CytoDataFrame(pd.DataFrame({"A": [1]}))
    checkbox_added: list[bool] = []

    class FakeActor:
        def SetVisibility(self, _value: int) -> None:
            return None

    class FakePlotter:
        def add_checkbox_button_widget(self, **_kwargs: object) -> None:
            checkbox_added.append(True)

    cdf._add_label_overlay_toggle_control(
        plotter=FakePlotter(),
        overlay_actors=[FakeActor()],
        display_options={"label_overlay_toggle": False},
    )

    assert checkbox_added == []


def test_show_trame_falls_back_to_ipywidgets(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_pyvista(
        monkeypatch,
        screenshot_image=np.zeros((2, 2, 3), dtype=np.uint8),
    )
    cdf = CytoDataFrame(pd.DataFrame({"A": [1]}), display_options={"view": "trame"})

    monkeypatch.setattr(
        cdf,
        "_get_3d_volume_from_cell",
        lambda row, column: (np.ones((2, 2, 2), dtype=np.uint8), (2, 2, 2)),
    )

    def fake_html_table() -> str:
        return "<table>t</table>"

    monkeypatch.setattr(cdf, "_generate_jupyter_dataframe_html", fake_html_table)
    monkeypatch.setattr(
        cdf,
        "_build_pyvista_viewer",
        lambda **_kwargs: __import__("ipywidgets").HTML("viewer"),
    )
    original_import = __import__

    def fake_import(name: str, *args: object, **kwargs: object):  # noqa: ANN202
        if name.startswith("trame"):
            raise ImportError("no trame")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    container = cdf.show_trame(row=0, column="A", backend=None)
    assert hasattr(container, "children")
    assert len(container.children) == 2


def test_show_trame_raises_when_ipywidgets_missing_in_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_pyvista(
        monkeypatch,
        screenshot_image=np.zeros((2, 2, 3), dtype=np.uint8),
    )
    cdf = CytoDataFrame(pd.DataFrame({"A": [1]}), display_options={"view": "trame"})
    monkeypatch.setattr(
        cdf,
        "_get_3d_volume_from_cell",
        lambda row, column: (np.ones((2, 2, 2), dtype=np.uint8), (2, 2, 2)),
    )
    monkeypatch.setattr(
        cdf,
        "_generate_jupyter_dataframe_html",
        lambda: "<table>t</table>",
    )
    monkeypatch.setattr(
        cdf,
        "_build_pyvista_viewer",
        lambda **_kwargs: types.SimpleNamespace(server=None, _server=None),
    )
    original_import = __import__

    def fake_import(name: str, *args: object, **kwargs: object):  # noqa: ANN202
        if name.startswith("trame"):
            raise ImportError("no trame")
        if name == "ipywidgets":
            raise ImportError("no ipywidgets")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    with pytest.raises(
        RuntimeError,
        match="ipywidgets is required for notebook layout",
    ):
        cdf.show_trame(row=0, column="A", backend=None)


def test_generate_jupyter_dataframe_html_info_repr_branch(
    monkeypatch: pytest.MonkeyPatch,
):
    cdf = CytoDataFrame(pd.DataFrame({"A": [1]}))
    monkeypatch.setattr(cdf, "_info_repr", lambda: True)
    html = cdf._generate_jupyter_dataframe_html()
    assert html.startswith("<pre>")
    assert "&lt;class" in html


def test_generate_jupyter_dataframe_html_with_joined_components(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
):
    base = pd.DataFrame(
        {"Image_FileName_DNA": ["dna.tiff"], "OMECol": [{"type": "ome.arrow"}]},
        index=[0],
    )
    cdf = CytoDataFrame(base, data_context_dir=str(tmp_path))
    cdf._custom_attrs["data_bounding_box"] = pd.DataFrame(
        {
            "Cells_AreaShape_BoundingBoxMinimum_X": [0],
            "Cells_AreaShape_BoundingBoxMinimum_Y": [0],
            "Cells_AreaShape_BoundingBoxMaximum_X": [1],
            "Cells_AreaShape_BoundingBoxMaximum_Y": [1],
        },
        index=[0],
    )
    cdf._custom_attrs["compartment_center_xy"] = pd.DataFrame(
        {"Cells_Location_Center_X": [0], "Cells_Location_Center_Y": [0]},
        index=[0],
    )
    cdf._custom_attrs["data_image_paths"] = pd.DataFrame(
        {"Image_PathName_DNA": [str(tmp_path)]},
        index=[0],
    )

    options = {
        "display.notebook_repr_html": True,
        "display.max_rows": 10,
        "display.min_rows": 10,
        "display.max_columns": 10,
        "display.show_dimensions": False,
    }

    monkeypatch.setattr("cytodataframe.frame.get_option", lambda name: options[name])
    monkeypatch.setattr(
        "cytodataframe.frame.CytoDataFrame.find_image_columns",
        lambda self: ["Image_FileName_DNA"],
    )
    monkeypatch.setattr(
        "cytodataframe.frame.CytoDataFrame.find_image_path_columns",
        lambda self, image_cols, all_cols: {"Image_FileName_DNA": "Image_PathName_DNA"},
    )
    monkeypatch.setattr(
        "cytodataframe.frame.CytoDataFrame.get_displayed_rows",
        lambda self: [0],
    )
    monkeypatch.setattr(
        cdf,
        "process_image_data_as_html_display",
        lambda **_kwargs: "<img src='x'/>",
    )
    monkeypatch.setattr(cdf, "find_ome_arrow_columns", lambda data: ["OMECol"])
    monkeypatch.setattr(
        cdf,
        "process_ome_arrow_data_as_html_display",
        lambda _value: "<div>OME</div>",
    )

    html = cdf._generate_jupyter_dataframe_html()
    assert "<img src='x'/>" in html
    assert "<div>OME</div>" in html


def test_render_output_displays_js_and_print_html(monkeypatch: pytest.MonkeyPatch):
    cdf = CytoDataFrame(pd.DataFrame({"A": [1]}))
    cdf._custom_attrs["_output"] = nullcontext()
    monkeypatch.setattr(
        cdf,
        "_generate_jupyter_dataframe_html",
        lambda: '<div class="cyto-3d-image" data-volume="a"></div>',
    )
    monkeypatch.setattr("cytodataframe.frame.get_option", lambda _name: False)

    displayed: list = []

    def capture_display(value: object) -> None:
        displayed.append(value)

    monkeypatch.setattr(
        "cytodataframe.frame.display",
        capture_display,
    )
    result = cdf._render_output()
    assert result is None
    assert len(displayed) == 3


def test_render_output_clears_output_before_display(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cdf = CytoDataFrame(pd.DataFrame({"A": [1]}))

    class DummyOutput:
        def __init__(self) -> None:
            self.clear_calls: list[bool] = []

        def clear_output(self, wait: bool = False) -> None:
            self.clear_calls.append(wait)

        def __enter__(self) -> "DummyOutput":
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:  # noqa: ANN001
            return False

    dummy_output = DummyOutput()
    cdf._custom_attrs["_output"] = dummy_output
    monkeypatch.setattr(cdf, "_generate_jupyter_dataframe_html", lambda: "<table/>")
    monkeypatch.setattr("cytodataframe.frame.get_option", lambda _name: True)
    monkeypatch.setattr("cytodataframe.frame.display", lambda _value: None)

    cdf._render_output()

    assert dummy_output.clear_calls == [True]


def test_generate_trame_snapshot_html_paths(monkeypatch: pytest.MonkeyPatch):
    cdf = CytoDataFrame(pd.DataFrame({"Image_FileName_DNA": ["dna.tiff"]}, index=[0]))
    monkeypatch.setattr(cdf, "_generate_jupyter_dataframe_html", lambda: "<table/>")

    # Early return when no bounding box.
    cdf._custom_attrs["data_bounding_box"] = None
    assert cdf._generate_trame_snapshot_html() == "<table/>"

    # Snapshot render path.
    cdf._custom_attrs["data_bounding_box"] = pd.DataFrame(
        {
            "Cells_AreaShape_BoundingBoxMinimum_X": [0],
            "Cells_AreaShape_BoundingBoxMinimum_Y": [0],
            "Cells_AreaShape_BoundingBoxMaximum_X": [1],
            "Cells_AreaShape_BoundingBoxMaximum_Y": [1],
        },
        index=[0],
    )
    cdf._custom_attrs["_snapshot_cache"] = {}
    cdf._custom_attrs["_snapshot_cache_lock"] = None
    monkeypatch.setattr(cdf, "find_image_columns", lambda: ["Image_FileName_DNA"])
    monkeypatch.setattr(cdf, "get_displayed_rows", lambda: [0])
    monkeypatch.setattr(
        cdf,
        "_get_3d_volume_from_cell",
        lambda row, column: (np.ones((2, 2, 2), dtype=np.uint8), (2, 2, 2)),
    )
    monkeypatch.setattr(
        cdf,
        "_get_3d_label_overlay_from_cell",
        lambda row, column, expected_shape: np.ones(expected_shape, dtype=np.uint8),
    )
    captured: dict[str, object] = {}

    def fake_snapshot(volume, dims, label_volume=None):  # noqa: ANN001, ANN202
        captured["label_volume"] = label_volume
        return "<img/>"

    monkeypatch.setattr(cdf, "_pyvista_volume_snapshot_html", fake_snapshot)
    out = cdf._generate_trame_snapshot_html()
    assert "<img/>" in out or "Snapshot unavailable" in out
    assert isinstance(captured.get("label_volume"), np.ndarray)


def test_pyvista_volume_snapshot_html_success(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_pyvista(
        monkeypatch,
        screenshot_image=np.zeros((2, 2, 3), dtype=np.uint8),
    )
    cdf = CytoDataFrame(
        pd.DataFrame({"A": [1]}),
        display_options={"width": "10px", "height": "10px"},
    )
    html = cdf._pyvista_volume_snapshot_html(
        volume=np.ones((2, 2, 2), dtype=np.uint8),
        dims=(2, 2, 2),
    )
    assert html is not None
    assert "data:image/png;base64" in html


def test_pyvista_volume_snapshot_html_returns_none_when_no_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_pyvista(monkeypatch, screenshot_image=None)
    cdf = CytoDataFrame(pd.DataFrame({"A": [1]}))
    html = cdf._pyvista_volume_snapshot_html(
        volume=np.ones((2, 2, 2), dtype=np.uint8),
        dims=(2, 2, 2),
    )
    assert html is None


def _install_fake_pyvista_with_records(  # noqa: C901
    monkeypatch: pytest.MonkeyPatch,
    records: dict[str, list[dict[str, object]]],
) -> None:
    """Install a lightweight fake PyVista module that records render calls."""

    class FakePointData:
        def __init__(self) -> None:
            self.data = {}

        def clear(self) -> None:
            self.data = {}

        def __setitem__(self, key: str, value: object) -> None:
            self.data[key] = value

        def set_active_scalars(self, _name: str) -> None:
            raise AttributeError

    class FakeImageData:
        def __init__(self) -> None:
            self.dimensions = None
            self.spacing = None
            self.origin = None
            self.point_data = FakePointData()

        def set_active_scalars(self, _name: str) -> None:
            return None

        def contour(self, *args: object, **kwargs: object) -> object:
            return {"args": args, "kwargs": kwargs}

    class FakeProp:
        def SetInterpolationTypeToNearest(self) -> None:
            return None

        def SetInterpolationTypeToLinear(self) -> None:
            return None

        def SetInterpolateScalarsBeforeMapping(self, _value: bool) -> None:
            return None

        def SetScalarOpacityUnitDistance(self, _value: float) -> None:
            return None

    class FakeMapper:
        def SetAutoAdjustSampleDistances(self, _value: bool) -> None:
            return None

        def SetUseJittering(self, _value: bool) -> None:
            return None

        def SetSampleDistance(self, _value: float) -> None:
            return None

    class FakeActor:
        def __init__(self) -> None:
            self.prop = FakeProp()
            self.mapper = FakeMapper()

        def GetProperty(self) -> FakeProp:
            return self.prop

        def GetMapper(self) -> FakeMapper:
            return self.mapper

    class FakePlotter:
        def __init__(self, notebook: bool = False, off_screen: bool = False) -> None:
            self.notebook = notebook
            self.off_screen = off_screen

        def set_background(self, _value: str) -> None:
            return None

        def add_volume(self, *args: object, **kwargs: object) -> FakeActor:
            records["add_volume"].append({"args": args, "kwargs": kwargs})
            return FakeActor()

        def add_mesh(self, *args: object, **kwargs: object) -> object:
            records["add_mesh"].append({"args": args, "kwargs": kwargs})
            return object()

        def screenshot(self, return_img: bool = True) -> np.ndarray | None:
            if not return_img:
                return None
            return np.zeros((2, 2, 3), dtype=np.uint8)

    fake_module = types.SimpleNamespace(
        ImageData=FakeImageData,
        Plotter=FakePlotter,
        set_jupyter_backend=lambda _backend: None,
        __spec__=ModuleSpec("pyvista", loader=None),
    )
    monkeypatch.setitem(sys.modules, "pyvista", fake_module)


def test_pyvista_volume_snapshot_html_surface_adds_mesh_overlay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = {"add_volume": [], "add_mesh": []}
    _install_fake_pyvista_with_records(monkeypatch, records)
    cdf = CytoDataFrame(
        pd.DataFrame({"A": [1]}),
        display_options={"label_overlay_mode": "surface"},
    )
    html = cdf._pyvista_volume_snapshot_html(
        volume=np.ones((2, 2, 2), dtype=np.uint8),
        dims=(2, 2, 2),
        label_volume=np.ones((2, 2, 2), dtype=np.uint8),
    )
    assert html is not None
    assert len(records["add_mesh"]) >= 2


def test_pyvista_volume_snapshot_html_filled_adds_volume_overlay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = {"add_volume": [], "add_mesh": []}
    _install_fake_pyvista_with_records(monkeypatch, records)
    cdf = CytoDataFrame(
        pd.DataFrame({"A": [1]}),
        display_options={"label_overlay_mode": "filled"},
    )
    html = cdf._pyvista_volume_snapshot_html(
        volume=np.ones((2, 2, 2), dtype=np.uint8),
        dims=(2, 2, 2),
        label_volume=np.ones((2, 2, 2), dtype=np.uint8),
    )
    assert html is not None
    assert len(records["add_volume"]) >= 2
    assert any(
        call["kwargs"].get("blending") == "maximum" for call in records["add_volume"]
    )


def test_show_trame_trame_layout_success(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_pyvista(
        monkeypatch,
        screenshot_image=np.zeros((2, 2, 3), dtype=np.uint8),
    )
    cdf = CytoDataFrame(pd.DataFrame({"A": [1]}), display_options={"view": "trame"})

    monkeypatch.setattr(
        cdf,
        "_get_3d_volume_from_cell",
        lambda row, column: (np.ones((2, 2, 2), dtype=np.uint8), (2, 2, 2)),
    )
    monkeypatch.setattr(
        cdf,
        "_generate_jupyter_dataframe_html",
        lambda: "<table>t</table>",
    )
    monkeypatch.setattr(
        cdf,
        "_build_pyvista_viewer",
        lambda **_kwargs: types.SimpleNamespace(server=None, _server=None),
    )

    class DummyCtx:
        def __enter__(self):  # noqa: ANN204
            return self

        def __exit__(self, *_exc: object) -> bool:
            return False

    class DummyLayout:
        def __init__(self, _server: object) -> None:
            self.content = DummyCtx()
            self.content.children = []

        def __enter__(self):  # noqa: ANN204
            return self

        def __exit__(self, *_exc: object) -> bool:
            return False

    class DummyServer:
        client_type = "vue2"

        def url(self) -> str:
            return "http://example"

    def fake_get_server() -> DummyServer:
        return DummyServer()

    html_mod = types.SimpleNamespace(Div=lambda *args, **kwargs: None)
    vuetify_mod = types.SimpleNamespace(
        VContainer=lambda *args, **kwargs: DummyCtx(),
        VRow=lambda *args, **kwargs: DummyCtx(),
        VCol=lambda *args, **kwargs: DummyCtx(),
    )
    monkeypatch.setitem(
        sys.modules,
        "trame.app",
        types.SimpleNamespace(
            get_server=fake_get_server,
            __spec__=ModuleSpec("trame.app", loader=None),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "trame.widgets",
        types.SimpleNamespace(
            html=html_mod,
            vuetify=vuetify_mod,
            __spec__=ModuleSpec("trame.widgets", loader=None),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "trame.ui.vuetify",
        types.SimpleNamespace(
            SinglePageLayout=DummyLayout,
            __spec__=ModuleSpec("trame.ui.vuetify", loader=None),
        ),
    )

    shown: list = []

    def capture_shown(value: object) -> None:
        shown.append(value)

    monkeypatch.setattr(
        "cytodataframe.frame.display",
        capture_shown,
    )
    layout = cdf.show_trame(row=0, column="A", backend=None)
    assert layout is not None
    assert shown


def test_repr_returns_expected_values(monkeypatch: pytest.MonkeyPatch) -> None:
    cdf = CytoDataFrame(pd.DataFrame({"A": [1]}))
    monkeypatch.setattr("cytodataframe.frame.get_option", lambda _name: True)
    assert cdf.__repr__() == ""
    debug_repr = cdf.__repr__(debug=True)
    assert isinstance(debug_repr, str)
    assert "A" in debug_repr


def test_enable_debug_mode_adds_handler_once() -> None:
    cdf = CytoDataFrame(pd.DataFrame({"A": [1]}))
    frame_logger = logging.getLogger("cytodataframe.frame")
    original_handlers = list(frame_logger.handlers)
    frame_logger.handlers = []
    frame_logger.setLevel(logging.INFO)
    try:
        cdf._enbable_debug_mode()
        assert frame_logger.level == logging.DEBUG
        assert len(frame_logger.handlers) == 1
        cdf._enbable_debug_mode()
        assert len(frame_logger.handlers) == 1
    finally:
        frame_logger.handlers = original_handlers
