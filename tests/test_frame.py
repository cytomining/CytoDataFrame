"""
Tests cosmicqc CytoDataFrame module
"""

import pathlib
import sys
import types

import imageio.v2 as imageio
import nbformat
import numpy as np
import pandas as pd
import pytest
from _pytest.monkeypatch import MonkeyPatch
from nbconvert.preprocessors import CellExecutionError, ExecutePreprocessor
from pyarrow import parquet

from cytodataframe.frame import CytoDataFrame
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

    class DummyOMEArrow:
        def __init__(self, data: str):  # noqa: ANN204
            self.data = data

    dummy_module = types.SimpleNamespace(
        OMEArrow=DummyOMEArrow,
        __version__="test",
        __spec__=types.SimpleNamespace(loader=None),
    )
    monkeypatch.setitem(sys.modules, "ome_arrow", dummy_module)

    captured: dict = {}

    def fake_write_table(table, file_path, **kwargs):  # noqa: ANN001, ANN202, ANN003
        captured["df"] = table.to_pandas()
        captured["file_path"] = file_path
        captured["kwargs"] = kwargs

    monkeypatch.setattr(
        "pyarrow.parquet.write_table", fake_write_table, raising=False
    )

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

    class DummyOMEArrow:
        def __init__(self, data: str):  # noqa: ANN204
            self.data = data

    dummy_module = types.SimpleNamespace(
        OMEArrow=DummyOMEArrow,
        __version__="test",
        __spec__=types.SimpleNamespace(loader=None),
    )
    monkeypatch.setitem(sys.modules, "ome_arrow", dummy_module)

    captured: dict = {}

    def fake_write_table(table, file_path, **kwargs):  # noqa: ANN001, ANN202, ANN003
        captured["df"] = table.to_pandas()

    monkeypatch.setattr(
        "pyarrow.parquet.write_table", fake_write_table, raising=False
    )

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

    # Minimal dummy dataframe
    df = pd.DataFrame({"Image_FileName_DNA": ["example.tif"]})
    cdf = CytoDataFrame(df)

    # Simulate the change dictionary sent by ipywidgets
    change = {"new": 75}

    # Track render calls using monkeypatch or a flag
    render_called = {}

    def mock_render_output() -> None:
        render_called["called"] = True

    monkeypatch.setattr(cdf, "_render_output", mock_render_output)

    # Call the method manually
    cdf._on_slider_change(change)

    # Check if internal widget state updated
    assert cdf._custom_attrs["_widget_state"]["scale"] == 75

    # Check if the render method was triggered
    assert render_called.get("called", False)


def test_example_notebook_execution():
    """
    Executes the example notebook to ensure it runs.
    """

    with open(
        (notebook_path := "docs/src/examples/cytodataframe_at_a_glance.ipynb")
    ) as f:
        nb = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=300, kernel_name="python3")

    try:
        ep.preprocess(
            nb, {"metadata": {"path": str(pathlib.Path(notebook_path).parent)}}
        )
    except CellExecutionError as e:
        pytest.fail(f"Notebook execution failed: {e}")
