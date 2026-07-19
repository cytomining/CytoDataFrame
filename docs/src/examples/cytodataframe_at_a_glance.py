# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: CytoDataFrame (3.13.5.final.0)
#     language: python
#     name: python3
# ---

# # CytoDataFrame at a Glance
#
# This notebook demonstrates various capabilities of
# [CytoDataFrame](https://github.com/cytomining/CytoDataFrame) using examples.
#
# CytoDataFrame is intended to provide you a Pandas-like
# DataFrame experience which is enhanced with single-cell
# visual information which can be viewed directly in a Jupyter notebook.

# +
import pathlib

import pandas as pd

from cytodataframe.frame import CytoDataFrame

# create paths for use with CytoDataFrames below
jump_data_path = "../../../tests/data/cytotable/JUMP_plate_BR00117006"
nf1_cellpainting_path = "../../../tests/data/cytotable/NF1_cellpainting_data_shrunken/"
nuclear_speckles_path = "../../../tests/data/cytotable/nuclear_speckles"
pediatric_cancer_atlas_path = (
    "../../../tests/data/cytotable/pediatric_cancer_atlas_profiling"
)
# -
# %%time
# view JUMP plate BR00117006 with images
frame = CytoDataFrame(
    data=f"{jump_data_path}/BR00117006_shrunken.parquet",
    data_context_dir=f"{jump_data_path}/images/orig",
)[
    [
        "Metadata_ImageNumber",
        "Cells_Number_Object_Number",
        "Nuclei_Texture_Variance_RNA_5_03_256",
        "Image_FileName_OrigAGP",
        "Image_FileName_OrigDNA",
        "Image_FileName_OrigRNA",
    ]
][:3]
frame

# %%time
# view JUMP plate BR00117006 with images and overlaid outlines for segmentation
frame = CytoDataFrame(
    data=f"{jump_data_path}/BR00117006_shrunken.parquet",
    data_context_dir=f"{jump_data_path}/images/orig",
    data_outline_context_dir=f"{jump_data_path}/images/outlines",
)[
    [
        "Metadata_ImageNumber",
        "Cells_Number_Object_Number",
        "Image_FileName_OrigAGP",
        "Image_FileName_OrigDNA",
        "Image_FileName_OrigRNA",
    ]
][:3]
frame


# %%time
# view JUMP plate BR00117006 with images and overlaid outlines for segmentation
# and changing the color to something besides the default (default is green).
CytoDataFrame(
    data=f"{jump_data_path}/BR00117006_shrunken.parquet",
    data_context_dir=f"{jump_data_path}/images/orig",
    data_outline_context_dir=f"{jump_data_path}/images/outlines",
    display_options={"outline_color": (200, 100, 255)},
)[
    [
        "Metadata_ImageNumber",
        "Cells_Number_Object_Number",
        "Image_FileName_OrigAGP",
        "Image_FileName_OrigDNA",
        "Image_FileName_OrigRNA",
    ]
][:3]

# %%time
# view JUMP plate BR00117006 with images and overlaid outlines for segmentation
# and adding scale bars which show how micrometers scale to the pixels displayed.
CytoDataFrame(
    data=f"{jump_data_path}/BR00117006_shrunken.parquet",
    data_context_dir=f"{jump_data_path}/images/orig",
    data_outline_context_dir=f"{jump_data_path}/images/outlines",
    display_options={
        "um_per_pixel": 0.1550,
        "scale_bar": {
            "length_um": 5,
            "location": "lower right",
            "color": (255, 255, 255),
            "thickness_px": 2,
            "margin_px": 5,
        },
    },
)[
    [
        "Metadata_ImageNumber",
        "Cells_Number_Object_Number",
        "Image_FileName_OrigAGP",
        "Image_FileName_OrigDNA",
        "Image_FileName_OrigRNA",
    ]
][:3]

# %%time
# view JUMP plate BR00117006 and merge multiple channels into a single
# single-cell crop composite (similar to a Fiji composite). Each channel is
# tinted a color and additively blended so colocalization is easy to see.
# Channels may be named by their column, their channel suffix (e.g. "OrigDNA"),
# or a substring. The merged image is added as a new "Image_Composite" column.
CytoDataFrame(
    data=f"{jump_data_path}/BR00117006_shrunken.parquet",
    data_context_dir=f"{jump_data_path}/images/orig",
    display_options={
        "composite_channels": {
            "OrigDNA": "blue",
            "OrigRNA": "green",
            "OrigAGP": "red",
        }
    },
)[
    [
        "Metadata_ImageNumber",
        "Cells_Number_Object_Number",
        "Image_FileName_OrigAGP",
        "Image_FileName_OrigDNA",
        "Image_FileName_OrigRNA",
    ]
][:3]

# %%time
# the same composite feature can merge *all* image channels at once using
# default (Fiji-like) colors by passing "all".
CytoDataFrame(
    data=f"{jump_data_path}/BR00117006_shrunken.parquet",
    data_context_dir=f"{jump_data_path}/images/orig",
    display_options={"composite_channels": "all"},
)[
    [
        "Metadata_ImageNumber",
        "Cells_Number_Object_Number",
        "Image_FileName_OrigAGP",
        "Image_FileName_OrigDNA",
        "Image_FileName_OrigRNA",
    ]
][:3]

# %%time
# composites also keep the segmentation outline and red center dot when those
# are configured, so a single merged view still shows where each object was
# segmented. Here we merge channels and overlay outlines from the segmentation
# directory (channels are colored to avoid the green outline so it stays clear).
CytoDataFrame(
    data=f"{jump_data_path}/BR00117006_shrunken.parquet",
    data_context_dir=f"{jump_data_path}/images/orig",
    data_outline_context_dir=f"{jump_data_path}/images/outlines",
    display_options={
        "composite_channels": {
            "OrigDNA": "blue",
            "OrigRNA": "magenta",
            "OrigAGP": "red",
        }
    },
)[
    [
        "Metadata_ImageNumber",
        "Cells_Number_Object_Number",
        "Image_FileName_OrigAGP",
        "Image_FileName_OrigDNA",
        "Image_FileName_OrigRNA",
    ]
][:3]

# %%time
# view JUMP plate BR00117006 with images and adjust the brightness
CytoDataFrame(
    data=f"{jump_data_path}/BR00117006_shrunken.parquet",
    data_context_dir=f"{jump_data_path}/images/orig",
    display_options={"brightness": 10},
)[
    [
        "Metadata_ImageNumber",
        "Cells_Number_Object_Number",
        "Image_FileName_OrigAGP",
        "Image_FileName_OrigDNA",
        "Image_FileName_OrigRNA",
    ]
][:3]

# %%time
# view JUMP plate BR00117006 with images and overlaid outlines for segmentation
# and removing the optional red center dot.
CytoDataFrame(
    data=f"{jump_data_path}/BR00117006_shrunken.parquet",
    data_context_dir=f"{jump_data_path}/images/orig",
    data_outline_context_dir=f"{jump_data_path}/images/outlines",
    display_options={"center_dot": False},
)[
    [
        "Metadata_ImageNumber",
        "Cells_Number_Object_Number",
        "Image_FileName_OrigAGP",
        "Image_FileName_OrigDNA",
        "Image_FileName_OrigRNA",
    ]
][:3]

# %%time
# view JUMP plate BR00117006 with images and change the display width
CytoDataFrame(
    data=f"{jump_data_path}/BR00117006_shrunken.parquet",
    data_context_dir=f"{jump_data_path}/images/orig",
    data_outline_context_dir=f"{jump_data_path}/images/outlines",
    display_options={"width": "100"},
)[
    [
        "Metadata_ImageNumber",
        "Cells_Number_Object_Number",
        "Image_FileName_OrigAGP",
        "Image_FileName_OrigDNA",
        "Image_FileName_OrigRNA",
    ]
][:3]

# %%time
# view JUMP plate BR00117006 with images, change the display height and width
# and also transpose for a different view of things.
CytoDataFrame(
    data=f"{jump_data_path}/BR00117006_shrunken.parquet",
    data_context_dir=f"{jump_data_path}/images/orig",
    data_outline_context_dir=f"{jump_data_path}/images/outlines",
    display_options={"width": "200px", "height": "auto"},
)[
    [
        "Metadata_ImageNumber",
        "Cells_Number_Object_Number",
        "Image_FileName_OrigAGP",
        "Image_FileName_OrigDNA",
        "Image_FileName_OrigRNA",
    ]
][:5].T

# +
# %%time
# export to OME Parquet, a format which uses OME Arrow
# to store OME-spec images as values within the table.
frame.to_ome_parquet(file_path="example.ome.parquet")

# read OME Parquet file into the CytoDataFrame
CytoDataFrame(data="example.ome.parquet")
# -

# %%time
# view JUMP plate BR00117006 with images, changing the bounding box
# using offsets so each image has roughly the same size.
CytoDataFrame(
    data=f"{jump_data_path}/BR00117006_shrunken.parquet",
    data_context_dir=f"{jump_data_path}/images/orig",
    data_outline_context_dir=f"{jump_data_path}/images/outlines",
    display_options={
        "offset_bounding_box": {
            "x_min": -20,
            "y_min": -20,
            "x_max": 20,
            "y_max": 20,
        },
    },
)[
    [
        "Metadata_ImageNumber",
        "Cells_Number_Object_Number",
        "Image_FileName_OrigAGP",
        "Image_FileName_OrigDNA",
        "Image_FileName_OrigRNA",
    ]
][:5]

# %%time
# CytoDataFrame can also crop images when the data does not include AreaShape
# bounding box columns (e.g. older CellProfiler outputs such as LINCS). Here we
# drop the bounding box columns to simulate this case; cropping then relies on
# offset_bounding_box applied to the compartment center coordinates.
jump_without_bounding_boxes = pd.read_parquet(
    f"{jump_data_path}/BR00117006_shrunken.parquet"
)
jump_without_bounding_boxes = jump_without_bounding_boxes.drop(
    columns=[
        column
        for column in jump_without_bounding_boxes.columns
        if "BoundingBox" in column
    ]
)
CytoDataFrame(
    data=jump_without_bounding_boxes,
    data_context_dir=f"{jump_data_path}/images/orig",
    display_options={
        "offset_bounding_box": {
            "x_min": -20,
            "y_min": -20,
            "x_max": 20,
            "y_max": 20,
        },
    },
)[
    [
        "Metadata_ImageNumber",
        "Cells_Number_Object_Number",
        "Image_FileName_OrigAGP",
        "Image_FileName_OrigDNA",
        "Image_FileName_OrigRNA",
    ]
][:5]

# %%time
# For image-level data that has neither bounding box nor compartment center
# columns (for example, whole-image quality control metrics), use
# render_whole_image to display the full field of view without cropping.
jump_image_level = pd.read_parquet(f"{jump_data_path}/BR00117006_shrunken.parquet")
jump_image_level = jump_image_level.drop(
    columns=[
        column
        for column in jump_image_level.columns
        if "BoundingBox" in column or "Location_Center" in column
    ]
)
CytoDataFrame(
    data=jump_image_level,
    data_context_dir=f"{jump_data_path}/images/orig",
    display_options={"render_whole_image": True},
)[
    [
        "Metadata_ImageNumber",
        "Cells_Number_Object_Number",
        "Image_FileName_OrigAGP",
        "Image_FileName_OrigDNA",
        "Image_FileName_OrigRNA",
    ]
][:3]

# %%time
# view NF1 Cell Painting data with images
CytoDataFrame(
    data=f"{nf1_cellpainting_path}/Plate_2_with_image_data_shrunken.parquet",
    data_context_dir=f"{nf1_cellpainting_path}/Plate_2_images",
)[
    [
        "Metadata_ImageNumber",
        "Metadata_Cells_Number_Object_Number",
        "Image_FileName_GFP",
        "Image_FileName_RFP",
        "Image_FileName_DAPI",
    ]
][:3]

# %%time
# view NF1 Cell Painting data with images and overlaid outlines from masks
frame = CytoDataFrame(
    data=f"{nf1_cellpainting_path}/Plate_2_with_image_data_shrunken.parquet",
    data_context_dir=f"{nf1_cellpainting_path}/Plate_2_images",
    data_mask_context_dir=f"{nf1_cellpainting_path}/Plate_2_masks",
)[
    [
        "Metadata_ImageNumber",
        "Metadata_Cells_Number_Object_Number",
        "Image_FileName_GFP",
        "Image_FileName_RFP",
        "Image_FileName_DAPI",
    ]
][:3]
frame

# +
# %%time
# add active paths on the local system to show how CytoDataFrame
# may be used without specifying a context directory for images.
# Note: normally these paths are local to the system where the
# profile data was generated, which often is not the same as the
# system which will be used to analyze the data.
parquet_path = f"{nf1_cellpainting_path}/Plate_2_with_image_data_shrunken.parquet"
nf1_dataset_with_modified_image_paths = pd.read_parquet(path=parquet_path)
nf1_dataset_with_modified_image_paths.loc[
    :, ["Image_PathName_DAPI", "Image_PathName_GFP", "Image_PathName_RFP"]
] = f"{pathlib.Path(parquet_path).parent}/Plate_2_images"

# view NF1 Cell Painting data with images and overlaid outlines from masks
CytoDataFrame(
    # note: we can read directly from an existing Pandas DataFrame
    data=nf1_dataset_with_modified_image_paths,
    data_mask_context_dir=f"{nf1_cellpainting_path}/Plate_2_masks",
)[
    [
        "Metadata_ImageNumber",
        "Metadata_Cells_Number_Object_Number",
        "Image_FileName_GFP",
        "Image_FileName_RFP",
        "Image_FileName_DAPI",
    ]
][:3]

# +
# %%time
# export to OME Parquet, a format which uses OME Arrow
# to store OME-spec images as values within the table.
frame.to_ome_parquet(file_path="example.ome.parquet")

# read OME Parquet file into the CytoDataFrame
CytoDataFrame(data="example.ome.parquet")
# -

# %%time
# view nuclear speckles data with images and overlaid outlines from masks
CytoDataFrame(
    data=f"{nuclear_speckles_path}/test_slide1_converted.parquet",
    data_context_dir=f"{nuclear_speckles_path}/images/plate1",
    data_mask_context_dir=f"{nuclear_speckles_path}/masks/plate1",
)[
    [
        "Metadata_ImageNumber",
        "Nuclei_Number_Object_Number",
        "Image_FileName_A647",
        "Image_FileName_DAPI",
        "Image_FileName_GOLD",
    ]
][:3]

# %%time
# view nuclear speckles data with images and overlaid outlines from masks
# and also apply a filter to only show rows where the value for
# "Nuclei_Texture_Variance_DAPI_3_03_256".
CytoDataFrame(
    data=f"{nuclear_speckles_path}/test_slide1_converted.parquet",
    data_context_dir=f"{nuclear_speckles_path}/images/plate1",
    data_mask_context_dir=f"{nuclear_speckles_path}/masks/plate1",
    display_options={
        "filter_columns": ["Nuclei_Texture_Variance_DAPI_3_03_256"],
    },
)[
    [
        "Metadata_ImageNumber",
        "Nuclei_Number_Object_Number",
        "Nuclei_Texture_Variance_DAPI_3_03_256",
        "Image_FileName_A647",
        "Image_FileName_DAPI",
        "Image_FileName_GOLD",
    ]
]

# %%time
# view ALSF pediatric cancer atlas plate BR00143976 with images
cdf = CytoDataFrame(
    data=f"{pediatric_cancer_atlas_path}/BR00143976_shrunken.parquet",
    data_context_dir=f"{pediatric_cancer_atlas_path}/images/orig",
    data_outline_context_dir=f"{pediatric_cancer_atlas_path}/images/outlines",
    segmentation_file_regex={
        r"CellsOutlines_BR(\d+)_C(\d{2})_\d+\.tiff": r".*ch3.*\.tiff",
        r"NucleiOutlines_BR(\d+)_C(\d{2})_\d+\.tiff": r".*ch5.*\.tiff",
    },
)[
    [
        "Metadata_ImageNumber",
        "Metadata_Nuclei_Number_Object_Number",
        "Image_FileName_OrigAGP",
        "Image_FileName_OrigDNA",
    ]
]
cdf

# %%time
# show that we can use the cytodataframe again
# by quick variable reference.
cdf

# +
# %%time
# export to OME Parquet, a format which uses OME Arrow
# to store OME-spec images as values within the table.
cdf.to_ome_parquet(file_path="example.ome.parquet")

# read OME Parquet file into the CytoDataFrame
CytoDataFrame(data="example.ome.parquet")

# +
# %%time
# 3D example dataset, showing how
# CytoDataFrame can be used with 3D data for visualization.
cp_3d_path = "../../../tests/data/CP_tutorial_3D_noise_nuclei_segmentation"

# send the data to CytoDataFrame
# note: because we have 3d input images, CytoDataFrame will automatically process
# using the 3D display options for interactive visualization.
cdf = CytoDataFrame(
    data=pathlib.Path(cp_3d_path) / "output/MyExpt_RealsizeNuclei.csv",
    data_context_dir=str(pathlib.Path(cp_3d_path) / "input"),
)

cdf[["ImageNumber", "ObjectNumber", "FileName_Nuclei"]][:3]
# +
# %%time
# read 3d images with segmentation masks and show the
# segmentation masks are also 3D.
cdf = CytoDataFrame(
    data=pathlib.Path(cp_3d_path) / "output/MyExpt_RealsizeNuclei.csv",
    data_context_dir=str(pathlib.Path(cp_3d_path) / "input"),
    data_mask_context_dir=str(pathlib.Path(cp_3d_path) / "output/masks"),
)

cdf[["ImageNumber", "ObjectNumber", "FileName_Nuclei"]][:3]
