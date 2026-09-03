# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: CytoDataFrame (3.13.5)
#     language: python
#     name: python3
# ---

# # OME-Arrow and CytoDataFrame
#
# This notebook demonstrates how [OME-Arrow](https://github.com/WayScience/ome-arrow) works in context with CytoDataFrame.

# + colab={"base_uri": "https://localhost:8080/", "height": 423} id="Y5e85t05nSLt" outputId="a0807c7c-4705-41a2-83b2-9fb7287c40f9"
import pyarrow as pa
import pyarrow.parquet as pq
from ome_arrow import OMEArrow

from cytodataframe import CytoDataFrame

# load a tiff using OME Arrow
oa_img = OMEArrow(
    "../../../tests/data/cytotable/JUMP_plate_BR00117006/images/orig/r01c01f01p01-ch2sk1fk1fl1.tiff"
)
oa_img

# + colab={"base_uri": "https://localhost:8080/", "height": 423} id="wkCpuZEBosNV" outputId="4adee883-9707-40a4-92ab-2ee748bbcf76"
# make a "slice" (crop) from the overall image
oa_img_slice = oa_img.slice(x_min=240, x_max=310, y_min=360, y_max=430)
oa_img_slice

# + colab={"base_uri": "https://localhost:8080/"} id="ZZ8LWIuQqQqz" outputId="f82de1df-3240-4dba-d0a0-6a4ce973f299"
# show the OME Arrow struct
oa_img_slice.data

# + colab={"base_uri": "https://localhost:8080/", "height": 1000} id="zcaDHeNorxCt" outputId="e64c061b-c56e-462b-dfe0-c73de76df0d3"
# create a pyarrow table for writing the data
table = pa.table(
    {
        # two columns, two values each, both int64 type
        "metadata_1": pa.array([0, 2], type=pa.int64()),
        "feature_1": pa.array([1, 2], type=pa.int64()),
        # add an image column with repeated entries based on the image slice
        "image": pa.repeat(oa_img_slice.data, 2),
    }
)

# write a parquet table
pq.write_table(table, "example.ome.parquet")

# read the parquet table with cytodataframe
# (showing the OME-Arrow image that was written)
CytoDataFrame("example.ome.parquet")
