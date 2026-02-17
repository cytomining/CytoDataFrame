# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: cytodataframe-shAZamSV-py3.12
#     language: python
#     name: python3
# ---

# + colab={"base_uri": "https://localhost:8080/", "height": 423} id="Y5e85t05nSLt" outputId="a0807c7c-4705-41a2-83b2-9fb7287c40f9"
from ome_arrow import OMEArrow

# load a tiff using OME Arrow
oa_img = OMEArrow("../../../tests/data/cytotable/JUMP_plate_BR00117006/images/orig/r01c01f01p01-ch2sk1fk1fl1.tiff")
oa_img

# + colab={"base_uri": "https://localhost:8080/", "height": 423} id="wkCpuZEBosNV" outputId="4adee883-9707-40a4-92ab-2ee748bbcf76"
# make a slice from the overall image
oa_img_slice = oa_img.slice(x_min=240,x_max=310,y_min=360,y_max=430)
oa_img_slice

# + colab={"base_uri": "https://localhost:8080/"} id="ZZ8LWIuQqQqz" outputId="f82de1df-3240-4dba-d0a0-6a4ce973f299"
# show the OME Arrow struct
oa_img_slice.data

# + colab={"base_uri": "https://localhost:8080/", "height": 1000, "referenced_widgets": ["2d299fff8e7d44d49adf62524c1063ba", "88a9126a2faf4bf0b80b20c3c7571e9a", "40f54afe0c3f4fe38a6b9ccd9a832414", "85bc1dacf14847ad84aa106c915cb360", "b4cfe791b5b24a8191f1dd020e4ca4ae", "f85e6ac6514b4d6f98fbdc7d6696a692", "aa2b037ccf0948db8be086b0e9d3b77a"]} id="zcaDHeNorxCt" outputId="e64c061b-c56e-462b-dfe0-c73de76df0d3"
import pyarrow as pa
import pyarrow.parquet as pq
from cytodataframe import CytoDataFrame

# create a pyarrow table for writing the data
table = pa.table({
    "metadata_1": pa.array([0, 2], type=pa.int64()),
    "feature_1": pa.array([1, 2], type=pa.int64()),
    # add an image column with repeated entries based on the image slice
    "image": pa.repeat(oa_img_slice.data, 2),
})

# write a parquet table
pq.write_table(table, "example.ome.parquet")

# read the parquet table with cytodataframe
CytoDataFrame("example.ome.parquet")
