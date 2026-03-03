"""
Creates and shrunken dataset for testing puproses
based on coSMicQC/tests/data/cytotable/NF1_cellpainting_data (Plate 2)

This file may be processed using the following command from the root
of the project repository:
`uv run python tests/data/cytotable/NF1_cellpainting_data_shrunken/create_image_data.py`
"""

import pathlib
import shutil

import pandas as pd

source_data_path = "tests/data/cytotable/NF1_cellpainting_data/"
target_data_path = "tests/data/cytotable/NF1_cellpainting_data_shrunken/"
source_image_data_path = source_data_path + "Plate_2_images"
source_parquet_path = source_data_path + "Plate_2_with_image_data.parquet"
target_image_data_path = target_data_path + "Plate_2_images"
target_parquet_path = target_data_path + "Plate_2_with_image_data_shrunken.parquet"

# create target image dir
pathlib.Path(target_image_data_path).mkdir(exist_ok=True)

# read source data
source_df = pd.read_parquet(source_parquet_path)

# get a sample of 5 from the source data
sampled_df = source_df.sample(n=5)

# send the sampled df to parquet file
sampled_df.to_parquet(target_parquet_path)


def check_and_copy_file(filename: str):
    """
    Checks for files in target dir and copies them if they don't already exist
    """
    source_path = f"{source_image_data_path}/{filename}"
    target_path = f"{target_image_data_path}/{filename}"

    # Check if the file already exists in the target directory
    if not pathlib.Path(target_path).is_file():
        # Copy the file if it doesn't exist
        shutil.copy(source_path, target_path)
        return f"Copied {filename} to {target_image_data_path}"
    else:
        return f"{filename} already exists in {target_image_data_path}"


# apply the file copy and collect status information
sampled_df["Image_FileName_DAPI_status"] = sampled_df["Image_FileName_DAPI"].apply(
    check_and_copy_file
)
sampled_df["Image_FileName_GFP_status"] = sampled_df["Image_FileName_GFP"].apply(
    check_and_copy_file
)
sampled_df["Image_FileName_RFP_status"] = sampled_df["Image_FileName_RFP"].apply(
    check_and_copy_file
)

sampled_df

# show the results using a mask on the dataframe for status
print(
    sampled_df[
        [
            "Image_FileName_DAPI_status",
            "Image_FileName_GFP_status",
            "Image_FileName_RFP_status",
        ]
    ]
)
