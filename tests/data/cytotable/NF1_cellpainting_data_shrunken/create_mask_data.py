"""
Creates image masks for images in
coSMicQC/tests/data/cytotable/NF1_cellpainting_data (Plate 2)

Note: expects Docker to be installed as a CLI on the system.

This file may be processed using the following command from the root
of the project repository:
`uv run python \
tests/data/cytotable/NF1_cellpainting_data_shrunken/create_mask_data.py`
"""

import os
import pathlib
import subprocess

# create a dir for segmentation masks
pathlib.Path("tests/data/cytotable/NF1_cellpainting_data_shrunken/Plate_2_masks").mkdir(
    exist_ok=True
)

# define docker command for CellProfiler use with provided pipeline file
command = [
    "docker",
    "run",
    "--platform",
    "linux/amd64",
    "--rm",
    "-v",
    f"{os.getcwd()}/tests/data/cytotable/NF1_cellpainting_data_shrunken:/app",
    "cellprofiler/cellprofiler:4.2.4",
    "cellprofiler",
    "-c",
    "-r",
    "-p",
    "/app/NF1_plate2_export_masks.cppipe",
    "-o",
    "/app/Plate_2_masks",
    "-i",
    "/app/Plate_2_images",
]

# Run the command and show output
try:
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    print("Command Output:\n", result.stdout)
    print("Command Error:\n", result.stderr)
except subprocess.CalledProcessError as e:
    print("Error:", e)
    print("Command Output:\n", e.stdout)
    print("Command Error:\n", e.stderr)
