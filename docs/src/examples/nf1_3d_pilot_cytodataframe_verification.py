# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: CytoDataFrame (3.13.5.final.0)
#     language: python
#     name: python3
# ---

# # NF1 3D pilot cropped voxel view
#
# View real 3D single-cell crops (with mask overlays) from two NF1 pilot wells in one `CytoDataFrame`.
#
# Most of the setup below isn't about `CytoDataFrame` -- it's working around this
# warehouse's data layout, which doesn't look like typical CellProfiler output:
#
# - There are no `Image_FileName_*`/`Image_PathName_*` columns to begin with --
#   raw image and mask paths have to be looked up from a separate metadata
#   table per image set (`load_image_set`).
# - Paths are recorded for the cluster filesystem (`/pl/active/koala/...`), not
#   this machine's mount, so every path needs remapping (`resolve_koala_path`).
# - Every well's mask file has the exact same generic name (`nuclei_mask.tiff`),
#   so combining two wells into one table needs `stage_mask` to tell them apart
#   -- a single-well notebook wouldn't need this at all.
#
# The actual `CytoDataFrame(...)` call at the end is just a handful of arguments.

# +
import os
import pathlib
import re
import tempfile

import pandas as pd

from cytodataframe import CytoDataFrame

# Local mount point for the koala cluster storage. This is machine-specific --
# e.g. bandicoot is a separate storage location from this machine's mount, and
# koala data (including this pilot's warehouse) may or may not be duplicated
# there. Rather than guess a path here, set the KOALA_LOCAL_BASE environment
# variable to wherever koala is reachable on your machine before running this
# notebook; the assertion below fails clearly (rather than a cryptic parquet
# read error) if it's wrong.
KOALA_LOCAL_BASE = pathlib.Path(
    os.environ.get("KOALA_LOCAL_BASE", "~/mnt/alpine/active/koala")
).expanduser()
KOALA_CLUSTER_PREFIX = "/pl/active/koala"
RESULT_DIR = "nf0055-nf0014-post-revert-20260821T150143Z"
# A "warehouse" is the structured parquet output of one nf1-3d-pilot-workflow-db
# run (one such directory per RESULT_DIR/run), holding the `ibp/` profile
# tables and `images/` metadata this notebook reads from below.
WAREHOUSE_DIR = (
    KOALA_LOCAL_BASE / f"nf1-3d-pilot-workflow-db/results/{RESULT_DIR}/warehouse"
)
assert WAREHOUSE_DIR.is_dir(), (
    f"Warehouse not found at {WAREHOUSE_DIR}. Set the KOALA_LOCAL_BASE "
    "environment variable to wherever koala's cluster storage is mounted "
    "on this machine (this pilot's data may need to be synced/duplicated "
    "here first if it isn't already)."
)

# An "image set" here is one well + field-of-view (one 3D image). Add every
# image set you want to view to this dict -- e.g. all wells/fields in a
# plate for per-plate QC -- not just the two shown here.
IMAGE_SETS = {
    "NF0055_T1__B10__F1": "NF0055_T1__NF0055_T1__B10__F1",
    "NF0014_T1__C4__F2": "NF0014_T1__NF0014_T1__C4__F2",
}
CHANNEL = "DNA"
COMPARTMENT = "Nuclei"

# Scratch dir for the mask symlinks `stage_mask` creates below (see its
# docstring) -- not part of the repo, just local working space.
MASK_LINK_DIR = pathlib.Path(tempfile.gettempdir()) / "cytodataframe_nf1_3d_mask_links"
MASK_LINK_DIR.mkdir(exist_ok=True)


def resolve_koala_path(source_uri: str) -> pathlib.Path:
    # Warehouse metadata records cluster paths, not this machine's mount.
    if source_uri.startswith(KOALA_CLUSTER_PREFIX):
        return KOALA_LOCAL_BASE / source_uri[len(KOALA_CLUSTER_PREFIX) :].lstrip("/")
    return pathlib.Path(source_uri)


def load_image_set(
    image_set: str, channel: str, compartment: str
) -> tuple[pd.DataFrame, pathlib.Path, pathlib.Path]:
    profiles = pd.read_parquet(
        WAREHOUSE_DIR / "ibp/sc_profiles_related" / f"{image_set}.parquet"
    ).head(1)
    # No Image_FileName_/PathName_ columns on the profile table -- raw image
    # and mask paths live in this separate per-image-set metadata table.
    assets = pd.read_parquet(
        WAREHOUSE_DIR / "images/image_assets" / f"{image_set}.parquet"
    )
    channel_uri = assets.loc[
        (assets["Metadata_ImageAsset_AssetType"] == "raw_image")
        & (assets["Metadata_ImageAsset_Channel"] == channel),
        "Metadata_ImageAsset_SourceURI",
    ].iloc[0]
    mask_uri = assets.loc[
        (assets["Metadata_ImageAsset_AssetType"] == "segmentation_mask")
        & (assets["Metadata_ImageAsset_Compartment"] == compartment),
        "Metadata_ImageAsset_SourceURI",
    ].iloc[0]
    channel_path = resolve_koala_path(channel_uri)
    mask_path = resolve_koala_path(mask_uri)
    profiles[f"Image_FileName_{channel}"] = str(channel_path)
    return profiles, channel_path, mask_path


def stage_mask(channel_path: pathlib.Path, mask_path: pathlib.Path) -> pathlib.Path:
    # Every well's mask has the same generic filename (e.g. "nuclei_mask.tiff"),
    # so symlink under a name that embeds the raw stem to tell rows apart.
    link = MASK_LINK_DIR / f"{channel_path.stem}__{mask_path.name}"
    if not link.exists():
        link.symlink_to(mask_path)
    return link


# -

# Load one object per well into a single two-row table.

# +
profile_rows = []
for image_set in IMAGE_SETS.values():
    profiles, channel_path, mask_path = load_image_set(image_set, CHANNEL, COMPARTMENT)
    stage_mask(channel_path, mask_path)
    profile_rows.append(profiles)

profiles = pd.concat(profile_rows, ignore_index=True)
mask_name = mask_path.name

# +
bbox_column_map = {
    "x_min": f"{COMPARTMENT}_NoChannel_VolumeSizeShape_MinX",
    "x_max": f"{COMPARTMENT}_NoChannel_VolumeSizeShape_MaxX",
    "y_min": f"{COMPARTMENT}_NoChannel_VolumeSizeShape_MinY",
    "y_max": f"{COMPARTMENT}_NoChannel_VolumeSizeShape_MaxY",
    "z_min": f"{COMPARTMENT}_NoChannel_VolumeSizeShape_MinZ",
    "z_max": f"{COMPARTMENT}_NoChannel_VolumeSizeShape_MaxZ",
}

voxel_view = CytoDataFrame(
    data=profiles[[f"Image_FileName_{CHANNEL}"]],
    data_bounding_box=profiles[list(bbox_column_map.values())],
    data_mask_context_dir=str(MASK_LINK_DIR),
    segmentation_file_regex={rf"__{re.escape(mask_name)}$": r"_\d+\.tif$"},
    display_options={
        "width": 260,
        "height": 260,
        "table_max_height": "580px",
        "label_overlay_mode": "filled",
        "volume_bbox_column_map": bbox_column_map,
    },
)
# The rendered widget includes a "Mask" checkbox to toggle the overlay on/off.
voxel_view
