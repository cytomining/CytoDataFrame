"""
Tests project integration with other packages
"""

import subprocess

import pandas as pd
import pytest

cosmicqc = pytest.importorskip(
    "cosmicqc",
    reason="project integration tests require cosmicqc to be installed",
)


def test_cosmicqc_find_outliers_cfret(cytotable_CFReT_data_df: pd.DataFrame):
    """
    Testing cosmicqc.find_outliers with CytoTable CFReT data.
    """

    # metadata columns to include in output data frame
    metadata_columns = [
        "Image_Metadata_Plate",
        "Image_Metadata_Well",
        "Image_Metadata_Site",
    ]

    # Set a negative threshold to identify both outlier small nuclei
    # and low formfactor representing non-circular segmentations.
    feature_thresholds = {
        "Nuclei_AreaShape_Area": -1,
        "Nuclei_AreaShape_FormFactor": -1,
    }

    # run function to identify outliers given conditions
    small_area_formfactor_outliers_df = cosmicqc.analyze.find_outliers(
        df=cytotable_CFReT_data_df,
        feature_thresholds=feature_thresholds,
        metadata_columns=metadata_columns,
    )

    # test that we we can receive a dict by from a chained operation
    # on the cytodataframe.
    assert (
        type(
            small_area_formfactor_outliers_df.sort_values(
                list(feature_thresholds)
            ).to_dict(orient="dict")
        )
        is dict
    )


def test_cosmicqc_find_outliers_cfret_cli():
    """
    Testing cosmicqc.find_outliers with CytoTable CFReT data
    and using the CLI.
    """

    # metadata columns to include in output data frame
    metadata_columns = [
        "Image_Metadata_Plate",
        "Image_Metadata_Well",
        "Image_Metadata_Site",
    ]

    # Set a negative threshold to identify both outlier small nuclei
    # and low formfactor representing non-circular segmentations.
    feature_thresholds = {
        "Nuclei_AreaShape_Area": -1,
        "Nuclei_AreaShape_FormFactor": -1,
    }

    # run function to identify outliers given conditions
    result = subprocess.run(
        [
            "cosmicqc",
            "find_outliers",
            "tests/data/cytotable/CFRet_data/test_localhost231120090001_converted.parquet",
            str(metadata_columns),
            str(feature_thresholds),
        ],
        capture_output=True,
        check=False,
    )

    # check that we don't have None and we do have something
    # that resembles a dataframe object in the tui
    assert result.stdout is not None
    assert all(
        check_str in str(result.stdout)
        for check_str in ["Nuclei_AreaShape_Area", "Nuclei_AreaShape_FormFactor"]
    )
