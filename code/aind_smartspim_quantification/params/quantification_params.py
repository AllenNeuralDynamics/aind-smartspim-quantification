""" Parameters used in the quantification script """
import os
from pathlib import Path

import yaml
from argschema import ArgSchema
from argschema.fields import Int, Str


class QuantificationParams(ArgSchema):
    """
    Quantification parameters
    """

    fused_folder = Str(
        required=True,
        metadata={"description": "Path where the data is located"},
    )

    ccf_registration_folder = Str(
        required=True,
        metadata={"description": "Path where the ccf registered data is located"},
    )

    cell_segmentation_folder = Str(
        required=True,
        metadata={"description": "Path where the cell segmentation data is located"},
    )

    channel_name = Str(
        required=True,
        metadata={"description": "Dataset's channel name"},
    )

    stitched_s3_path = Str(
        required=True,
        metadata={
            "description": "Path where the stitched data is located in the cloud"
        },
    )

    save_path = Str(
        required=False,
        metadata={"description": "Folder where we want to output files"},
        dump_default="../results/",
    )

    downsample_res = Int(
        required=False,
        metadata={"description": "Zarr scale that was used in CCF step"},
        dump_default=3,
    )

    reference_microns_ccf = Int(
        required=False,
        metadata={"description": "Reference resolution in um used in CCF step"},
        dump_default=25,
    )

    bucket_path = Str(
        required=True,
        metadata={"description": "Amazon Bucket or Google Bucket name"},
    )


def get_yaml_config(filename: str) -> dict:
    """
    Get default configuration from a YAML file.
    Parameters
    ------------------------
    filename: str
        String where the YAML file is located.
    Returns
    ------------------------
    Dict
        Dictionary with the configuration
    """

    filename = Path(os.path.dirname(__file__)).joinpath(filename)

    config = None
    try:
        with open(filename, "r") as stream:
            config = yaml.safe_load(stream)
    except Exception as error:
        print(error)

    return config
