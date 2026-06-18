"""
Main file to execute the smartspim segmentation
in code ocean
"""

import logging
import os
import sys
import time
from glob import glob
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import zarr
from log_schema import setup_logging

from aind_smartspim_quantification import (__pipeline_name__, __title__,
                                            __version__, quantification)
from aind_smartspim_quantification.params.quantification_params import \
    get_yaml_config
from aind_smartspim_quantification.utils import utils

logger = logging.getLogger(__name__)


def get_data_config(
    data_folder: str,
    processing_manifest_path: str = "segmentation_processing_manifest*",
    data_description_path: str = "data_description.json",
) -> Tuple:
    """
    Returns the first smartspim dataset found
    in the data folder

    Parameters
    -----------
    data_folder: str
        Path to the folder that contains the data

    processing_manifest_path: str
        Path for the processing manifest

    data_description_path: str
        Path for the data description

    Returns
    -----------
    Tuple[Dict, str]
        Dict: Empty dictionary if the path does not exist,
        dictionary with the data otherwise.

        Str: Empty string if the processing manifest
        was not found
    """

    # Returning first smartspim dataset found
    # Doing this because of Code Ocean, ideally we would have
    # a single dataset in the pipeline

    derivatives_dict = utils.read_json_as_dict(
        glob(f"{data_folder}/{processing_manifest_path}")[0]
    )
    data_description_dict = utils.read_json_as_dict(
        f"{data_folder}/{data_description_path}"
    )

    smartspim_dataset = data_description_dict["name"]
    institution_abbreviation = data_description_dict["institution"]["abbreviation"]

    return derivatives_dict, smartspim_dataset, institution_abbreviation


def set_up_pipeline_parameters(
    pipeline_config: dict, default_config: dict, smartspim_dataset_name: str
):
    """
    Sets up smartspim stitching parameters that come from the
    pipeline configuration

    Parameters
    -----------
    smartspim_dataset: str
        String with the smartspim dataset name

    pipeline_config: dict
        Dictionary that comes with the parameters
        for the pipeline described in the
        processing_manifest.json

    default_config: dict
        Dictionary that has all the default
        parameters to execute this capsule with
        smartspim data

    smartspim_dataset_name: str
        Smartspim dataset name for the s3 path.

    Returns
    -----------
    Dict
        Dictionary with the combined parameters
    """

    default_config["fused_folder"] = os.path.abspath(
        f"{pipeline_config['quantification']['fused_folder']}"
    )

    # Added to handle registration testing
    s3_path = pipeline_config["stitching"].get(
        "s3_path", f"s3://aind-open-data/{smartspim_dataset_name}"
    )

    if "test" in s3_path:
        s3_seg_path = s3_path.replace("test", "stitched")
    else:
        s3_seg_path = s3_path

    default_config["stitched_s3_path"] = s3_path
    default_config["registration_channel"] = pipeline_config["stitching"]["channel"]
    default_config["channel_name"] = pipeline_config["quantification"]["channel"]
    default_config["save_path"] = os.path.abspath(
        f"{pipeline_config['quantification']['save_path']}/quant_{pipeline_config['quantification']['channel']}"
    )

    if default_config["input_params"]["mode"] == "detect":
        default_config["input_params"][
            "detected_cells_csv_path"
        ] = f"{default_config['cell_segmentation_folder']}/"
    elif default_config["input_params"]["mode"] == "reprocess":
        default_config["input_params"]["detected_cells_csv_path"] = (
            s3_seg_path.split("/")[-1]
            + "/"
            + default_config["cell_segmentation_folder"]
        )

    default_config["input_params"][
        "ccf_transforms_path"
    ] = f"{default_config['ccf_registration_folder']}/"

    return default_config


def validate_capsule_inputs(input_elements: List[str]) -> List[str]:
    """
    Validates input elemts for a capsule in
    Code Ocean.

    Parameters
    -----------
    input_elements: List[str]
        Input elements for the capsule. This
        could be sets of files or folders.

    Returns
    -----------
    List[str]
        List of missing files
    """

    missing_inputs = []
    for required_input_element in input_elements:
        required_input_element = Path(required_input_element)

        if not required_input_element.exists():
            missing_inputs.append(str(required_input_element))

    return missing_inputs


def get_estimated_downsample(
    voxel_resolution: List[float], registration_res: Tuple[float] = (16.0, 14.4, 14.4)
) -> int:
    """
    Get the estimated multiscale based on the provided
    voxel resolution. This is used for image stitching.

    e.g., if the original resolution is (1.8. 1.8, 2.0)
    in XYZ order, and you provide (3.6, 3.6, 4.0) as
    image resolution, then the picked resolution will be
    1.

    Parameters
    ----------
    voxel_resolution: List[float]
        Image original resolution. This would be the resolution
        in the multiscale "0".
    registration_res: Tuple[float]
        Approximated resolution that was used for registration
        in the computation of the transforms. Default: (16.0, 14.4, 14.4)
    """

    downsample_versions = []
    for idx in range(len(voxel_resolution)):
        downsample_versions.append(
            registration_res[idx] // float(voxel_resolution[idx])
        )

    downsample_res = int(min(downsample_versions))
    return round(np.log2(downsample_res))


def run():
    """
    Main function to execute the smartspim quantification
    in code ocean
    """

    # Absolute paths of common Code Ocean folders
    data_folder = os.path.abspath("../data")
    results_folder = os.path.abspath("../results")
    scratch_folder = os.path.abspath("../scratch")

    mode = str(sys.argv[1:])
    mode = mode.replace("[", "").replace("]", "").casefold()

    process_name = f"{__title__}-{mode}"
    setup_logging(
        model={
            "pipeline_name": __pipeline_name__,
            "process_name": process_name,
            "software_name": __title__,
            "software_version": __version__,
        }
    )
    start_time = time.monotonic()

    # It is assumed that these files
    # will be in the data folder
    required_input_elements = []

    missing_files = validate_capsule_inputs(required_input_elements)

    if len(missing_files):
        raise ValueError(
            f"We miss the following files in the capsule input: {missing_files}"
        )

    pipeline_config, smartspim_dataset_name, institute_abbreviation = get_data_config(
        data_folder=data_folder
    )

    quantification_info = pipeline_config.get("quantification")

    logger.info(
        "SmartSPIM quantification stage started",
        extra={
            "event_type": "stage_start",
            "dataset_name": smartspim_dataset_name,
            "data_folder": data_folder,
            "results_folder": results_folder,
            "scratch_folder": scratch_folder,
            "mode": mode,
        },
    )

    try:
        _run_quantification(
            mode=mode,
            data_folder=data_folder,
            results_folder=results_folder,
            scratch_folder=scratch_folder,
            pipeline_config=pipeline_config,
            quantification_info=quantification_info,
            smartspim_dataset_name=smartspim_dataset_name,
            institute_abbreviation=institute_abbreviation,
        )
    except Exception:
        duration_seconds = round(time.monotonic() - start_time, 3)
        logger.error(
            "SmartSPIM quantification stage failed",
            exc_info=True,
            extra={
                "event_type": "stage_failure",
                "dataset_name": smartspim_dataset_name,
                "duration_seconds": duration_seconds,
            },
        )
        raise

    duration_seconds = round(time.monotonic() - start_time, 3)
    logger.info(
        "SmartSPIM quantification stage completed",
        extra={
            "event_type": "stage_complete",
            "dataset_name": smartspim_dataset_name,
            "duration_seconds": duration_seconds,
        },
    )


def _run_quantification(
    mode: str,
    data_folder: str,
    results_folder: str,
    scratch_folder: str,
    pipeline_config: dict,
    quantification_info: Optional[dict],
    smartspim_dataset_name: str,
    institute_abbreviation: str,
):
    """
    Runs the smartspim quantification processing body.

    Parameters
    -----------
    mode: str
        Capsule run mode, either "detect" or "reprocess"

    data_folder: str
        Absolute path to the Code Ocean data folder

    results_folder: str
        Absolute path to the Code Ocean results folder

    scratch_folder: str
        Absolute path to the Code Ocean scratch folder

    pipeline_config: dict
        Pipeline configuration coming from the processing manifest

    quantification_info: Optional[dict]
        Quantification section of the pipeline configuration,
        None if this dataset has no quantification channels

    smartspim_dataset_name: str
        Name of the smartspim dataset

    institute_abbreviation: str
        Institution abbreviation for the dataset
    """

    if quantification_info is not None:
        logger.debug("Pipeline config: %s", pipeline_config)
        logger.debug("Data folder contents: %s", os.listdir(data_folder))

        # get default configs
        default_config = get_yaml_config(
            os.path.abspath(
                "aind_smartspim_quantification/params/default_quantify_configs.yaml"
            )
        )

        ccf_folder = glob(
            f"{data_folder}/ccf_{pipeline_config['quantification']['channel']}"
        )

        if len(ccf_folder):
            ccf_folder = ccf_folder[0]

        # add paths to default_config
        default_config["ccf_registration_folder"] = os.path.abspath(ccf_folder)

        # add mode information
        if "detect" in mode:
            default_config["cell_segmentation_folder"] = os.path.abspath(
                f"{data_folder}/cell_{pipeline_config['quantification']['channel']}"
            )
            default_config["input_params"]["mode"] = "detect"
        elif "reprocess" in mode:
            default_config["cell_segmentation_folder"] = (
                f"image_cell_segmentation/{pipeline_config['quantification']['channel']}"
            )
            default_config["input_params"]["mode"] = "reprocess"
        else:
            raise NotImplementedError(f"The mode {mode} has not been implemented")

        # add paths to ls_to_template transforms
        default_config["input_params"]["template_transforms"] = [
            os.path.abspath(
                glob(f"{data_folder}/ccf_*/ls_to_template_SyN_0GenericAffine.mat")[0]
            ),
            os.path.abspath(
                glob(f"{data_folder}/ccf_*/ls_to_template_SyN_1InverseWarp.nii.gz")[0]
            ),
        ]

        # add paths to template_to_ccf transforms
        default_config["input_params"]["ccf_transforms"] = [
            os.path.abspath(
                f"{data_folder}/lightsheet_template_ccf_registration/spim_template_to_ccf_syn_0GenericAffine_25.mat"
            ),
            os.path.abspath(
                f"{data_folder}/lightsheet_template_ccf_registration/spim_template_to_ccf_syn_1InverseWarp_25.nii.gz"
            ),
        ]

        # add paths for reverse transforms for calculating metrics
        default_config["reverse_transforms"] = {
            "template_transforms": [
                os.path.abspath(
                    glob(f"{data_folder}/ccf_*/ls_to_template_SyN_1Warp.nii.gz")[0]
                ),
                os.path.abspath(
                    glob(f"{data_folder}/ccf_*/ls_to_template_SyN_0GenericAffine.mat")[
                        0
                    ]
                ),
            ],
            "ccf_transforms": [
                os.path.abspath(
                    f"{data_folder}/lightsheet_template_ccf_registration/spim_template_to_ccf_syn_1Warp_25.nii.gz"
                ),
                os.path.abspath(
                    f"{data_folder}/lightsheet_template_ccf_registration/spim_template_to_ccf_syn_0GenericAffine_25.mat"
                ),
            ],
        }

        # add paths to the nifti files of the template and ccf
        default_config["input_params"]["image_files"] = {
            "ccf_template": os.path.abspath(
                f"{data_folder}/lightsheet_template_ccf_registration/ccf_average_template_25.nii.gz"
            ),
            "smartspim_template": os.path.abspath(
                f"{data_folder}/lightsheet_template_ccf_registration/smartspim_lca_template_25.nii.gz"
            ),
        }

        # add orientation information to default_config
        acquisition_path = os.path.abspath(f"{data_folder}/acquisition.json")
        acquisition_configs = utils.read_json_as_dict(acquisition_path)
        ccf_res_microns = 25

        default_config["ng_config"] = {
            "base_url": "https://neuroglancer-demo.appspot.com/#!",
            "crossSectionScale": 1,
            "projectionScale": 512,
            "orientation": acquisition_configs,
            "dimensions": {
                "z": [ccf_res_microns * 10**-6, "m"],
                "y": [ccf_res_microns * 10**-6, "m"],
                "x": [ccf_res_microns * 10**-6, "m"],
                "t": [0.001, "s"],
            },
            "rank": 3,
            "gpuMemoryLimit": 1500000000,
        }

        logger.debug("Pipeline config: %s", pipeline_config)
        logger.debug("Data folder contents: %s", os.listdir(data_folder))

        # combine configs
        smartspim_config = set_up_pipeline_parameters(
            pipeline_config=pipeline_config,
            default_config=default_config,
            smartspim_dataset_name=smartspim_dataset_name,
        )

        smartspim_config["name"] = smartspim_dataset_name
        smartspim_config["institute_abbreviation"] = institute_abbreviation
        smartspim_config["input_params"]["orientation"] = acquisition_configs["axes"]

        # get zarr resolution
        zarr_attrs_path = f"{smartspim_config['fused_folder']}/{smartspim_config['channel_name']}.zarr/.zattrs"
        zarr_attrs = utils.read_json_as_dict(zarr_attrs_path)
        acquisition_res = zarr_attrs["multiscales"][0]["datasets"][0][
            "coordinateTransformations"
        ][0]["scale"][2:]
        reg_scale = get_estimated_downsample(acquisition_res)
        reg_res = [float(res) * 2**reg_scale for res in acquisition_res]

        smartspim_config["input_params"]["downsample_res"] = reg_scale
        smartspim_config["input_params"]["scaling"] = [
            res / ccf_res_microns for res in reg_res
        ]
        smartspim_config["reverse_scaling"] = [ccf_res_microns / res for res in reg_res]

        quantification.main(
            data_folder=Path(data_folder),
            output_quantified_folder=Path(results_folder),
            intermediate_quantified_folder=Path(scratch_folder),
            smartspim_config=smartspim_config,
        )

    else:
        logger.info("No quantification channels, pipeline config: %s", pipeline_config)
        utils.save_dict_as_json(
            filename=f"{results_folder}/segmentation_processing_manifest_empty.json",
            dictionary=pipeline_config,
        )


if __name__ == "__main__":
    run()
