#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 15:55:37 2023

@author: nicholas.lusk
@modified by: camilo.laiton
"""

import json
import logging
import multiprocessing
import os
import re
import time
from glob import glob
from pathlib import Path

import ants
import numpy as np
import pandas as pd
import pims
from aind_data_schema.core.processing import DataProcess, ProcessName
from imlib.cells.cells import Cell
from imlib.IO.cells import get_cells, save_cells
from tqdm import tqdm

from ._shared.types import PathLike
from .utils import utils
from .utils.generate_ccf_cell_count import generate_25_um_ccf_cells


def read_xml(
    seg_path: PathLike, reg_dims: list, ds: int, orient: str, institute: str
) -> list:
    """
    Imports cell locations from segmentation output

    Parameters
    -------------
    seg_path: PathLike
        Path where the .xml file is located
    reg_dims: list
        Resolution (pixels) of the image used for segmentation. ordered
        relative to zarr
    ds: int
        factor by which image for registration was downsampled from input_dims
    orient: str
        the orientation the brain was imaged
    insititute: str
        the institution that imaged the dataset

    Returns
    -------------
    list
        List with cell locations as tuples. orientation needs to be relative
        to zarr for proper scaling
    """

    cell_file = glob(os.path.join(seg_path, "classified_*.xml"))[0]
    file_cells = get_cells(cell_file)

    cells = []

    for cell in file_cells:
        if orient == "spr":
            cells.append(
                (
                    cell.z / ds,
                    reg_dims[1] - (cell.y / ds),
                    reg_dims[2] - (cell.x / ds),
                )
            )
        elif orient == "spl" and institute == "AIBS":
            cells.append(
                (
                    cell.z / ds,
                    reg_dims[1] - (cell.y / ds),
                    cell.x / ds,
                )
            )
        elif orient == "spl" and institute == "AIND":
            cells.append((cell.z / ds, cell.x / ds, cell.y / ds))
        elif orient == "sal":
            cells.append((cell.z / ds, cell.x / ds, cell.y / ds))
        elif orient == "rpi":
            cells.append(
                (
                    reg_dims[0] - (cell.z / ds),
                    reg_dims[1] - (cell.y / ds),
                    reg_dims[2] - (cell.x / ds),
                )
            )

    return cells


def scale_cells(cells, scale):
    """
    Takes the downsampled cells, scales and orients them in smartspim template
    space.

    Parameters
    ----------
    cells : list
        list of cell locations that has been downsampled

    scale : list
        the scaling metric between the raw image being downsampled to level 3
        and the image after being placed into 25um state space for

    Returns
    -------
    scaled_cells: np.ndarray
        list of scaled cells that have been scaled

    """

    scaled_cells = []
    for cell in cells:
        scaled_cells.append(
            (cell[0] * scale[0], cell[1] * scale[1], cell[2] * scale[2])
        )

    return np.array(scaled_cells)


def convert_to_ants_space(template_parameters: dict, cells: np.ndarray):
    """
    Convert points from "index" space and places them into the physical space
    required for applying ants transforms for a given ANTsImage

    Parameters
    ----------
    template_parameters : dict
        parameters of the ANTsImage physical space that you are converting
        the points
    cells : np.ndarray
        the location of cells in index space that have been oriented to the
        ANTs image that you are converting into

    Returns
    -------
    ants_pts : np.ndarray
        pts converted into ANTsPy physical space

    """

    ants_pts = cells.copy()

    for dim in range(template_parameters["dims"]):
        ants_pts[:, dim] *= template_parameters["scale"][dim]
        ants_pts[:, dim] *= template_parameters["direction"][dim]
        ants_pts[:, dim] += template_parameters["origin"][dim]

    return ants_pts


def convert_from_ants_space(template_parameters: dict, cells: np.ndarray):
    """
    Convert points from the physical space of an ANTsImage and places
    them into the "index" space required for visualizing

    Parameters
    ----------
    template_parameters : dict
        parameters of the ANTsImage physical space from where you are
        converting the points
    cells : np.ndarray
        the location of cells in physical space

    Returns
    -------
    pts : np.ndarray
        pts converted for ANTsPy physical space to "index" space

    """

    pts = cells.copy()

    for dim in range(template_parameters["dims"]):
        pts[:, dim] -= template_parameters["origin"][dim]
        pts[:, dim] *= template_parameters["direction"][dim]
        pts[:, dim] /= template_parameters["scale"][dim]

    return pts


def apply_transforms_to_points(ants_pts: np.ndarray, transforms: list) -> np.ndarray:
    """
    Takes the cell locations that have been converted into the correct
    physical space needed for the provided transforms and registers the points

    Parameters
    ----------
    ants_pts: np.ndarray
        array with cell locations placed into ants physical space
    transforms: list
        list of the file locations for the transformations

    Returns
    -------
    transformed_pts
        list of point locations in CCF state space

    """

    df = pd.DataFrame(ants_pts, columns=["x", "y", "z"])
    transformed_pts = ants.apply_transforms_to_points(
        3, df, transforms, whichtoinvert=(False, True)
    )

    return np.array(transformed_pts)


def create_visualization_folders(save_path: PathLike):
    """
    Creates visualization folder structure

    Parameters
    ----------
    save_path : PathLike
        save path from smartspim_config dict

    Returns
    -------
    ccf_cells_precomputed_output : Path
        path to where the precomputed data for ccf registered
        segmentation
    cells_precomputed_output : Path
        path to where the precomputed data for raw registered
        cells will be saved

    """

    utils.create_folder(f"{save_path}/visualization")

    ccf_cells_precomputed_output = os.path.join(
        save_path, "visualization/ccf_cell_precomputed"
    )
    cells_precomputed_output = os.path.join(
        save_path, "visualization/cell_points_precomputed"
    )

    # Creating folders

    utils.create_folder(ccf_cells_precomputed_output)
    utils.create_folder(cells_precomputed_output)

    return ccf_cells_precomputed_output, cells_precomputed_output


def write_transformed_cells(
    cell_transformed: list, save_path: PathLike, logger: logging.Logger
) -> str:
    """
    Save transformed cell coordinates to xml for napari compatability

    Parameters
    -------------
    cell_transformed: list
        list of Cell objects with transformed cell locations

    save_path: PathLike
        save path from smartspim_config dict

    Returns
    -------------
    str
        Path to the generated xml
    """

    cells = []

    logger.info("Saving transformed cell locations to XML")
    for coord in tqdm(cell_transformed, total=len(cell_transformed)):
        coord = [dim if dim > 1 else 1.0 for dim in coord]
        coord_dict = {"x": coord[0], "y": coord[1], "z": coord[2]}
        cells.append(Cell(coord_dict, "cell"))

    transformed_cells_path = os.path.join(save_path, "transformed_cells.xml")
    save_cells(cells, transformed_cells_path)
    return transformed_cells_path


def generate_neuroglancer_link(
    data_folder: PathLike,
    csv_path: PathLike,
    transformed_cells_path: PathLike,
    ccf_cells_precomputed_output: PathLike,
    cells_precomputed_output: PathLike,
    smartspim_config: dict,
    logger: logging.Logger,
):
    """
    Creates and saves neuroglancer link for ccf registered annotations
    and segmentation

    Parameters
    ----------
    data_folder : PathLike
        location of input data to the capsule
    csv_path : PathLike
        location of ccf reference file with region acronyms and
        ID pairings
    transformed_cells_path : PathLike
        location of .xml file that has cells transformed into
        ccf space
    ccf_cells_precomputed_output : PathLike
        location to save the precomputed ccf segmenation layer
    cells_precomputed_output : PathLike
        location to save the precomputed annotation layer
    smartspim_config : dict
        parameterizations from capsules
    logger : logging.Logger
        logging object

    Returns
    -------
    None.

    """

    # neuroglancer link visualization
    params = {
        "ccf_cells_precomputed": {  # Parameters to generate CCF + Cells precomputed format
            "input_path": csv_path,  # Path where the cell_count.csv is located
            "output_path": ccf_cells_precomputed_output,  # Path where we want to save the CCF + cell location precomputed
            "ccf_reference_path": None,  # Path where the CCF reference csv is located, set None to get from tissuecyte
        },
        "cells_precomputed": {  # Parameters to generate cell points precomputed format
            "xml_path": transformed_cells_path,  # Path where the cell points are located
            "output_precomputed": cells_precomputed_output,  # Path where the precomputed format will be stored
        },
        "zarr_path": f"{smartspim_config['ccf_registration_folder']}/OMEZarr/image.zarr".replace(
            str(data_folder), ""
        ),  # Path where the 25 um zarr image is stored, output from CCF capsule
        "output_ng_link": smartspim_config["save_path"],
    }

    logger.info("Generating precomputed formats and visualization link")
    neuroglancer_link = generate_25_um_ccf_cells(params)
    json_state = neuroglancer_link.state

    # Updating json to visualize data on S3
    process_output_filename = f"image_cell_quantification/{smartspim_config['channel_name']}/visualization/neuroglancer_config.json"
    dataset_path = smartspim_config["stitched_s3_path"]

    json_state[
        "ng_link"
    ] = f"https://aind-neuroglancer-sauujisjxq-uw.a.run.app#!{dataset_path}/{process_output_filename}"
    logger.info(f"Neuroglancer link: {json_state['ng_link']}")
    # Updating s3 paths of layers

    # Updating S3 registered brain to future S3 path
    # Getting registration channel name
    ccf_reg_channel_name = re.search(
        "(Ex_[0-9]*_Em_[0-9]*)", smartspim_config["input_params"]["ccf_transforms_path"]
    ).group()

    ccf_registered_s3_path = f"zarr://{dataset_path}/image_atlas_alignment/{ccf_reg_channel_name}/OMEZarr/image.zarr"
    json_state["layers"][0]["source"] = ccf_registered_s3_path

    # Updating S3 cell points to future S3 path
    cell_points_s3_path = f"precomputed://{dataset_path}/image_cell_quantification/{smartspim_config['channel_name']}/visualization/cell_points_precomputed"
    json_state["layers"][1]["source"] = cell_points_s3_path

    # Updating CCF + cells to future S3 Path
    ccf_cells_s3_path = f"precomputed://{dataset_path}/image_cell_quantification/{smartspim_config['channel_name']}/visualization/ccf_cell_precomputed"
    json_state["layers"][2]["source"] = ccf_cells_s3_path

    with open(
        f"{smartspim_config['save_path']}/visualization/neuroglancer_config.json", "w"
    ) as outfile:
        json.dump(json_state, outfile, indent=2)


def cell_quantification(
    input_res: list,
    detected_cells_xml_path: PathLike,
    ccf_transforms_path: PathLike,
    save_path: PathLike,
    downsample_res: int,
    reference_microns_ccf: int,
    institute_abbreviation: str,
    orientation: list,
    scaling: list,
    template_transforms: list,
    ccf_transforms: list,
    image_files: dict,
    logger: logging.Logger,
):
    """
    Runs quantification of registered cells

    Parameters
    --------------
    input_res: list
        Original image resolution in XZY order

    detected_cells_xml_path: PathLike
        Path to the folder where the cell segmentation
        output the XML(s)

    ccf_transforms_path: PathLike
        Path to the folder where the CCF capsule
        output the transformations for the channel

    save_path: PathLike
        Path where we want to save the CSV with the
        cell counts per region

    downsample_res: int
        Integer that indicates the downsample resolution
        that was used in the CCF alignment. Default 3

    reference_microns_ccf_int: int
        Integer that indicates to which um space the
        downsample image was taken to. Default 25 um.

    institute_abbreviation: str
        Institution abbreviation

    orientation: list
        Info on the orientation that the brain was
        aquired during imaging

    scaling: list
        The scaling between resolution of the image and that of
        the 25um templates

    template_transforms: list
        Pathways to the registration transforms from the registration
        capsule ordered [InverseWarp, Affine]

    ccf_transforms: list
        Pathways to ccf transforms data asset ordered [InverseWarp, Affine]

    image_files: dict
        Pathways to the nifti files for the smartspim template and ccf

    logger: logging.Logger
        logging object

    Returns
    ----------
    csv_path: PathLike
        Path to the generated csv with cell counts per
        CCF region
    transformed_cells_path: PathLike
        Path to the points in CCF space
    """
    logger.info(f"input image resolution is {input_res}, and this is considered XZY")

    # Getting downsample res
    ds = 2**downsample_res
    reg_dims = [dim / ds for dim in input_res]

    logger.info(f"Downsample res: {ds}, reg dims: {reg_dims}")

    # Getting cell locations and ccf transformations
    orient = utils.get_orientation(orientation)
    raw_cells = read_xml(
        detected_cells_xml_path, reg_dims, ds, orient, institute_abbreviation
    )

    scaled_cells = scale_cells(raw_cells, scaling)
    template_params = utils.get_template_info(image_files["smartspim_template"])

    logger.info(
        f"Reorient cells from {orient} to template {template_params['orientation']} "
    )
    _, swapped = utils.get_orientation_transform(orient, template_params["orientation"])
    orient_cells = scaled_cells[:, swapped]

    logger.info("Converting oriented cells into ANTs physical space")
    template_params = utils.get_template_info(image_files["smartspim_template"])
    ants_cells = convert_to_ants_space(template_params, orient_cells)

    logger.info("Registering Cells to SmartSPIM template")
    template_cells = apply_transforms_to_points(ants_cells, template_transforms)

    logger.info("Convert template cells into CCF space and orientation")
    ccf_pts = apply_transforms_to_points(template_cells, ccf_transforms)

    logger.info("Conver cells back into index space")
    ccf_params = utils.get_template_info(image_files["ccf_template"])
    ccf_cells = convert_from_ants_space(ccf_params, ccf_pts)

    _, swapped = utils.get_orientation_transform(
        template_params["orientation"], ccf_params["orientation"]
    )
    cells_transformed = ccf_cells[:, swapped]

    # Getting annotation map and meshes path
    ccf_dir = os.path.dirname(os.path.realpath(__file__))
    count = utils.CellCounts(ccf_dir, reference_microns_ccf)

    # removing cells that are outside the brain
    cells_array = np.array(cells_transformed) * reference_microns_ccf
    cells_cropped = count.crop_cells(cells_array) / reference_microns_ccf

    transformed_cropped = []
    for cell in cells_cropped:
        transformed_cropped.append(cell)

    # Writing CSV
    transformed_cells_path = write_transformed_cells(
        transformed_cropped, save_path, logger
    )

    logger.info("Calculating cell counts per brain region and generating CSV")

    # count cells
    count_df = count.create_counts(transformed_cropped)

    fname = "cell_count_by_region.csv"
    csv_path = os.path.join(save_path, fname)
    count_df.to_csv(csv_path)

    return csv_path, transformed_cells_path


def main(
    data_folder: PathLike,
    output_quantified_folder: PathLike,
    intermediate_quantified_folder: PathLike,
    smartspim_config: dict,
):
    """
    This function quantifies detected cells

    Parameters
    -----------
    data_folder: PathLike
        Path where the image data is located

    output_quantified_path: PathLike
        Path where the OMEZarr and metadata will
        live after fusion

    intermediate_quantified_folder: PathLike
        Path where the intermediate files
        will live. These will not be in the final
        folder structure. e.g., 3D fused chunks
        from TeraStitcher

    smartspim_config: dict
        Dictionary with the smartspim configuration
        for that dataset

    """
    data_processes = []
    metadata_path_res = f"{smartspim_config['fused_folder']}/{smartspim_config['channel_name']}.zarr/0/.zarray"
    input_res = utils.read_json_as_dict(metadata_path_res)["shape"]

    # input res is returned in order tczyx, here we use xzy
    # where z is the imaging axis
    smartspim_config["input_params"]["input_res"] = [
        input_res[-3],
        input_res[-2],
        input_res[-1],
    ]

    metadata_folder = Path(smartspim_config["save_path"]).joinpath("metadata")

    # Logger pointing everything to the metadata path
    utils.create_folder(dest_dir=str(metadata_folder))
    utils.create_folder(dest_dir=smartspim_config["save_path"])
    logger = utils.create_logger(output_log_path=metadata_folder)
    utils.print_system_information(logger)

    # Tracking compute resources
    # Subprocess to track used resources
    manager = multiprocessing.Manager()
    time_points = manager.list()
    cpu_percentages = manager.list()
    memory_usages = manager.list()

    profile_process = multiprocessing.Process(
        target=utils.profile_resources,
        args=(
            time_points,
            cpu_percentages,
            memory_usages,
            20,
        ),
    )
    profile_process.daemon = True
    profile_process.start()

    start_time = time.time()
    # Calculate cell counts per region
    csv_path, transformed_cells_path = cell_quantification(
        logger=logger,
        save_path=smartspim_config["save_path"],
        institute_abbreviation=smartspim_config["institute_abbreviation"],
        **smartspim_config["input_params"],
    )

    # Create visualization folders
    ccf_cells_precomputed, cells_precomputed = create_visualization_folders(
        smartspim_config["save_path"]
    )

    # Generate neuroglancer links
    generate_neuroglancer_link(
        data_folder=data_folder,
        csv_path=csv_path,
        transformed_cells_path=transformed_cells_path,
        ccf_cells_precomputed_output=ccf_cells_precomputed,
        cells_precomputed_output=cells_precomputed,
        smartspim_config=smartspim_config,
        logger=logger,
    )

    end_time = time.time()

    data_processes.append(
        DataProcess(
            name=ProcessName.IMAGE_CELL_QUANTIFICATION,
            software_version="1.5.0",
            start_date_time=start_time,
            end_date_time=end_time,
            input_location=f"{smartspim_config['fused_folder']}/{smartspim_config['channel_name']}.zarr/0",
            output_location=str(output_quantified_folder),
            outputs={"output_folder": str(output_quantified_folder)},
            code_url="https://github.com/AllenNeuralDynamics/aind-smartspim-quantification",
            code_version="1.5.0",
            parameters=smartspim_config,
            notes="The output folder contains the precomputed format to visualize and count cells per CCF region",
        )
    )

    utils.generate_processing(
        data_processes=data_processes,
        dest_processing=metadata_folder,
        processor_full_name="Nicholas Lusk",
        pipeline_version="1.5.0",
    )

    # Getting tracked resources and plotting image
    utils.stop_child_process(profile_process)

    if len(time_points):
        utils.generate_resources_graphs(
            time_points,
            cpu_percentages,
            memory_usages,
            metadata_folder,
            "smartspim_quantification",
        )


if __name__ == "__main__":
    main()
