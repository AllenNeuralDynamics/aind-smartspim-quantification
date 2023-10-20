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
from glob import glob

import ants
import pims

from imlib.cells.cells import Cell
from imlib.IO.cells import get_cells, save_cells
from tqdm import tqdm

from .utils import utils, logging_utils, generate_25_um_ccf_cells

from ._shared import PathLike


def read_transform(reg_path: PathLike) -> tuple:
    """
    Imports ants transformation from registration output

    Parameters
    -------------
    seg_path: PathLike
        Path to .gz file from registration

    Returns
    -------------
    ants.transform
        affine transform nonlinear warp field from ants.registration()
    """
    affine_file = os.path.abspath(os.path.join(reg_path, "affine_transforms.mat"))
    affine = ants.read_transform(affine_file)
    affinetx = affine.invert()

    warp_file = os.path.abspath(os.path.join(reg_path, "ls_ccf_warp_transforms.nii.gz"))

    # To make it compatible with the older CCF version
    if not os.path.exists(warp_file):
        warp_file = os.path.abspath(os.path.join(reg_path, "warp_transforms.nii.gz"))

    warp = ants.image_read(warp_file)
    warptx = ants.transform_from_displacement_field(warp)

    return affinetx, warptx


def read_xml(seg_path: PathLike, reg_dims: list, ds: int) -> list:
    """
    Imports cell locations from segmentation output

    Parameters
    -------------
    seg_path: PathLike
        Path where the .xml file is located
    reg_dims: list
        Resolution (pixels) of the image used for segmentation. Orientation [x (ML), z (DV), y (AP)]
    ds: int
        factor by which image for registration was downsampled from input_dims

    Returns
    -------------
    list
        List with cell locations as tuples (x (ML), y (AP), z (DV))
    """

    cell_file = glob(os.path.join(seg_path, "*.xml"))[0]
    file_cells = get_cells(cell_file)

    cells = []

    for cell in file_cells:
        cells.append((cell.x / ds, cell.z / ds, reg_dims[2] - (cell.y / ds)))

    return cells

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
    cell_transformed: list, 
    save_path: PathLike,
    logger: logging.Logger
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
        logger: logging.Logger
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
            data_folder, ""
        ),  # Path where the 25 um zarr image is stored, output from CCF capsule
        "output_ng_link": smartspim_config["save_path"],
    }

    logger.info("Generating precomputed formats and visualization link")
    neuroglancer_link = generate_25_um_ccf_cells(params)
    json_state = neuroglancer_link.state

    # Updating json to visualize data on S3
    dataset_path = smartspim_config["stitched_s3_path"]
    process_output_filename = f"image_cell_quantification/{smartspim_config['channel_name']}/visualization/neuroglancer_config.json"

    json_state[
        "ng_link"
    ] = f"https://aind-neuroglancer-sauujisjxq-uw.a.run.app#!{dataset_path}/{process_output_filename}"
    logger.info(f"Neuroglancer link: {json_state['ng_link']}")
    # Updating s3 paths of layers

    # Updating S3 registered brain to future S3 path
    ccf_registered_s3_path = f"zarr://{dataset_path}/image_atlas_alignment/{smartspim_config['channel_name']}/OMEZarr/image.zarr"
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
    logger: logging.Logger
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
    ds = 2 ** downsample_res
    reg_dims = [dim / ds for dim in input_res]

    logger.info(f"Downsample res: {ds}, reg dims: {reg_dims}")

    # Getting cell locations and ccf transformations
    raw_cells = read_xml(detected_cells_xml_path, reg_dims, ds)
    affinetx, warptx = read_transform(ccf_transforms_path)

    # Getting transformed res which is the original image in the 3rd multiscale
    # rescaled to the 25 um resolution equal to the Allen CCF Atlas
    transformed_res_path = f"{ccf_transforms_path}/metadata/downsampled_16.tiff"
    transform_res = None

    with pims.open(transformed_res_path) as imgs:
        transform_res = [
            imgs.frame_shape[-1],
            imgs.frame_shape[-2],
            len(imgs),
        ]  # output is in [ML, DV, AP] which is the same as the input array

    logger.info(f"Image shape of transformed image: {transform_res}")
    scale = [raw / trans for raw, trans in zip(transform_res, reg_dims)]
    logger.info(f"Scale: {scale}")

    logger.info("Processing cell location using registration transform")
    cells_transformed = []

    for cell in tqdm(raw_cells, total=len(raw_cells)):
        scaled_cell = [dim * scale for dim, scale in zip(cell, scale)]
        affine_pt = affinetx.apply_to_point(scaled_cell)
        warp_pt = warptx.apply_to_point(affine_pt)
        cells_transformed.append(warp_pt)

    # Writing CSV
    transformed_cells_path = write_transformed_cells(cells_transformed,
                                                     save_path,
                                                     logger)

    logger.info("Calculating cell counts per brain region and generating CSV")

    # Getting annotation map and meshes path
    ccf_dir = os.path.dirname(os.path.realpath(__file__))
    count = utils.CellCounts(ccf_dir, reference_microns_ccf)
    count_df = count.create_counts(cells_transformed)

    fname = "cell_count_by_region.csv"
    csv_path = os.path.join(save_path, fname)
    count_df.to_csv(csv_path)

    return csv_path, transformed_cells_path


def main(
    data_folder: PathLike,
    output_quantified_folder: PathLike,
    intermediate_quantified_folder: PathLike,
    smartspim_config: dict
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

    metadata_path_res = f"{smartspim_config['fused_folder']}/{smartspim_config['channel_name']}.zarr/0/.zarray"
    input_res = utils.read_json_as_dict(metadata_path_res)["shape"]
    
    # input res is returned in order tczyx, here we use xzy
    smartspim_config['input_params']['input_res'] = [input_res[-1], input_res[-3], input_res[-2]]
    
    metadata_folder = output_quantified_folder.joinpath("metadata")
    
    # Logger pointing everything to the metadata path
    logger = logging_utils.create_logger(output_log_path=metadata_folder)
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

    # Calculate cell counts per region
    csv_path, transformed_cells_path = cell_quantification(logger = logger,
                                                           **smartspim_config['input_params']
                                                           )
    
    # Create visualization folders
    ccf_cells_precomputed, cells_precomputed = create_visualization_folders(smartspim_config['save_path'])
    
    
    # Generate neuroglancer links
    generate_neuroglancer_link(
        data_folder = data_folder,
        csv_path = csv_path,
        transformed_cells_path = transformed_cells_path,
        ccf_cells_precomputed_output = ccf_cells_precomputed,
        cells_precomputed_output = cells_precomputed,
        smartspim_config = smartspim_config,
        logger = logger
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
