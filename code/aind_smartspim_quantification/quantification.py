#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 15:55:37 2023

@author: nicholas.lusk
@modified by: camilo.laiton
"""

import copy
import json
import logging
import multiprocessing
import os
import re
import time
import xmltodict
from glob import glob
from pathlib import Path

import ants
import boto3
import numpy as np
import pandas as pd
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
            cells.append((cell.z / ds, cell.y / ds, cell.x / ds))
        elif orient == "sal":
            cells.append(
                (
                    cell.z / ds,
                    cell.y / ds, 
                    reg_dims[2] - (cell.x / ds)
                )
            )
        elif orient == "rpi":
            cells.append(
                (
                    reg_dims[0] - (cell.z / ds),
                    reg_dims[1] - (cell.y / ds),
                    cell.x / ds,
                )
            )

    return cells

def read_aws_xml(seg_path: PathLike, reg_dims: list, ds: int, orient: str, institute: str) -> list:
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
    orient: str
        the orientation the brain was imaged
    insititute: str
        the institution that imaged the dataset

    Returns
    -------------
    list
        List with cell locations as tuples (x (ML), y (AP), z (DV))
    """
    print(f"seg_path being used is: {seg_path}")
    client = boto3.client('s3')

    res = client.get_object(
        Bucket='aind-open-data',
        Key=seg_path + '/detected_cells.xml'
    )
    xml_file = res['Body'].read()
    file_cells = xmltodict.parse(xml_file)['CellCounter_Marker_File']['Marker_Data']['Marker_Type']['Marker']

    cells = []

    for cell in file_cells:
        # spl is not a real orientation. but a bug from early acquisition script
        if orient == 'spr':
            cells.append(
                (    
                    int(cell['MarkerZ']) / ds,
                    reg_dims[1] - (int(cell['MarkerY']) / ds),
                    reg_dims[2] - (int(cell['MarkerX']) / ds)
                )
            )
        elif orient == 'spl' and institute == 'AIBS':
            cells.append(
                (
                    int(cell["MarkerZ"]) / ds,
                    reg_dims[1] - (int(cell["MarkerY"]) / ds),
                    int(cell["MarkerX"]) / ds,
                )
            )
        elif orient == 'spl' and institute == 'AIND':
            cells.append(
                (
                    int(cell['MarkerZ']) / ds, 
                    int(cell['MarkerY']) / ds, 
                    int(cell['MarkerX']) / ds
                )
            )
        elif orient == 'sal':
            cells.append(
                (
                    int(cell['MarkerZ']) / ds, 
                    int(cell['MarkerY']) / ds, 
                    int(cell['MarkerX']) / ds
                )
            )
        elif orient == 'rpi':
            cells.append(
                (
                    int(cell['MarkerZ']) / ds,
                    reg_dims[1] - (int(cell['MarkerY']) / ds),
                    reg_dims[2] - (int(cell['MarkerX']) / ds)
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


def apply_transforms_to_points(ants_pts: np.ndarray, transforms: list, invert: tuple) -> np.ndarray:
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
        3, df, transforms, whichtoinvert=invert
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
    mode: str,
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

    reference_microns_ccf: int
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

    mode: str
        the condition this script is being run: Options ["detect", "reprocess"]

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
    if 'detect' in mode:
        raw_cells = read_xml(
            detected_cells_xml_path, reg_dims, ds, orient, institute_abbreviation
        )
    elif 'reprocess' in mode:
        raw_cells = read_aws_xml(
            detected_cells_xml_path, reg_dims, ds, orient, institute_abbreviation
        )

    # This is beacuse of a bug in registration
    if orient == 'rpi':
        scaling = scaling[::-1]

    scaled_cells = scale_cells(raw_cells, scaling)
    template_params = utils.get_template_info(image_files["smartspim_template"])

    logger.info(
        f"Reorient cells from {orient} to template {template_params['orientation']} "
    )
    _, swapped, _ = utils.get_orientation_transform(orient, template_params["orientation"])
    orient_cells = np.array(scaled_cells)[:, swapped]

    logger.info("Converting oriented cells into ANTs physical space")
    template_params = utils.get_template_info(image_files["smartspim_template"])
    ants_cells = convert_to_ants_space(template_params, orient_cells)

    logger.info("Registering Cells to SmartSPIM template")
    template_cells = apply_transforms_to_points(
        ants_cells, 
        template_transforms,
        invert = (False, True)
    )

    logger.info("Convert template cells into CCF space and orientation")
    ccf_pts = apply_transforms_to_points(
        template_cells,
        ccf_transforms,
        invert = (False, True)
    )

    logger.info("Conver cells back into index space")
    ccf_params = utils.get_template_info(image_files["ccf_template"])
    ccf_cells = convert_from_ants_space(ccf_params, ccf_pts)

    _, swapped, _ = utils.get_orientation_transform(
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

def quantification_metrics(
        region_list: list,
        reference_microns_ccf: int,
        reverse_transforms: list,
        image_files: dict,
        orientation: list,
        reverse_scaling: list,
        image_path: PathLike,
        registered_path: PathLike
) -> pd.DataFrame:
    """
    Reverse transform ccf regions and get volume and intensity metrics 
    

    Parameters
    ----------
    region_list : list
        list of ccf ids for regions that you want to get quantification
        metrics
        
    reference_microns_ccf: int
        Integer that indicates to which um space the
        downsample image was taken to. Default 25 um.
        
    reverse_transforms: dict
        Pathways for reverse transfrom data assets for ccf_to_template and
        template_to_ls ordered [Warp, Affine]
        
    image_files: dict
        Pathways to the nifti files for the smartspim template and ccf
    
    reverse_scaling: list
        List of scaling from 25um to downsampled 3
    
    orientation: list
        Info on the orientation that the brain was
        aquired during imaging
    
    image_path: str
        The location of the stitched zarr
        
    registered_path: str
        Location to the registered zarr

    Returns
    -------
    metric_df: pd.dataframe
        metrics for regions reverse transforms and nmi

    """
    ccf_dir = os.path.dirname(os.path.realpath(__file__))
    count = utils.CellCounts(ccf_dir, reference_microns_ccf)
    region_info = count.get_metric_region_info(region_list)
    
    img = utils.__read_zarr_image(image_path)
    img = np.array(img).squeeze()
    
    registered_img = utils.__read_zarr_image(registered_path)
    registered_img = np.array(registered_img).squeeze()
    
    ccf_img = ants.image_read(image_files["ccf_template"]).numpy()
    
    metrics = []
    for region in region_list:
        verts, faces = count.get_CCF_mesh_points(region)
        
        if region_info[region][1] == 'hemi':
            vertices_right = copy.copy(verts)
            vertices_right[:, 0] = (
                vertices_right[:, 0] + (5700 - vertices_right[:, 0]) * 2
            )
        
            verts = np.vstack((verts, vertices_right))
    
        #Get the mesh oriented in the same direction as the ccf
        scaled_verts = verts / reference_microns_ccf
        oriented_verts = scaled_verts[:, [0, 2, 1]]
        
        mask = np.zeros(shape = ccf_img.shape, dtype = np.int8)
        mask = utils.get_intensity_mask(
            scaled_verts[:, [2, 1, 0]],
            faces,
            mask, 
            split = region_info[region][1]
        )
        
        norm_mutual_info = utils.normalized_mutual_information(
            ccf_img, 
            registered_img,
            mask
        )
        
        #Transform to template
        ccf_params = utils.get_template_info(image_files["ccf_template"])
        ants_verts = convert_to_ants_space(ccf_params, oriented_verts)
        template_verts = apply_transforms_to_points(
            ants_verts,
            reverse_transforms['ccf_transforms'],
            invert = (False, False)
        )
        
        #Transform to lightsheet
        template_params = utils.get_template_info(image_files["smartspim_template"])
        ls_verts = apply_transforms_to_points(
            template_verts,
            reverse_transforms['template_transforms'],
            invert = (False, False)
        )
        converted_verts = convert_from_ants_space(template_params, ls_verts)
        
        # convert to orientation of the zarr image
        orient = utils.get_orientation(orientation)
        _, swapped, mat = utils.get_orientation_transform(template_params['orientation'], orient)
        converted_verts  = converted_verts [:, swapped]

        # this is also because of the bug in registration
        if orient == 'rpi':
            reverse_scaling = reverse_scaling[::-1]

        out_verts = scale_cells(converted_verts , reverse_scaling)
        
        # get metrics
        volume = utils.get_volume(
            out_verts,
            faces, 
            region_info[region][1]
        )

        img_oriented = utils.orient_image(img, mat)
        
        mask = np.zeros(shape = img_oriented.shape, dtype = np.int8)
        mask = utils.get_intensity_mask(out_verts, faces, mask, region_info[region][1])
        intensity = utils.get_region_intensity(img_oriented, mask)

        metrics.append(
            [
                region_info[region][0],
                region,
                volume,
                np.sum(intensity),
                norm_mutual_info
            ]
        )
    
    cols = [
        'Acronym',
        'Region_ID',
        'Volume',
        'Total_Intensity',
        "Normalized_Mutual_Info"
    ]
    
    metric_df = pd.DataFrame(metrics, columns=cols)
    
    return metric_df

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

    image_path = os.path.abspath(
        f"{smartspim_config['fused_folder']}/{smartspim_config['registration_channel']}.zarr/3/"
    )
    
    registered_zarr = os.path.abspath(
        f'{smartspim_config["input_params"]["ccf_transforms_path"]}/OMEZarr/image.zarr/0/'   
    )
    
    metric_params = {
        'region_list': smartspim_config['region_list'],
        'reference_microns_ccf': smartspim_config['input_params']['reference_microns_ccf'],
        'reverse_transforms': smartspim_config['reverse_transforms'],
        'image_files': smartspim_config["input_params"]['image_files'],
        'orientation': smartspim_config['input_params']['orientation'],
        'reverse_scaling': smartspim_config['reverse_scaling'],
        'image_path': image_path,
        'registered_path': registered_zarr
    }
    
    metrics = quantification_metrics(**metric_params)
    
    metric_path = os.path.abspath(
        f'{smartspim_config["save_path"]}/region_metrics.csv'
    )
    
    metrics.to_csv(metric_path)

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
