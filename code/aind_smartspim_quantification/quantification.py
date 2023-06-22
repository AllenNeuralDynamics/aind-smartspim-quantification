#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 15:55:37 2023

@author: nicholas.lusk
@modified by: camilo.laiton
"""

import json
import logging
import os
from glob import glob
from pathlib import Path
from typing import Union

import ants
import numpy as np
import pims
import yaml
from argschema import ArgSchemaParser
from imlib.cells.cells import Cell
from imlib.IO.cells import get_cells, save_cells
from tqdm import tqdm

from .generate_ccf_cell_count import generate_25_um_ccf_cells
from .quantification_params import QuantificationParams
from .utils import CellCounts, create_folder, read_json_as_dict

PathLike = Union[str, Path]

LOG_FMT = "%(asctime)s %(message)s"
LOG_DATE_FMT = "%Y-%m-%d %H:%M"

logging.basicConfig(format=LOG_FMT, datefmt=LOG_DATE_FMT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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
        logger.error(error)

    return config


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
    affine_file = glob(os.path.join(reg_path, "*.mat"))[0]
    affine = ants.read_transform(affine_file)
    affinetx = affine.invert()

    warp_file = glob(os.path.join(reg_path, "*.gz"))[0]
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
    input_dims: list
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


def write_transformed_cells(cell_transformed: list, save_path: PathLike) -> str:
    """
    Save transformed cell coordinates to xml for napari compatability

    Parameters
    -------------
    cell_transformed: list
        list of Cell objects with transformed cell locations

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


def run(
    input_res: list,
    detected_cells_xml_path: PathLike,
    ccf_transforms_path: PathLike,
    save_path: PathLike,
    downsample_res: int = 3,
    reference_microns_ccf: int = 25,
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

        transform_res_dtype = np.dtype(imgs.pixel_type)

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
    transformed_cells_path = write_transformed_cells(cells_transformed, save_path)

    logger.info("Calculating cell counts per brain region and generating CSV")

    # Getting annotation map and meshes path
    ccf_dir = os.path.dirname(os.path.realpath(__file__))
    count = CellCounts(ccf_dir, reference_microns_ccf)
    count_df = count.create_counts(cells_transformed)

    fname = "cell_count_by_region.csv"
    csv_path = os.path.join(save_path, fname)
    count_df.to_csv(csv_path)

    return csv_path, transformed_cells_path


def main(input_data: dict):
    """
    Main function to run quantification in a single channel

    Parameters
    -------------
    input_data: dict
        Parameters that the quantification capsule needs.
        These are built using the pipeline configuration.

    """

    mod = ArgSchemaParser(schema_type=QuantificationParams, input_data=input_data)
    args = mod.args

    dataset_path = os.path.abspath(args["fused_folder"])

    ccf_folder = os.path.abspath(args["ccf_registration_folder"])

    cell_folder = os.path.abspath(args["cell_segmentation_folder"])

    channel_name = args["channel_name"]  # "Ex_445_Em_469"
    downsample_res = args["downsample_res"]  # 3
    reference_microns_ccf = args["reference_microns_ccf"]  # 25
    args["save_path"] = os.path.abspath(args["save_path"])
    data_folder_path = os.path.abspath("../data")
    results_folder_path = os.path.abspath(f"../results")

    metadata_path_res = f"{dataset_path}/{channel_name}.zarr/0/.zarray"

    input_res = read_json_as_dict(metadata_path_res)["shape"]

    # input res is returned in order tczyx, here we use xzy
    input_res = [input_res[-1], input_res[-3], input_res[-2]]

    input_params = {
        "input_res": input_res,  # [x (ML), y (DV), z(AP)]
        "detected_cells_xml_path": f"{cell_folder}/",
        "ccf_transforms_path": f"{ccf_folder}/",
        "save_path": args["save_path"],
        "downsample_res": args["downsample_res"],
        "reference_microns_ccf": args["reference_microns_ccf"],
    }

    create_folder(f"{args['save_path']}/visualization")

    csv_path, transformed_cells_path = run(**input_params)

    ccf_cells_precomputed_output = os.path.join(
        args["save_path"], "visualization/ccf_cell_precomputed"
    )
    cells_precomputed_output = os.path.join(
        args["save_path"], "visualization/cell_points_precomputed"
    )

    # Creating folders
    create_folder(ccf_cells_precomputed_output)
    create_folder(cells_precomputed_output)

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
        "zarr_path": f"{ccf_folder}/OMEZarr/image.zarr".replace(
            data_folder_path, ""
        ),  # Path where the 25 um zarr image is stored, output from CCF capsule
        "output_ng_link": args["save_path"],
    }

    logger.info("Generating precomputed formats and visualization link")
    neuroglancer_link = generate_25_um_ccf_cells(params)
    json_state = neuroglancer_link.state

    # Updating json to visualize data on S3
    dataset_path = args["stitched_s3_path"]
    process_output_filename = f"processed/Quantification/{channel_name}/visualization/neuroglancer_config.json"

    json_state[
        "ng_link"
    ] = f"https://aind-neuroglancer-sauujisjxq-uw.a.run.app#!{dataset_path}/{process_output_filename}"
    logger.info(f"Neuroglancer link: {json_state['ng_link']}")
    # Updating s3 paths of layers

    # Updating S3 cell points to future S3 path
    cell_points_s3_path = f"{dataset_path}/processed/Quantification/{channel_name}/visualization/cell_points_precomputed"
    json_state["layers"][1]["source"] = cell_points_s3_path

    # Updating CCF + cells to future S3 Path
    ccf_cells_s3_path = f"{dataset_path}/processed/Quantification/{channel_name}/visualization/ccf_cell_precomputed"
    json_state["layers"][2]["source"] = ccf_cells_s3_path

    with open(
        f"{args['save_path']}/visualization/neuroglancer_config.json", "w"
    ) as outfile:
        json.dump(json_state, outfile, indent=2)


if __name__ == "__main__":
    main()
