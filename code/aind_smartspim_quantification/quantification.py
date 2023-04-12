#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 15:55:37 2023

@author: nicholas.lusk
@modified by: camilo.laiton
"""

import logging
import os
from glob import glob
from pathlib import Path
from typing import Union

import ants
import numpy as np
import pims
import yaml
from argschema import ArgSchema, ArgSchemaParser, InputFile
from argschema.fields import Int, Str
from imlib.cells.cells import Cell
from imlib.IO.cells import get_cells, save_cells
from tqdm import tqdm

from .quantification_params import QuantificationParams
from .utils import CellCounts, read_json_as_dict

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
        List with cell locations as tuples (x, y, z)
    """

    cell_file = glob(os.path.join(seg_path, "*.xml"))[0]
    file_cells = get_cells(cell_file)

    cells = []

    for cell in file_cells:
        cells.append((cell.x / ds, cell.z / ds, reg_dims[2] - (cell.y / ds)))

    return cells


def write_transformed_cells(cell_transformed: list, save_path: PathLike):
    """
    Save transformed cell coordinates to xml for napari compatability

    Parameters
    -------------
    cell_transformed: list
        list of Cell objects with transformed cell locations

    Returns
    -------------
    xml
        writes an .xml file with registered locations
    """

    cells = []

    logger.info("Saving transformed cell locations to XML")
    for coord in tqdm(cell_transformed, total=len(cell_transformed)):
        coord = [dim if dim > 1 else 1.0 for dim in coord]
        coord_dict = {"x": coord[0], "y": coord[1], "z": coord[2]}
        cells.append(Cell(coord_dict, "cell"))

    save_cells(cells, os.path.join(save_path, "transformed_cells.xml"))


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
    """

    # Getting downsample res
    ds = 2**downsample_res
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
            imgs.frame_shape[1],
            len(imgs),
            imgs.frame_shape[0],
        ]  # ZYX -> XZY
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
    write_transformed_cells(cells_transformed, save_path)

    logger.info("Calculating cell counts per brain region and generating CSV")

    # Getting annotation map and meshes path
    ccf_dir = os.path.dirname(os.path.realpath(__file__))
    count = CellCounts(ccf_dir, reference_microns_ccf)
    count_df = count.create_counts(cells_transformed)

    fname = "cell_count_by_region.csv"
    count_df.to_csv(os.path.join(save_path, fname))


def main():
    mod = ArgSchemaParser(schema_type=QuantificationParams)
    args = mod.args

    dataset_path = args[
        "dataset_path"
    ]  # "/data/SmartSPIM_656374_2023-01-27_12-41-55_stitched_2023-01-31_17-28-34"
    channel_name = args["channel_name"]  # "Ex_445_Em_469"
    intermediate_folder = args["intermediate_folder"]  # "processed/OMEZarr"
    downsample_res = args["downsample_res"]  # 3
    reference_microns_ccf = args["reference_microns_ccf"]  # 25

    metadata_path_res = (
        f"{dataset_path}/{intermediate_folder}/{channel_name}.zarr/0/.zarray"
    )
    input_res = read_json_as_dict(metadata_path_res)["shape"]

    # input res is returned in order tczyx, here we use xzy
    input_res = [input_res[-1], input_res[-3], input_res[-2]]

    input_params = {
        "input_res": input_res,  # x z y
        "detected_cells_xml_path": f"{dataset_path}/processed/Cell_Segmentation/{channel_name}/",
        "ccf_transforms_path": f"{dataset_path}/processed/CCF_Atlas_Registration/{channel_name}/",
        "save_path": args["save_path"],
        "downsample_res": args["downsample_res"],
        "reference_microns_ccf": args["reference_microns_ccf"],
    }

    run(**input_params)


if __name__ == "__main__":
    main()
