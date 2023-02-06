#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 15:55:37 2023

@author: nicholas.lusk
"""

import os
import ants
import yaml
import utils
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from glob import glob
from tqdm import tqdm
from pathlib import Path
from typing import Union

from imlib.cells.cells import Cell
from imlib.IO.cells import get_cells, save_cells

from argschema.fields import Int, Str
from argschema import ArgSchema, ArgSchemaParser, InputFile


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

class QuantSchema(ArgSchema):
    """
    Schema format for Cell Quantification
    """

    config_file = InputFile(
        required=True,
        metadata={"description": "Path to the YAML config file."},
        dump_default="smartspim_config.yaml",
    )

    input_segmentation = Str(
        metadata={
            "required": True,
            "description": "Path to the .xml file output from segmentation capsule",
        }
    )
    
    input_registration = Str(
        metadata={
            "required": True,
            "description": "Path to the .tiff and .gz files from registration capsule",
        }
    )

    downsample = Int(
        metadata={
            "required": True, 
            "description": "Level which the image was downsampled for registration",
        },
        dump_default=3
    )

    input_res = Int(
        metadata={
            "required": True, 
            "description": "Resolution of original SmartSPIM image",
        },
        dump_default=[7400, 4200, 10240]
    )

    transform_res = Int(
        metadata={
            "required": True,
            "description": "Resolution of \"downsampled_16.tiff\" output from registration",
        },

    )

    save_path = Str(
        metadata={
            "required": True,
            "description": "Location to save output files",
        }
    )

class quantify(ArgSchemaParser):
    """
    Class for quantifying lightsheet cell data
    """
    
    default_params = QuantSchema
    

    def __read_xml(self, seg_path: PathLike, reg_dims: list, ds: int) -> list:
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
            cells.append((cell.x / ds, 
                          cell.z / ds, 
                          reg_dims[2] - (cell.y / ds)))
        
        return cells
    
    def __read_transform(self, reg_path: PathLike) -> tuple:
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
        affine_file = glob(os.path.join(reg_path, '*.mat'))[0]
        affine = ants.read_transform(affine_file)
        affinetx = affine.invert()
        
        warp_file = glob(os.path.join(reg_path, '*.gz'))[0]
        warp = ants.image_read(warp_file)
        warptx = ants.transform_from_displacement_field(warp)
        
        return affinetx, warptx
    
    def __write_transformed_cells(self, cell_transformed: list):
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
            
        print('Saving transformed cell locations to XML')
        for coord in tqdm(self.cell_transformed, total = len(self.cell_transformed)):
            
            coord = [dim if dim > 1 else 1.0 for dim in coord]     
            coord_dict = {'x': coord[0], 'y': coord[1], 'z': coord[2]}
            cells.append(Cell(coord_dict, 'cell'))
            
        save_cells(cells, os.path.join(self.args['save_path'], 'transformed_cells.xml'))
                
    def run(self):
        """
        Runs quantification of registered cells

        """
        
        ds = 2**self.args['downsample']
        reg_dims = [dim / ds for dim in self.args['input_res']]
        
        raw_cells = self.__read_xml(self.args['input_segmentation'], reg_dims, ds)
        affinetx, warptx = self.__read_transform(self.args['input_registration'])
        
        scale = [raw/trans for raw, trans in zip(self.args['transform_res'], reg_dims)]
        cells_transformed = []
        
        print('Processing cell location using registration transform')
        for cell in tqdm(raw_cells, total = len(raw_cells)):  
            scaled_cell = [dim * scale for dim, scale in zip(cell, scale)]
            affine_pt = affinetx.apply_to_point(scaled_cell)
            warp_pt = warptx.apply_to_point(affine_pt)
            cells_transformed.append(warp_pt)
            
        self.__write_transformed_cells(cells_transformed)
        
        print('Calculating cell counts per brain region')
        count = utils.CellCounts(25)
        count_df = count.create_counts(cells_transformed)
        
        print('Saving counts')
        fname = 'Cell_count_by_region.csv'
        count_df.to_csv(os.path.join(self.args['save_path'], fname))
        
def main():

    default_params = {
        "input_segmentation": '/path/to/segmentation/output',
        "input_registration": '/path/to/registration/output',
        "downsample": 3,
        "input_res": [7400, 4200, 10240],
        "transform_res": [533, 302, 819],
        "save_path": os.getcwd(),
    }

    quant = quantify(default_params)
    quant.run()
    
if __name__ == "__main__":
    main()

