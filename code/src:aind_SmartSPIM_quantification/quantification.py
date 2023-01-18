#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import logging
from pathlib import Path
from typing import Union

import numpy as np

from glob import glob
from argschema.fields import Str
from imlib.IO.cells import get_cells
from argschema import ArgSchemaParser, ArgSchema, InputFile

PathLike = Union[str, Path]

LOG_FMT = "%(asctime)s %(message)s"
LOG_DATE_FMT = "%Y-%m-%d %H:%M"

logging.basicConfig(format=LOG_FMT, datefmt=LOG_DATE_FMT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)



class QuantSchema(ArgSchema):
    """
    Schema format for Cell Quantification
    """

    config_file = InputFile(
        required=True,
        metadata={"description": "Path to the YAML config file."},
        dump_default="smartspim_config.yaml",
    )

    cell_data = Str(
        metadata={
            "required": True,
            "description": "Path where .xml output of cell segmentation is located",
        }
    )

    register_data = Str(
        metadata={"required": True, "description": "Path to transfrom data from registration"}
    )

    save_path = Str(
        metadata={
            "required": True,
            "description": "Location to save .csv of ",
        }
    )
    
class Quantify(ArgSchemaParser):

    """
    Class for quantifying cell count for SmartSPIM lightsheet data
    """

    default_schema = QuantSchema
    
    def get_xml(self, cell_data: PathLike) -> np.array:
        """
        Reads a zarr image
        Parameters
        -------------
        cell_data: PathLike
            Path where the .xml from the segmentation module is located
        Returns
        -------------
        np.array
            numpy array [n x 3] with cell centroids. Coordinate order: AP (y), ML (x), DV (z)
        """
        cell_list = get_cells(cell_data)
        cell_locations = []
        
        for c in cell_list:
            cell_locations.append([c.y, c.x, c.z])
            
        return np.array(cell_locations)
    
    ###### Not sure what this will do yet, waiting on sharmi #####
    def get_transform(self, register_data: PathLike) -> np.array:
        """
        Reads a zarr image
        Parameters
        -------------
        register_data: PathLike
            Path where the output from registration is located
        Returns
        -------------
        np.array
            numpy array with affine transforms and deformation fields from registration module 
        """
        
        return transformed_locations 
    
    def apply_transforms(self, cells: np.array, transforms: np.array) -> np.array:   
        """
        Reads a zarr image
        Parameters
        -------------
        cells: np.array
            Array with raw coordinate data
        transforms: np.array
            Array with affine transform and warp information from registration
        Returns
        -------------
        np.array
            numpy array coordinate locations registered to the CCFv3
        """
        
        return
    
    def count_by_region(self, registered_cells: np.array) -> None:
        return
        
    
    def run(self) -> None:
        
        cell_loc = self.get_xml(self.args['cell_data'])
        transforms = self.get_transforms(self.args['register_data'])
        registered_cells = self.apply_transforms(cell_loc, transforms)
        
        counts self.count_by_region(registered_cells)        
        
        if not os.path.exists(self.args['save_path']):
            print('Creating directory for ')
        
        

def main():
    
    default_params = {
        'cell_data': '',
        'register_data': '',
        'save_path': ''
        }
    
    quant = Quantify(default_params)  
    quant.run()
        
if __name__ == "__main__":
    main()