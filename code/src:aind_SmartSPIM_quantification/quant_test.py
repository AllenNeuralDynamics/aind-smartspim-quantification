#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 12:47:37 2023

@author: nicholas.lusk
"""

import os
import ants
import utils

import pandas as pd

from glob import glob
from tqdm import tqdm

from imlib.cells.cells import Cell
from imlib.IO.cells import get_cells, save_cells

class quantify():
    """
    Class for quantifying lightsheet cell data
    """
    
    def __init__(self, params):
        self.input_segmentation = params["input_segmentation"]
        self.input_registration = params["input_registration"]
        self.downsample  = params["downsample"]
        self.input_res = params['input_res']
        self.transform_res = params['transform_res']
        self.save_path = params['save_path']
    
    

    def __read_xml(self, seg_path, reg_dims, ds):
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
        

        cell_file = glob(os.path.join(seg_path, "detected_cells.xml"))[0]
        file_cells = get_cells(cell_file)
        
        cells = []

        for cell in file_cells:
            cells.append((cell.x / ds, 
                          cell.z / ds, 
                          reg_dims[2] - (cell.y / ds)))
        
        return cells
    
    def __read_transform(self, reg_path):
        """
        Imports ants transformation from registration output
        
        Parameters
        -------------
        seg_path: PathLike
            Path to the affine (.mat) and deformation field (.gz) file from registration

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
    
    def __write_transformed_cells(self, cell_transformed):
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
        for coord in tqdm(cell_transformed, total = len(cell_transformed)):
            
            coord = [dim if dim > 1 else 1.0 for dim in coord]     
            coord_dict = {'x': coord[0], 'y': coord[1], 'z': coord[2]}
            cells.append(Cell(coord_dict, 'cell'))
            
        save_cells(cells, os.path.join(self.save_path, 'transformed_cells.xml'))
        
    def __write_registered_counts(self, counts: pd.DataFrame):
        """
        Save cell counts for brain regions using CCFv3 from brainreg library
        
        Parameters
        -------------
        cell_transformed: list
            list of Cell objects with transformed cell locations
            
        Returns
        -------------
        xlsx
            writes excel file with cell counts for each brain region of atlas with registration resolution
        """        
        
        
    def run(self, import_cells = True):
        """
        Runs quantification of registered cells

        """
        
        # can make more options if needed
        if self.downsample == 3:
            micron_scale = 25
        
        if not import_cells:
            ds = 2**self.downsample
            reg_dims = [dim / ds for dim in self.input_res]
        
            raw_cells = self.__read_xml(self.input_segmentation, reg_dims, ds)
            affinetx, warptx = self.__read_transform(self.input_registration)
        
            scale = [raw/trans for raw, trans in zip(self.transform_res, reg_dims)]
            cells_transformed = []
        
            print('Processing cell location using registration transform')
            for cell in tqdm(raw_cells, total = len(raw_cells)):  
                scaled_cell = [dim * scale for dim, scale in zip(cell, scale)]
                affine_pt = affinetx.apply_to_point(scaled_cell)
                warp_pt = warptx.apply_to_point(affine_pt)
                cells_transformed.append(warp_pt)
        
            print('Saving transformed cell coordinates')
            self.__write_transformed_cells(cells_transformed)
        else:
            print('Getting cells form file')
            cell_file = glob(os.path.join(self.input_segmentation, "transformed_cells.xml"))[0]
            file_cells = get_cells(cell_file)
            
            cells_transformed = []
            for cell in file_cells:
                cells_transformed.append((cell.x, 
                                          cell.y, 
                                          cell.z))
        
        
        print('Calculating cell counts per brain region')
        count = utils.CellCounts(micron_scale)
        count_df = count.create_counts(cells_transformed)
        
        print('Saving regional counts')
        fname = 'Cell_count_by_region.csv'
        count_df.to_csv(os.path.join(self.save_path, fname))
        return count_df
        
def main():

    default_params = {
        "input_segmentation": '/Users/nicholas.lusk/Documents/LightSheet/SmartSPIM_631680/',
        "input_registration": '/Users/nicholas.lusk/Documents/LightSheet/SmartSPIM_631680/',
        "downsample": 3,
        "input_res": [7400, 4200, 10240],
        "transform_res": [533, 302, 819],
        "save_path": os.getcwd(),
    }

    quant = quantify(default_params)
    out = quant.run()
    return out
    
if __name__ == "__main__":
    out = main()