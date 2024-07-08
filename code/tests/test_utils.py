#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 13:25:17 2024

@author: nicholas.lusk
"""
import os
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pd_testing

sys.path.insert(
    0, "/Users/nicholas.lusk/Documents/Github/aind-smartspim-quantification/code/"
)

from aind_smartspim_quantification.utils import utils


class TestSmartspimUtils(unittest.TestCase):
    """Tests utility methods for smartspim quantification capsule"""

    def setUp(self):
        """Setting up unit test"""
        current_path = Path(os.path.abspath(__file__)).parent
        self.ccf_files = current_path.joinpath("./resources/")
        self.test_local_json_path = current_path.joinpath("./resources/local_json.json")
        self.test_structureID = "test"
        self.resolution = 25
        self.CellCounts = utils.CellCounts(self.ccf_files, self.resolution)

    def tearDown(self):
        """Tearing down utils unit test"""

    def test_get_annotation_map(self):
        """
        Test successful loading of annotation mapping
        """

        expected_result = {"1": "TMv", "512": "CB"}
        self.assertEqual(self.CellCounts.annot_map, expected_result)

    def test_get_CCF_mesh_points(self):
        """
        Test successful loading of mesh points for region
        """

        expected_result_1 = [
            [0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [2.5, 4.33, 0.0],
            [2.5, 1.44, 4.1],
        ]
        expected_result_2 = [[0, 1, 2], [1, 2, 3], [0, 2, 3], [0, 1, 3]]

        result_1, result_2 = self.CellCounts.get_CCF_mesh_points(self.test_structureID)
        self.assertTrue((result_1 == expected_result_1).all())
        self.assertTrue((result_2 == expected_result_2).all())

    def test_reflect_about_midline(self):
        """
        Test method reflecting points around a midline
        """

        expected_result = np.array([[5710, 0, 0]])
        result = self.CellCounts.reflect_about_midline(np.array([[5690, 0, 0]]))
        self.assertTrue((result == expected_result).all())

    def test_get_region_lists(self):
        """
        Tests successful loading of structure files
        """

        expected_result = [(1, "hemi"), (512, "mid")]
        self.CellCounts.get_region_lists()
        self.assertEqual(self.CellCounts.structs, expected_result)

    def test_crop_cells(self):
        """
        Tests method for cell cropping if cells are in micron state space
        """

        expected_result = np.array([[5700, 4072, 7623]], dtype=np.float32)

        test_pts = np.array([[0, 0, 0], [7623, 4072, 5700]])

        result = self.CellCounts.crop_cells(test_pts)
        
        self.assertTrue((result == expected_result).all())

    def test_create_counts(self):
        """
        Tests method for detecting cells within CCF meshes
        """

        example_data = {
            "Id": [1, 512],
            "Structure": ["TMv", "CB"],
            "Struct_Info": ["hemi", "mid"],
            "Struct_area_um3": [63031250.0, 55975232132.2824],
            "Left": [1, 1],
            "Right": [0, 0],
            "Total": [1, 1],
            "Left_Density": [1 / 63031250.0, 1 / 55975232132.2824],
            "Right_Density": [0.0, 0.0],
            "Total_Density": [1 / 63031250.0, 1 / 55975232132.2824],
        }

        expected_result = pd.DataFrame.from_dict(example_data)

        # locations have one outside brain and one in each region
        test_pts = [
            [0, 0, 0],
            [188, 267, 299],
            [227, 143, 462],
        ]

        result = self.CellCounts.create_counts(test_pts)
        pd_testing.assert_frame_equal(result, expected_result)

    def test_get_orientation(self):
        """
        Tests method for reading of orientation from processing manifest
        """

        expected_result = "sal"

        test_params = [
            {"direction": "Superior_to_inferior", "dimension": 0},
            {"direction": "Anterior_to_posterior", "dimension": 1},
            {"direction": "Left_to_right", "dimension": 2},
        ]

        result = utils.get_orientation(test_params)
        self.assertEqual(result, expected_result)
    
    def test_get_intensity_mask(self):
        """
        Tests method for creating intensity mask for metrics
        """
        
        expected_result = np.load(
            os.path.join(self.ccf_files, 'mask.npy')
        )
        
        print(expected_result)
        
        verticies = [
            [0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [2.5, 4.33, 0.0],
            [2.5, 1.44, 4.1],
        ]
        faces = [[0, 1, 2], [1, 2, 3], [0, 2, 3], [0, 1, 3]]
        mask = np.zeros((10, 10, 10), dtype = int)
        split = 'mid'
        
        result = utils.get_intensity_mask(verticies, faces, mask, split)
        print(result)
        self.assertTrue((result == expected_result).all())
        
    def test_normalized_mutual_information(self):
        """
        Tests regional mutual information metric
        """
        
        expected_result = 1.0
        
        patch_1 = np.ones((9,9,9), dtype = int)
        patch_2 = np.ones((9,9,9), dtype = int)
        mask = np.pad(
            np.ones((3,3,3), dtype = int),
            (3,3),
            mode = "constant",
            constant_values = 0
        )
        
        result = utils.normalized_mutual_information(patch_1, patch_2, mask)
        self.assertEqual(result, expected_result)

    
    def test_read_json_as_dict(self):
        """
        Tests successful reading of dictionary
        """

        expected_result = {"some_key": "some_value"}
        result = utils.read_json_as_dict(self.test_local_json_path)
        self.assertEqual(result, expected_result)


if __name__ == "__main__":
    unittest.main()
