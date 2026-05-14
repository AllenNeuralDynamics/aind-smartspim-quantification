#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests utility methods for smartspim quantification capsule"""

import os
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pd_testing
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

    # ------------------------------------------------------------------
    # CellCounts.get_annotation_map
    # ------------------------------------------------------------------

    def test_get_annotation_map(self):
        """Test successful loading of annotation mapping"""
        expected_result = {"1": "TMv", "512": "CB"}
        self.assertEqual(self.CellCounts.annot_map, expected_result)

    # ------------------------------------------------------------------
    # CellCounts.get_CCF_mesh_points
    # ------------------------------------------------------------------

    def test_get_CCF_mesh_points(self):
        """Test successful loading of mesh points for region"""
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

    # ------------------------------------------------------------------
    # CellCounts.reflect_about_midline
    # ------------------------------------------------------------------

    def test_reflect_about_midline(self):
        """Test method reflecting points around a midline"""
        expected_result = np.array([[5710, 0, 0]])
        result = self.CellCounts.reflect_about_midline(np.array([[5690, 0, 0]]))
        self.assertTrue((result == expected_result).all())

    # ------------------------------------------------------------------
    # CellCounts.get_region_lists
    # ------------------------------------------------------------------

    def test_get_region_lists(self):
        """Tests successful loading of structure files"""
        expected_result = [(1, "hemi"), (512, "mid")]
        self.CellCounts.get_region_lists()
        self.assertEqual(self.CellCounts.structs, expected_result)

    # ------------------------------------------------------------------
    # CellCounts.crop_cells
    # ------------------------------------------------------------------

    def test_crop_cells(self):
        """Tests cell cropping removes points outside the brain"""
        # crop_cells reorders columns as [2, 1, 0]; the point [7623, 4072, 5700]
        # becomes [5700, 4072, 7623] after reordering.
        expected_cells = np.array([[5700, 4072, 7623]], dtype=np.float32)

        test_pts = np.array([[0, 0, 0], [7623, 4072, 5700]], dtype=np.float32)
        test_metrics = np.array([[0.9, 0.1, 1], [0.8, 0.2, 2]], dtype=np.float32)

        result_cells, result_metrics = self.CellCounts.crop_cells(
            test_pts, test_metrics
        )

        self.assertTrue((result_cells == expected_cells).all())
        self.assertEqual(len(result_metrics), 1)

    # ------------------------------------------------------------------
    # CellCounts.create_counts
    # ------------------------------------------------------------------

    def test_create_counts(self):
        """Tests that create_counts returns a DataFrame with the correct shape and counts"""
        expected_cols = [
            "Id",
            "Acronym",
            "Struct_Info",
            "Struct_area_um3",
            "Left",
            "Right",
            "Total",
            "Left_Density",
            "Right_Density",
            "Total_Density",
            "Left_Median_Foreground",
            "Right_Median_Foreground",
            "Total_Median_Foreground",
            "Left_Median_Background",
            "Right_Median_Background",
            "Total_Median_Background",
        ]

        # One cell outside brain, one per region; fg=1.0, bg=0.0 for all
        test_pts = [
            [0, 0, 0],
            [188, 267, 299],
            [227, 143, 462],
        ]
        test_metrics = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=np.float32)

        result = self.CellCounts.create_counts(test_pts, test_metrics)

        # Correct column set
        self.assertListEqual(list(result.columns), expected_cols)

        # Two regions: TMv (Id=1) and CB (Id=512)
        self.assertEqual(len(result), 2)

        # Each region should have exactly 1 detected cell
        row_tmv = result[result["Acronym"] == "TMv"].iloc[0]
        self.assertEqual(row_tmv["Total"], 1)
        self.assertEqual(row_tmv["Right"], 0)

        row_cb = result[result["Acronym"] == "CB"].iloc[0]
        self.assertEqual(row_cb["Total"], 1)
        self.assertEqual(row_cb["Right"], 0)

        # Densities must be positive where there are cells
        self.assertGreater(row_tmv["Left_Density"], 0)
        self.assertGreater(row_cb["Left_Density"], 0)

        # Median foreground should equal 1.0 for cells with fg=1.0
        self.assertAlmostEqual(row_tmv["Total_Median_Foreground"], 1.0)
        self.assertAlmostEqual(row_cb["Total_Median_Foreground"], 1.0)

    # ------------------------------------------------------------------
    # get_orientation
    # ------------------------------------------------------------------

    def test_get_orientation(self):
        """Tests method for reading of orientation from processing manifest"""
        expected_result = "sal"

        test_params = [
            {"direction": "Superior_to_inferior", "dimension": 0},
            {"direction": "Anterior_to_posterior", "dimension": 1},
            {"direction": "Left_to_right", "dimension": 2},
        ]

        result = utils.get_orientation(test_params)
        self.assertEqual(result, expected_result)

    # ------------------------------------------------------------------
    # get_orientation_transform
    # ------------------------------------------------------------------

    def test_get_orientation_transform_identity(self):
        """Identity transform when input == output orientation"""
        original, swapped, mat = utils.get_orientation_transform("spr", "spr")

        np.testing.assert_array_equal(original, [0, 1, 2])
        np.testing.assert_array_equal(swapped, [0, 1, 2])
        np.testing.assert_array_equal(mat, np.eye(3))

    def test_get_orientation_transform_swap(self):
        """Axes are correctly permuted for a transposed orientation"""
        # SPR → RPS: s↔r swap (s at index 2 in output, r at index 0 in output)
        original, swapped, mat = utils.get_orientation_transform("spr", "rps")

        # s(0)→2, p(1)→1, r(2)→0  means swapped = [2, 1, 0]
        np.testing.assert_array_equal(original, [0, 1, 2])
        np.testing.assert_array_equal(swapped, [2, 1, 0])

    # ------------------------------------------------------------------
    # get_intensity_mask
    # ------------------------------------------------------------------

    def test_get_intensity_mask(self):
        """Tests method for creating intensity mask for metrics"""
        expected_result = np.load(os.path.join(self.ccf_files, "mask.npy"))

        vertices = [
            [0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [2.5, 4.33, 0.0],
            [2.5, 1.44, 4.1],
        ]
        faces = [[0, 1, 2], [1, 2, 3], [0, 2, 3], [0, 1, 3]]
        mask = np.zeros((10, 10, 10), dtype=int)
        split = "mid"

        result = utils.get_intensity_mask(vertices, faces, mask, split)
        self.assertTrue((result == expected_result).all())

    # ------------------------------------------------------------------
    # normalized_mutual_information
    # ------------------------------------------------------------------

    def test_normalized_mutual_information(self):
        """Tests that NMI of identical images equals 1.0"""
        expected_result = 1.0

        patch_1 = np.ones((9, 9, 9), dtype=int)
        patch_2 = np.ones((9, 9, 9), dtype=int)
        mask = np.pad(
            np.ones((3, 3, 3), dtype=int), (3, 3), mode="constant", constant_values=0
        )

        result = utils.normalized_mutual_information(patch_1, patch_2, mask)
        self.assertEqual(result, expected_result)

    # ------------------------------------------------------------------
    # read_json_as_dict
    # ------------------------------------------------------------------

    def test_read_json_as_dict(self):
        """Tests successful reading of dictionary"""
        expected_result = {"some_key": "some_value"}
        result = utils.read_json_as_dict(self.test_local_json_path)
        self.assertEqual(result, expected_result)

    def test_read_json_as_dict_missing_file(self):
        """read_json_as_dict returns empty dict for a non-existent path"""
        result = utils.read_json_as_dict("/nonexistent/path/that/does/not/exist.json")
        self.assertEqual(result, {})


if __name__ == "__main__":
    unittest.main()
