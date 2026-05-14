"""Tests SmartSPIM Pipeline Quantification"""

import os
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from aind_smartspim_quantification import quantification


class TestCaseBase(unittest.TestCase):
    def assertIsFile(self, path):
        if not Path(path).resolve().is_file():
            raise AssertionError("File does not exist: %s" % str(path))


class TestSmartspimQuantification(TestCaseBase):
    """Tests quantification methods for smartspim quantification capsule"""

    def setUp(self):
        """Setting up unit test"""
        current_path = Path(os.path.abspath(__file__)).parent
        self.test_dir = current_path.joinpath("./resources/")
        self.detected_cells_csv = self.test_dir / "detected_cells.csv"
        self.downsample = 8
        self.ds = 2**self.downsample
        self.reg_dims = [929.125, 458.125, 1103.375]
        # test_pts shape: (n_cells, 3) — needed by write_transformed_cells
        self.test_pts = np.array([[384, 56, 51]])
        # dummy metrics: [Foreground, Background, Cell_ID]
        self.test_metrics = np.array([[0.9, 0.1, 42]])

    def tearDown(self):
        """Tearing down unit test"""

    # ------------------------------------------------------------------
    # read_cells_from_csv
    # ------------------------------------------------------------------

    def test_read_cells_from_csv_spr(self):
        """CSV reader with SPR orientation and identity orient matrix"""
        # CSV has x=3072, y=412, z=448; ds=8
        # y = 412/8 = 51.5 (no flip for non-spl/AIBS)
        # cell = (z/ds, y, x/ds) = (56.0, 51.5, 384.0)
        # identity orient_matrix → no axis flipping
        orient_matrix = np.eye(3)

        result = quantification.read_cells_from_csv(
            self.detected_cells_csv,
            self.reg_dims,
            self.ds,
            "spr",
            orient_matrix,
            "AIBS",
        )

        expected = np.array([[56.0, 51.5, 384.0]])
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_read_cells_from_csv_spl_aibs(self):
        """CSV reader applies AIBS SPL bug-correction on the y axis"""
        # For spl + AIBS: y = reg_dims[1] - (y / ds) = 458.125 - 51.5 = 406.625
        orient_matrix = np.eye(3)

        result = quantification.read_cells_from_csv(
            self.detected_cells_csv,
            self.reg_dims,
            self.ds,
            "spl",
            orient_matrix,
            "AIBS",
        )

        expected_y = self.reg_dims[1] - (412 / self.ds)  # 406.625
        expected = np.array([[56.0, expected_y, 384.0]])
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_read_cells_from_csv_file_not_found(self):
        """read_cells_from_csv raises FileNotFoundError for missing path"""
        with self.assertRaises(FileNotFoundError):
            quantification.read_cells_from_csv(
                self.test_dir / "nonexistent.csv",
                self.reg_dims,
                self.ds,
                "spr",
                np.eye(3),
                "AIBS",
            )

    def test_read_cells_from_csv_negative_orient(self):
        """Axis with negative orient_matrix direction gets flipped"""
        # orient_matrix with the first axis negated → cells[:, 0] = reg_dims[0] - cells[:, 0]
        orient_matrix = np.diag([-1.0, 1.0, 1.0])

        result = quantification.read_cells_from_csv(
            self.detected_cells_csv,
            self.reg_dims,
            self.ds,
            "spr",
            orient_matrix,
            "AIBS",
        )

        base = np.array([[56.0, 51.5, 384.0]])
        expected_flipped_0 = self.reg_dims[0] - base[0, 0]  # 929.125 - 56.0
        self.assertAlmostEqual(result[0, 0], expected_flipped_0, places=5)
        self.assertAlmostEqual(result[0, 1], base[0, 1], places=5)
        self.assertAlmostEqual(result[0, 2], base[0, 2], places=5)

    # ------------------------------------------------------------------
    # scale_cells
    # ------------------------------------------------------------------

    def test_scale_cells_identity(self):
        """scale_cells with scale=[1,1,1] returns unchanged cells"""
        cells = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = quantification.scale_cells(cells, [1.0, 1.0, 1.0])
        np.testing.assert_array_almost_equal(result, cells)

    def test_scale_cells_known_scale(self):
        """scale_cells applies per-dimension scaling correctly"""
        cells = np.array([[1.0, 2.0, 3.0]])
        result = quantification.scale_cells(cells, [2.0, 3.0, 4.0])
        expected = np.array([[2.0, 6.0, 12.0]])
        np.testing.assert_array_almost_equal(result, expected)

    # ------------------------------------------------------------------
    # convert_to_ants_space / convert_from_ants_space
    # ------------------------------------------------------------------

    def test_convert_ants_space_roundtrip(self):
        """convert_to then convert_from ANTs space recovers original cells"""
        template_params = {
            "dims": 3,
            "scale": [0.1, 0.2, 0.3],
            "origin": [1.0, -2.0, 0.5],
            "direction": [1.0, -1.0, 1.0],
        }
        cells = np.array([[10.0, 20.0, 30.0], [5.0, 15.0, 25.0]])
        ants_cells = quantification.convert_to_ants_space(template_params, cells)
        recovered = quantification.convert_from_ants_space(template_params, ants_cells)
        np.testing.assert_array_almost_equal(recovered, cells, decimal=10)

    # ------------------------------------------------------------------
    # write_transformed_cells
    # ------------------------------------------------------------------

    @patch("logging.Logger")
    def test_write_transformed_cells(self, mock_log):
        """write_transformed_cells saves a CSV and returns the path"""
        out_path = quantification.write_transformed_cells(
            self.test_pts, self.test_metrics, self.test_dir, mock_log
        )

        self.assertIsFile(out_path)
        self.assertTrue(out_path.endswith(".csv"))

        os.remove(out_path)


if __name__ == "__main__":
    unittest.main()
