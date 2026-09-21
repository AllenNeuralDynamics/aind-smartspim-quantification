"""Tests SmartSPIM Pipeline Quantification"""

import os
import tempfile
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
        # downsample is the exponent (downsample_res); ds = 2**downsample is the factor.
        # Default production value is 3 → ds = 8.
        self.downsample = 3
        self.ds = 2**self.downsample  # = 8
        self.reg_dims = [929.125, 458.125, 1103.375]
        # test_pts shape: (n_cells, 3) — needed by write_transformed_cells
        self.test_pts = np.array([[384, 56, 51]])
        # dummy metrics: [Foreground, Background, Cell_ID]
        self.test_metrics = np.array([[0.9, 0.1, 42]])

    def tearDown(self):
        """Tearing down unit test"""

    def test_read_cells_from_csv_spr(self):
        """CSV reader with SPR orientation and identity orient matrix"""
        # CSV has x=3072, y=412, z=448.
        # read_cells_from_csv returns (z/ds, y/ds, x/ds) for non-spl orientations.
        orient_matrix = np.eye(3)

        result = quantification.read_cells_from_csv(
            self.detected_cells_csv,
            self.reg_dims,
            self.ds,
            "spr",
            orient_matrix,
            "AIBS",
        )

        expected = np.array([[448 / self.ds, 412 / self.ds, 3072 / self.ds]])
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_read_cells_from_csv_spl_aibs(self):
        """CSV reader applies AIBS SPL bug-correction on the y axis"""
        # For spl + AIBS: y = reg_dims[1] - (raw_y / ds)
        orient_matrix = np.eye(3)

        result = quantification.read_cells_from_csv(
            self.detected_cells_csv,
            self.reg_dims,
            self.ds,
            "spl",
            orient_matrix,
            "AIBS",
        )

        expected = np.array(
            [[448 / self.ds, self.reg_dims[1] - 412 / self.ds, 3072 / self.ds]]
        )
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

        # Without the flip, dim 0 would be z/ds; the negative orient reflects it.
        unflipped_dim0 = 448 / self.ds
        expected_dim0 = self.reg_dims[0] - unflipped_dim0
        self.assertAlmostEqual(result[0, 0], expected_dim0, places=5)
        # Dims 1 and 2 are unaffected by the single-axis flip.
        self.assertAlmostEqual(result[0, 1], 412 / self.ds, places=5)
        self.assertAlmostEqual(result[0, 2], 3072 / self.ds, places=5)

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

    @patch("logging.Logger")
    def test_write_transformed_cells(self, mock_log):
        """write_transformed_cells saves a CSV and returns the path"""
        out_path = quantification.write_transformed_cells(
            self.test_pts, self.test_metrics, self.test_dir, mock_log
        )

        self.assertIsFile(out_path)
        self.assertTrue(out_path.endswith(".csv"))

        os.remove(out_path)

    def test_write_transformed_cells_content(self):
        """write_transformed_cells CSV has expected column names and row count"""
        with patch("logging.Logger") as mock_log:
            with tempfile.TemporaryDirectory() as tmpdir:
                out_path = quantification.write_transformed_cells(
                    self.test_pts, self.test_metrics, tmpdir, mock_log
                )
                import pandas as pd

                df = pd.read_csv(out_path, index_col=0)
                self.assertListEqual(
                    list(df.columns),
                    ["x", "y", "z", "Foreground", "Background", "Cell ID"],
                )
                self.assertEqual(len(df), 1)

    def test_read_cells_from_xml_spr(self):
        """XML reader returns correct coordinates for SPR orientation"""
        xml_path = self.test_dir / "classified_test_multi.xml"
        result = quantification.read_cells_from_xml(
            xml_path,
            self.reg_dims,
            self.ds,
            "spr",
            np.eye(3),
            "AIBS",
        )
        # Two markers: (3072,412,448) and (1000,200,300); returns (z/ds, y/ds, x/ds)
        expected = np.array(
            [
                [448 / self.ds, 412 / self.ds, 3072 / self.ds],
                [300 / self.ds, 200 / self.ds, 1000 / self.ds],
            ]
        )
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_read_cells_from_xml_spl_aibs(self):
        """XML reader applies AIBS SPL bug-correction on the y axis"""
        xml_path = self.test_dir / "classified_test_multi.xml"
        result = quantification.read_cells_from_xml(
            xml_path,
            self.reg_dims,
            self.ds,
            "spl",
            np.eye(3),
            "AIBS",
        )
        # y = reg_dims[1] - raw_y/ds for both cells
        expected = np.array(
            [
                [448 / self.ds, self.reg_dims[1] - 412 / self.ds, 3072 / self.ds],
                [300 / self.ds, self.reg_dims[1] - 200 / self.ds, 1000 / self.ds],
            ]
        )
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_read_cells_from_xml_file_not_found(self):
        """read_cells_from_xml raises FileNotFoundError for missing path"""
        with self.assertRaises(FileNotFoundError):
            quantification.read_cells_from_xml(
                self.test_dir / "nonexistent.xml",
                self.reg_dims,
                self.ds,
                "spr",
                np.eye(3),
                "AIBS",
            )

    def test_get_cell_metrics(self):
        """get_cell_metrics returns foreground/background/cell_id for Class==1 rows"""
        csv_path = self.test_dir / "cell_likelihoods.csv"
        result = quantification.get_cell_metrics(csv_path)

        # CSV has 2 Class==1 rows: (0.8, 0.2, 101) and (0.9, 0.1, 102)
        self.assertEqual(result.shape, (2, 3))
        np.testing.assert_array_almost_equal(result[0], [0.8, 0.2, 101], decimal=5)
        np.testing.assert_array_almost_equal(result[1], [0.9, 0.1, 102], decimal=5)

    def test_get_cell_metrics_file_not_found(self):
        """get_cell_metrics raises FileNotFoundError for missing path"""
        with self.assertRaises(FileNotFoundError):
            quantification.get_cell_metrics(self.test_dir / "nonexistent.csv")

    def test_create_visualization_folders(self):
        """create_visualization_folders creates the expected subdirectory tree"""
        with tempfile.TemporaryDirectory() as tmpdir:
            ccf_path, cells_path = quantification.create_visualization_folders(tmpdir)

            self.assertTrue(os.path.isdir(ccf_path))
            self.assertTrue(os.path.isdir(cells_path))
            self.assertTrue(os.path.isdir(os.path.join(tmpdir, "visualization")))
            self.assertIn("ccf_cell_precomputed", ccf_path)
            self.assertIn("cell_points_precomputed", cells_path)


if __name__ == "__main__":
    unittest.main()
