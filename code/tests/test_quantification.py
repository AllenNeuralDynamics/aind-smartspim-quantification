"""Tests SmartSPIM Pipeline Quantification"""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(
    0, "/Users/nicholas.lusk/Documents/Github/aind-smartspim-quantification/code/"
)

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
        self.downsample = 8
        self.reg_dims = [929.125, 458.125, 1103.375]
        self.test_pts = [[384, 56, 51]]

    def tearDown(self):
        """Tearing down unit test"""

    def test_read_xml_spr(self):
        """Test importing cell locations for SPR orientation"""

        expected_result = [(384.0, 56.0, 1051.875)]
        result = quantification.read_xml(
            self.test_dir, self.reg_dims, self.downsample, "spr", "AIBS"
        )

        self.assertListEqual(result, expected_result)

    def test_read_xml_spl_AIBS(self):
        """Test importing cell locations for SPL orientaion from AIBS"""
        expected_result = [(545.125, 56.0, 1051.875)]
        result = quantification.read_xml(
            self.test_dir, self.reg_dims, self.downsample, "spl", "AIBS"
        )

        self.assertListEqual(result, expected_result)

    def test_read_xml_spl_AIND(self):
        """Test importing cell locations for SPL orientaion from AIND"""
        expected_result = [(384.0, 56.0, 51.5)]
        result = quantification.read_xml(
            self.test_dir, self.reg_dims, self.downsample, "spl", "AIND"
        )

        self.assertListEqual(result, expected_result)

    def test_read_xml_sal(self):
        """Test importing cell locations for SAL"""
        expected_result = [(384.0, 56.0, 51.5)]
        result = quantification.read_xml(
            self.test_dir, self.reg_dims, self.downsample, "sal", "AIBS"
        )

        self.assertListEqual(result, expected_result)

    def test_read_xml_rpi(self):
        """Test importing cell locations for SAL"""
        expected_result = [(402.125, 545.125, 1051.875)]
        result = quantification.read_xml(
            self.test_dir, self.reg_dims, self.downsample, "rpi", "AIBS"
        )

        self.assertListEqual(result, expected_result)

    # need to figure out how to test functions without writing file
    @patch("logging.Logger")
    def test_write_transformed_cells(self, mock_log):
        quantification.write_transformed_cells(self.test_pts, self.test_dir, mock_log)

        self.assertIsFile(os.path.join(self.test_dir, "transformed_cells.xml"))

        os.remove(os.path.join(self.test_dir, "transformed_cells.xml"))


if __name__ == "__main__":
    unittest.main()
