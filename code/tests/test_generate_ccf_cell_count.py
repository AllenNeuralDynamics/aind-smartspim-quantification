#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for generate_ccf_cell_count utilities"""

import os
import struct
import unittest
from pathlib import Path
from unittest.mock import patch

import dask.array as da
import numpy as np
from aind_smartspim_quantification.utils import generate_ccf_cell_count as gcc


class TestGenerateCCFCellCount(unittest.TestCase):
    """Tests for generate_ccf_cell_count module"""

    def setUp(self):
        current_path = Path(os.path.abspath(__file__)).parent
        self.resources = current_path / "resources"
        self.multi_xml = self.resources / "classified_test_multi.xml"

    def test_get_points_from_xml_returns_list(self):
        """get_points_from_xml returns a list of dicts with x/y/z keys"""
        result = gcc.get_points_from_xml(self.multi_xml)
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        for point in result:
            self.assertIn("x", point)
            self.assertIn("y", point)
            self.assertIn("z", point)

    def test_get_points_from_xml_correct_values(self):
        """get_points_from_xml extracts the correct marker coordinates"""
        result = gcc.get_points_from_xml(self.multi_xml)
        # First marker: X=3072, Y=412, Z=448
        self.assertEqual(result[0]["x"], "3072")
        self.assertEqual(result[0]["y"], "412")
        self.assertEqual(result[0]["z"], "448")
        # Second marker: X=1000, Y=200, Z=300
        self.assertEqual(result[1]["x"], "1000")
        self.assertEqual(result[1]["y"], "200")
        self.assertEqual(result[1]["z"], "300")

    def test_get_points_from_xml_count(self):
        """get_points_from_xml returns exactly two points for the multi-marker fixture"""
        result = gcc.get_points_from_xml(self.multi_xml)
        self.assertEqual(len(result), 2)

    def test_buf_builder_extends_buffer(self):
        """buf_builder appends exactly 12 bytes (3 × 4-byte float) to the buffer"""
        buf = bytearray()
        gcc.buf_builder(1.0, 2.0, 3.0, buf)
        self.assertEqual(len(buf), 12)

    def test_buf_builder_correct_values(self):
        """buf_builder encodes coordinates as little-endian floats"""
        buf = bytearray()
        gcc.buf_builder(4.5, 9.0, 13.5, buf)
        x, y, z = struct.unpack("<3f", bytes(buf))
        self.assertAlmostEqual(x, 4.5, places=4)
        self.assertAlmostEqual(y, 9.0, places=4)
        self.assertAlmostEqual(z, 13.5, places=4)

    def test_buf_builder_accumulates(self):
        """buf_builder called twice produces 24 bytes total"""
        buf = bytearray()
        gcc.buf_builder(1.0, 2.0, 3.0, buf)
        gcc.buf_builder(4.0, 5.0, 6.0, buf)
        self.assertEqual(len(buf), 24)
        x1, y1, z1, x2, y2, z2 = struct.unpack("<6f", bytes(buf))
        self.assertAlmostEqual(x1, 1.0, places=4)
        self.assertAlmostEqual(x2, 4.0, places=4)

    @patch("aind_smartspim_quantification.utils.generate_ccf_cell_count.da.from_zarr")
    def test_calculate_dynamic_range_returns_two_ints(self, mock_from_zarr):
        """calculate_dynamic_range returns [range_max, window_max] as ints"""
        data = np.arange(1, 101, dtype=np.float32)
        mock_from_zarr.return_value = da.from_array(data[np.newaxis, :])

        result = gcc.calculate_dynamic_range("/fake/path", percentile=99, level=3)

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], int)
        self.assertIsInstance(result[1], int)

    @patch("aind_smartspim_quantification.utils.generate_ccf_cell_count.da.from_zarr")
    def test_calculate_dynamic_range_window_larger_than_range(self, mock_from_zarr):
        """window_max is always >= range_max"""
        data = np.arange(1, 101, dtype=np.float32)
        mock_from_zarr.return_value = da.from_array(data[np.newaxis, :])

        result = gcc.calculate_dynamic_range("/fake/path", percentile=50, level=3)

        self.assertGreaterEqual(result[1], result[0])

    @patch("aind_smartspim_quantification.utils.generate_ccf_cell_count.da.from_zarr")
    def test_calculate_dynamic_range_positive(self, mock_from_zarr):
        """range_max is positive for non-zero image data"""
        data = np.full((5, 5, 5), 100.0, dtype=np.float32)
        mock_from_zarr.return_value = da.from_array(data)

        result = gcc.calculate_dynamic_range("/fake/path", percentile=99, level=3)

        self.assertGreater(result[0], 0)


if __name__ == "__main__":
    unittest.main()
