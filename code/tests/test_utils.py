#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests utility methods for smartspim quantification capsule"""

import logging
import multiprocessing
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pandas.testing as pd_testing

from aind_smartspim_quantification.utils import utils


def _worker_sleep():
    time.sleep(60)


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
        """Test successful loading of annotation mapping"""
        expected_result = {"1": "TMv", "512": "CB"}
        self.assertEqual(self.CellCounts.annot_map, expected_result)

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

    def test_reflect_about_midline(self):
        """Test method reflecting points around a midline"""
        expected_result = np.array([[5710, 0, 0]])
        result = self.CellCounts.reflect_about_midline(np.array([[5690, 0, 0]]))
        self.assertTrue((result == expected_result).all())

    def test_get_region_lists(self):
        """Tests successful loading of structure files"""
        expected_result = [(1, "hemi"), (512, "mid")]
        self.CellCounts.get_region_lists()
        self.assertEqual(self.CellCounts.structs, expected_result)

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

    def test_get_intensity_mask(self):
        """Tests that get_intensity_mask marks interior voxels of a tetrahedron"""
        vertices = [
            [0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [2.5, 4.33, 0.0],
            [2.5, 1.44, 4.1],
        ]
        faces = [[0, 1, 2], [1, 2, 3], [0, 2, 3], [0, 1, 3]]
        mask = np.zeros((10, 10, 10), dtype=int)

        result = utils.get_intensity_mask(vertices, faces, mask, split="mid")

        # Shape must be unchanged
        self.assertEqual(result.shape, (10, 10, 10))
        # Only 0 and 1 values allowed
        self.assertTrue(np.all(np.isin(result, [0, 1])))
        # At least one interior voxel must have been found
        self.assertGreater(result.sum(), 0)

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

    def test_read_json_as_dict(self):
        """Tests successful reading of dictionary"""
        expected_result = {"some_key": "some_value"}
        result = utils.read_json_as_dict(self.test_local_json_path)
        self.assertEqual(result, expected_result)

    def test_read_json_as_dict_missing_file(self):
        """read_json_as_dict returns empty dict for a non-existent path"""
        result = utils.read_json_as_dict("/nonexistent/path/that/does/not/exist.json")
        self.assertEqual(result, {})

    def test_orient_image_identity(self):
        """orient_image with an identity matrix returns the original array"""
        img = np.arange(24).reshape(2, 3, 4)
        result = utils.orient_image(img, np.eye(3))
        np.testing.assert_array_equal(result, img)

    def test_orient_image_axis_swap(self):
        """orient_image with a transpose matrix swaps the first and last axes"""
        img = np.arange(24).reshape(2, 3, 4)
        # Matrix that maps dim 0→2 and dim 2→0 (swap first and last axes)
        mat = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=float)
        result = utils.orient_image(img, mat)
        expected = np.moveaxis(img, [0, 1, 2], [2, 1, 0])
        np.testing.assert_array_equal(result, expected)

    def test_get_volume_mid(self):
        """get_volume returns a positive float for a mid-split tetrahedron"""
        vertices = [
            [0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [2.5, 4.33, 0.0],
            [2.5, 1.44, 4.1],
        ]
        faces = [[0, 1, 2], [1, 2, 3], [0, 2, 3], [0, 1, 3]]
        volume = utils.get_volume(np.array(vertices), np.array(faces), split="mid")
        self.assertGreater(volume, 0)
        self.assertIsInstance(float(volume), float)

    def test_get_volume_hemi(self):
        """get_volume sums left and right sub-mesh volumes for hemi split"""
        # Duplicate the tetrahedron vertices so break_pt splits cleanly
        verts = np.array(
            [
                [0.0, 0.0, 0.0],
                [5.0, 0.0, 0.0],
                [2.5, 4.33, 0.0],
                [2.5, 1.44, 4.1],
                [0.0, 0.0, 0.0],
                [5.0, 0.0, 0.0],
                [2.5, 4.33, 0.0],
                [2.5, 1.44, 4.1],
            ]
        )
        faces = np.array([[0, 1, 2], [1, 2, 3], [0, 2, 3], [0, 1, 3]])
        volume = utils.get_volume(verts, faces, split="hemi")
        self.assertGreater(volume, 0)

    def test_get_mesh_interior_points(self):
        """get_mesh_interior_points returns coordinate tuples inside the mesh"""
        import vedo

        vertices = [
            [0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [2.5, 4.33, 0.0],
            [2.5, 1.44, 4.1],
        ]
        faces = [[0, 1, 2], [1, 2, 3], [0, 2, 3], [0, 1, 3]]
        mesh = vedo.Mesh([vertices, faces])
        xs, ys, zs = utils.get_mesh_interior_points(mesh)
        # The function should return at least one interior point
        self.assertGreater(len(xs), 0)
        self.assertEqual(len(xs), len(ys))
        self.assertEqual(len(xs), len(zs))

    def test_get_region_intensity_masks_correctly(self):
        """get_region_intensity zeroes out voxels where mask == 0"""
        img = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        mask = np.array([[[1, 0], [0, 1]], [[1, 1], [0, 0]]])
        result = utils.get_region_intensity(img, mask)
        expected = np.array([[[1, 0], [0, 4]], [[5, 6], [0, 0]]])
        np.testing.assert_array_equal(result, expected)

    def test_get_region_intensity_all_masked(self):
        """get_region_intensity with all-zero mask returns all-zero array"""
        img = np.ones((3, 3, 3))
        mask = np.zeros((3, 3, 3))
        result = utils.get_region_intensity(img, mask)
        np.testing.assert_array_equal(result, np.zeros((3, 3, 3)))

    def test_get_metric_region_info_all(self):
        """get_metric_region_info returns info for all requested regions"""
        result = self.CellCounts.get_metric_region_info(["1", "512"])
        self.assertIn("1", result)
        self.assertIn("512", result)
        self.assertEqual(result["1"], ["TMv", "hemi"])
        self.assertEqual(result["512"], ["CB", "mid"])

    def test_get_metric_region_info_subset(self):
        """get_metric_region_info returns only the requested subset"""
        result = self.CellCounts.get_metric_region_info(["1"])
        self.assertIn("1", result)
        self.assertNotIn("512", result)

    def test_get_metric_region_info_no_match(self):
        """get_metric_region_info returns empty dict when no regions match"""
        result = self.CellCounts.get_metric_region_info(["9999"])
        self.assertEqual(result, {})

    def test_get_size_bytes(self):
        """get_size formats values below 1 KB as bytes"""
        result = utils.get_size(512)
        self.assertIn("B", result)
        self.assertIn("512", result)

    def test_get_size_kilobytes(self):
        """get_size formats 1024 bytes as 1 KB"""
        result = utils.get_size(1024)
        self.assertIn("KB", result)
        self.assertIn("1.00", result)

    def test_get_size_megabytes(self):
        """get_size formats 1 MiB correctly"""
        result = utils.get_size(1024 * 1024)
        self.assertIn("MB", result)
        self.assertIn("1.00", result)

    def test_get_size_gigabytes(self):
        """get_size formats 1 GiB correctly"""
        result = utils.get_size(1024**3)
        self.assertIn("GB", result)
        self.assertIn("1.00", result)

    def test_check_path_instance_with_path(self):
        """check_path_instance returns True for a pathlib.Path"""
        self.assertTrue(utils.check_path_instance(Path("/tmp")))

    def test_check_path_instance_with_string(self):
        """check_path_instance returns False for a plain string"""
        self.assertFalse(utils.check_path_instance("/tmp"))

    def test_check_path_instance_with_int(self):
        """check_path_instance returns False for a non-path object"""
        self.assertFalse(utils.check_path_instance(42))

    def test_save_dict_as_json_roundtrip(self):
        """save_dict_as_json writes a dict that read_json_as_dict can recover"""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "out.json")
            data = {"key1": "value1", "key2": 42}
            utils.save_dict_as_json(filepath, data)
            result = utils.read_json_as_dict(filepath)
            self.assertEqual(result, data)

    def test_save_dict_as_json_converts_path_to_str(self):
        """save_dict_as_json serialises pathlib.Path values as strings"""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "out.json")
            utils.save_dict_as_json(filepath, {"p": Path("/tmp/test")})
            result = utils.read_json_as_dict(filepath)
            self.assertEqual(result["p"], "/tmp/test")

    def test_save_dict_as_json_none_dict(self):
        """save_dict_as_json with None writes an empty JSON object"""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "out.json")
            utils.save_dict_as_json(filepath, None)
            result = utils.read_json_as_dict(filepath)
            self.assertEqual(result, {})

    def test_save_string_to_txt(self):
        """save_string_to_txt writes the given string to a file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "out.txt")
            utils.save_string_to_txt("hello world", filepath)
            with open(filepath) as f:
                content = f.read()
            self.assertIn("hello world", content)

    def test_create_folder_creates_new(self):
        """create_folder creates a directory that did not exist"""
        with tempfile.TemporaryDirectory() as tmpdir:
            new_dir = os.path.join(tmpdir, "new_subdir")
            self.assertFalse(os.path.exists(new_dir))
            utils.create_folder(new_dir)
            self.assertTrue(os.path.isdir(new_dir))

    def test_create_folder_no_error_if_exists(self):
        """create_folder does not raise if the directory already exists"""
        with tempfile.TemporaryDirectory() as tmpdir:
            utils.create_folder(tmpdir)  # should not raise

    def test_create_logger_returns_logger(self):
        """create_logger returns a logging.Logger instance"""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = utils.create_logger(tmpdir)
            self.assertIsInstance(logger, logging.Logger)

    def test_create_logger_creates_log_file(self):
        """create_logger creates exactly one .log file in the output directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            utils.create_logger(tmpdir)
            log_files = [f for f in os.listdir(tmpdir) if f.endswith(".log")]
            self.assertEqual(len(log_files), 1)

    def test_generate_resources_graphs_creates_png(self):
        """generate_resources_graphs saves a PNG file to the output path"""
        import matplotlib.pyplot as plt

        plt.switch_backend("Agg")
        with tempfile.TemporaryDirectory() as tmpdir:
            utils.generate_resources_graphs(
                [0.0, 1.0, 2.0],
                [10.0, 20.0, 15.0],
                [50.0, 55.0, 52.0],
                tmpdir,
                "test_prefix",
            )
            expected_png = os.path.join(tmpdir, "test_prefix_compute_resources.png")
            self.assertTrue(os.path.isfile(expected_png))

    def test_generate_resources_graphs_empty_lists(self):
        """generate_resources_graphs returns early without error on empty inputs"""
        import matplotlib.pyplot as plt

        plt.switch_backend("Agg")
        utils.generate_resources_graphs([], [], [], "/nonexistent", "prefix")

    def test_stop_child_process(self):
        """stop_child_process terminates a running subprocess"""
        p = multiprocessing.Process(target=_worker_sleep)
        p.start()
        self.assertTrue(p.is_alive())
        utils.stop_child_process(p)
        self.assertFalse(p.is_alive())

    @patch.dict(os.environ, {"CO_CPUS": "16"})
    def test_get_cpu_limit_co_cpus_env(self):
        """get_cpu_limit returns CO_CPUS value when set"""
        result = utils.get_cpu_limit()
        self.assertEqual(result, "16")

    @patch.dict(os.environ, {"AWS_BATCH_JOB_ID": "some-job-id"}, clear=False)
    def test_get_cpu_limit_aws_batch(self):
        """get_cpu_limit returns 1 when AWS_BATCH_JOB_ID is set (no CO_CPUS)"""
        env = {"AWS_BATCH_JOB_ID": "some-job-id"}
        with patch.dict(os.environ, env):
            # ensure CO_CPUS is not set
            os.environ.pop("CO_CPUS", None)
            result = utils.get_cpu_limit()
            self.assertEqual(result, 1)

    def test_get_cpu_limit_fallback(self):
        """get_cpu_limit returns a positive integer when no env vars are set"""
        keys_to_remove = ["CO_CPUS", "AWS_BATCH_JOB_ID", "SLURM_JOB_CPUS_PER_NODE"]
        clean_env = {k: v for k, v in os.environ.items() if k not in keys_to_remove}
        with patch.dict(os.environ, clean_env, clear=True):
            result = utils.get_cpu_limit()
            self.assertIsNotNone(result)

    @patch.dict(os.environ, {"CO_MEMORY": "64000000000"})
    def test_get_memory_limit_bytes_co_memory(self):
        """get_memory_limit_bytes returns the CO_MEMORY integer value when set"""
        result = utils.get_memory_limit_bytes()
        self.assertEqual(result, 64000000000)

    def test_get_memory_limit_bytes_fallback(self):
        """get_memory_limit_bytes returns a positive int via psutil fallback"""
        keys_to_remove = ["CO_MEMORY", "SLURM_MEM_PER_NODE", "SLURM_MEM_PER_CPU"]
        clean_env = {k: v for k, v in os.environ.items() if k not in keys_to_remove}
        with patch.dict(os.environ, clean_env, clear=True):
            result = utils.get_memory_limit_bytes()
            self.assertGreater(result, 0)

    def test_resource_monitor_collects_samples(self):
        """ResourceMonitor collects CPU and RAM samples and produces valid ResourceUsage"""
        from aind_data_schema.core.processing import ResourceUsage

        monitor = utils.ResourceMonitor(interval_seconds=0.05).start()
        time.sleep(0.25)
        monitor.stop()
        usage = monitor.to_resource_usage(cpu_cores=2)
        self.assertIsInstance(usage, ResourceUsage)
        self.assertGreater(len(usage.cpu_usage), 0)
        self.assertGreater(len(usage.ram_usage), 0)
        self.assertIsNotNone(usage.ram_unit)
        self.assertIsNotNone(usage.system_memory)

    def test_resource_monitor_context_manager(self):
        """ResourceMonitor works as a context manager and stops cleanly"""
        from aind_data_schema.core.processing import ResourceUsage

        with utils.ResourceMonitor(interval_seconds=0.05) as monitor:
            time.sleep(0.15)
        usage = monitor.to_resource_usage()
        self.assertIsInstance(usage, ResourceUsage)
        self.assertGreater(len(usage.cpu_usage), 0)


if __name__ == "__main__":
    unittest.main()
