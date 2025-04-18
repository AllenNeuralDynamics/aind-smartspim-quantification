#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 13:30:28 2023

@author: nicholas.lusk
@modified by: camilo.laiton
"""

import copy
import json
import logging
import multiprocessing
import os
import pickle
import platform
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import ants
import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import ray
import vedo
from aind_data_schema.core.processing import (DataProcess, PipelineProcess,
                                              Processing)
from skimage import measure
from sklearn.metrics import normalized_mutual_info_score

from .._shared.types import PathLike

# initialize for multiprocessing
if not ray.is_initialized():
    ray.init(ignore_reinit_error=True, _plasma_directory=os.path.abspath("../scratch/"))


@ray.remote
def parallel_func(shared_coords, shared_path, struct, struct_tup):
    """
    Function to run multiprocess counting across regions

    Parameters
    ------------------------
    shared_coords: array
        array of registered cell locations in shared memory
    shared_path: Pathlike
        pathway to the CCF mesh files folder
    struct: str
        the CCF acromym for the given structure
    struct_tup: tuple
        contains the CCF structure ID and whether crosses the midline


    Returns
    ------------------------
    data_out: tuple
        has identification and count information for structure being processed

    """

    mesh_dir = os.path.join(shared_path, "json_verts_float/")
    with open("{}{}.json".format(mesh_dir, struct_tup[0])) as f:
        structure_data = json.loads(f.read())
        vertices, faces = (
            np.array(structure_data[str(struct_tup[0])]["vertices"]),
            np.array(structure_data[str(struct_tup[0])]["faces"]),
        )

        region = vedo.Mesh([vertices, faces])
        locations = region.inside_points(shared_coords).points()
        count = len(locations)

        # volume is in um**3 and density in cells/um**3
        region_vol = region.volume()
        count_density = count / region_vol

        if struct_tup[1] == "hemi":
            L_count = copy.copy(count)
            L_density = copy.copy(count_density)

            vertices_right = copy.copy(vertices)
            vertices_right[:, 0] = (
                vertices_right[:, 0] + (5700 - vertices_right[:, 0]) * 2
            )

            R_region = vedo.Mesh([vertices_right, faces])
            R_count = len(R_region.inside_points(shared_coords).points())
            R_density = R_count / region_vol

            count = L_count + R_count
            count_density = (L_density + R_density) / 2

        else:
            if count > 0:
                L_count = len(locations[np.where(locations[:, 0] < 5700)])
                R_count = len(locations[np.where(locations[:, 0] >= 5700)])
                L_density = L_count / (region_vol / 2)
                R_density = R_count / (region_vol / 2)
            else:
                L_count, R_count, L_density, R_density = 0, 0, 0, 0

        data_out = (
            struct_tup[0],
            struct,
            struct_tup[1],
            region_vol,
            L_count,
            R_count,
            count,
            L_density,
            R_density,
            count_density,
        )

        return data_out


class CellCounts:
    """
    Class of getting regional cell counts

    Parameters
    ------------------------
    resolution: int
        resolution of registration in microns

    """

    def __init__(self, ccf_dir: str, resolution: int = 25):
        """
        Initialization method of the CellCounts class

        Parameters
        ------------
        ccf_dir: str
            Path where the CCF directory with the
            meshes is located

        resolution: int
            CCF atlas resolution in microns
        """
        self.resolution = resolution
        self.annot_map = self.get_annotation_map(
            os.path.join(ccf_dir, "ccf_files/annotation_map.json")
        )
        self.CCF_dir = os.path.join(ccf_dir, "ccf_files/CCF_meshes")
        self.region_files = ["non_crossing_structures", "mid_crossing_structures"]

    def get_annotation_map(self, annotation_map_path: str):
        """
        Load .json will anotation dictionary {structure_id: structure_acronym}
        """

        with open(annotation_map_path) as json_file:
            mapping = json.load(json_file)

        return mapping

    def get_CCF_mesh_points(self, structure_id):
        """
        Gets mesh of specific brain region based on structure ID
        """

        mesh_dir = os.path.join(self.CCF_dir, "json_verts_float/")
        with open("{}{}.json".format(mesh_dir, structure_id)) as f:
            structure_data = json.loads(f.read())
            vertices, faces = (
                np.array(structure_data[str(structure_id)]["vertices"]),
                np.array(structure_data[str(structure_id)]["faces"]),
            )

        return vertices, faces

    def reflect_about_midline(self, vertices):
        """
        Brain regions that do not span the midline need to be flipped to get counts on rightside
        """

        vertices_right = copy.copy(vertices)
        vertices_right[:, 0] = vertices_right[:, 0] + (5700 - vertices_right[:, 0]) * 2
        return vertices_right

    def get_region_lists(self):
        """
        Import list of acronyms of brain regions
        """

        # Reading non-crossing structures to get acronyms
        with open(os.path.join(self.CCF_dir, self.region_files[0]), "rb") as f:
            hemi_struct = pickle.load(f)
            hemi_struct.remove(1051)  # don't know why this is being done
            hemi_labeled = [(s, "hemi") for s in hemi_struct]

        # Reading mid-crossing structures to get acronyms
        with open(os.path.join(self.CCF_dir, self.region_files[1]), "rb") as f:
            u = pickle._Unpickler(f)
            u.encoding = "latin1"
            mid_struct = u.load()
            mid_labeled = [(s, "mid") for s in mid_struct]

        self.structs = hemi_labeled + mid_labeled

    def get_metric_region_info(self, regions):
        """
        Get subest of CCF regions that are used for quantification

        Parameters
        ----------
        regions : list
            region IDs

        Returns
        -------
        info: dict
            pairing of region ID with info

        """
        self.get_region_lists()

        info = {}
        for struct_id, location in self.structs:
            if str(struct_id) in regions:
                acronym = self.annot_map[str(struct_id)]
                info[str(struct_id)] = [acronym, location]

        return info

    def crop_cells(self, cells, factor=0.99):
        """
        Removes cells outside and on the boundary of the CCF

        Parameters
        ------------------------
        cells: list
            list of cell locations after applying registration transformations

        factor: float
            factor by which you shrink the boundary of the CCF for removing edge cells

        Returns
        ------------------------
        cells_out: list
            list of cells that are within the scaled CCF
        """

        cells = cells[:, [2, 1, 0]]

        verts, faces = self.get_CCF_mesh_points("997")

        region = vedo.Mesh([verts, faces])
        com = region.center_of_mass()

        verts_scaled = com + factor * (verts - com)
        region_scaled = vedo.Mesh([verts_scaled, faces])

        cells_out = region_scaled.inside_points(cells).points()

        return cells_out

    def create_counts(self, cells):
        """
        Import list of acronyms of brain regions

        Parameters
        ------------------------
        cells: list
            list of cell locations after applying registration transformations
        Returns
        ------------------------
        struct_count: df
            dataframe with cell counts for each brain region
        """

        # convert cells to np.array() and convert to microns for counting
        cells = np.array(cells) * self.resolution

        # get list of all regions and region IDs from .json file
        self.get_region_lists()

        shared_coords = ray.put(cells)
        shared_path = ray.put(self.CCF_dir)

        results = [
            parallel_func.remote(
                shared_coords, shared_path, self.annot_map[str(s[0])], s
            )
            for s in self.structs
        ]
        data_out = ray.get(results)

        cols = [
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
        ]
        df_out = pd.DataFrame(data_out, columns=cols)
        return df_out


def get_orientation(params: dict) -> str:
    """
    Fetch aquisition orientation to identify origin for cell locations
    from cellfinder. Important for read_xml function in quantification
    script

    Parameters
    ----------
    params : dict
        The orientation information from processing_manifest.json

    Returns
    -------
    orient : str
        string that indicates axes order and direction current available
        options are:
            'spr'
            'sal'
        But more may be used later
    """

    orient = ["", "", ""]
    for vals in params:
        direction = vals["direction"].lower()
        dim = vals["dimension"]
        orient[dim] = direction[0]

    return "".join(orient)


def get_orientation_transform(orientation_in: str, orientation_out: str) -> tuple:
    """
    Takes orientation acronyms (i.e. spr) and creates a convertion matrix for
    converting from one to another

    Parameters
    ----------
    orientation_in : str
        the current orientation of image or cells (i.e. spr)
    orientation_out : str
        the orientation that you want to convert the image or
        cells to (i.e. ras)

    Returns
    -------
    tuple
        the location of the values in the identity matrix with values
        (original, swapped)
    """

    reverse_dict = {"r": "l", "l": "r", "a": "p", "p": "a", "s": "i", "i": "s"}

    input_dict = {dim.lower(): c for c, dim in enumerate(orientation_in)}
    output_dict = {dim.lower(): c for c, dim in enumerate(orientation_out)}

    transform_matrix = np.zeros((3, 3))
    for k, v in input_dict.items():
        if k in output_dict.keys():
            transform_matrix[v, output_dict[k]] = 1
        else:
            k_reverse = reverse_dict[k]
            transform_matrix[v, output_dict[k_reverse]] = -1

    if orientation_in.lower() == "spl" or orientation_out.lower() == "spl":
        transform_matrix = abs(transform_matrix)

    original, swapped = np.where(transform_matrix.T)

    return original, swapped, transform_matrix


def orient_image(img: np.array, orient_mat: np.array) -> np.array:
    """
    Orients array based on orientation matrix

    Parameters
    ----------
    img : np.array
        The image that is being oriented
    orient_mat : np.array
        identity matrix outlining the tranforms

    Returns
    -------
    img_out
        reoriented image
    """

    original, swapped = np.where(orient_mat)
    img_out = np.moveaxis(img, original, swapped)

    for c, row in enumerate(orient_mat.T):
        val = np.where(row)[0][0]
        if row[val] == -1:
            img_out = np.flip(img_out, c)

    return img_out


def get_template_info(file_path: PathLike) -> dict:
    """
    Collect relevent information from AntsImage for transforming points

    Parameters
    ----------
    file_path : PathLike
        path to an nifti file that contains an ANTsImage template

    Returns
    -------
    params: dict
        information from file needed to convert cells into correct physical
        space

    """

    ants_img = ants.image_read(file_path)

    params = {
        "orientation": ants_img.orientation,
        "dims": ants_img.dimension,
        "scale": ants_img.spacing,
        "origin": ants_img.origin,
        "direction": ants_img.direction[np.where(ants_img.direction != 0)],
    }

    return params


def get_volume(vertices, faces, split):
    if split == "hemi":
        break_pt = len(vertices) // 2
        vert_L, vert_R = vertices[:break_pt], vertices[break_pt:]

        region_L = vedo.Mesh([vert_L, faces])
        region_R = vedo.Mesh([vert_R, faces])

        volume = region_L.volume() + region_R.volume()
    else:
        region = vedo.Mesh([vertices, faces])
        volume = region.volume()

    return volume


def get_mesh_interior_points(mesh):
    """
    Collects all points that within a given vedo.Mesh

    Parameters:
    -----------
    mesh: vedo.mesh
        Mesh object of a given CCF region

    Returns:
    --------
    tuple
        The x, y and z coordinates of the points that fall within The
        mesh

    """

    bounds = mesh.bounds()
    region_array = mesh.binarize(spacing=(1, 1, 1)).tonumpy()

    indecies = np.where(region_array == 255)
    xs = indecies[0] + int(bounds[0])
    ys = indecies[1] + int(bounds[2])
    zs = indecies[2] + int(bounds[4])

    return (xs, ys, zs)


def get_intensity_mask(vertices, faces, mask, split):
    """
    Create binary mask of a given CCF region using the verticies and
    faces from JSON file

    Parameters:
    -----------
    vertices: list
        the location of nodes for a given mesh
    faces: list
        the conncetivity of the mesh nodes
    split: str
        indicates if the mesh is for a region that crosses the midline
        or is localized to a hemisphere. Options: ['mid', 'hemi']

    Returns:
    --------
    mask: np.array
        a 3D array in CCF space that masks the region being processed
    """
    if split == "hemi":
        break_pt = len(vertices) // 2
        vert_L, vert_R = vertices[:break_pt], vertices[break_pt:]

        region_L = vedo.Mesh([vert_L, faces])
        indicies = get_mesh_interior_points(region_L)
        mask[indicies] = 1

        region_R = vedo.Mesh([vert_R, faces])
        indicies = get_mesh_interior_points(region_R)
        mask[indicies] = 1
    else:
        region = vedo.Mesh([vertices, faces])
        indicies = get_mesh_interior_points(region)
        mask[indicies] = 1

    return mask


def normalized_mutual_information(
    ccf_img: np.array, img: np.array, mask: np.array
) -> float:
    """
    Method to compute the mutual information error metric using numpy.
    Note: Check the used dtype to reach a higher precision in the metric

    See: Normalised Mutual Information of: A normalized entropy
    measure of 3-D medical image alignment,
    Studholme,  jhill & jhawkes (1998).

    Parameters
    ------------------------
    ccf_img: np.array
        2D/3D patch of extracted from the image 1
        and based on a windowed point.

    img: np.array
        2D/3D patch of extracted from the image 2
        and based on a windowed point.

    mask: np.array
        2D/

    Returns
    ------------------------
    float
        Float with the value of the mutual information error.
    """

    ccf_img = ccf_img.astype(int)

    # mutual information is invariant to scaling so this should not matter
    if img.dtype == np.dtype(np.float32):
        img = (img - img.min()) / (img.max() - img.min()) * ccf_img.max()
        img = img.astype(int)

    patch_1 = np.where(mask > 0, ccf_img, 0)
    patch_2 = np.where(mask > 0, img, 0)

    # flatten arrays
    patch_1 = patch_1.flatten()
    patch_2 = patch_2.flatten()

    # Compute the Normalized Mutual Information between the pixel distributions
    nmi = normalized_mutual_info_score(patch_1, patch_2, average_method="geometric")

    return nmi


def get_region_intensity(img, mask):
    """
    Convert region mask back to original values
    """
    masked_img = np.where(mask > 0, img, 0)
    return masked_img


def get_plot_planes(mask, split):
    if split == "hemi":
        mask = mask[:, :, : mask.shape[3] // 2]

    props = measure.regionprops(mask)
    planes = props[0].centroid

    return [int(p) for p in planes]


def plot_overlays(img: np.array, mask: np.array, planes: list):
    mask = np.where(mask == 0, np.nan, mask)
    vmax = mask.max()

    fig, ax = plt.subplots(nrows=1, ncols=3)

    i = img[planes[0], :, :]
    m = mask[planes[0], :, :]
    ax[0].imshow(i)
    ax[0].imshow(m, cmap="jet_r", vmax=vmax, alpha=0.6)

    i = img[:, planes[1], :]
    m = mask[:, planes[1], :]
    ax[1].imshow(i)
    ax[1].imshow(m, cmap="jet_r", vmax=vmax, alpha=0.6)

    i = img[:, :, planes[2]]
    m = mask[:, :, planes[2]]
    ax[2].imshow(i)
    ax[2].imshow(m, cmap="jet_r", vmax=vmax, alpha=0.6)


def __read_zarr_image(image_path: PathLike):
    """
    Reads a zarr image

    Parameters
    -------------
    image_path: PathLike
        Path where the zarr image is located

    Returns
    -------------
    da.core.Array
        Dask array with the zarr image
    """

    image_path = str(image_path)
    signal_array = da.from_zarr(image_path)

    return signal_array


def save_string_to_txt(txt: str, filepath: str, mode="w") -> None:
    """
    Saves a text in a file in the given mode.

    Parameters
    ------------------------
    txt: str
        String to be saved.

    filepath: PathLike
        Path where the file is located or will be saved.

    mode: str
        File open mode.

    """

    with open(filepath, mode) as file:
        file.write(txt + "\n")


def read_json_as_dict(filepath: str) -> dict:
    """
    Reads a json as dictionary.
    Parameters
    ------------------------
    filepath: PathLike
        Path where the json is located.
    Returns
    ------------------------
    dict:
        Dictionary with the data the json has.
    """

    dictionary = {}

    if os.path.exists(filepath):
        with open(filepath) as json_file:
            dictionary = json.load(json_file)

    return dictionary


def create_folder(dest_dir: str, verbose: bool = False) -> None:
    """
    Create new folders.

    Parameters
    ------------------------

    dest_dir: str
        Path where the folder will be created if it does not exist.

    verbose: Optional[bool]
        If we want to show information about the folder status. Default False.

    Raises
    ------------------------

    OSError:
        if the folder exists.

    """

    if not (os.path.exists(dest_dir)):
        try:
            if verbose:
                print(f"Creating new directory: {dest_dir}")
            os.makedirs(dest_dir)
        except OSError as e:
            if e.errno != os.errno.EEXIST:
                raise


def generate_processing(
    data_processes: List[DataProcess],
    dest_processing: PathLike,
    processor_full_name: str,
    pipeline_version: str,
):
    """
    Generates data description for the output folder.

    Parameters
    ------------------------

    data_processes: List[dict]
        List with the processes aplied in the pipeline.

    dest_processing: PathLike
        Path where the processing file will be placed.

    processor_full_name: str
        Person in charged of running the pipeline
        for this data asset

    pipeline_version: str
        Terastitcher pipeline version

    """
    # flake8: noqa: E501
    processing_pipeline = PipelineProcess(
        data_processes=data_processes,
        processor_full_name=processor_full_name,
        pipeline_version=pipeline_version,
        pipeline_url="https://github.com/AllenNeuralDynamics/aind-smartspim-pipeline",
        note="Metadata for fusion step",
    )

    processing = Processing(
        processing_pipeline=processing_pipeline,
        notes="This processing only contains metadata about fusion \
            and needs to be compiled with other steps at the end",
    )

    processing.write_standard_file(output_directory=dest_processing)


def profile_resources(
    time_points: List,
    cpu_percentages: List,
    memory_usages: List,
    monitoring_interval: int,
):
    """
    Profiles compute resources usage.

    Parameters
    ----------
    time_points: List
        List to save all the time points
        collected

    cpu_percentages: List
        List to save the cpu percentages
        during the execution

    memory_usage: List
        List to save the memory usage
        percentages during the execution

    monitoring_interval: int
        Monitoring interval in seconds
    """
    start_time = time.time()

    while True:
        current_time = time.time() - start_time
        time_points.append(current_time)

        # CPU Usage
        cpu_percent = psutil.cpu_percent(interval=monitoring_interval)
        cpu_percentages.append(cpu_percent)

        # Memory usage
        memory_info = psutil.virtual_memory()
        memory_usages.append(memory_info.percent)

        time.sleep(monitoring_interval)


def generate_resources_graphs(
    time_points: List,
    cpu_percentages: List,
    memory_usages: List,
    output_path: str,
    prefix: str,
):
    """
    Profiles compute resources usage.

    Parameters
    ----------
    time_points: List
        List to save all the time points
        collected

    cpu_percentages: List
        List to save the cpu percentages
        during the execution

    memory_usage: List
        List to save the memory usage
        percentages during the execution

    output_path: str
        Path where the image will be saved

    prefix: str
        Prefix name for the image
    """
    time_len = len(time_points)
    memory_len = len(memory_usages)
    cpu_len = len(cpu_percentages)

    min_len = min([time_len, memory_len, cpu_len])
    if not min_len:
        return

    plt.figure(figsize=(10, 6))

    plt.subplot(2, 1, 1)
    plt.plot(time_points[:min_len], cpu_percentages[:min_len], label="CPU Usage")
    plt.xlabel("Time (s)")
    plt.ylabel("CPU Usage (%)")
    plt.title("CPU Usage Over Time")
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(time_points[:min_len], memory_usages[:min_len], label="Memory Usage")
    plt.xlabel("Time (s)")
    plt.ylabel("Memory Usage (%)")
    plt.title("Memory Usage Over Time")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{output_path}/{prefix}_compute_resources.png", bbox_inches="tight")


def stop_child_process(process: multiprocessing.Process):
    """
    Stops a process

    Parameters
    ----------
    process: multiprocessing.Process
        Process to stop
    """
    process.terminate()
    process.join()


def get_size(bytes, suffix: str = "B") -> str:
    """
    Scale bytes to its proper format
    e.g:
        1253656 => '1.20MB'
        1253656678 => '1.17GB'

    Parameters
    ----------
    bytes: bytes
        Bytes to scale

    suffix: str
        Suffix used for the conversion
    """
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if bytes < factor:
            return f"{bytes:.2f}{unit}{suffix}"
        bytes /= factor


def get_code_ocean_cpu_limit():
    """
    Gets the Code Ocean capsule CPU limit

    Returns
    -------
    int:
        number of cores available for compute
    """
    # Checks for environmental variables
    co_cpus = os.environ.get("CO_CPUS")
    aws_batch_job_id = os.environ.get("AWS_BATCH_JOB_ID")

    if co_cpus:
        return co_cpus
    if aws_batch_job_id:
        return 1

    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as fp:
            cfs_quota_us = int(fp.read())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as fp:
            cfs_period_us = int(fp.read())

        container_cpus = cfs_quota_us // cfs_period_us

    except FileNotFoundError as e:
        container_cpus = 0

    # For physical machine, the `cfs_quota_us` could be '-1'
    return psutil.cpu_count(logical=False) if container_cpus < 1 else container_cpus


def print_system_information(logger: logging.Logger):
    """
    Prints system information

    Parameters
    ----------
    logger: logging.Logger
        Logger object
    """
    co_memory = os.environ.get("CO_MEMORY")
    co_memory = int(co_memory) if co_memory else None

    # System info
    sep = "=" * 40
    logger.info(f"{sep} Code Ocean Information {sep}")
    logger.info(f"Code Ocean assigned cores: {get_code_ocean_cpu_limit()}")

    if co_memory:
        logger.info(f"Code Ocean assigned memory: {get_size(co_memory)}")

    logger.info(f"Computation ID: {os.environ.get('CO_COMPUTATION_ID')}")
    logger.info(f"Capsule ID: {os.environ.get('CO_CAPSULE_ID')}")
    logger.info(f"Is pipeline execution?: {bool(os.environ.get('AWS_BATCH_JOB_ID'))}")

    logger.info(f"{sep} System Information {sep}")
    uname = platform.uname()
    logger.info(f"System: {uname.system}")
    logger.info(f"Node Name: {uname.node}")
    logger.info(f"Release: {uname.release}")
    logger.info(f"Version: {uname.version}")
    logger.info(f"Machine: {uname.machine}")
    logger.info(f"Processor: {uname.processor}")

    # Boot info
    logger.info(f"{sep} Boot Time {sep}")
    boot_time_timestamp = psutil.boot_time()
    bt = datetime.fromtimestamp(boot_time_timestamp)
    logger.info(
        f"Boot Time: {bt.year}/{bt.month}/{bt.day} {bt.hour}:{bt.minute}:{bt.second}"
    )

    # CPU info
    logger.info(f"{sep} CPU Info {sep}")
    # number of cores
    logger.info(f"Physical node cores: {psutil.cpu_count(logical=False)}")
    logger.info(f"Total node cores: {psutil.cpu_count(logical=True)}")

    # CPU frequencies
    cpufreq = psutil.cpu_freq()
    logger.info(f"Max Frequency: {cpufreq.max:.2f}Mhz")
    logger.info(f"Min Frequency: {cpufreq.min:.2f}Mhz")
    logger.info(f"Current Frequency: {cpufreq.current:.2f}Mhz")

    # CPU usage
    logger.info("CPU Usage Per Core before processing:")
    for i, percentage in enumerate(psutil.cpu_percent(percpu=True, interval=1)):
        logger.info(f"Core {i}: {percentage}%")
    logger.info(f"Total CPU Usage: {psutil.cpu_percent()}%")

    # Memory info
    logger.info(f"{sep} Memory Information {sep}")
    # get the memory details
    svmem = psutil.virtual_memory()
    logger.info(f"Total: {get_size(svmem.total)}")
    logger.info(f"Available: {get_size(svmem.available)}")
    logger.info(f"Used: {get_size(svmem.used)}")
    logger.info(f"Percentage: {svmem.percent}%")
    logger.info(f"{sep} Memory - SWAP {sep}")
    # get the swap memory details (if exists)
    swap = psutil.swap_memory()
    logger.info(f"Total: {get_size(swap.total)}")
    logger.info(f"Free: {get_size(swap.free)}")
    logger.info(f"Used: {get_size(swap.used)}")
    logger.info(f"Percentage: {swap.percent}%")

    # Network information
    logger.info(f"{sep} Network Information {sep}")
    # get all network interfaces (virtual and physical)
    if_addrs = psutil.net_if_addrs()
    for interface_name, interface_addresses in if_addrs.items():
        for address in interface_addresses:
            logger.info(f"=== Interface: {interface_name} ===")
            if str(address.family) == "AddressFamily.AF_INET":
                logger.info(f"  IP Address: {address.address}")
                logger.info(f"  Netmask: {address.netmask}")
                logger.info(f"  Broadcast IP: {address.broadcast}")
            elif str(address.family) == "AddressFamily.AF_PACKET":
                logger.info(f"  MAC Address: {address.address}")
                logger.info(f"  Netmask: {address.netmask}")
                logger.info(f"  Broadcast MAC: {address.broadcast}")
    # get IO statistics since boot
    net_io = psutil.net_io_counters()
    logger.info(f"Total Bytes Sent: {get_size(net_io.bytes_sent)}")
    logger.info(f"Total Bytes Received: {get_size(net_io.bytes_recv)}")


def create_logger(output_log_path: PathLike):
    """
    Creates a logger that generates
    output logs to a specific path.

    Parameters
    ------------
    output_log_path: PathLike
        Path where the log is going
        to be stored

    Returns
    -----------
    logging.Logger
        Created logger pointing to
        the file path.
    """

    CURR_DATE_TIME = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    LOGS_FILE = f"{output_log_path}/fusion_log_{CURR_DATE_TIME}.log"

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s : %(message)s",
        datefmt="%Y-%m-%d %H:%M",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOGS_FILE, "a"),
        ],
        force=True,
    )

    logging.disable("DEBUG")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    return logger


def check_path_instance(obj: object) -> bool:
    """
    Checks if an objects belongs to pathlib.Path subclasses.

    Parameters
    ------------------------

    obj: object
        Object that wants to be validated.

    Returns
    ------------------------

    bool:
        True if the object is an instance of Path subclass, False otherwise.
    """

    for childclass in Path.__subclasses__():
        if isinstance(obj, childclass):
            return True

    return False


def save_dict_as_json(
    filename: str, dictionary: dict, verbose: Optional[bool] = False
) -> None:
    """
    Saves a dictionary as a json file.

    Parameters
    ------------------------

    filename: str
        Name of the json file.

    dictionary: dict
        Dictionary that will be saved as json.

    verbose: Optional[bool]
        True if you want to print the path where the file was saved.

    """

    if dictionary is None:
        dictionary = {}

    else:
        for key, value in dictionary.items():
            # Converting path to str to dump dictionary into json
            if check_path_instance(value):
                # TODO fix the \\ encode problem in dump
                dictionary[key] = str(value)

    with open(filename, "w") as json_file:
        json.dump(dictionary, json_file, indent=4)

    if verbose:
        print(f"- Json file saved: {filename}")
