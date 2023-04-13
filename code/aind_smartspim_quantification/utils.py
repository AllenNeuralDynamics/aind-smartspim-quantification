#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 13:30:28 2023

@author: nicholas.lusk
@modified by: camilo.laiton
"""

import copy
import json
import os
import pickle

import numpy as np
import pandas as pd
import ray
import vtkplotter

# initialize for multiprocessing
if not ray.is_initialized():
    ray.init(num_cpus=os.cpu_count(), ignore_reinit_error=True)


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

        region = vtkplotter.Mesh([vertices, faces])
        count = len(region.insidePoints(shared_coords))  # .points())

        if struct_tup[1] == "hemi":
            L_count = copy.copy(count)

            vertices_right = copy.copy(vertices)
            vertices_right[:, 0] = (
                vertices_right[:, 0] + (5700 - vertices_right[:, 0]) * 2
            )

            R_region = vtkplotter.Mesh([vertices_right, faces])
            R_count = len(R_region.insidePoints(shared_coords))  # .points())

            count = L_count + R_count

        else:
            L_count, R_count = np.NaN, np.NaN

        data_out = (struct_tup[0], struct, struct_tup[1], L_count, R_count, count)

        return data_out


class CellCounts:
    """
    Class of getting regional cell counts

    Parameters
    ------------------------
    resolution: int
        resolution of registration in microns

    """

    def __init__(self, ccf_dir: str, annotation_map_path: str, resolution: int = 25):
        self.resolution = resolution
        self.annot_map = self.get_annotation_map(
            os.path.join(ccf_dir, "annotation_map.json")
        )
        self.CCF_dir = os.path.join(ccf_dir, "CCF_meshes")
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

    def create_counts(self, cells):
        """
        Import list of acronyms of brain regions

        Parameters
        ------------------------
        cells: list
            list of cell locations after applying registration transformations
        Parameters
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

        cols = ["Id", "Structure", "Struct_Info", "Left", "Right", "Total"]
        df_out = pd.DataFrame(data_out, columns=cols)
        return df_out


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
