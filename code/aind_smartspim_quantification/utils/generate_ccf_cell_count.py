"""
Script to generate CCF + cell counts
"""

import inspect
import struct
import json
import logging
import multiprocessing
import os
import time
from pathlib import Path
from typing import Optional, Union, List

import boto3
import numpy as np
import pandas as pd
import xmltodict
import dask.array as da
from multiprocessing.managers import BaseManager, NamespaceProxy

from .utils import create_folder

# IO types
PathLike = Union[str, Path]


def get_ccf(
    out_path: str,
    bucket_name: Optional[str] = "tissuecyte-visualizations",
    s3_folder: Optional[str] = "data/221205/ccf_annotations/",
):
    """
    Parameters
    ----------
    out_path : str
        path to where the precomputed segmentation map will be stored
    bucket_name: Optional[str]
        Bucket name where the precomputed annotation is stored for the
        CCF
    s3_folder: Optional[str]
        Path inside of the bucket where the annotations are stored

    """

    # location of the data from tissueCyte, but can get our own and change to aind-open-data

    s3_resource = boto3.resource("s3")
    bucket = s3_resource.Bucket(bucket_name)

    for obj in bucket.objects.filter(Prefix=s3_folder):
        target = os.path.join(out_path, os.path.relpath(obj.key, s3_folder))

        # dont currently need 10um data so we should skip
        if "10000_10000_10000" in obj.key:
            continue

        if not os.path.exists(os.path.dirname(target)):
            os.makedirs(os.path.dirname(target))

        # dont try and download folders
        if obj.key[-1] == "/":
            continue

        bucket.download_file(obj.key, target)
        
def get_points_from_xml(path: PathLike, encoding: str = "utf-8") -> List[dict]:
    """
    Function to parse the points from the
    cell segmentation capsule.

    Parameters
    -----------------

    Path: PathLike
        Path where the XML is stored.

    encoding: str
        XML encoding. Default: "utf-8"

    Returns
    -----------------
    List[dict]
        List with the location of the points.
    """

    with open(path, "r", encoding=encoding) as xml_reader:
        xml_file = xml_reader.read()

    xml_dict = xmltodict.parse(xml_file)
    cell_data = xml_dict["CellCounter_Marker_File"]["Marker_Data"][
        "Marker_Type"
    ]["Marker"]

    new_cell_data = []
    for cell in cell_data:
        new_cell_data.append(
            {"x": cell["MarkerX"], "y": cell["MarkerY"], "z": cell["MarkerZ"],}
        )

    return new_cell_data

def calculate_dynamic_range(image_path: PathLike, percentile = 99, level = 3):
    """
    Calculates the default dynamic range for teh neuroglancer link
    using a defined percentile from the downsampled zarr

    Parameters
    ----------
    image_path : PathLike
        location of the zarr used for classification
    percentile : int
        The top percentile value for setting the dynamic range
    level : int
        level of zarr to use for calculating percentile

    Returns
    -------
    dynamic_ranges : list
        The dynamic range and window range values for zarr

    """

    img = da.from_zarr(image_path, str(level)).squeeze()
    range_max = da.percentile(img.flatten(), percentile).compute()[0]
    window_max = int(range_max * 1.5)
    dynamic_ranges = [int(range_max), window_max]

    return dynamic_ranges

class ObjProxy(NamespaceProxy):
    """Returns a proxy instance for any user defined data-type. The proxy instance will have the namespace and
    functions of the data-type (except private/protected callables/attributes). Furthermore, the proxy will be
    pickable and can its state can be shared among different processes."""

    @classmethod
    def populate_obj_attributes(cls, real_cls):
        """
        Populates attributes of the proxy object
        """
        DISALLOWED = set(dir(cls))
        ALLOWED = [
            "__sizeof__",
            "__eq__",
            "__ne__",
            "__le__",
            "__repr__",
            "__dict__",
            "__lt__",
            "__gt__",
        ]
        DISALLOWED.add("__class__")
        new_dict = {}
        for attr, value in inspect.getmembers(real_cls, callable):
            if attr not in DISALLOWED or attr in ALLOWED:
                new_dict[attr] = cls._proxy_wrap(attr)
        return new_dict

    @staticmethod
    def _proxy_wrap(attr):
        """
        This method creates function that calls the proxified object's method.
        """

        def f(self, *args, **kwargs):
            """
            Function that calls the proxified object's method.
            """
            return self._callmethod(attr, args, kwargs)

        return f


def buf_builder(x, y, z, buf_):
    """builds the buffer"""
    pt_buf = struct.pack("<3f", x, y, z)
    buf_.extend(pt_buf)


attributes = ObjProxy.populate_obj_attributes(bytearray)
bytearrayProxy = type("bytearrayProxy", (ObjProxy,), attributes)


def generate_precomputed_cells(cells, precompute_path, configs):
    """
    Function for saving precomputed annotation layer

    Parameters
    -----------------

    cells: dict
        output of the xmltodict function for importing cell locations
    precomputed_path: str
        path to where you want to save the precomputed files
    comfigs: dict
        data on the space that the data will be viewed

    """

    BaseManager.register(
        "bytearray",
        bytearray,
        bytearrayProxy,
        exposed=tuple(dir(bytearrayProxy)),
    )
    manager = BaseManager()
    manager.start()

    buf = manager.bytearray()

    cell_list = []
    for idx, cell in cells.iterrows():
        cell_list.append([int(cell["z"]), int(cell["y"]), int(cell["x"])])

    l_bounds = np.min(cell_list, axis=0)
    u_bounds = np.max(cell_list, axis=0)

    output_path = os.path.join(precompute_path, "spatial0")
    create_folder(output_path)

    metadata = {
        "@type": "neuroglancer_annotations_v1",
        "dimensions": dict((key, configs['dimensions'][key]) for key in ('z', 'y', 'x')),
        "lower_bound": [float(x) for x in l_bounds],
        "upper_bound": [float(x) for x in u_bounds],
        "annotation_type": "point",
        "properties": [],
        "relationships": [],
        "by_id": {"key": "by_id",},
        "spatial": [
            {
                "key": "spatial0",
                "grid_shape": [1] * configs['rank'],
                "chunk_size": [max(1, float(x)) for x in u_bounds - l_bounds],
                "limit": len(cell_list),
            },
        ],
    }

    with open(os.path.join(precompute_path, "info"), "w") as f:
        f.write(json.dumps(metadata))

    with open(os.path.join(output_path, "0_0_0"), "wb") as outfile:
        start_t = time.time()

        total_count = len(cell_list)  # coordinates is a list of tuples (x,y,z)

        print("Running multiprocessing")

        if not isinstance(buf, type(None)):
            buf.extend(struct.pack("<Q", total_count))

            with multiprocessing.Pool(processes=os.cpu_count()) as p:
                p.starmap(
                    buf_builder, [(x, y, z, buf) for (x, y, z) in cell_list]
                )

            # write the ids at the end of the buffer as increasing integers
            id_buf = struct.pack(
                "<%sQ" % len(cell_list), *range(len(cell_list))
            )
            buf.extend(id_buf)
        else:
            buf = struct.pack("<Q", total_count)

            for x, y, z in cell_list:
                pt_buf = struct.pack("<3f", x, y, z)
                buf += pt_buf

            # write the ids at the end of the buffer as increasing integers
            id_buf = struct.pack(
                "<%sQ" % len(cell_list), *range(len(cell_list))
            )
            buf += id_buf

        print(
            "Building file took {0} minutes".format(
                (time.time() - start_t) / 60
            )
        )

        outfile.write(bytes(buf))


def generate_cff_segmentation(
    input_path: str, output_path: str, ccf_reference_path: Optional[str] = None
):
    """
    Function for creating segmentation layer with cell counts

    Parameters
    -----------------

    input_path: str
        path to file cell_count_by_region.csv from generated from "aind-smartspim-quantification"
    output_path: str
        path to where you want to save the precomputed files
    """
    # check that save path exists and if not create
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    df_count = pd.read_csv(input_path, index_col=0)
    include = list(df_count["Acronym"].values)

    # get CCF id-struct pairings
    if ccf_reference_path is None:
        ccf_reference_path = os.path.join(
            Path(os.path.dirname(os.path.realpath(__file__))).parent,
            "ccf_files/ccf_ref.csv",
        )

    df_ccf = pd.read_csv(ccf_reference_path)

    keep_ids = []
    keep_struct = []
    for r, irow in df_ccf.iterrows():
        if irow["struct"] in include:
            keep_ids.append(str(irow["id"]))
            total = df_count.loc[
                df_count["Acronym"] == irow["struct"], ["Total"]
            ].values.squeeze()
            keep_struct.append(irow["struct"] + " cells: " + str(total))

    # download ccf procomputed format
    get_ccf(output_path)

    # currently using 25um resolution so need to drop 10um data or NG finicky
    with open(os.path.join(output_path, "info"), "r") as f:
        info_file = json.load(f)

    info_file["scales"].pop(0)

    with open(os.path.join(output_path, "info"), "w") as f:
        json.dump(info_file, f, indent=2)

    # build json for segmentation properties
    data = {
        "@type": "neuroglancer_segment_properties",
        "inline": {
            "ids": keep_ids,
            "properties": [{"id": "label", "type": "label", "values": keep_struct}],
        },
    }

    with open(os.path.join(output_path, "segment_properties/info"), "w") as outfile:
        json.dump(data, outfile, indent=2)
        
def generate_25_um_ccf_cells(
    cells_df: pd.DataFrame,
    ng_configs: dict,
    smartspim_config: dict,
    dynamic_range: list,
    logger: logging.Logger,
    bucket = 'aind-open-data'
):
    """
    Creates the json state dictionary for the neuroglancer link

    Parameters
    ----------
    cells_df: pd.DataFrame
        the location of all the cells from proposal phase
    ng_configs : dict
        Parameters for creating neuroglancer link defined in run_capsule.py
    smartspim_config : dict
        Dataset specific parameters from processing_manifest
    dynamic_range : list
        The intensity range calculated from the zarr
    logger: logging.Logger
    bucket: str
        Location on AWS where the data lives
    
    Returns
    -------
    json_state : dict
        fully configured JSON for neuroglancer visualization
    """
    
    generate_cff_segmentation(
        smartspim_config["ccf_overlay_precomputed"]["input_path"],
        smartspim_config["ccf_overlay_precomputed"]["output_path"],
        smartspim_config["ccf_overlay_precomputed"]["ccf_reference_path"],
    )

    output_precomputed = os.path.join(smartspim_config["save_path"], "visualization/cell_points_precomputed")
    create_folder(output_precomputed)
    print(f"Output cells precomputed: {output_precomputed}")
 
    generate_precomputed_cells(
        cells_df, 
        precompute_path = output_precomputed, 
        configs = ng_configs
    )
    
    ng_path = f"s3://{bucket}/{smartspim_config['name']}/image_cell_quantification/{smartspim_config['channel_name']}/visualization/neuroglancer_config.json"

    json_state = {
        "ng_link": f"{ng_configs['base_url']}{ng_path}",
        "title": smartspim_config['name'].split("_")[1],
        "dimensions": ng_configs["dimensions"],
        "crossSectionOrientation": [0.0, 1.0, 0.0, 0.0],
        "crossSectionScale": ng_configs["crossSectionScale"],
        "projectionScale": ng_configs['projectionScale'],
        "layers": [
            {
                "source": f"zarr://s3://{bucket}/{smartspim_config['name']}/image_atlas_alignment/{smartspim_config['channel_name']}/OMEZarr/image.zarr",
                "type": "image",
                "tab": "rendering",
                "shader": '#uicontrol vec3 color color(default="#ffffff")\n#uicontrol invlerp normalized\nvoid main() {\nemitRGB(color * normalized());\n}',
                "shaderControls": {
                    "normalized": {
                        "range": [0, dynamic_range[0]], 
                        "window": [0, dynamic_range[1]],
                    },
                },
                "name": f"Channel: {smartspim_config['channel_name']}"
            },
            {
                "source": f"precomputed://s3://{bucket}/{smartspim_config['name']}/image_cell_quantification/{smartspim_config['channel_name']}/visualization/cell_points_precomputed",
                "type": "annotation",
                "tool": "annotatePoint",
                "tab": "annotations",
                "crossSectionAnnotationSpacing": 1.0,
                "name": "Registered Cells" 
            },
            {
                "source": f"precomputed://s3://{bucket}/{smartspim_config['name']}/image_cell_quantification/{smartspim_config['channel_name']}/visualization/ccf_cell_precomputed",
                "type": "segmentation",
                "tab": "segments",
                "name": "CCF Overlay" 
            },
        ],
        "gpuMemoryLimit": ng_configs['gpuMemoryLimit'],
        "selectedLayer": {"visible": True, "layer": f"Channel: {smartspim_config['channel_name']}"},
        "layout": "4panel",
    }
    
    logger.info(f"Visualization link: {json_state['ng_link']}")
    output_path = os.path.join(smartspim_config['save_path'], "visualization/neuroglancer_config.json")

    with open(output_path, "w") as outfile:
        json.dump(json_state, outfile, indent=2)
        
    return output_path

if __name__ == "__main__":
    params  = {
        "ccf_cells_precomputed": {  # Parameters to generate CCF + Cells precomputed format
            "input_path": "/Users/camilo.laiton/Downloads/cell_count_by_region.csv",  # Path where the cell_count.csv is located
            "output_path": "/Users/camilo.laiton/repositories/new_ng_link/aind-ng-link/src/ng_link/scripts/CCF_Cells_Test",  # Path where we want to save the CCF + cell location precomputed
            "ccf_reference_path": None,  # Path where the CCF reference csv is located
        },
        "cells_precomputed": {  # Parameters to generate cell points precomputed format
            "xml_path": "/Users/camilo.laiton/Downloads/transformed_cells.xml",  # Path where the cell points are located
            "output_precomputed": "/Users/camilo.laiton/repositories/new_ng_link/aind-ng-link/src/ng_link/scripts/Cells_Test",  # Path where the precomputed format will be stored
        },
        "zarr_path": "s3://aind-open-data/SmartSPIM_656374_2023-01-27_12-41-55_stitched_2023-01-31_17-28-34/processed/CCF_Atlas_Registration/Ex_445_Em_469/OMEZarr/image.zarr",  # Path where the 25 um zarr image is stored, output from CCF capsule
    }

    generate_25_um_ccf_cells(params)
