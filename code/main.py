"""
Main file to execute the smartspim segmentation
in code ocean
"""

import json
import logging
import os
import subprocess
import sys
from glob import glob

from aind_smartspim_quantification import quantification

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s : %(message)s",
    datefmt="%Y-%m-%d %H:%M",
    handlers=[
        logging.StreamHandler(),
        # logging.FileHandler("test.log", "a"),
    ],
)
logging.disable("DEBUG")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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


def execute_command_helper(command: str, print_command: bool = False) -> None:
    """
    Execute a shell command.

    Parameters
    ------------------------
    command: str
        Command that we want to execute.
    print_command: bool
        Bool that dictates if we print the command in the console.

    Raises
    ------------------------
    CalledProcessError:
        if the command could not be executed (Returned non-zero status).

    """

    if print_command:
        print(command)

    popen = subprocess.Popen(
        command, stdout=subprocess.PIPE, universal_newlines=True, shell=True
    )
    for stdout_line in iter(popen.stdout.readline, ""):
        yield str(stdout_line).strip()
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def main():
    """
    Main function to execute the smartspim quantification
    in code ocean
    """
    data_folder = os.path.abspath("../data/")
    processing_manifest_path = glob(f"{data_folder}/processing_manifest_*")[0]

    if not os.path.exists(processing_manifest_path):
        raise ValueError("Processing manifest path does not exist!")

    pipeline_config = read_json_as_dict(processing_manifest_path)

    # Creating the quantification config based on pipeline parameters
    quantification_config = {
        "fused_folder": os.path.abspath(
            f"{pipeline_config['quantification']['fused_folder']}"
        ),
        "ccf_registration_folder": os.path.abspath(
            f"../data/ccf_{pipeline_config['quantification']['channel']}"
        ),
        "cell_segmentation_folder": os.path.abspath(
            f"../data/cell_{pipeline_config['quantification']['channel']}"
        ),
        "stitched_s3_path": pipeline_config["stitching"]["s3_path"],
        "channel_name": pipeline_config["quantification"]["channel"],
        "save_path": f"{pipeline_config['quantification']['save_path']}/quant_{pipeline_config['quantification']['channel']}",
        "downsample_res": pipeline_config["registration"]["input_scale"],
        "reference_microns_ccf": 25,
        "bucket_path": pipeline_config["bucket"],
    }
    logger.info(f"Provided quantification configuration: {quantification_config}")
    sys.argv = [sys.argv[0]]

    image_path = quantification.main(quantification_config)


if __name__ == "__main__":
    main()
