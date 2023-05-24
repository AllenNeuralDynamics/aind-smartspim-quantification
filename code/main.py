"""
Main file to execute the smartspim segmentation
in code ocean
"""

import os
import subprocess
import sys

from aind_smartspim_quantification import quantification


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
    """
    Main function to execute the smartspim quantification
    in code ocean
    """
    image_path = quantification.main()
    bucket_path = "aind-open-data"

    output_folder = os.path.abspath("../results")

    dataset_folder = str(sys.argv[2]).split("/")[2]
    channel_name = str(sys.argv[4])

    dataset_name = dataset_folder + f"/processed/Quantification/{channel_name}"
    s3_path = f"s3://{bucket_path}/{dataset_name}"

    # Moving data to the quantification folder
    for out in execute_command_helper(
        f"aws s3 mv --recursive {output_folder} {s3_path}"
    ):
        print(out)

    save_string_to_txt(
        f"Results of cell quantification saved in: {s3_path}",
        f"{output_folder}/output_quantification.txt",
    )


if __name__ == "__main__":
    main()
