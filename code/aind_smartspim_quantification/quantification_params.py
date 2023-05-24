"""
Defines the parameters used in the quantification script
"""
from argschema import ArgSchema
from argschema.fields import InputDir, Int, Str


class QuantificationParams(ArgSchema):
    """
    Quantification parameters
    """

    dataset_path = InputDir(
        required=True, metadata={"description": "Path where the data is located"},
    )

    channel_name = Str(
        required=True, metadata={"description": "Dataset's channel name"},
    )

    save_path = Str(
        required=False,
        metadata={"description": "Folder where we want to output files"},
        dump_default="/results/",
    )

    intermediate_folder = Str(
        required=False,
        metadata={"description": "Intermediate folder where the image data is stored"},
        dump_default="processed/OMEZarr",
    )

    downsample_res = Int(
        required=False,
        metadata={"description": "Zarr scale that was used in CCF step"},
        dump_default=3,
    )

    reference_microns_ccf = Int(
        required=False,
        metadata={"description": "Reference resolution in um used in CCF step"},
        dump_default=25,
    )
