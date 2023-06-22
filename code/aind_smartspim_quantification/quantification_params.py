""" Parameters used in the quantification script """
from argschema import ArgSchema
from argschema.fields import Int, Str


class QuantificationParams(ArgSchema):
    """
    Quantification parameters
    """

    fused_folder = Str(
        required=True,
        metadata={"description": "Path where the data is located"},
    )

    ccf_registration_folder = Str(
        required=True,
        metadata={"description": "Path where the ccf registered data is located"},
    )

    cell_segmentation_folder = Str(
        required=True,
        metadata={"description": "Path where the cell segmentation data is located"},
    )

    channel_name = Str(
        required=True,
        metadata={"description": "Dataset's channel name"},
    )

    stitched_s3_path = Str(
        required=True,
        metadata={
            "description": "Path where the stitched data is located in the cloud"
        },
    )

    save_path = Str(
        required=False,
        metadata={"description": "Folder where we want to output files"},
        dump_default="../results/",
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

    bucket_path = Str(
        required=True,
        metadata={"description": "Amazon Bucket or Google Bucket name"},
    )
