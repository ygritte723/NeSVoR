"""nesvor entrypoints"""


import argparse
import sys
from typing import Union, Sequence, Optional
import torch
import random
import numpy as np
import string
from . import commands
from .formatter import CommandHelpFormatter, MainHelpFormatter
from ..utils import setup_logger
from .. import __version__


# parents parsers


def build_parser_training() -> argparse.ArgumentParser:
    """arguments related to the training of INR"""
    _parser = argparse.ArgumentParser(add_help=False)
    parser = _parser.add_argument_group("model architecture")
    # hash grid encoding
    parser.add_argument(
        "--n-features-per-level",
        default=2,
        type=int,
        help="Length of the feature vector at each level.",
    )
    parser.add_argument(
        "--log2-hashmap-size",
        default=19,
        type=int,
        help="Max log2 size of the hash grid per level.",
    )
    parser.add_argument(
        "--level-scale",
        default=1.3819,
        type=float,
        help="Scaling factor between two levels.",
    )
    parser.add_argument(
        "--coarsest-resolution",
        default=16.0,
        type=float,
        help="Resolution of the coarsest grid in millimeter.",
    )
    parser.add_argument(
        "--finest-resolution",
        default=0.5,
        type=float,
        help="Resolution of the finest grid in millimeter.",
    )
    parser.add_argument(
        "--n-levels-bias",
        default=0,
        type=int,
        help="Number of levels used for bias field estimation.",
    )
    # implicit network
    parser.add_argument(
        "--depth", default=1, type=int, help="Number of hidden layers in MLPs."
    )
    parser.add_argument(
        "--width", default=64, type=int, help="Number of neuron in each hidden layer."
    )
    parser.add_argument(
        "--n-features-z",
        default=15,
        type=int,
        help="Length of the intermediate feature vector z.",
    )
    parser.add_argument(
        "--n-features-slice",
        default=16,
        type=int,
        help="Length of the slice embedding vector e.",
    )
    parser.add_argument(
        "--no-transformation-optimization",
        action="store_true",
        help="Disable optimization for rigid slice transfromation, i.e., the slice transformations are fixed",
    )
    parser.add_argument(
        "--no-slice-scale",
        action="store_true",
        help="Disable adaptive scaling for slices.",
    )
    parser.add_argument(
        "--no-pixel-variance",
        action="store_true",
        help="Disable pixel-level variance.",
    )
    parser.add_argument(
        "--no-slice-variance",
        action="store_true",
        help="Disable slice-level variance.",
    )
    parser = _parser.add_argument_group("model architecture (deformable part)")
    # deformable net
    parser.add_argument(
        "--deformable",
        action="store_true",
        help="Enable implicit deformation field.",
    )
    parser.add_argument(
        "--n-features-deform",
        default=8,
        type=int,
        help="Length of the deformation embedding vector.",
    )
    parser.add_argument(
        "--n-features-per-level-deform",
        default=4,
        type=int,
        help="Length of the feature vector at each level (deformation field).",
    )
    parser.add_argument(
        "--level-scale-deform",
        default=1.3819,
        type=float,
        help="Scaling factor between two levels (deformation field).",
    )
    parser.add_argument(
        "--coarsest-resolution-deform",
        default=32.0,
        type=float,
        help="Resolution of the coarsest grid in millimeter (deformation field).",
    )
    parser.add_argument(
        "--finest-resolution-deform",
        default=8.0,
        type=float,
        help="Resolution of the finest grid in millimeter (deformation field).",
    )

    # loss function
    parser = _parser.add_argument_group("loss and regularization")
    # rigid transformation
    parser.add_argument(
        "--weight-transformation",
        default=0.1,
        type=float,
        help="Weight of transformation regularization.",
    )
    # bias field
    parser.add_argument(
        "--weight-bias",
        default=100.0,
        type=float,
        help="Weight of bias field regularization.",
    )
    # image regularization
    parser.add_argument(
        "--image-regularization",
        default="edge",
        type=str,
        choices=["TV", "edge", "L2", "none"],
        help=(
            "Type of image regularization. `TV`: total variation (L1 regularization of image gradient); "
            "`edge`: edge-preserving regularization; "
            "`L2`: L2 regularization of image gradient; "
            "`none`: not image regularization."
        ),
    )
    parser.add_argument(
        "--weight-image",
        default=1.0,
        type=float,
        help="Weight of image regularization.",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=0.2,
        help=(
            "Parameter to define intensity of an edge in edge-preserving regularization."
            "The edge-preserving regularization becomes L1 when delta -> 0."
        ),
    )
    parser.add_argument(
        "--img-reg-autodiff",
        action="store_true",
        help="Use auto diff to compute the image graident in the image regularization. By default, a finite difference is used.",
    )
    # deformation regularization
    parser.add_argument(
        "--weight-deform",
        default=0.1,
        type=float,
        help="Weight of deformation regularization ",
    )

    # training
    parser = _parser.add_argument_group("training")
    parser.add_argument(
        "--learning-rate",
        default=5e-3,
        type=float,
        help="Learning rate of Adam optimizer.",
    )
    parser.add_argument(
        "--gamma",
        default=0.33,
        type=float,
        help="Multiplicative factor of learning rate decay.",
    )
    parser.add_argument(
        "--milestones",
        nargs="+",
        type=float,
        default=[0.5, 0.75, 0.9],
        help="List of milestones of learning rate decay. Must be in (0, 1) and increasing.",
    )
    parser.add_argument(
        "--n-iter", default=6000, type=int, help="Number of iterations for training."
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        help="Number of epochs for training. If provided, will ignore --n-iter",
    )
    parser.add_argument(
        "--batch-size", default=1024 * 4, type=int, help="Batch size for training."
    )
    parser.add_argument(
        "--n-samples",
        default=128 * 2,
        type=int,
        help="Number of sample for PSF during training.",
    )
    parser.add_argument(
        "--single-precision",
        action="store_true",
        help="use float32 training (default: float16/float32 mixed trainig)",
    )
    return _parser


def build_parser_inputs(
    input_stacks: Union[bool, str] = False,
    input_slices: Union[bool, str] = False,
    input_model: Union[bool, str] = False,
    input_volume: Union[bool, str] = False,
    segmentation: bool = False,
    bias_field: bool = False,
) -> argparse.ArgumentParser:
    """arguments related to input data"""
    _parser = argparse.ArgumentParser(add_help=False)
    parser = _parser.add_argument_group("input")
    # stack input
    if input_stacks:
        parser.add_argument(
            "--input-stacks",
            nargs="+",
            type=str,
            required=input_stacks == "required",
            help="Paths to the input stacks (NIfTI).",
        )
        if not segmentation:
            if not bias_field:
                parser.add_argument(
                    "--thicknesses",
                    nargs="+",
                    type=float,
                    help=(
                        "Slice thickness of each input stack. "
                        "If not provided, use the slice gap of the input stack. "
                        "If only a single number is provided, Assume all input stacks have the same thickness."
                    ),
                )
            parser.add_argument(
                "--stack-masks",
                nargs="+",
                type=str,
                help="Paths to masks of input stacks.",
            )
            parser.add_argument(
                "--volume-mask",
                type=str,
                help="Paths to a 3D mask which will be applied to each input stack.",
            )
            parser.add_argument(
                "--stacks-intersection",
                action="store_true",
                help="Only consider the region defined by the intersection of input stacks. Will be ignored if --volume-mask is provided.",
            )
    # slices input
    if input_slices:
        parser.add_argument(
            "--input-slices",
            type=str,
            required=input_slices == "required",
            help="Folder of the input slices.",
        )
    # input model
    if input_model:
        parser.add_argument(
            "--input-model",
            type=str,
            required=input_model == "required",
            help="Path to the trained NeSVoR model.",
        )
    # input volume
    if input_volume:
        parser.add_argument(
            "--input-volume",
            type=str,
            required=input_volume == "required",
            help="Path to the input 3D volume.",
        )
        parser.add_argument(
            "--volume-mask",
            type=str,
            help=(
                "Paths to a 3D mask of ROI in the volume. "
                "Will use the non-zero region of the input volume if not provided"
            ),
        )

    return _parser


def build_parser_outputs(
    output_volume: Union[bool, str] = False,
    output_slices: Union[bool, str] = False,
    simulate_slices: Union[bool, str] = False,
    output_model: Union[bool, str] = False,
    output_stack_masks: Union[bool, str] = False,
    output_corrected_stacks: Union[bool, str] = False,
    output_folder: Union[bool, str] = False,
    output_json: Union[bool, str] = True,
    **kwargs,
) -> argparse.ArgumentParser:
    """arguments related to ouptuts"""
    _parser = argparse.ArgumentParser(add_help=False)
    parser = _parser.add_argument_group("output")
    # output volume
    if output_volume:
        parser.add_argument(
            "--output-volume",
            type=str,
            required=output_volume == "required",
            help="Paths to the reconstructed volume",
        )
        parser.add_argument(
            "--output-resolution",
            default=0.8,
            type=float,
            help="Isotropic resolution of the reconstructed volume",
        )
        parser.add_argument(
            "--output-intensity-mean",
            default=700.0,
            type=float,
            help="mean intensity of the output volume",
        )
        parser.add_argument(
            "--inference-batch-size", type=int, help="batch size for inference"
        )
        parser.add_argument(
            "--n-inference-samples",
            type=int,
            help="number of sample for PSF during inference",
        )
        parser.add_argument(
            "--output-psf-factor",
            type=float,
            default=1.0,
            help="Determind the psf for generating output volume: FWHM = output-resolution * output-psf-factor",
        )
        parser.add_argument(
            "--sample-orientation",
            type=str,
            help="Path to a nii file. The sampled volume will be reoriented according to the transformatio in this file.",
        )
    # output slices
    if output_slices:
        parser.add_argument(
            "--output-slices",
            required=output_slices == "required",
            type=str,
            help="Folder to save the motion corrected slices",
        )
    # simulate slices
    if simulate_slices:
        parser.add_argument(
            "--simulated-slices",
            required=simulate_slices == "required",
            type=str,
            help="Folder to save the simulated slices from the reconstructed volume",
        )
    # output model
    if output_model:
        parser.add_argument(
            "--output-model",
            type=str,
            required=output_model == "required",
            help="Path to save the output model (.pt)",
        )
    if output_volume or simulate_slices:
        parser.add_argument(
            "--sample-mask",
            type=str,
            help="3D Mask for sampling INR. If not provided, will use a mask esitmated from the input data.",
        )
    # output seg masks
    if output_stack_masks:
        parser.add_argument(
            "--output-stack-masks",
            type=str,
            nargs="+",
            required=output_stack_masks == "required",
            help="Path to the output folder or list of pathes to the output masks",
        )
    if output_corrected_stacks:
        parser.add_argument(
            "--output-corrected-stacks",
            type=str,
            nargs="+",
            required=output_corrected_stacks == "required",
            help="Path to the output folder or list of pathes to the output corrected stacks",
        )
    if output_json:
        parser.add_argument(
            "--output-json",
            type=str,
            help="Path to a json file for saving the inputs and results of the command.",
        )
    if output_folder:
        parser.add_argument(
            "--output-folder",
            type=str,
            required=output_folder == "required",
            help="Path to save outputs.",
        )

    update_defaults(_parser, **kwargs)
    return _parser


def build_parser_svort() -> argparse.ArgumentParser:
    """arguments related to rigid registration before reconstruction"""
    _parser = argparse.ArgumentParser(add_help=False)
    parser = _parser.add_argument_group("rigid registration")
    parser.add_argument(
        "--registration",
        default="svort",
        type=str,
        choices=["svort", "svort-only", "svort-stack", "stack", "none"],
        help=(
            "The type of registration method applied before reconstruction. "
            "`svort`: try SVoRT and stack-to-stack registration and choose the one with better NCC; "
            "`svort-only`: only apply the SVoRT model"
            "`svort-stack`: only apply the stack transformations of SVoRT; "
            "`stack`: stack-to-stack rigid registration; "
            "`none`: no registration. "
            "[Note] The SVoRT model can be only applied to fetal brain data. "
        ),
    )
    parser.add_argument(
        "--svort-version",
        default="v2",
        type=str,
        choices=["v1", "v2"],
        help="version of SVoRT",
    )
    parser.add_argument(
        "--scanner-space",
        action="store_true",
        help=(
            "Perform registration in the scanner space. "
            "Default: register the data to the atlas space when svort or svort-stack are used."
        ),
    )
    return _parser


def build_parser_segmentation(optional: bool = False) -> argparse.ArgumentParser:
    """arguments related to 2D brain segmentaion/masking"""
    _parser = argparse.ArgumentParser(add_help=False)
    parser = _parser.add_argument_group("brain segmentation")
    if optional:
        parser.add_argument(
            "--segmentation",
            action="store_true",
            help="Perform fetal brain segmentation (brain ROI masking) for each input stack.",
        )
    parser.add_argument(
        "--batch-size-seg",
        type=int,
        default=64,
        help="batch size for segmentation",
    )
    parser.add_argument(
        "--no-augmentation-seg",
        action="store_true",
        help="disable inference data augmentation in segmentation",
    )
    parser.add_argument(
        "--dilation-radius-seg",
        type=float,
        default=1.0,
        help="dilation radius for segmentation mask in millimeter.",
    )
    parser.add_argument(
        "--threshold-small-seg",
        type=float,
        default=0.1,
        help=(
            "Threshold for removing small segmetation mask (between 0 and 1). "
            "A mask will be removed if its area < threshold * max area in the stack."
        ),
    )
    return _parser


def build_parser_bias_field_correction(
    optional: bool = False,
) -> argparse.ArgumentParser:
    """arguments related to N4 bias field correction"""
    _parser = argparse.ArgumentParser(add_help=False)
    parser = _parser.add_argument_group("N4 bias field correction")
    if optional:
        parser.add_argument(
            "--bias-field-correction",
            action="store_true",
            help="Perform bias field correction using the N4 algorithm.",
        )
    parser.add_argument(
        "--n-proc-n4",
        type=int,
        default=8,
        help="number of workers for the N4 algorithm.",
    )
    parser.add_argument(
        "--shrink-factor-n4",
        type=int,
        default=2,
        help="The shrink factor used to reduce the size and complexity of the image.",
    )
    parser.add_argument(
        "--tol-n4",
        type=float,
        default=0.001,
        help="The convergence threshold in N4.",
    )
    parser.add_argument(
        "--spline-order-n4",
        type=int,
        default=3,
        help="The order of B-spline.",
    )
    parser.add_argument(
        "--noise-n4",
        type=float,
        default=0.01,
        help="The noise estimate defining the Wiener filter.",
    )
    parser.add_argument(
        "--n-iter-n4",
        type=int,
        default=50,
        help="The maximum number of iterations specified at each fitting level.",
    )
    parser.add_argument(
        "--n-levels-n4",
        type=int,
        default=4,
        help="The maximum number of iterations specified at each fitting level.",
    )
    parser.add_argument(
        "--n-control-points-n4",
        type=int,
        default=4,
        help=(
            "The control point grid size in each dimension. "
            "The B-spline mesh size is equal to the number of control points in that dimension minus the spline order."
        ),
    )
    parser.add_argument(
        "--n-bins-n4",
        type=int,
        default=200,
        help="The number of bins in the log input intensity histogram.",
    )
    return _parser


def build_parser_assessment(**kwargs) -> argparse.ArgumentParser:
    """arguments related to image quality and motion assessment of input data"""
    _parser = argparse.ArgumentParser(add_help=False)
    parser = _parser.add_argument_group("stack assessment")
    parser.add_argument(
        "--metric",
        type=str,
        choices=["ncc", "matrix-rank", "volume", "iqa2d", "iqa3d", "none"],
        default="none",
        help=(
            "Metric for assessing input stacks. "
            "`ncc` (\u2191): cross correlaiton between adjacent slices; "
            "`matrix-rank` (\u2193): motion metric based on the rank of the data matrix; "
            "`volume` (\u2191): volume of the masked ROI; "
            "`iqa2d` (\u2191): image quality score generated by a 2D CNN (only for fetal brain), the score of a stack is the average score of the images in it; "
            "`iqa3d` (\u2191): image quality score generated by a 3D CNN (only for fetal brain); "
            "`none`: no metric. "
            "`\u2191` means a stack with a higher score will have a better rank."
        ),
    )
    parser.add_argument(
        "--filter-method",
        type=str,
        choices=["top", "bottom", "threshold", "percentage", "none"],
        default="none",
        help=(
            "Method to remove low-quality stacks. "
            "`top`: keep the top C stacks; "
            "`bottom`: remove the bottom C stacks; "
            "`threshold`: remove a stack if the metric is worse than C; "
            "`percentatge`: remove the bottom (num_stack * C) stacks; "
            "`none`: no filtering. The value of `C` is specified by --cutoff"
        ),
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        help="The cutoff value for filtering, i.e., the value `C` in --filter-method",
    )
    update_defaults(_parser, **kwargs)
    return _parser


def build_parser_volume_segmentation() -> argparse.ArgumentParser:
    """arguments related to 3D brain segmentation"""
    _parser = argparse.ArgumentParser(add_help=False)
    parser = _parser.add_argument_group("TWAI brain segmentation")
    parser.add_argument(
        "--ga",
        type=float,
        help="Gestational age at the time of acquisition of the fetal brain 3D MRI to be segmented.",
    )
    parser.add_argument(
        "--condition",
        type=str,
        choices=["Neurotypical", "Spina Bifida", "Pathological"],
        default="Neurotypical",
        help="Brain condition of the fetal brain 3D MRI to be segmented.",
    )
    parser.add_argument(
        "--bias-field-correction",
        action="store_true",
        help="Perform bias field correction before segmentation.",
    )
    return _parser


def build_parser_common() -> argparse.ArgumentParser:
    """miscellaneous arguments"""
    _parser = argparse.ArgumentParser(add_help=False)
    parser = _parser.add_argument_group("common")
    parser.add_argument("--device", type=int, default=0, help="Id of the GPU to use.")
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="level of verbosity: (0: warning/error, 1: info, 2: debug)",
    )
    parser.add_argument("--output-log", type=str, help="Path to the output log file")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--debug", action="store_true", help="Debug mode.")
    return _parser


def update_defaults(parser: argparse.ArgumentParser, **kwargs):
    # a helper function to update the default values in a parser
    parser.set_defaults(**kwargs)


# command parsers


def add_subcommand(
    subparsers: argparse._SubParsersAction,
    name: str,
    help: Optional[str],
    description: Optional[str],
    parents: Sequence,
) -> argparse.ArgumentParser:
    # a helper function to create a subcommand
    parser = subparsers.add_parser(
        name=name,
        help=help,
        description=description,
        parents=parents,
        formatter_class=CommandHelpFormatter,
        add_help=False,
    )
    parser.add_argument("-h", "--help", action="help", help=argparse.SUPPRESS)
    return parser


def build_command_reconstruct(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    # reconstruct
    parser_reconstruct = add_subcommand(
        subparsers,
        name="reconstruct",
        help="slice-to-volume reconstruction using NeSVoR",
        description=(
            "Use the NeSVoR algorithm to reconstuct a high-quality and coherent 3D volume from multiple stacks of 2D slices. "
            "This command can be applied to both rigid (e.g., brain) and non-rigid (e.g. uterus) motion. "
            "It also includes several optional preprocessing stpes: "
            "1: ROI masking / segmentation from each input stack with a CNN (only for fetal brain); "
            "2: N4 bias filed correction for each stack; "
            "3: Assess quality and motion of each stack, which can be used to rank and filter the data; "
            "4: Motion correction with SVoRT (only for fetal brain) or stack-to-stack registration. "
        ),
        parents=[
            build_parser_inputs(input_stacks=True, input_slices=True),
            build_parser_outputs(
                output_volume=True,
                output_slices=True,
                simulate_slices=True,
                output_model=True,
            ),
            build_parser_segmentation(optional=True),
            build_parser_bias_field_correction(optional=True),
            build_parser_assessment(),
            build_parser_svort(),
            build_parser_training(),
            build_parser_common(),
        ],
    )
    return parser_reconstruct


def build_command_sample_volume(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    # sample-volume
    parser_sample_volume = add_subcommand(
        subparsers,
        name="sample-volume",
        help="sample a volume from a trained NeSVoR model",
        description="sample a volume from a trained NeSVoR model",
        parents=[
            build_parser_inputs(input_model="required"),
            build_parser_outputs(
                output_volume="required",
                inference_batch_size=1024 * 4 * 8,
                n_inference_samples=128 * 2 * 2,
            ),
            build_parser_common(),
        ],
    )
    return parser_sample_volume


def build_command_sample_slices(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    # sample-slices
    parser_sample_slices = add_subcommand(
        subparsers,
        name="sample-slices",
        help="sample slices from a trained NeSVoR model",
        description="sample slices from a trained NeSVoR model",
        parents=[
            build_parser_inputs(input_slices="required", input_model="required"),
            build_parser_outputs(simulate_slices="required"),
            build_parser_common(),
        ],
    )
    return parser_sample_slices


def build_command_register(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    # register
    parser_register = add_subcommand(
        subparsers,
        name="register",
        help="slice-to-volume registration",
        description="Perform inital (rigid) motion correction using SVoRT (only for fetal brain) or stack-to-stack registration.",
        parents=[
            build_parser_inputs(input_stacks="required"),
            build_parser_outputs(output_slices="required"),
            build_parser_svort(),
            build_parser_common(),
        ],
    )
    return parser_register


def build_command_segment_stack(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    # segment-stack
    parser_segment_stack = add_subcommand(
        subparsers,
        name="segment-stack",
        help="2D fetal brain segmentation/masking",
        description=(
            "Segment the fetal brain ROI from each stack using a CNN model (MONAIfbs). "
            "See https://github.com/gift-surg/MONAIfbs for details. "
        ),
        parents=[
            build_parser_inputs(input_stacks="required", segmentation=True),
            build_parser_outputs(output_stack_masks="required"),
            build_parser_segmentation(optional=False),
            build_parser_common(),
        ],
    )
    return parser_segment_stack


def build_command_correct_bias_field(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    # correct-bias-field
    parser_correct_bias_field = add_subcommand(
        subparsers,
        name="correct-bias-field",
        help="bias field correction",
        description="Perform bias field correction for each input stack with the N4 algorithm.",
        parents=[
            build_parser_inputs(input_stacks="required", bias_field=True),
            build_parser_outputs(output_corrected_stacks="required"),
            build_parser_bias_field_correction(optional=False),
            build_parser_common(),
        ],
    )
    return parser_correct_bias_field


def build_command_assess(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    # assess
    parser_assess = add_subcommand(
        subparsers,
        name="assess",
        help="quality assessment of input stacks",
        description=(
            "Assess the quality and motion of each input stack. "
            "The output metrics can be used for determining the template stack or removing low-quality data"
        ),
        parents=[
            build_parser_inputs(input_stacks="required", bias_field=True),
            build_parser_outputs(),
            build_parser_assessment(metric="ncc"),
            build_parser_common(),
        ],
    )
    return parser_assess


def build_command_segment_volume(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    # segment-volume
    parser_segment_volume = add_subcommand(
        subparsers,
        name="segment-volume",
        help="3D fetal brain segmentation",
        description=(
            "TWAI brain segmentation of reconstructed 3D volume. Segmentation labels: "
            "1: white matter (excluding corpus callosum); "
            "2: intra-axial cerebrospinal fluid (CSF); "
            "3: cerebellum; "
            "4: extra-axial CSF; "
            "5: cortical gray matter; "
            "6: deep gray matter; "
            "7: brainstem; "
            "8: corpus callosum. "
            "Check out https://github.com/LucasFidon/trustworthy-ai-fetal-brain-segmentation for details."
        ),
        parents=[
            build_parser_inputs(input_volume="required"),
            build_parser_volume_segmentation(),
            build_parser_outputs(output_json=False, output_folder="required"),
            build_parser_common(),
        ],
    )
    return parser_segment_volume


def main() -> None:
    # main parser
    parser = argparse.ArgumentParser(
        prog="nesvor",
        description=f"NeSVoR: a toolkit for neural slice-to-volume reconstruction (v{__version__})",
        epilog="Run 'nesvor COMMAND --help' for more information on a command.\n\n"
        + "To learn more about NeSVoR, check out our repo at "
        + "https://github.com/daviddmc/NeSVoR",
        formatter_class=MainHelpFormatter,
    )
    parser.add_argument(
        "-v", "--version", action="version", version="%(prog)s v" + __version__
    )
    # commands
    subparsers = parser.add_subparsers(
        title="commands", metavar="COMMAND", dest="command"
    )
    build_command_reconstruct(subparsers)
    build_command_sample_volume(subparsers)
    build_command_sample_slices(subparsers)
    build_command_register(subparsers)
    build_command_segment_stack(subparsers)
    build_command_correct_bias_field(subparsers)
    build_command_assess(subparsers)
    build_command_segment_volume(subparsers)
    # print help if no args are provided
    if len(sys.argv) == 1:
        parser.print_help(sys.stdout)
        return
    if len(sys.argv) == 2:
        if sys.argv[-1] in subparsers.choices:
            subparsers.choices[sys.argv[-1]].print_help(sys.stdout)
            return
    # parse args and setup
    args = parser.parse_args()
    args.device = torch.device(args.device)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
    if args.debug:
        args.verbose = 2
    setup_logger(args.output_log, args.verbose)
    # execute command
    command_class = "".join(string.capwords(w) for w in args.command.split("-"))
    getattr(commands, command_class)(args).main()


if __name__ == "__main__":
    main()
