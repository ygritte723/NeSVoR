"""nesvor entrypoints"""


import sys
import string
import logging
from parsers import main_parser

# Set paths for default arguments
DEFAULT_SUBJECT = "20035"
INPUT_PATH = f"/home/xzhon54/xinliuz/imgs/input/ori/{DEFAULT_SUBJECT}/"
MASK_PATH = f"/home/xzhon54/xinliuz/imgs/input/segs/{DEFAULT_SUBJECT}/"
OUTPUT_PATH = f"/home/xzhon54/xinliuz/imgs/output/{DEFAULT_SUBJECT}/"


def setup_default_args():
    """Set default command and arguments for debugging."""
    sys.argv.extend([
        "reconstruct",
        "--input-stacks", f"{INPUT_PATH}{DEFAULT_SUBJECT}_AX_T2w_fetal.nii",
                        f"{INPUT_PATH}{DEFAULT_SUBJECT}_CORO_T2w_fetal.nii",
                        f"{INPUT_PATH}{DEFAULT_SUBJECT}_SAG_T2w_fetal.nii",
        "--output-volume", f"{OUTPUT_PATH}volume_m_{DEFAULT_SUBJECT}_debug.nii.gz",
        "--bias-field-correction",
        "--stack-masks", f"{MASK_PATH}{DEFAULT_SUBJECT}_AX_T2w_fetal_seg.nii.gz",
                         f"{MASK_PATH}{DEFAULT_SUBJECT}_CORO_T2w_fetal_seg.nii.gz",
                         f"{MASK_PATH}{DEFAULT_SUBJECT}_SAG_T2w_fetal_seg.nii.gz",
        "--output-resolution", "0.8"
    ])
    
    
def main() -> None:
    parser, subparsers = main_parser()  # Initialize parser early

    if len(sys.argv) == 1:
        # print help if no args are provided
        parser.print_help(sys.stdout)
        setup_default_args()

    if len(sys.argv) == 2 and sys.argv[-1] in subparsers.choices:
        subparsers.choices[sys.argv[-1]].print_help(sys.stdout)
        return
    # parse args
    args = parser.parse_args()
    print('args:', args)
    run(args)


def run(args) -> None:
    import torch
    # from . import commands
    import commands
    from nesvor import utils

    # setup logger
    if args.debug:
        args.verbose = 2
    utils.setup_logger(args.output_log, args.verbose)
    
    # Setup device
    device = torch.device(args.device if args.device >= 0 else "cpu")
    if device.type == "cpu":
        logging.warning(
            "NeSVoR is running in CPU mode. The performance will be suboptimal. Try to use a GPU instead."
        )
    args.device = device
    # setup seed
    utils.set_seed(args.seed)

    # execute command
    command_class = "".join(string.capwords(w) for w in args.command.split("-"))
    getattr(commands, command_class)(args).main()


if __name__ == "__main__":
    main()
