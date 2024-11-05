"""nesvor entrypoints"""


import sys
import string
import logging
from parsers import main_parser
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

def main() -> None:
    parser, subparsers = main_parser()  # Initialize parser early

    # print help if no args are provided
    if len(sys.argv) == 1:
        parser.print_help(sys.stdout)
        
        subject = "20035"
        input_path = f"/home/xzhon54/xinliuz/imgs/input/ori/{subject}/"
        mask_path = f"/home/xzhon54/xinliuz/imgs/input/segs/{subject}/"
        output_path = f"/home/xzhon54/xinliuz/imgs/output/{subject}/"

        
        # Insert default command and arguments into sys.argv
        sys.argv.extend([
            "reconstruct",
            "--input-stacks", f"{input_path}{subject}_AX_T2w_fetal.nii",
                            f"{input_path}{subject}_CORO_T2w_fetal.nii",
                            f"{input_path}{subject}_SAG_T2w_fetal.nii",
            "--output-volume", f"{output_path}volume_m_{subject}_debug.nii.gz",
            "--bias-field-correction",
            "--stack-masks", f"{mask_path}{subject}_AX_T2w_fetal_seg.nii.gz",
                             f"{mask_path}{subject}_CORO_T2w_fetal_seg.nii.gz",
                             f"{mask_path}{subject}_SAG_T2w_fetal_seg.nii.gz",
            "--output-resolution", "0.8"
        ])

    if len(sys.argv) == 2:
        if sys.argv[-1] in subparsers.choices:
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
    # setup device
    if args.device >= 0:
        args.device = torch.device(args.device)
    else:
        args.device = torch.device("cpu")
        logging.warning(
            "NeSVoR is running in CPU mode. The performance will be suboptimal. Try to use a GPU instead."
        )
    # setup seed
    utils.set_seed(args.seed)

    # execute command
    command_class = "".join(string.capwords(w) for w in args.command.split("-"))
    getattr(commands, command_class)(args).main()


if __name__ == "__main__":
    main()
