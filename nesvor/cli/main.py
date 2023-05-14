"""nesvor entrypoints"""


import sys
import string
from .parsers import main_parser


def main() -> None:
    parser, subparsers = main_parser()
    # print help if no args are provided
    if len(sys.argv) == 1:
        parser.print_help(sys.stdout)
        return
    if len(sys.argv) == 2:
        if sys.argv[-1] in subparsers.choices:
            subparsers.choices[sys.argv[-1]].print_help(sys.stdout)
            return
    # parse args
    args = parser.parse_args()

    run(args)


def run(args) -> None:
    import torch
    from . import commands
    from .. import utils

    # setup
    args.device = torch.device(args.device)
    utils.set_seed(args.seed)
    if args.debug:
        args.verbose = 2
    utils.setup_logger(args.output_log, args.verbose)
    # execute command
    command_class = "".join(string.capwords(w) for w in args.command.split("-"))
    getattr(commands, command_class)(args).main()


if __name__ == "__main__":
    main()
