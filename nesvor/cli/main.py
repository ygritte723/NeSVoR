"""nesvor entrypoints"""


import sys
import torch
import string
from . import commands
from .parsers import main_parser
from ..utils import setup_logger, set_seed
from .. import __version__


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
    # parse args and setup
    args = parser.parse_args()
    args.device = torch.device(args.device)
    set_seed(args.seed)
    if args.debug:
        args.verbose = 2
    setup_logger(args.output_log, args.verbose)
    # execute command
    command_class = "".join(string.capwords(w) for w in args.command.split("-"))
    getattr(commands, command_class)(args).main()


if __name__ == "__main__":
    main()
