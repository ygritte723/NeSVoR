"""Formatters of help info for the command line interface"""


import argparse


class Formatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawDescriptionHelpFormatter,
):
    def __init__(self, prog: str) -> None:
        super().__init__(prog, max_help_position=50, width=None)


class CommandHelpFormatter(Formatter, argparse.MetavarTypeHelpFormatter):
    pass


class MainHelpFormatter(Formatter):
    def _format_action(self, action):
        parts = super()._format_action(action)
        L = self._max_help_position - 2
        if action.nargs == 0:
            parts_list = parts.split()
            i = 0
            while parts_list[i][0] == "-":
                i += 1
            parts1 = " ".join(parts_list[:i])
            parts2 = " ".join(parts_list[i:])
            parts = "  " + parts1 + " " * (L - len(parts1)) + parts2 + "\n"
        if action.nargs == argparse.PARSER:
            parts_list = parts.split("\n")[1:]
            parts_list_combined = []
            i = 0
            while i < len(parts_list):
                if not parts_list[i]:
                    i += 1
                elif len(parts_list[i].split()) > 1:
                    parts_list_combined.append(parts_list[i])
                    i += 1
                else:
                    parts_list_combined.append(parts_list[i] + " " + parts_list[i + 1])
                    i += 2
            for i in range(len(parts_list_combined)):
                parts_list_list = parts_list_combined[i].split()
                parts1 = parts_list_list[0]
                parts2 = " ".join(parts_list_list[1:])
                parts_list_combined[i] = (
                    "  " + parts1 + " " * (L - len(parts1)) + parts2
                )
            parts = "\n".join(parts_list_combined) + "\n"
        return parts
