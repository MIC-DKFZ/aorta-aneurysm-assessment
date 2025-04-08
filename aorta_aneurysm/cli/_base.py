"""
SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and Institute of Radiology, Uniklinikum Erlangen, Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU).
SPDX-License-Identifier: CC BY-NC 4.0
"""

import argparse

class CliBase:
    name = None # to override in subclass
    desc = None # to override in subclass

    def __init__(self, subparsers):
        self.subparsers = subparsers
        self.parser = subparsers.add_parser(self.name, help=self.desc, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.register_args(self.parser)

    def register_args(self, parser):
        raise NotImplementedError("Method register_args() must be implemented in subclass")

    def run(self, args):
        raise NotImplementedError("Method run() must be implemented in subclass")
