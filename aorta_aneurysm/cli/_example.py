"""
SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and Institute of Radiology, Uniklinikum Erlangen, Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU).
SPDX-License-Identifier: CC BY-NC 4.0
"""

from aorta_aneurysm.cli._base import CliBase

class CliExample(CliBase):
    name = "example"
    desc = "Example pipeline."

    def register_args(self, parser):
        parser.add_argument("parameter1", type=str, help="Help message")


    def run(self, args):
        parameter1 = args.parameter1
        ...
