#!/usr/bin/env python

# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and Institute of Radiology, Uniklinikum Erlangen, Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU).
# SPDX-License-Identifier: CC BY-NC 4.0

import argparse, sys, os, warnings
from pathlib import Path

from aorta_aneurysm.cli.cli_preprocess_batch import CliPreprocessBatch
from aorta_aneurysm.cli.cli_diam_batch import CliDiamBatch

def main():
    sys.path.append(Path(__file__).resolve().parent.parent.as_posix())

    if os.getenv("DEBUG", "0") == "0":
        warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(
                description="Aorta aneurysm diameter calculation for abbreviated DCE-MRI. Check each subcommand with '-h' for more help.", 
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(dest="command")

    ## Add all the subcommands here
    clis = [
        CliPreprocessBatch(subparsers),
        CliDiamBatch(subparsers),
    ]

    ## Parse
    args = parser.parse_args()
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    ## Call subcommand
    name_to_cli = {cli.name: cli for cli in clis}
    cli = name_to_cli[args.command]
    cli.run(args)

if __name__=="__main__":
    main()