"""
SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and Institute of Radiology, Uniklinikum Erlangen, Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU).
SPDX-License-Identifier: CC BY-NC 4.0
"""

import os, json
from pathlib import Path

from aorta_aneurysm.cli._base import CliBase
from aorta_aneurysm.diameter_calculation.diameter_calculation import DIAMETER_METHOD_STR_TO_FUNC, diameter_results_to_csv, diameter_calculation_batch
from aorta_aneurysm.util.paths import is_nifti_or_nrrd, image_stem

class CliDiamBatch(CliBase):
    name = "diam_batch"
    desc = "Batch diameter calculation (could be used on segmentations)."

    def register_args(self, parser):
        parser.add_argument("segs_dir", type=str, help="Directory containing the segmentations.")
        parser.add_argument("output_csv", type=str, help="Output path to csv")
        parser.add_argument("--method", 
                                    type=str, default="diam_centerline3v4", 
                                    choices=list(DIAMETER_METHOD_STR_TO_FUNC.keys()),
                                    help="Method to use for the diameter calculation.")
        parser.add_argument("-j", "--num_workers", 
                                    type=int, default=1, 
                                    help="Number of workers to use.")
        parser.add_argument("--rm2", action="store_true", 
                                    help="Allow a variance of 2mm in the diameter calculation.")


    def run(self, args):
        seg_paths = [
            f for f in Path(args.segs_dir).iterdir() 
            if f.is_file() and is_nifti_or_nrrd(f)
        ]
        print(f"[INFO] >> Found {len(seg_paths)} segmentations for processing...")

        print("[INFO] >> Calculating diameters...")
        names = []
        results = []
        results.extend( diameter_calculation_batch(seg_paths, args.method, None, None, args.num_workers, verbose=True, rm2=args.rm2) )
        names.extend( [image_stem(f) for f in seg_paths] )
        print("[INFO] >> Saving results to csv...")
        diameter_results_to_csv(names, results, Path(args.output_csv))

        print("[INFO] >> Done.")
