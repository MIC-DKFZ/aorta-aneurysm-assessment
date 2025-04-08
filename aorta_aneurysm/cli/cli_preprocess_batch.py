"""
SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and Institute of Radiology, Uniklinikum Erlangen, Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU).
SPDX-License-Identifier: CC BY-NC 4.0
"""

from pathlib import Path

from aorta_aneurysm.cli._base import CliBase
from aorta_aneurysm.util.paths import is_nifti_or_nrrd
from aorta_aneurysm.pre_and_post_process.model_input_output import image_to_model_input_image_batch

class CliPreprocessBatch(CliBase):
    name = "preprocess_batch"
    desc = "Batch preprocess segmentations."

    def register_args(self, parser):
        parser.add_argument("input_dir", type=str, help="Directory containing the nifti input images")
        parser.add_argument("output_dir_for_model", type=str, help="Output directory that will be fed to the model")
        parser.add_argument("-j", "--num_workers", type=int, default=1, help="Number of workers to use.")
        parser.add_argument("--seg", action="store_true", help="If the input is a label image, not an patient image.")
        parser.add_argument("--nocrash", action="store_true", help="If set, the program will not crash if an error occurs but will not preprocess problematic case.")
        parser.add_argument("--file_ending", type=str, default=".nii.gz", help="File ending of the images to be created. Default is .nii.gz")


    def run(self, args):
        print("[INFO] >> Preprocessing images...")
        input_paths = [x for x in Path(args.input_dir).glob("**/*") if is_nifti_or_nrrd(x)]
        output_dir = Path(args.output_dir_for_model)
        parents = set([x.parent for x in input_paths])
        print(f"[INFO] >> Found and will preprocess {len(parents)} directories:")
        for parent in parents:
            print(f"[INFO] \t\t{str(parent)}")
        image_to_model_input_image_batch(input_paths, output_dir, args.num_workers, verbose=True, is_label=args.seg, nocrash=args.nocrash, file_ending=args.file_ending)
        print(f"[INFO] >> Done. Output at {output_dir}.")
        print("[INFO] >> Done.")
