"""
SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and Institute of Radiology, Uniklinikum Erlangen, Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU).
SPDX-License-Identifier: CC BY-NC 4.0
"""

import shutil
from pathlib import Path

def image_stem(image_path: Path):
    if image_path.name.endswith('.nii.gz'):
        return image_path.name[:-len('.nii.gz')]
    else:
        return image_path.stem

def image_fileending(image_path: Path):
    if image_path.name.endswith('.nii.gz'):
        return '.nii.gz'
    else:
        return image_path.suffix

def is_nifti_or_nrrd(path: Path) -> bool:
    return path.is_file() and (path.name.endswith(".nii.gz") or path.name.endswith(".nii") or path.name.endswith(".nrrd"))
