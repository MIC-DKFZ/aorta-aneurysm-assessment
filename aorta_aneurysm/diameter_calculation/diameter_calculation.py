"""
SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and Institute of Radiology, Uniklinikum Erlangen, Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU).
SPDX-License-Identifier: CC BY-NC 4.0
"""

from pathlib import Path
from typing import List

import SimpleITK as sitk
from typing import Union

from aorta_aneurysm.diameter_calculation.diameter_calculation_result import DiameterCalculationResult
from aorta_aneurysm.diameter_calculation.method_diam_centerline3v4 import diam_centerline3v4
from aorta_aneurysm.util.parallel import run_in_parallel
from aorta_aneurysm.util.paths import image_stem, is_nifti_or_nrrd

DIAMETER_METHOD_STR_TO_FUNC = {
    "diam_centerline3v4": diam_centerline3v4,
}

def diameter_calculation(
        seg: Union[sitk.Image,str,Path], 
        method: str = "diam_circ_seg",
        ) -> DiameterCalculationResult:
    """Calculate the diameter of the aorta segmentation.

    Parameters
    ----------
    seg : Union[sitk.Image,str,Path]
        3D binary segmentation of the aorta or its Path.
    method : str, optional
        The method to use for the diameter calculation. Default is "diam_circ_seg".

    Returns
    -------
    DiameterCalculationResult
        The result of the diameter calculation.
    """
    if isinstance(seg, (str, Path)):
        seg = sitk.ReadImage(str(seg))
    return DIAMETER_METHOD_STR_TO_FUNC[method](seg)

def diameter_calculation_batch(
        seg_paths: List[Path], 
        method: str = "diam_circ_seg",
        save_and_delete_plot_dir: Path = None,
        input_dir: Path = None,
        num_workers: int = 1,
        verbose: bool = False,
        rm2: bool = False,
        ) -> List[DiameterCalculationResult]:
    """Calculate the diameter of the aorta segmentation.

    Parameters
    ----------
    seg_paths : List[Path]
        Paths to 3D binary segmentation of the aorta.
    method : str, optional
        The method to use for the diameter calculation. Default is "diam_circ_seg".
    num_workers : int, optional
        The number of workers to use. Default is 1.
    verbose : bool, optional
        Whether to print progress. Default is False.
    input_dir : Path, optional
        The input patient images directory. Default is None.
    rm2 : bool, optional
        Allow a variance of 2mm in the diameter calculation. Default is False.

    Returns
    -------
    List[DiameterCalculationResult]
        The results of the diameter calculation.
    """
    method = DIAMETER_METHOD_STR_TO_FUNC[method]
    work = []
    for seg_path in seg_paths:
        w = [seg_path]
        if save_and_delete_plot_dir is not None:
            w.append(save_and_delete_plot_dir / f"{image_stem(seg_path)}.pdf")
        else:
            w.append(None)
        if input_dir is not None:
            name = image_stem(seg_path)
            candidates = [x for x in input_dir.iterdir() if is_nifti_or_nrrd(x) and name in image_stem(x)]
            if len(candidates) > 1:
                raise ValueError(f"Multiple candidates for {name}: {candidates}")
            w.append(candidates[0])
        else:
            w.append(None)
        w.append(rm2)
        work.append(tuple(w))
    return run_in_parallel(method, work, num_workers=num_workers, verbose=verbose)

def diameter_results_to_csv(names: List[str], results: List[DiameterCalculationResult], output_csv: Path):
    """Write the results of the diameter calculation to a csv file.

    Parameters
    ----------
    names : List[str]
        The names of the segmentations.
    results : List[DiameterCalculationResult]
        The results of the diameter calculation.
    output_csv : Path
        The path to the output csv file.
    """
    output_csv.parent.mkdir(exist_ok=True, parents=True)
    with open(output_csv, "w") as f:
        f.write("name,calculation_success,diameter,slice_num,error,diameter_asc,slice_num_asc,diameter_dsc,slice_num_dsc,diameter_arch,slice_num_arch\n")
        for name,result in zip(names,results):
            diam = round(result.diameter, 1) if result.diameter != -1 else -1
            diam_asc =  round(result.diameter_asc, 1) if result.diameter_asc != -1 else -1
            diam_dsc =  round(result.diameter_dsc, 1) if result.diameter_dsc != -1 else -1
            diam_arch = round(result.diameter_arch, 1) if result.diameter_arch != -1 else -1
            succ = 1 if result.calculation_success else 0
            f.write(f"{name},{succ},{diam},{result.slice_num},{result.error},{diam_asc},{result.slice_num_asc},{diam_dsc},{result.slice_num_dsc},{diam_arch},{result.slice_num_arch}\n")
