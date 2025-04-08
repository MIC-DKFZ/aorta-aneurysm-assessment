"""
SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and Institute of Radiology, Uniklinikum Erlangen, Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU).
SPDX-License-Identifier: CC BY-NC 4.0
"""

import os, json
from pathlib import Path
from typing import List, Tuple, Union, Dict
from dataclasses import dataclass, asdict

import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

from aorta_aneurysm.diameter_calculation.diameter_calculation_result import DiameterCalculationResult
from aorta_aneurysm.diameter_calculation.diam_centerline3_helper.calculate import calculate
from aorta_aneurysm.util.image import image_resample_to_spacing, image_flip_with_array, image_seg_binarize
from aorta_aneurysm.util.paths import image_stem

def diam_centerline3v4(
        seg: Union[sitk.Image, str, Path], 
        save_and_delete_plot_path: Path = None,
        img: Union[sitk.Image, str, Path] = None,
        rm2: bool = False,
        add_plot_to_result: bool = False,
        metadata: Dict = None,
        fast: bool = False,
        verbose: bool = False,
        ) -> DiameterCalculationResult:
    """
    Calculate the diameter of a segmented object using the 'diam_centerline3' method.
    In short, calculate planes and normals and a bunch of optimizations to get the diameter.
    
    Parameters
    ----------
    seg : Union[sitk.Image, str, Path]
        Segmentation or its path.
    save_and_delete_plot_path : Path, optional
        Path to save and delete the plot, by default None.
    img: Union[sitk.Image, str, Path], optional
        Image or its path, by default None.
    rm2 : bool, optional
        Allow a variance of 2mm in the diameter calculation, by default False.
    add_plot_to_result : bool, optional
        Add the plot to the result, by default False.
    metadata : Dict, optional
        Metadata to be used for report, by default None.
    fast : bool, optional
        Whether to use a faster version. Will take 1/27th of the time, by default False.
    verbose : bool, optional
        Whether to print verbose output, by default False.
        
    Returns
    -------
    DiameterCalculationResult
        Diameter calculation result (see class for more info).
    """
    ## Load if path
    if isinstance(seg, str) or isinstance(seg, Path):
        patient_name = image_stem( Path(seg) )
        seg = sitk.ReadImage(str(seg))
        seg = image_seg_binarize(seg, 1)
    else:
        patient_name = "N/A"

    ## Collect metadata
    if metadata is None:
        patient_id = "N/A"
        study_id = "N/A"
        study_desc = "N/A"
        study_date = "N/A"
        picked_series_id = "N/A"
        picked_series_desc = "N/A"
    else:
        patient_name = metadata.get("(0010, 0010)", "N/A")
        patient_id = metadata.get("(0010, 0020)", "N/A")
        study_id = metadata.get("(0020, 000d)", "N/A")
        study_desc = metadata.get("(0008, 1030)", "N/A")
        study_date = metadata.get("(0008, 0020)", "N/A")
        picked_series_id = metadata.get("(0020, 000e)", "N/A")
        picked_series_desc = metadata.get("(0008, 103e)", "N/A")
    
    try:
        calculate_result = calculate(seg, fast=fast, verbose=verbose, rm2=rm2)

        seg_iso_clean           = calculate_result["seg_iso_clean"]
        min_sp                  = calculate_result["min_sp"]
        original_spacing        = calculate_result["original_spacing"]
        image_shape_iso_xyz     = calculate_result["image_shape_iso_xyz"]
        max_diameter_asc        = calculate_result["max_diameter_asc"]
        max_diameter_dsc        = calculate_result["max_diameter_dsc"]
        max_diameter_arch       = calculate_result["max_diameter_arch"]
        max_diam_plane_asc      = calculate_result["max_diam_plane_asc"]
        max_diam_plane_dsc      = calculate_result["max_diam_plane_dsc"]
        max_diam_plane_arch     = calculate_result["max_diam_plane_arch"]
        max_diam_slice_arr_asc  = calculate_result["max_diam_slice_arr_asc"]
        max_diam_slice_arr_dsc  = calculate_result["max_diam_slice_arr_dsc"]
        max_diam_slice_arr_arch = calculate_result["max_diam_slice_arr_arch"]
        max_diam_slice_point_2D_asc  = calculate_result["max_diam_slice_point_2D_asc"]
        max_diam_slice_point_2D_dsc  = calculate_result["max_diam_slice_point_2D_dsc"]
        max_diam_slice_point_2D_arch = calculate_result["max_diam_slice_point_2D_arch"]
        planes                  = calculate_result["planes"]
        planes_diameters        = calculate_result["planes_diameters"]
        planes_categorization   = calculate_result["planes_categorization"]
        centerline_points_original = calculate_result["centerline_points_original"]
        centerline_normals_original = calculate_result["centerline_normals_original"]
        is_z_flipped           = calculate_result["is_z_flipped"]

        # Axial slices
        max_diam_ax_slice_asc = (
            (max_diam_plane_asc.point.z * min_sp / original_spacing[2]) 
            if max_diameter_asc != -1 else -1
        )
        max_diam_ax_slice_dsc = (
            (max_diam_plane_dsc.point.z * min_sp / original_spacing[2]) 
            if max_diameter_dsc != -1 else -1
        )
        max_diam_ax_slice_arch = (
            (max_diam_plane_arch.point.z * min_sp / original_spacing[2]) 
            if max_diameter_arch != -1 else -1
        )

        if is_z_flipped:
            max_diam_ax_slice_asc = image_shape_iso_xyz[2] - max_diam_ax_slice_asc
            max_diam_ax_slice_dsc = image_shape_iso_xyz[2] - max_diam_ax_slice_dsc
            max_diam_ax_slice_arch = image_shape_iso_xyz[2] - max_diam_ax_slice_arch

        # Overall results (across ASC, DSC, ARCH)
        max_diam_overall, max_slice_overall = -1, -1
        for d,s in zip([max_diameter_asc, max_diameter_dsc, max_diameter_arch], 
                       [max_diam_ax_slice_asc, max_diam_ax_slice_dsc, max_diam_ax_slice_arch]):
            if d > max_diam_overall:
                max_diam_overall = d
                max_slice_overall = s

        other_info_analytical_results = [
            {
                "category": cat,
                "diameter": round(float(diam),1),
                "plane_point": plane.point.as_np().tolist(),
                "plane_normal": plane.normal.as_np().tolist(),
                "plane_normal_normalized": plane.normal.as_np_normalized().tolist(),
            }
            for plane,diam,cat in zip(planes, planes_diameters, planes_categorization)
        ]

        # Construct result
        result = DiameterCalculationResult(
            diameter=max_diam_overall, 
            slice_num=max_slice_overall, 
            diameter_asc=max_diameter_asc,
            slice_num_asc=max_diam_ax_slice_asc,
            diameter_dsc=max_diameter_dsc,
            slice_num_dsc=max_diam_ax_slice_dsc,
            diameter_arch=max_diameter_arch,
            slice_num_arch=max_diam_ax_slice_arch,
            other_info={
                "analytical_results": other_info_analytical_results,
                "patient_name": patient_name,
                "min_sp": float(min_sp),
                "is_z_flipped": is_z_flipped,
                "original_spacing": [float(x) for x in original_spacing],
                "image_shape_iso_xyz": [int(x) for x in image_shape_iso_xyz],
                "centerline_points_original": [p.as_np().tolist() for p in centerline_points_original],
                "centerline_normals_normalized_original": [n.as_np_normalized().tolist() for n in centerline_normals_original],
            }
        )

        ## Create figure if needed
        if save_and_delete_plot_path or add_plot_to_result:
            raise NotImplementedError("Plotting not implemented.")
        
        if add_plot_to_result and not save_and_delete_plot_path:
            result.plot = fig

        return result

    except Exception as e:
        print(f"Error in diam_centerline3 for {patient_name}: {e}")
        return DiameterCalculationResult(calculation_success=False, error=str(e))
