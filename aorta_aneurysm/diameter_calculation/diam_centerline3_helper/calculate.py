"""
SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and Institute of Radiology, Uniklinikum Erlangen, Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU).
SPDX-License-Identifier: CC BY-NC 4.0
"""

import copy
from typing import Any, Dict
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

from aorta_aneurysm.diameter_calculation.diam_centerline3_helper.calculate_helper import is_flipped_sanity_check
from aorta_aneurysm.diameter_calculation.diam_centerline3_helper.Segment import create_segments
from aorta_aneurysm.diameter_calculation.diam_centerline3_helper.Point import Point, centerline_dijkstra, remove_close_centerline_points, reduce_similar_centerline_points
from aorta_aneurysm.diameter_calculation.diam_centerline3_helper.calc_centerline import get_aorta_centerline
from aorta_aneurysm.util.image import image_resample_to_spacing, image_keep_at_most_2_axial_cc, get_xyz_array_from_image
from aorta_aneurysm.util.array import cc, calc_circularity, keep_largest_cc
from aorta_aneurysm.diameter_calculation.diam_centerline3_helper.Normal import Normal, calculate_robust_normals2
from aorta_aneurysm.diameter_calculation.diam_centerline3_helper.calc_plane_and_diameter import calc_plane_and_diameter
from aorta_aneurysm.diameter_calculation.diam_centerline3_helper.categorize_planes import categorize_planes

def calculate(
        seg: sitk.Image, 
        fast: bool = False,
        verbose: bool = False,
        rm2: bool = False,
    ) -> Dict[Any,Any]:
    """
    Calculate the diameters of the aorta (& planes etc) using the 'diam_centerline3' updated method.

    The seg should be in the format of the output of the model (because then it's correctly oriented)
    
    Parameters
    ----------
    seg : sitk.Image
        Segmentation.
    fast : bool, optional
        Whether to use a faster version. Will take 1/27th of the time, by default False.
    verbose : bool, optional
        Whether to print verbose output, by default False.
    rm2 : bool, optional
        Allow a variance of 2mm in the diameter calculation, by default False.
    """

    ## Find the minimum spacing of the segmentation
    original_spacing = seg.GetSpacing()
    min_sp = min(original_spacing)

    ## Resample to isotropic spacing
    seg_iso = image_resample_to_spacing(seg, [min_sp, min_sp, min_sp], is_label=True)

    ## Keep only the 2 largest connected components at every slice
    seg_iso_clean = image_keep_at_most_2_axial_cc(seg_iso)

    ## Get numpy array
    seg_arr   = get_xyz_array_from_image(seg_iso_clean)

    is_z_flipped = is_flipped_sanity_check(seg_iso_clean)
    if is_z_flipped:
        seg_arr = np.flip(seg_arr, axis=2)

    image_shape_iso_xyz = seg_arr.shape

    ## Centerline and normals
    centerline_points, _ = get_aorta_centerline(seg_arr, min_sp)
    centerline_normals = calculate_robust_normals2(centerline_points, window_size=2)
    centerline_points_original = copy.deepcopy(centerline_points)
    centerline_normals_original = copy.deepcopy(centerline_normals)

    ## Reduce even further (best to happen after normals calculation)
    centerline_points_new = reduce_similar_centerline_points(centerline_points)
    centerline_normals_new = [n for p,n in zip(centerline_points, centerline_normals) if p in centerline_points_new]
    centerline_points = centerline_points_new
    centerline_normals = centerline_normals_new
    print(f"Centerline normals count after reducing similar: {len(centerline_points)}") if verbose else None

    # Keep the first and last at each direction
    cpoints_sorted_x = sorted(centerline_points, key=lambda p: p.x)
    cpoints_sorted_y = sorted(centerline_points, key=lambda p: p.y)
    cpoints_sorted_z = sorted(centerline_points, key=lambda p: p.z)
    centerline_points_new2 = []
    for cpoint in cpoints_sorted_x[0:5] + cpoints_sorted_x[-5:]:
        if cpoint not in centerline_points_new2:
            centerline_points_new2.append(cpoint)
    for cpoint in cpoints_sorted_y[0:7] + cpoints_sorted_y[-7:]:
        if cpoint not in centerline_points_new2:
            centerline_points_new2.append(cpoint)
    for cpoint in cpoints_sorted_z[0:7] + cpoints_sorted_z[-7:]:
        if cpoint not in centerline_points_new2:
            centerline_points_new2.append(cpoint)
    # Keep 12 extra
    rest = [p for p in centerline_points if p not in centerline_points_new2]
    step = len(rest) / 12
    for i in range(12):
        idx = int(i * step)
        if idx < len(rest):
            centerline_points_new2.append(rest[idx])
    # Make them in the same order as centerline_points
    centerline_points_new2 = [p for p in centerline_points if p in centerline_points_new2]
    centerline_normals_new2 = [n for p,n in zip(centerline_points, centerline_normals) if p in centerline_points_new2]
    centerline_points = centerline_points_new2
    centerline_normals = centerline_normals_new2

    ## Find planes and diameters
    planes = []
    planes_slice_arrs = []
    planes_slice_2D_points = []
    planes_diameters = []
    for p,n in zip(centerline_points, centerline_normals):
        plane, slice, slice_2D_point, diameter = calc_plane_and_diameter(p, n, seg_arr, min_sp, fast=fast)
        print(f"p,n {'kept' if diameter!=-1 else 'disc'}: {p}, {n}") if verbose else None
        if diameter > 0:
            planes.append(plane)
            planes_slice_arrs.append(slice)
            planes_slice_2D_points.append(slice_2D_point)
            planes_diameters.append(diameter)
    planes = [p for p in planes if p is not None]
    planes_slice_arrs = [s for s in planes_slice_arrs if s is not None]
    planes_slice_2D_points = [p for p in planes_slice_2D_points if p is not None]
    planes_diameters = [d for d in planes_diameters if d != -1]
    print(f"Planes kept: {len(planes)}") if verbose else None

    if rm2:
        ## Allow a variance of 2mm
        planes_diameters = [(d-2) for d in planes_diameters]

    ## Categorize
    planes_categorization = categorize_planes(planes, voxel_size=min_sp, image_shape=seg_arr.shape)

    max_diameter_asc, max_diameter_dsc, max_diameter_arch = -1, -1, -1
    max_diam_plane_asc, max_diam_plane_dsc, max_diam_plane_arch = None, None, None
    max_diam_slice_arr_asc, max_diam_slice_arr_dsc, max_diam_slice_arr_arch = None, None, None
    max_diam_slice_point_2D_asc, max_diam_slice_point_2D_dsc, max_diam_slice_point_2D_arch = None, None, None   

    for plane, pl_slice_arr, pl_slice_point_2D, pl_diam, pl_cat in zip(
        planes, planes_slice_arrs, planes_slice_2D_points, planes_diameters, planes_categorization
    ):
        if pl_cat == "ASC":
            if pl_diam > max_diameter_asc:
                max_diameter_asc = pl_diam
                max_diam_plane_asc = plane
                max_diam_slice_arr_asc = pl_slice_arr
                max_diam_slice_point_2D_asc = pl_slice_point_2D
        elif pl_cat == "DSC":
            if pl_diam > max_diameter_dsc:
                max_diameter_dsc = pl_diam
                max_diam_plane_dsc = plane
                max_diam_slice_arr_dsc = pl_slice_arr
                max_diam_slice_point_2D_dsc = pl_slice_point_2D
        elif pl_cat == "ARCH":
            if pl_diam > max_diameter_arch:
                max_diameter_arch = pl_diam
                max_diam_plane_arch = plane
                max_diam_slice_arr_arch = pl_slice_arr
                max_diam_slice_point_2D_arch = pl_slice_point_2D

    ## Fix: 3v4.10 Make sure there are no ASC points and DSC points close to each other
    asc_points = []
    dsc_points = []
    for plane, pl_cat in zip(planes, planes_categorization):
        if pl_cat == 'ASC':
            asc_points.append(plane.point)
        elif pl_cat == 'DSC':
            dsc_points.append(plane.point)
    # Check distances
    should_be_dismissed = False
    for asc_point in asc_points:
        for dsc_point in dsc_points:
            d = asc_point.distance_to(dsc_point)
            if d < 25:
                should_be_dismissed = True
    if should_be_dismissed:
        print('> Dismissed case because ASC and DSC points are too close')
        return {
            "seg_iso_clean": seg_iso_clean,
            "min_sp": min_sp,
            "original_spacing": original_spacing,
            "image_shape_iso_xyz": image_shape_iso_xyz,
            "max_diameter_asc": -1,
            "max_diameter_dsc": -1,
            "max_diameter_arch": -1,
            "max_diam_plane_asc": None,
            "max_diam_plane_dsc": None,
            "max_diam_plane_arch": None,
            "max_diam_slice_arr_asc": None,
            "max_diam_slice_arr_dsc": None,
            "max_diam_slice_arr_arch": None,
            "max_diam_slice_point_2D_asc": None,
            "max_diam_slice_point_2D_dsc": None,
            "max_diam_slice_point_2D_arch": None,
            "planes": [],
            "planes_diameters": [],
            "planes_categorization": [],
            "centerline_points_original": centerline_points_original,
            "centerline_normals_original": centerline_normals_original,
            "is_z_flipped": is_z_flipped,
        }

    return {
        "seg_iso_clean": seg_iso_clean,
        "min_sp": min_sp,
        "original_spacing": original_spacing,
        "image_shape_iso_xyz": image_shape_iso_xyz,
        "max_diameter_asc": max_diameter_asc,
        "max_diameter_dsc": max_diameter_dsc,
        "max_diameter_arch": max_diameter_arch,
        "max_diam_plane_asc": max_diam_plane_asc,
        "max_diam_plane_dsc": max_diam_plane_dsc,
        "max_diam_plane_arch": max_diam_plane_arch,
        "max_diam_slice_arr_asc": max_diam_slice_arr_asc,
        "max_diam_slice_arr_dsc": max_diam_slice_arr_dsc,
        "max_diam_slice_arr_arch": max_diam_slice_arr_arch,
        "max_diam_slice_point_2D_asc": max_diam_slice_point_2D_asc,
        "max_diam_slice_point_2D_dsc": max_diam_slice_point_2D_dsc,
        "max_diam_slice_point_2D_arch": max_diam_slice_point_2D_arch,
        "planes": planes,
        "planes_diameters": planes_diameters,
        "planes_categorization": planes_categorization,
        "centerline_points_original": centerline_points_original,
        "centerline_normals_original": centerline_normals_original,
        "is_z_flipped": is_z_flipped,
    }