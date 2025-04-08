"""
SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and Institute of Radiology, Uniklinikum Erlangen, Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU).
SPDX-License-Identifier: CC BY-NC 4.0
"""

import itertools
from typing import Tuple

import SimpleITK as sitk
import numpy as np

from aorta_aneurysm.diameter_calculation.diam_centerline3_helper.Point import Point, rotate_point
from aorta_aneurysm.diameter_calculation.diam_centerline3_helper.Normal import Normal
from aorta_aneurysm.diameter_calculation.diam_centerline3_helper.Plane import Plane, slice_volume_with_plane
from aorta_aneurysm.util.array import cc, calc_circularity

def __keep_only_part_of_slice_that_has_overlap_with_point(x : int, y : int, slice_arr: np.ndarray) -> np.ndarray:
    """
    From a 2D slice keep only the part that has overlap with a point (out of all possible components)
    """
    slice_arrs = cc(slice_arr, threshold_to_keep=50) # thres arbitrary, just in case
    for s in slice_arrs:
        s_points = [(x, y) for x, y in zip(*np.where(s))]
        if (x, y) in s_points:
            return s
    return np.zeros_like(slice_arr)

def __calculate_slice_diameter(slice_arr: np.ndarray, voxel_size: float) -> float:
    slice_arr_yx = np.transpose(slice_arr, (1, 0)).astype(np.uint8)
    slice_seg = sitk.GetImageFromArray(slice_arr_yx)
    slice_seg.SetSpacing((voxel_size, voxel_size))
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.ComputeFeretDiameterOn()
    stats.Execute(slice_seg)
    max_diameter = stats.GetFeretDiameter(1)
    return max_diameter

def calc_plane_and_diameter(
    point: Point,
    normal: Normal,
    seg_arr: np.ndarray,
    voxel_size: float,
    fast: bool = False,
    domain: str = 'aorta',
) -> Tuple[ Plane, np.ndarray, Point, float ]:
    """
    Calculate the best plane, corresponding slice, and diameter for a given point and normal.

    Parameters
    ----------
    point : Point
        The point to calculate the plane and diameter.
    normal : Normal
        The normal of the plane.
    seg_arr : np.ndarray
        The segmentation array.
    voxel_size : float
        The voxel size.
    fast : bool
        Whether to use a faster version. Will take 1/27th of the time

    Returns
    -------
    Tuple[ Plane, np.ndarry, float ]:
        The best plane, corresponding 2D slice, corresponding 2D point, and diameter.
    """
    n_orig_x, n_orig_y, n_orig_z = normal.as_np_normalized().tolist()
    
    best_circularity = -1
    best_plane = None
    best_slice = None
    best_slice_new_point = None # after rotation/translation

    checked = set()

    iterables_dist = [ -0.15, 0, 0.15] if not fast else [0]
    it = [(0,0,0)] + [(dx, dy, dz) for dx, dy, dz in itertools.product(iterables_dist, repeat=3) if dx != 0 or dy != 0 or dz != 0] # do 0,0,0 first
    for dx, dy, dz in it:
        normal_variation = Normal(n_orig_x + dx, n_orig_y + dy, n_orig_z + dz)
        
        ## Check if basically already handled
        str_repr = ",".join([ str(q) for q in [round(v,1) for v in normal_variation.as_np_normalized()] ])
        if str_repr in checked:
            continue
        checked.add(str_repr)

        plane_variation = Plane(point, normal_variation)

        if not normal_variation.is_ok():
            continue
        
        # Get image slice
        slice_arr, R, translation = slice_volume_with_plane(seg_arr, 
                                                            plane_variation, 
                                                            return_rotation_and_translation=True,
                                                            is_seg=True)
        new_plane_point_2D_np = rotate_point(point, R).as_np()[0:2] + translation

        # Remove other components
        slice_arr = __keep_only_part_of_slice_that_has_overlap_with_point(new_plane_point_2D_np[0], 
                                                                        new_plane_point_2D_np[1], 
                                                                        slice_arr)

        # Calculate circularity
        circularity = calc_circularity(slice_arr)
        
        if domain=='aorta' and circularity < 0.45 and best_circularity == -1:
            break # No reason to continue, it's not getting much better
        elif domain.startswith('pa') and circularity < 0.45 and best_circularity == -1:
            break
        
        if circularity > best_circularity:
            best_circularity = circularity
            best_plane = plane_variation
            best_slice = slice_arr
            best_slice_new_point = Point(new_plane_point_2D_np[0], new_plane_point_2D_np[1], 0)
            if dx == 0 and dy == 0 and dz == 0 and circularity > 0.90:
                break # Seems good enough at expect one

    ## Calculate diameter if ok
    diameter = -1
    if best_slice is not None:
        if domain == 'aorta':
            n_x, n_y, n_z = best_plane.normal.as_np_normalized().tolist()
            if abs(n_z) < 0.5+1e-6:
                thres = 0.74 # probably arch
            else:
                thres = 0.84
            if best_circularity > thres:
                diameter = __calculate_slice_diameter(best_slice, voxel_size)
        elif domain == 'pa':
            # print(f"best_circularity: {best_circularity:.2f}")
            if best_circularity > 0.64:
                diameter = __calculate_slice_diameter(best_slice, voxel_size)
        elif domain == 'pa-alt':
            # print(f"best_circularity: {best_circularity:.2f}")
            if best_circularity > 0.54:
                diameter = __calculate_slice_diameter(best_slice, voxel_size)

    return best_plane, best_slice, best_slice_new_point, diameter