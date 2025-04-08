"""
SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and Institute of Radiology, Uniklinikum Erlangen, Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU).
SPDX-License-Identifier: CC BY-NC 4.0
"""

import os, copy
from typing import List

import numpy as np
from skimage.morphology import skeletonize

from aorta_aneurysm.diameter_calculation.diam_centerline3_helper.Point import Point
from aorta_aneurysm.util.array import cc, calc_circularity, erode_only_in_xy

def extend_seg_by_mirroring_for_centerline_calc(seg_arr: np.ndarray) -> np.ndarray:
    """
    Go over slices in the z-direction.
    It mirrors parts of the centerline so that the centerline extends to the whole structure if possible.
    """
    seg_arr_copy = copy.deepcopy(seg_arr)
    to_add = []

    ## FOR ONLY 1cc
    idx_1cc_slice = -1
    for idx_slice in range(seg_arr_copy.shape[2]):
        slice = seg_arr_copy[:, :, idx_slice]
        ccs = cc(slice, threshold_to_keep=10)

        if len(ccs) == 0:
            continue
        if len(ccs) == 2:
            break

        if idx_1cc_slice == -1:
            idx_1cc_slice = idx_slice
            continue

        cc_arr = ccs[0]
        circularity = calc_circularity(cc_arr)
        if circularity > 0.75:
            diff = idx_slice - idx_1cc_slice
            to_add_index = idx_1cc_slice - diff
            to_add.append((to_add_index, cc_arr))
        else:
            break

    ## FOR ONLY THE SMALLEST 2cc
    idx_2cc_slice = -1
    for idx_slice in range(seg_arr_copy.shape[2]):
        slice = seg_arr_copy[:, :, idx_slice]
        ccs = cc(slice, threshold_to_keep=10)

        if len(ccs) == 2:
            if idx_2cc_slice == -1:
                idx_2cc_slice = idx_slice
                continue
            
            cc_arr = ccs[1] # smallest
            circularity = calc_circularity(cc_arr)
            if circularity > 0.75:
                diff = idx_slice - idx_2cc_slice
                to_add_index = idx_2cc_slice - diff
                to_add.append((to_add_index, cc_arr))
            else:
                break

    ## FOR ONLY THE LARGEST 2cc
    idx_2cc_slice = -1
    for idx_slice in range(seg_arr_copy.shape[2]):
        slice = seg_arr_copy[:, :, idx_slice]
        ccs = cc(slice, threshold_to_keep=10)

        if len(ccs) == 2:
            if idx_2cc_slice == -1:
                idx_2cc_slice = idx_slice
                continue
            
            cc_arr = ccs[0] # largest
            circularity = calc_circularity(cc_arr)
            if circularity > 0.75:
                diff = idx_slice - idx_2cc_slice
                to_add_index = idx_2cc_slice - diff
                to_add.append((to_add_index, cc_arr))
            else:
                break

    for to_add_elem in to_add:
        idx, cc_arr = to_add_elem
        if idx>0:
            slice = seg_arr_copy[:, :, idx]
            and_op = np.logical_and(slice, cc_arr)
            if np.sum(and_op) == 0: # so we don't overlap with anything
                seg_arr_copy[:, :, idx] = np.logical_or(slice, cc_arr)

    return seg_arr_copy

def handle_small_boxes(seg_arr: np.ndarray, centerline_points: List[Point], voxel_size: float):
    """
    Handle small boxes in the segmentation array. This are typically not handled well by the regular centerline calculation.
    Happens sometimes at the ascending aorta.
    """

    for seg_c in cc(seg_arr, threshold_to_keep=10):
        coords = np.argwhere(seg_c == 1)
        
        ## Skip if somehow empty (shouldn't happen but you know..)
        if len(coords) == 0:
            continue
        
        centerline_points_coords = np.array([p.as_np() for p in centerline_points])
        
        ## Skip if there are centerline points in here
        if len(centerline_points_coords) > 0:
            matches = np.any(np.all(coords == centerline_points_coords[:, np.newaxis], axis=2), axis=1)
            if np.any(matches):
                continue

        # If it looks roughly square
        op_min_x, op_max_x = min([p[0] for p in coords]), max([p[0] for p in coords])
        op_min_z, op_max_z = min([p[2] for p in coords]), max([p[2] for p in coords])        
        if (op_max_z - op_min_z)*voxel_size < 30 and (op_max_x - op_min_x)*voxel_size > 15 and (op_max_x - op_min_x)*voxel_size < 55:
            # Try to erode in xy direction and add new centerline
            _eroded_xy_seg_s_arr = erode_only_in_xy(seg_c)
            _centerline_c_arr = skeletonize(_eroded_xy_seg_s_arr)
            centerline_points.extend( [Point(x[0], x[1], x[2]) for x in np.argwhere(_centerline_c_arr == 1)] )

    return centerline_points


def remove_random_small_pieces(seg_arr: np.ndarray, min_sp: float) -> np.ndarray:
    """Remove small mis-segmentations that are not connected to the main structure."""
    try:
        seg_b_arrs = cc(seg_arr, threshold_to_keep=50)
        if len(seg_b_arrs) >= 2:
            ## Bounding box for rest (all but the smallest connected component)
            points_rest_nested = [np.argwhere(seg_b_arr) for seg_b_arr in seg_b_arrs[:-1]]
            points_rest = np.concatenate(points_rest_nested)
            _min_x_rest = np.min(points_rest[:, 0])
            _max_x_rest = np.max(points_rest[:, 0])
            _min_y_rest = np.min(points_rest[:, 1])
            _max_y_rest = np.max(points_rest[:, 1])
            _min_z_rest = np.min(points_rest[:, 2])
            _max_z_rest = np.max(points_rest[:, 2])
            bb_rest = (_min_x_rest, _max_x_rest, _min_y_rest, _max_y_rest, _min_z_rest, _max_z_rest)
            area_large = (bb_rest[1] - bb_rest[0]) * (bb_rest[3] - bb_rest[2]) * (bb_rest[5] - bb_rest[4])

            ## Small bounding box
            points_small = np.argwhere(seg_b_arrs[-1])
            _min_x_small = np.min(points_small[:, 0])
            _max_x_small = np.max(points_small[:, 0])
            _min_y_small = np.min(points_small[:, 1])
            _max_y_small = np.max(points_small[:, 1])
            _min_z_small = np.min(points_small[:, 2])
            _max_z_small = np.max(points_small[:, 2])
            bb_small = (_min_x_small, _max_x_small, _min_y_small, _max_y_small, _min_z_small, _max_z_small)
            area_small = (bb_small[1] - bb_small[0]) * (bb_small[3] - bb_small[2]) * (bb_small[5] - bb_small[4])

            # Find distance between the two bounding boxes
            min_diff_x = min(
                abs(bb_rest[0] - bb_small[0]), abs(bb_rest[0] - bb_small[1]),
                abs(bb_rest[1] - bb_small[0]), abs(bb_rest[1] - bb_small[1]),
            )
            min_diff_y = min(
                abs(bb_rest[2] - bb_small[2]), abs(bb_rest[2] - bb_small[3]),
                abs(bb_rest[3] - bb_small[2]), abs(bb_rest[3] - bb_small[3]),
            )
            min_diff_z = min(
                abs(bb_rest[4] - bb_small[4]), abs(bb_rest[4] - bb_small[5]),
                abs(bb_rest[5] - bb_small[4]), abs(bb_rest[5] - bb_small[5]),
            )
            distance = np.sqrt(min_diff_x**2 + min_diff_y**2 + min_diff_z**2)

            ## Check that bb_small is not "touching" bb_large
            if not (
                ((bb_small[0] >= bb_rest[0] and bb_small[0] <= bb_rest[1]) or
                (bb_small[1] >= bb_rest[0] and bb_small[1] <= bb_rest[1])) and
                ((bb_small[2] >= bb_rest[2] and bb_small[2] <= bb_rest[3]) or
                (bb_small[3] >= bb_rest[2] and bb_small[3] <= bb_rest[3])) and
                ((bb_small[4] >= bb_rest[4] and bb_small[4] <= bb_rest[5]) or
                (bb_small[5] >= bb_rest[4] and bb_small[5] <= bb_rest[5]))
            ):
                ## If not too wide (covering both asc/dsc), skip
                if (bb_rest[3] - bb_rest[2]) * min_sp > 60:    
                    ## If not too massive, skip
                    if area_large >= 5e5 and area_small <= 2e4:
                        ## If not far apart, skip
                        if distance >= 40:
                            ## Remove small piece
                            seg_b_arrs_new = seg_b_arrs[:-1]
                            seg_arr_new = np.zeros_like(seg_arr)
                            for seg_b_arr in seg_b_arrs_new:
                                seg_arr_new += seg_b_arr
                            seg_arr = seg_arr_new
    except:
        pass
    finally:
        return seg_arr