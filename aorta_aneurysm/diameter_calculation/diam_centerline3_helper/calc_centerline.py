"""
SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and Institute of Radiology, Uniklinikum Erlangen, Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU).
SPDX-License-Identifier: CC BY-NC 4.0
"""

import copy
from typing import Tuple, List

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from scipy.ndimage import binary_erosion, binary_fill_holes, binary_closing

from aorta_aneurysm.diameter_calculation.diam_centerline3_helper.calc_centerline_helper import extend_seg_by_mirroring_for_centerline_calc, handle_small_boxes, remove_random_small_pieces
from aorta_aneurysm.diameter_calculation.diam_centerline3_helper.Point import Point, centerline_dijkstra, remove_close_centerline_points
from aorta_aneurysm.util.image import image_resample_to_spacing, image_keep_at_most_2_axial_cc, get_xyz_array_from_image
from aorta_aneurysm.util.array import cc, calc_circularity, keep_largest_cc


def get_aorta_centerline(seg_arr: np.ndarray, voxel_size: float) -> Tuple[ List[Point], np.ndarray ]:
    """
    Get the centerline of the aorta from the segmentation array. 
    INPUT IN XYZ FORM!! USE np.transpose(arr, (2, 1, 0)) TO CONVERT FROM ZYX TO XYZ
    ALSO THIS EXPECTS THE SEGMENTATION TO BE CLEANED 
    (NO SMALL CONNECTED COMPONENTS (see: image_keep_at_most_2_axial_cc))
    AND ISOTROPIC (spacing voxel_size in all dimensions).
    LASTLY THIS EXPECTS THE PATIENT TO ALWAYS BE SITTING IN THE SAME POSITION, 
    AS DICTATED BY THE PREPROCESSING PIPELINE. I.E. FEED THE OUTPUT SEGMENTATION OF THE MODEL
    BUT ISOTROPIC, XYZ, AND CLEANED OF SMALL CONNECTED COMPONENTS .

    The main part goes over slices in the z-direction.
    It mirrors parts of the centerline so that the centerline extends to the whole structure if possible.

    Parameters
    ----------
    seg_arr : np.ndarray
        Segmentation array. See description above.
    voxel_size : float
        Voxel size in mm of the isotropic image.
    
    Returns
    -------
    Tuple[ List[Point], np.ndarray ]
        List of centerline points and the centerline array.
    """

    seg_arr_copy = copy.deepcopy(seg_arr)
    seg_arr_copy = binary_closing(seg_arr_copy).astype(np.uint8)
    seg_arr_copy = binary_fill_holes(seg_arr_copy).astype(np.uint8)

    ## Mirror parts so that the centerline extends to the end of the structure
    seg_arr_copy = extend_seg_by_mirroring_for_centerline_calc(seg_arr_copy)

    ## Remove potential mis-segmentations far apart
    ## TODO: DOUBLE CHECK
    seg_arr_copy = remove_random_small_pieces(seg_arr_copy, voxel_size)

    ## Skeletonize
    centerline_arr = skeletonize(seg_arr_copy)

    ### Fix 3v4.10: THE ABOVE SOMEHOW STOPPED WORKING, SO I'M USING THIS INSTEAD
    centerline_points = [Point(*p) for p in np.argwhere(centerline_arr == 1)]
    centerline_points = [p for p in centerline_points if seg_arr[p.as_np()[0], p.as_np()[1], p.as_np()[2]] == 1]
    
    ## Reduce points using dijkstra and distance
    centerline_points = centerline_dijkstra(centerline_points, voxel_size, distance_to_check=3)
    centerline_points = remove_close_centerline_points(centerline_points, 2)

    centerline_points = handle_small_boxes(seg_arr_copy, centerline_points, voxel_size)

    ## Fix 3v4.10: ALSO REDO HERE
    centerline_points = [p for p in centerline_points if seg_arr[p.as_np()[0], p.as_np()[1], p.as_np()[2]] == 1]

    ## Update array from points
    centerline_arr = np.zeros_like(seg_arr)
    for p in centerline_points:
        centerline_arr[p.x, p.y, p.z] = 1

    return centerline_points, centerline_arr