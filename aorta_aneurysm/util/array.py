"""
SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and Institute of Radiology, Uniklinikum Erlangen, Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU).
SPDX-License-Identifier: CC BY-NC 4.0
"""

import copy
from typing import List

import numpy as np
import cc3d
from scipy.ndimage import binary_erosion

def calc_circularity(binary_2D_seg_arr: np.ndarray, destructive : bool = False, largest_cc_only: bool = True) -> float:
    """
    Calculate how circular the largest connected component of a 2D binary segmentation is.

    Parameters
    ----------
    binary_2D_seg_arr : np.ndarray
        2D binary seg.
    destructive : bool
        If set to False a copy of the array will be made. Otherwise the seg will be edited directly for speed.
    largest_cc_only : bool
        If set to True only the largest connected component will be kept.
    
    Returns
    -------
    float
        Circularity in [0,1]    
    """
    ## Keep largest CC
    if largest_cc_only:
        binary_2D_seg_arr = keep_largest_cc(binary_2D_seg_arr, destructive=destructive)
    ## Actually calculate
    try:
        y, x = np.where(binary_2D_seg_arr == 1)
        centroid_x = np.mean(x)
        centroid_y = np.mean(y)
        area_shape = np.sum(binary_2D_seg_arr)
        radius = np.max(np.sqrt((x - centroid_x)**2 + (y - centroid_y)**2))
        area_circle = np.pi * radius**2
        if area_circle < 1e-6: # would have been division by zero
            return 0.0
        circularity = area_shape / area_circle
        return float(circularity)
    except:
        return 0.0

def keep_largest_cc(binary_seg_arr : np.ndarray, destructive : bool = False, connectivity=6) -> np.ndarray:
    """
    Keep only the largest connected component in a 2D/3D binary seg.

    Parameters
    ----------
    binary_seg_arr : np.ndarray
        2D/3D binary seg
    destructive : bool
        If set to False a copy of the array will be made. Otherwise the seg will be edited directly for speed.
    
    Returns
    -------
    np.ndarray 
        3D binary seg with only the largest connected component
    """
    binary_seg_arr_copy = binary_seg_arr if destructive else copy.deepcopy(binary_seg_arr)
    ## Get connected components
    labeled = cc3d.connected_components(binary_seg_arr_copy, connectivity=connectivity)
    labels, counts = np.unique(labeled, return_counts=True)
    ## Remove background
    labels, counts = labels[1:], counts[1:]
    ## Remove everything that is not the largest cc
    if len(labels) > 1:
        for l in labels[1:]:
            binary_seg_arr_copy[labeled == l] = 0
    ## Finish
    return binary_seg_arr_copy

def remove_small_axial_cc(binary_3D_seg_arr : np.ndarray, small_voxel_thres : int, destructive : bool = False) -> np.ndarray:
    """
    Calculate all connected components and remove the small ones (based on the threshold)

    Parameters
    ----------
    binary_3D_seg_arr : np.ndarray
        3D binary seg
    small_voxel_thres : int
        At which threshold to remove CCs
    destructive : bool
        If set to False a copy of the array will be made. Otherwise the seg will be edited directly for speed.
    
    Returns
    -------
    np.ndarray 
        3D binary seg with removed small connected components
    """
    binary_3D_seg_arr_copy = binary_3D_seg_arr if destructive else copy.deepcopy(binary_3D_seg_arr)
    ## Iterate slices
    for i in range(binary_3D_seg_arr_copy.shape[0]):   
        slice_arr = binary_3D_seg_arr_copy[i]
        ## Get connected components
        labeled = cc3d.connected_components(slice_arr, connectivity=6)
        labels, counts = np.unique(labeled, return_counts=True)
        ## Remove background
        labels, counts = labels[1:], counts[1:]
        ## Check that sth is there
        if len(labels) == 0:
            continue
        if len(labels) > 1:
            for l in labels:
                if counts[l-1] < small_voxel_thres:
                    slice_arr[labeled == l] = 0
            # Make binary again
            slice_arr[slice_arr > 0] = 1
            binary_3D_seg_arr_copy[i] = slice_arr
    ## Finish
    return binary_3D_seg_arr_copy

def keep_at_most_2_axial_cc(binary_3D_seg_arr : np.ndarray, destructive : bool = False) -> np.ndarray:
    """
    Keep only the 2 largest connected component in each axial slice of the 3D binary seg.

    Parameters
    ----------
    binary_3D_seg_arr : np.ndarray
        3D binary seg
    destructive : bool
        If set to False a copy of the array will be made. Otherwise the seg will be edited directly for speed.
    
    Returns
    -------
    np.ndarray 
        3D binary seg with only the largest connected component in each axial slice
    """
    binary_3D_seg_arr_copy = binary_3D_seg_arr if destructive else copy.deepcopy(binary_3D_seg_arr)
    ## Iterate slices
    for i in range(binary_3D_seg_arr_copy.shape[0]):   
        slice_arr = binary_3D_seg_arr_copy[i]
        ## Get connected components
        labeled = cc3d.connected_components(slice_arr, connectivity=6)
        labels, counts = np.unique(labeled, return_counts=True)
        ## Remove background
        labels, counts = labels[1:], counts[1:]
        ## Check that sth is there
        if len(labels) in [0,1,2]:
            continue
        if len(labels) > 2:
            for l in labels[2:]:
                slice_arr[labeled == l] = 0
            slice_arr[labeled == labels[1]] = 1
            binary_3D_seg_arr_copy[i] = slice_arr
    ## Finish
    return binary_3D_seg_arr_copy

def cc(binary_seg_arr : np.ndarray, threshold_to_keep : int = 0) -> List[np.ndarray]:
    """
    Calculate connected components of a 2D/3D binary seg. Get them as list of binary segs

    Parameters
    ----------
    binary_seg_arr : np.ndarray
        2D/3D binary seg
    threshold_to_keep : int, optional
        At which threshold to keep CCs
    
    Returns
    -------
    List[np.ndarray]
        List of 3D binary segs with connected components
    """
    connectivity = 6 if len(binary_seg_arr.shape) == 3 else 4
    result = []
    labeled = cc3d.connected_components(binary_seg_arr, connectivity=connectivity)
    labels, counts = np.unique(labeled, return_counts=True)
    labels, counts = zip(*sorted(zip(labels, counts), key=lambda x: x[1], reverse=True))
    labels = [label for label,c in zip(labels,counts) if label > 0 and c >= threshold_to_keep]
    ## Split
    for label in labels:
        new_seg = np.zeros_like(binary_seg_arr)
        new_seg[labeled == label] = 1
        result.append(new_seg)
    ## Finish
    return result


def erode_only_in_xy(data_xyz: np.ndarray, iterations: int = 'auto') -> np.ndarray:
    """
    Erode a 3D image only in the XY plane.

    Parameters
    ----------
    data_xyz : np.ndarray
        The 3D image in XYZ.
    iterations : int, optional
        The number of iterations for the erosion. Default is 'auto' and is automatically determined.

    Returns
    -------
    np.ndarray
        The eroded image.
    """
    if iterations == 'auto':
        # Find the z index of the highest voxel that is 1 in the arr
        z_max = np.max(np.where(data_xyz == 1)[2])
        z_min = np.min(np.where(data_xyz == 1)[2])
        z_len = z_max - z_min
        # Find the x and y index of the highest voxel that is 1 in the arr
        x_max = np.max(np.where(data_xyz == 1)[0])
        x_min = np.min(np.where(data_xyz == 1)[0])
        x_len = x_max - x_min
        y_max = np.max(np.where(data_xyz == 1)[1])
        y_min = np.min(np.where(data_xyz == 1)[1])
        y_len = y_max - y_min
        
        if z_len == 0:
            # Return gravity center
            # print("[WARNING] Returning gravity center")
            eroded = np.empty_like(data_xyz)
            eroded[np.average([x_min, x_max]).astype(int)][np.average([y_min, y_max]).astype(int)][z_min] = 1
            return eroded        
        if z_len <= 4:
            # Return basically gravity center
            # print("[WARNING] Returning basically center")
            eroded = np.empty_like(data_xyz)
            for z in range(z_min, z_max+1):
                eroded[np.average([x_min, x_max]).astype(int)][np.average([y_min, y_max]).astype(int)][z] = 1
            return eroded
        else:
            opposite_len = min(x_len, y_len)
            # Iterations are set so that the opposite len becomes a bit smaller than z_len
            iterations = (opposite_len - z_len)//2 + 1
            # iterations = 4
            # print(f"[INFO] Setting iterations to {iterations}")

    # Define a structuring element that affects only the x and y dimensions
    structure_2d = np.array([[0, 1, 0],
                             [1, 1, 1],
                             [0, 1, 0]])
    eroded = np.empty_like(data_xyz)
    for z in range(data_xyz.shape[2]):
        eroded[:, :, z] = binary_erosion(data_xyz[:, :, z], structure=structure_2d, iterations=iterations)
    return eroded